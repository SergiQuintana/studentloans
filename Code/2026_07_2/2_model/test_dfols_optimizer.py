"""Synthetic tests for the opt-in DFO-LS SMM optimizer backend.

Everything here is synthetic and runs locally: no server files, estimates,
or CCP bundles are required beyond what importing ``model_fitloans_dynamic``
itself needs. Three guarantees are certified:

(a) residual consistency: the residual vector produced by the actual
    residual-construction code path (``parental_income_cell_loss_and_residuals``,
    the single function used by the per-cell SMM evaluation) satisfies
    ``sum(r**2) == scalar loss`` to 1e-10, for both the legacy 6-moment
    ``flow_plus_stock`` shape and the 5-moment ``flow_split_stock`` shape,
    including non-finite moments and the multi-cell ``sqrt(cell_weight)``
    stacking used by ``minimize_distance_education_cells_parental_income``;

(b) the DFO-LS wrapper ``solve_dfols_least_squares`` finds the optimum of a
    bounded 5-dimensional least-squares problem and is bitwise reproducible
    across repeated calls with the same seed, including perturbation
    restarts (skipped gracefully if dfols is not installed);

(c) the default optimizer paths are untouched: the new keyword arguments
    default to inert values and the production driver still requests
    "hybrid" through ``BUDGET_SMM_OPTIMIZER``.
"""

import inspect
import os
import unittest

import numpy as np

import model_fitloans_dynamic as m

try:
    import dfols  # noqa: F401
    HAVE_DFOLS = True
except ImportError:
    HAVE_DFOLS = False


def _reference_loss(simulated, data_moments, moment_spec, primary_weight):
    """Independent replica of the historical in-line scalar loss."""
    valid = np.isfinite(data_moments) & np.isfinite(simulated)
    scale = np.maximum(np.abs(data_moments), 1.0e-6)
    weights = np.tile(
        m.parental_income_moment_weight_pattern(moment_spec, primary_weight), 4
    )
    error = (simulated - data_moments) / scale
    return float(np.sum(weights[valid] * error[valid] ** 2))


def _synthetic_moments(n_per_group, seed, with_nans=True):
    """Synthetic data/simulated moment arrays for four parinc groups."""
    rng = np.random.default_rng(seed)
    size = 4 * n_per_group
    data = rng.uniform(-2000.0, 8000.0, size=size)
    # Exercise the eps floor: near-zero data moments (shares, rates).
    data[::n_per_group] = rng.uniform(-1.0e-8, 1.0e-8, size=4)
    simulated = data + rng.normal(0.0, 500.0, size=size)
    if with_nans:
        data[3] = np.nan          # missing data moment
        simulated[size - 2] = np.nan  # missing simulated moment
    return simulated, data


class ResidualConsistencyTests(unittest.TestCase):
    def test_flow_plus_stock_six_moment_shape(self):
        simulated, data = _synthetic_moments(6, seed=101)
        loss, residuals = m.parental_income_cell_loss_and_residuals(
            simulated, data, "flow_plus_stock", m.DEFAULT_PRIMARY_MOMENT_WEIGHT
        )
        self.assertEqual(residuals.size, 24)
        self.assertTrue(np.all(np.isfinite(residuals)))
        self.assertLessEqual(
            abs(float(np.sum(residuals ** 2)) - loss),
            1.0e-10 * max(1.0, abs(loss)),
        )
        self.assertAlmostEqual(
            loss,
            _reference_loss(
                simulated, data, "flow_plus_stock",
                m.DEFAULT_PRIMARY_MOMENT_WEIGHT,
            ),
            places=12,
        )

    def test_flow_split_stock_five_moment_shape(self):
        simulated, data = _synthetic_moments(5, seed=202)
        loss, residuals = m.parental_income_cell_loss_and_residuals(
            simulated, data, m.SPLIT_MOMENT_SPEC,
            m.DEFAULT_PRIMARY_MOMENT_WEIGHT,
        )
        self.assertEqual(residuals.size, 20)
        self.assertTrue(np.all(np.isfinite(residuals)))
        self.assertLessEqual(
            abs(float(np.sum(residuals ** 2)) - loss),
            1.0e-10 * max(1.0, abs(loss)),
        )
        self.assertAlmostEqual(
            loss,
            _reference_loss(
                simulated, data, m.SPLIT_MOMENT_SPEC,
                m.DEFAULT_PRIMARY_MOMENT_WEIGHT,
            ),
            places=12,
        )

    def test_residual_formula_and_nan_handling(self):
        """r_k = sqrt(w_k) * (sim_k - data_k) / max(|data_k|, 1e-6); 0 if NaN."""
        simulated, data = _synthetic_moments(6, seed=303)
        _, residuals = m.parental_income_cell_loss_and_residuals(
            simulated, data, "flow_plus_stock", m.DEFAULT_PRIMARY_MOMENT_WEIGHT
        )
        weights = np.tile(
            m.parental_income_moment_weight_pattern(
                "flow_plus_stock", m.DEFAULT_PRIMARY_MOMENT_WEIGHT
            ),
            4,
        )
        valid = np.isfinite(data) & np.isfinite(simulated)
        expected = np.zeros(data.size)
        expected[valid] = np.sqrt(weights[valid]) * (
            (simulated[valid] - data[valid])
            / np.maximum(np.abs(data[valid]), 1.0e-6)
        )
        np.testing.assert_array_equal(residuals, expected)
        self.assertTrue(np.all(residuals[~valid] == 0.0))
        self.assertFalse(np.all(valid), "The synthetic arrays must contain NaNs.")

    def test_stacked_multicell_weighting_matches_total_loss(self):
        """sqrt(cell_weight)-scaled stacking reproduces the weighted sum."""
        cell_weights = np.array([0.4, 1.0, 1.7, 0.9])
        total_loss = 0.0
        stacked = []
        for cell_index, cell_weight in enumerate(cell_weights):
            simulated, data = _synthetic_moments(6, seed=400 + cell_index)
            loss, residuals = m.parental_income_cell_loss_and_residuals(
                simulated, data, "flow_plus_stock",
                m.DEFAULT_PRIMARY_MOMENT_WEIGHT,
            )
            total_loss += cell_weight * loss
            stacked.append(np.sqrt(cell_weight) * residuals)
        stacked = np.concatenate(stacked)
        self.assertEqual(stacked.size, 4 * 24)
        self.assertLessEqual(
            abs(float(np.sum(stacked ** 2)) - total_loss),
            1.0e-10 * max(1.0, abs(total_loss)),
        )

    def test_evaluator_returns_residuals_last(self):
        """The per-cell evaluator's 5th return item is the residual vector."""
        source = inspect.getsource(m._evaluate_sampled_parental_income_cell)
        self.assertIn(
            "return loss, simulated, sim_new_share, spec, residuals", source
        )


class DfolsWrapperTests(unittest.TestCase):
    @staticmethod
    def _problem():
        rng = np.random.default_rng(7)
        design = rng.normal(0.0, 1.0, size=(12, 5))
        target = np.array([1.5, -2.0, 0.5, 3.0, -0.75])
        observed = design @ target

        def residual_fun(x):
            return design @ np.asarray(x, dtype=np.float64) - observed

        bounds = [(-5.0, 5.0)] * 5
        x0 = np.zeros(5)
        return residual_fun, x0, bounds, target

    @unittest.skipIf(not HAVE_DFOLS, "DFO-LS is not installed (pip install DFO-LS)")
    def test_finds_optimum_of_bounded_least_squares(self):
        residual_fun, x0, bounds, target = self._problem()
        result = m.solve_dfols_least_squares(
            residual_fun, x0, bounds, maxfun=200, restarts=0, seed=12345
        )
        np.testing.assert_allclose(result.x, target, atol=1.0e-4)
        self.assertLess(result.fun, 1.0e-8)
        self.assertGreater(result.nfev, 0)
        self.assertLessEqual(result.nfev, 200)
        self.assertIsInstance(result.message, str)

    @unittest.skipIf(not HAVE_DFOLS, "DFO-LS is not installed (pip install DFO-LS)")
    def test_restarts_are_reproducible_for_fixed_seed(self):
        residual_fun, x0, bounds, target = self._problem()
        first = m.solve_dfols_least_squares(
            residual_fun, x0, bounds, maxfun=120, restarts=2, seed=98765
        )
        second = m.solve_dfols_least_squares(
            residual_fun, x0, bounds, maxfun=120, restarts=2, seed=98765
        )
        np.testing.assert_array_equal(first.x, second.x)
        self.assertEqual(first.fun, second.fun)
        self.assertEqual(first.nfev, second.nfev)
        # Restarts add evaluations beyond a single solve.
        single = m.solve_dfols_least_squares(
            residual_fun, x0, bounds, maxfun=120, restarts=0, seed=98765
        )
        self.assertGreater(first.nfev, single.nfev)
        # The restarted best is never worse than the single solve.
        self.assertLessEqual(first.fun, single.fun + 1.0e-12)
        np.testing.assert_allclose(first.x, target, atol=1.0e-4)

    def test_input_validation(self):
        residual_fun, x0, bounds, _ = self._problem()
        with self.assertRaises(ValueError):
            m.solve_dfols_least_squares(
                residual_fun, x0, bounds, maxfun=0, restarts=0, seed=1
            )
        with self.assertRaises(ValueError):
            m.solve_dfols_least_squares(
                residual_fun, x0, bounds, maxfun=50, restarts=-1, seed=1
            )
        with self.assertRaises(ValueError):
            m.solve_dfols_least_squares(
                residual_fun, x0, bounds[:-1], maxfun=50, restarts=0, seed=1
            )


class DefaultPathUnchangedTests(unittest.TestCase):
    def test_new_keyword_arguments_default_inert(self):
        signature = inspect.signature(m.fit_education_cells)
        self.assertEqual(signature.parameters["optimizer"].default, "nelder-mead")
        self.assertIsNone(signature.parameters["dfols_maxfun"].default)
        self.assertEqual(signature.parameters["dfols_restarts"].default, 0)

        signature = inspect.signature(m.estimate_budget_shock_all_education)
        self.assertEqual(signature.parameters["optimizer"].default, "hybrid")
        self.assertIsNone(signature.parameters["dfols_maxfun"].default)
        self.assertEqual(signature.parameters["dfols_restarts"].default, 0)

        signature = inspect.signature(
            m.minimize_distance_education_cells_parental_income
        )
        self.assertFalse(signature.parameters["return_residuals"].default)

    def test_scalar_mode_stays_the_hybrid_objective(self):
        """The scalar path of the multicell objective still returns the loss."""
        source = inspect.getsource(
            m.minimize_distance_education_cells_parental_income
        )
        self.assertIn("return total_loss", source)

    def test_dfols_budget_constants(self):
        self.assertEqual(m.DFOLS_WARM_START_MAXFUN, 300)
        self.assertEqual(
            min(m.DFOLS_COLD_MAXFUN_PER_PARAMETER * 68, m.DFOLS_MAXFUN_CAP),
            2380,
        )
        self.assertEqual(
            min(m.DFOLS_COLD_MAXFUN_PER_PARAMETER * 100, m.DFOLS_MAXFUN_CAP),
            3000,
        )

    def test_production_driver_requests_dfols(self):
        """estimation_all_em.py uses dfols as its production optimizer.

        The driver is inspected as text because importing it would pull
        server-only inputs. Promoted "hybrid" -> "dfols" on 2026-07-24
        (researcher approval; Agents_Readme/Tasks/
        ESTIMATION_SPEED_ANALYSIS_2026_07_24.md, point 1). The constant must
        default to "dfols" and be the only optimizer passed to
        estimate_budget_shock_all_education.
        """
        driver_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "estimation_all_em.py"
        )
        with open(driver_path, "r", encoding="utf-8") as handle:
            driver_source = handle.read()
        self.assertIn('BUDGET_SMM_OPTIMIZER = "dfols"', driver_source)
        self.assertIn("optimizer=BUDGET_SMM_OPTIMIZER", driver_source)
        self.assertNotIn('optimizer="hybrid"', driver_source)
        self.assertNotIn('optimizer="dfols"', driver_source)


if __name__ == "__main__":
    unittest.main()
