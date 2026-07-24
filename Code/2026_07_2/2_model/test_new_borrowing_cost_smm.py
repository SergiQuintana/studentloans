"""Synthetic tests for the new-borrowing (kappa) SMM extension.

Everything here is synthetic: no server files, estimates, or CCP bundles are
required beyond what importing ``model_fitloans_dynamic`` itself needs. The
brute-force references reproduce the pre-kappa kernel objective in plain
numpy, so test (a) certifies that zero kappa arrays leave the kernels'
numerical output identical to the pre-change code, and test (b) certifies the
kappa charge against the explicit accrued-debt SMM convention
``debt_grid[j] > (1+r)*prev_debt`` with entry/continuation selected by
``prev_debt <= 0``.
"""

import unittest

import numpy as np

import budget_shock as bs
import model_fitloans_dynamic as m
from debt_limits import CONSUMPTION_FLOOR, INTEREST_RATE


def _crra_utility(consumption, sigma):
    if abs(sigma - 1.0) < 1e-8:
        return 0.1 * np.log(0.00001 * consumption)
    return 0.1 * ((0.00001 * consumption) ** (1.0 - sigma)) / (1.0 - sigma)


def _brute_force_pooled(
    budget_by_draw, shock_by_draw, debt_grid, sigma_i, debtpen_i,
    ccp_path_row, terminal_row, b_idx, max_idx, beta_term_i,
    kappa_entry_i, kappa_cont_i, prev_debt_i,
    c_floor=CONSUMPTION_FLOOR,
):
    """Plain-numpy reference for the pooled kernel, kappa included.

    With all kappas zero this is exactly the pre-change objective: CRRA flow
    utility, the stock debt penalty on positive candidate debt, the CCP and
    discounted terminal continuation, the consumption floor, and the
    hi-index fallback when no candidate is feasible.
    """
    draws, n = budget_by_draw.shape
    B = debt_grid.size
    out = np.empty((draws, n), dtype=np.int64)
    for draw in range(draws):
        for i in range(n):
            lo = max(int(b_idx[i]), 0)
            hi = min(int(max_idx[i]), B - 1)
            if hi < lo:
                hi = lo
            kappa = (
                kappa_entry_i[i] if prev_debt_i[i] <= 0.0 else kappa_cont_i[i]
            )
            accrued = (1.0 + INTEREST_RATE) * prev_debt_i[i]
            best_v = -1e30
            best_j = lo
            found = False
            for j in range(lo, hi + 1):
                c = budget_by_draw[draw, i] + shock_by_draw[draw, i] + debt_grid[j]
                if c < c_floor:
                    continue
                found = True
                u = _crra_utility(c, sigma_i[i])
                if debt_grid[j] > 0.0:
                    u += debtpen_i[i]
                if kappa != 0.0 and debt_grid[j] > accrued:
                    u += kappa
                v = u + ccp_path_row[i, j] + beta_term_i[i] * terminal_row[i, j]
                if v > best_v:
                    best_v = v
                    best_j = j
            out[draw, i] = hi if not found else best_j
    return out


def _synthetic_kernel_inputs(n=40, draws=3, seed=20260723):
    """Random, tie-free inputs covering entry, continuation, and infeasible rows."""
    rng = np.random.default_rng(seed)
    debt_grid = np.linspace(0.0, 30000.0, 16)
    budget_by_draw = rng.uniform(-5000.0, 30000.0, size=(draws, n))
    # Force a few individuals to be infeasible at every candidate so the
    # hi-index fallback path is exercised.
    budget_by_draw[:, :3] = -100000.0
    shock_by_draw = rng.normal(0.0, 2000.0, size=(draws, n))
    sigma_i = rng.uniform(0.3, 2.5, size=n)
    sigma_i[5] = 1.0  # log-utility branch
    debtpen_i = rng.uniform(-2.0, 0.0, size=n)
    ccp_path_row = rng.normal(0.0, 0.5, size=(n, debt_grid.size))
    terminal_row = rng.normal(0.0, 0.5, size=(n, debt_grid.size))
    b_idx = rng.integers(0, 6, size=n).astype(np.int64)
    max_idx = (b_idx + rng.integers(1, 9, size=n)).astype(np.int64)
    beta_term_i = rng.uniform(0.7, 1.0, size=n)
    # Half the individuals start with zero debt (entry), half with positive
    # debt (continuation). Previous debt sits between grid points so the
    # accrued-debt threshold does not coincide with a candidate value.
    prev_debt_i = np.where(
        np.arange(n) % 2 == 0, 0.0, rng.uniform(500.0, 12000.0, size=n)
    )
    return {
        "budget_by_draw": budget_by_draw,
        "shock_by_draw": shock_by_draw,
        "debt_grid": debt_grid,
        "sigma_i": sigma_i,
        "debtpen_i": debtpen_i,
        "ccp_path_row": ccp_path_row,
        "terminal_row": terminal_row,
        "b_idx": b_idx,
        "max_idx": max_idx,
        "beta_term_i": beta_term_i,
        "prev_debt_i": prev_debt_i,
    }


class PooledKernelKappaTests(unittest.TestCase):
    def test_zero_kappa_matches_pre_change_reference(self):
        inputs = _synthetic_kernel_inputs()
        n = inputs["sigma_i"].size
        zeros = np.zeros(n, dtype=np.float64)
        observed = m.solve_all_draws_debt_idx_pooled(
            kappa_entry_i=zeros, kappa_cont_i=zeros,
            prev_debt_i=inputs["prev_debt_i"],
            c_floor=CONSUMPTION_FLOOR, fallback_idx=15,
            **{k: v for k, v in inputs.items() if k != "prev_debt_i"},
        )
        expected = _brute_force_pooled(
            kappa_entry_i=zeros, kappa_cont_i=zeros, **inputs
        )
        np.testing.assert_array_equal(observed, expected)

    def test_zero_helper_matches_pre_change_reference(self):
        inputs = _synthetic_kernel_inputs(seed=7)
        n = inputs["sigma_i"].size
        kappa_entry_i, kappa_cont_i, prev_debt_i = (
            m._zero_new_borrowing_kernel_arrays(n)
        )
        kernel_inputs = dict(inputs)
        kernel_inputs["prev_debt_i"] = prev_debt_i
        observed = m.solve_all_draws_debt_idx_pooled(
            kappa_entry_i=kappa_entry_i, kappa_cont_i=kappa_cont_i,
            **kernel_inputs,
        )
        expected = _brute_force_pooled(
            kappa_entry_i=np.zeros(n), kappa_cont_i=np.zeros(n), **kernel_inputs
        )
        np.testing.assert_array_equal(observed, expected)

    def test_nonzero_kappa_matches_reference_and_binds(self):
        inputs = _synthetic_kernel_inputs(seed=99)
        n = inputs["sigma_i"].size
        kappa_entry_i = np.full(n, -1.5)
        kappa_cont_i = np.full(n, -0.75)
        observed = m.solve_all_draws_debt_idx_pooled(
            kappa_entry_i=kappa_entry_i, kappa_cont_i=kappa_cont_i, **inputs
        )
        expected = _brute_force_pooled(
            kappa_entry_i=kappa_entry_i, kappa_cont_i=kappa_cont_i, **inputs
        )
        np.testing.assert_array_equal(observed, expected)
        zeros = np.zeros(n)
        baseline = m.solve_all_draws_debt_idx_pooled(
            kappa_entry_i=zeros, kappa_cont_i=zeros,
            **{k: v for k, v in inputs.items()},
        )
        self.assertTrue(
            np.any(observed != baseline),
            "A large kappa should change at least one debt choice.",
        )

    def test_accrued_debt_threshold_convention(self):
        """kappa applies above (1+r)*prev_debt, not above prev_debt itself."""
        debt_grid = np.array([0.0, 10000.0, 10400.0, 20000.0])
        n = 1
        prev_debt = np.array([10000.0])  # accrued: 10500
        shared = dict(
            budget_by_draw=np.array([[30000.0]]),
            shock_by_draw=np.zeros((1, 1)),
            debt_grid=debt_grid,
            sigma_i=np.array([1.5]),
            debtpen_i=np.zeros(1),
            ccp_path_row=np.zeros((n, debt_grid.size)),
            terminal_row=np.zeros((n, debt_grid.size)),
            b_idx=np.array([1], dtype=np.int64),
            max_idx=np.array([3], dtype=np.int64),
            beta_term_i=np.ones(1),
            prev_debt_i=prev_debt,
        )
        # Candidate 10400 rolls over less than accrued debt (no new loan);
        # candidate 20000 is a new loan. Utility is increasing in debt here,
        # so without kappa the choice is 20000; a large continuation cost
        # must push the choice down to 10400, NOT to 10000 or the threshold.
        no_kappa = m.solve_all_draws_debt_idx_pooled(
            kappa_entry_i=np.zeros(1), kappa_cont_i=np.zeros(1), **shared
        )
        self.assertEqual(int(no_kappa[0, 0]), 3)
        with_kappa = m.solve_all_draws_debt_idx_pooled(
            kappa_entry_i=np.zeros(1), kappa_cont_i=np.array([-50.0]), **shared
        )
        self.assertEqual(int(with_kappa[0, 0]), 2)

    def test_entry_versus_continuation_selection(self):
        """prev_debt == 0 uses the entry cost; prev_debt > 0 the continuation."""
        debt_grid = np.array([0.0, 5000.0, 15000.0])
        n = 2
        shared = dict(
            budget_by_draw=np.full((1, n), 20000.0),
            shock_by_draw=np.zeros((1, n)),
            debt_grid=debt_grid,
            sigma_i=np.full(n, 1.5),
            debtpen_i=np.zeros(n),
            ccp_path_row=np.zeros((n, debt_grid.size)),
            terminal_row=np.zeros((n, debt_grid.size)),
            b_idx=np.zeros(n, dtype=np.int64),
            max_idx=np.full(n, 2, dtype=np.int64),
            beta_term_i=np.ones(n),
            prev_debt_i=np.array([0.0, 1000.0]),
        )
        # A prohibitive entry cost with a free continuation cost must deter
        # only the zero-debt individual, and vice versa.
        entry_only = m.solve_all_draws_debt_idx_pooled(
            kappa_entry_i=np.full(n, -50.0), kappa_cont_i=np.zeros(n), **shared
        )
        self.assertEqual(int(entry_only[0, 0]), 0)
        self.assertEqual(int(entry_only[0, 1]), 2)
        continuation_only = m.solve_all_draws_debt_idx_pooled(
            kappa_entry_i=np.zeros(n), kappa_cont_i=np.full(n, -50.0), **shared
        )
        self.assertEqual(int(continuation_only[0, 0]), 2)
        # With prev debt 1000 the accrued stock is 1050 < 5000, so the
        # deterred continuation borrower rolls down to the lowest candidate.
        self.assertEqual(int(continuation_only[0, 1]), 0)

    def test_one_draw_kernel_matches_pooled(self):
        inputs = _synthetic_kernel_inputs(seed=31, draws=1)
        n = inputs["sigma_i"].size
        kappa_entry_i = np.full(n, -0.8)
        kappa_cont_i = np.full(n, -0.4)
        pooled = m.solve_all_draws_debt_idx_pooled(
            kappa_entry_i=kappa_entry_i, kappa_cont_i=kappa_cont_i, **inputs
        )
        single = m.solve_one_draw_debt_idx_terminal_only(
            budget=inputs["budget_by_draw"][0],
            e=inputs["shock_by_draw"][0],
            debt_grid=inputs["debt_grid"],
            sigma_i=inputs["sigma_i"],
            debtpen_i=inputs["debtpen_i"],
            ccp_path_row=inputs["ccp_path_row"],
            terminal_row=inputs["terminal_row"],
            b_idx=inputs["b_idx"],
            max_idx=inputs["max_idx"],
            beta_term=1.0,
            kappa_entry_i=kappa_entry_i,
            kappa_cont_i=kappa_cont_i,
            prev_debt_i=inputs["prev_debt_i"],
        )
        expected = _brute_force_pooled(
            kappa_entry_i=kappa_entry_i, kappa_cont_i=kappa_cont_i,
            **{**inputs, "beta_term_i": np.ones(n)},
        )
        np.testing.assert_array_equal(single, expected[0])
        # The pooled kernel with the same per-individual beta agrees too.
        np.testing.assert_array_equal(
            pooled,
            _brute_force_pooled(
                kappa_entry_i=kappa_entry_i, kappa_cont_i=kappa_cont_i, **inputs
            ),
        )


class SplitMomentTests(unittest.TestCase):
    def _tiny_dataset(self):
        parinc = np.array([1, 1, 1, 1, 1, 2, 2, 2, 3, 4])
        begin_debt = np.array(
            [0.0, 0.0, 0.0, 500.0, 500.0, 0.0, 0.0, 1000.0, 0.0, 2000.0]
        )
        flow = np.array(
            [1000.0, 0.0, 2000.0, 0.0, 3000.0, 8391.0, 0.0, 500.0, 0.0, 0.0]
        )
        annual_cap = np.full(10, 8391.0)
        return parinc, flow, begin_debt, annual_cap

    def test_hand_computed_values(self):
        parinc, flow, begin_debt, annual_cap = self._tiny_dataset()
        output, flow_share, weight, labels = m.parental_income_split_moments(
            parinc, flow, begin_debt, annual_cap
        )
        eps = 0.01
        expected = np.array([
            # parinc 1: entry {1000, 0, 2000}, continuation {0, 3000}
            2.0 / 3.0, 0.5, 1500.0, 3000.0, 0.0,
            # parinc 2: entry {8391, 0}, continuation {500};
            # positive flows {8391, 500}, one at the 8391 cap.
            0.5, 1.0, 8391.0, 500.0, 0.5,
            # parinc 3: one zero-flow entry observation, no continuation group.
            0.0, eps, eps, eps, eps,
            # parinc 4: no entry group, one zero-flow continuation observation.
            eps, 0.0, eps, eps, eps,
        ])
        np.testing.assert_allclose(output, expected)
        np.testing.assert_allclose(flow_share, [0.6, 2.0 / 3.0, 0.0, 0.0])
        np.testing.assert_allclose(weight, [5.0, 3.0, 1.0, 1.0])
        self.assertEqual(labels, (1, 2, 3, 4))

    def test_at_cap_uses_99_percent_threshold(self):
        parinc = np.array([1, 1, 1])
        begin_debt = np.zeros(3)
        cap = 10000.0
        flow = np.array([0.99 * cap, 0.98 * cap, 0.0])
        output, _, _, _ = m.parental_income_split_moments(
            parinc, flow, begin_debt, np.full(3, cap)
        )
        # Exactly one of the two positive flows reaches 99% of its cap.
        self.assertAlmostEqual(output[4], 0.5)

    def test_input_validation(self):
        parinc, flow, begin_debt, annual_cap = self._tiny_dataset()
        with self.assertRaises(ValueError):
            m.parental_income_split_moments(
                parinc, flow[:-1], begin_debt, annual_cap
            )
        with self.assertRaises(ValueError):
            m.parental_income_split_moments(
                parinc, flow, begin_debt, np.zeros(10)
            )

    def test_weight_pattern_is_4_4_2_2_1(self):
        np.testing.assert_array_equal(
            m.parental_income_moment_weight_pattern(
                m.SPLIT_MOMENT_SPEC, m.DEFAULT_PRIMARY_MOMENT_WEIGHT
            ),
            np.array([4.0, 4.0, 2.0, 2.0, 1.0]),
        )
        # Existing specifications are untouched.
        np.testing.assert_array_equal(
            m.parental_income_moment_weight_pattern("flow_plus_stock", 4.0),
            np.array([4.0, 4.0, 1.0, 1.0, 2.0, 2.0]),
        )
        np.testing.assert_array_equal(
            m.parental_income_moment_weight_pattern("fast_stock", 4.0),
            np.array([4.0, 4.0, 1.0, 1.0]),
        )

    def test_existing_moment_functions_reject_split_spec(self):
        parinc, flow, begin_debt, annual_cap = self._tiny_dataset()
        with self.assertRaises(ValueError):
            m.parental_income_distribution_moments(
                parinc, flow, begin_debt, moment_spec=m.SPLIT_MOMENT_SPEC
            )


class ExtendedVectorPathTests(unittest.TestCase):
    N_CELLS = 10

    def _vector(self, extended):
        blocks = []
        for index in range(self.N_CELLS):
            blocks.extend([1000.0 + index] * 4 + [50.0 + index, 0.0])
        vector = np.asarray(
            blocks + [2.0, 2.1, 2.2, 2.3] + [-1.0, -1.1, -1.2, -1.3]
        )
        if extended:
            vector = np.concatenate((vector, [-0.5, -0.25, -0.1]))
        return vector

    def test_shared_tail_slicing_legacy(self):
        risk, debt, kappa = m._split_multicell_shared_tail(
            self._vector(extended=False), self.N_CELLS, False
        )
        np.testing.assert_array_equal(risk, [2.0, 2.1, 2.2, 2.3])
        np.testing.assert_array_equal(debt, [-1.0, -1.1, -1.2, -1.3])
        self.assertIsNone(kappa)

    def test_shared_tail_slicing_extended(self):
        risk, debt, kappa = m._split_multicell_shared_tail(
            self._vector(extended=True), self.N_CELLS, True
        )
        np.testing.assert_array_equal(risk, [2.0, 2.1, 2.2, 2.3])
        np.testing.assert_array_equal(debt, [-1.0, -1.1, -1.2, -1.3])
        np.testing.assert_array_equal(kappa, [-0.5, -0.25, -0.1])

    def test_shared_tail_extended_requires_71_entries(self):
        with self.assertRaises(ValueError):
            m._split_multicell_shared_tail(
                self._vector(extended=False), self.N_CELLS, True
            )

    def test_unpack_to_kernel_arrays_roundtrip(self):
        """71-vector -> canonical spec -> per-individual kappa kernel inputs."""
        cells = [
            bs.budget_education_cell_code(education, year)
            for education, year in bs.BUDGET_EDUCATION_CELLS
        ]
        vector = self._vector(extended=True)
        self.assertEqual(
            vector.size,
            bs.estimation_vector_size_multicell(
                len(cells), include_new_borrowing=True
            ),
        )
        spec = bs.unpack_parental_income_multicell_estimation_vector(
            vector, cells, index_kind="education_cell"
        )
        loan_type = np.array([0, 1, 1, 0])
        prev_debt = np.array([0.0, 250.0, 0.0, 1000.0])
        kappa_entry_i, kappa_cont_i, prev_out = m._new_borrowing_kernel_arrays(
            spec, loan_type, prev_debt
        )
        np.testing.assert_array_equal(
            kappa_entry_i, [-0.5, -0.25, -0.25, -0.5]
        )
        np.testing.assert_array_equal(kappa_cont_i, np.full(4, -0.1))
        np.testing.assert_array_equal(prev_out, prev_debt)

    def test_legacy_spec_yields_zero_kernel_arrays(self):
        cells = [
            bs.budget_education_cell_code(education, year)
            for education, year in bs.BUDGET_EDUCATION_CELLS
        ]
        spec = bs.unpack_parental_income_multicell_estimation_vector(
            self._vector(extended=False), cells, index_kind="education_cell"
        )
        kappa_entry_i, kappa_cont_i, _ = m._new_borrowing_kernel_arrays(
            spec, np.array([0, 1]), np.array([0.0, 100.0])
        )
        np.testing.assert_array_equal(kappa_entry_i, np.zeros(2))
        np.testing.assert_array_equal(kappa_cont_i, np.zeros(2))

    def test_evaluate_cell_injection_slots_match_shared_tail(self):
        """The kappa injection in the per-cell evaluation mirrors bs unpack."""
        cells = [
            bs.budget_education_cell_code(education, year)
            for education, year in bs.BUDGET_EDUCATION_CELLS
        ]
        vector = self._vector(extended=True)
        risk, debt, kappa = m._split_multicell_shared_tail(
            vector, self.N_CELLS, True
        )
        block = vector[0:6]
        full_params = np.concatenate((block[0:5], risk, debt, block[5:6]))
        cell_spec = bs.unpack_parental_income_estimation_vector(
            full_params, [cells[0]], index_kind="education_cell"
        )
        cell_spec["new_borrow_cost_entry_by_loan_type"] = kappa[:2].copy()
        cell_spec["new_borrow_cost_continuation"] = float(kappa[2])
        multicell_spec = bs.unpack_parental_income_multicell_estimation_vector(
            vector, cells, index_kind="education_cell"
        )
        np.testing.assert_array_equal(
            bs.new_borrowing_cost_parameters(cell_spec)[0],
            bs.new_borrowing_cost_parameters(multicell_spec)[0],
        )
        self.assertEqual(
            bs.new_borrowing_cost_parameters(cell_spec)[1],
            bs.new_borrowing_cost_parameters(multicell_spec)[1],
        )
        np.testing.assert_array_equal(
            cell_spec["risk_aversion"], multicell_spec["risk_aversion"]
        )
        np.testing.assert_array_equal(
            cell_spec["debt_pen_parinc"], multicell_spec["debt_pen_parinc"]
        )


class FlagDefaultTests(unittest.TestCase):
    def test_flag_is_off_by_default(self):
        self.assertFalse(m.ESTIMATE_NEW_BORROWING_COST)

    def test_existing_moment_specs_unchanged(self):
        self.assertEqual(
            m.PARENTAL_INCOME_MOMENT_SPECS,
            ("fast_stock", "flow_stock", "fast_flow", "flow_plus_stock"),
        )
        self.assertEqual(
            m.EXTENDED_PARENTAL_INCOME_MOMENT_SPECS,
            m.PARENTAL_INCOME_MOMENT_SPECS + (m.SPLIT_MOMENT_SPEC,),
        )


if __name__ == "__main__":
    unittest.main()
