# -*- coding: utf-8 -*-
"""Synthetic tests for the loan-type debt-penalty shift (Spec B).

Heterogeneous debt aversion: one additional parameter
``debt_penalty_loan_type_shift`` — the additive per-period debt-penalty
shift of the debt-averse latent loan type — applied uniformly across the
four parental-income levels. Loan type 0 is the LOW-borrowing type (the
loan-type moments order type 0 as "low" and type 1 as "high"), so the shift
(more negative = more averse) attaches to loan type 0
(``budget_shock.DEBT_PENALTY_SHIFT_LOAN_TYPE``).

Everything here is synthetic: no server files, estimates, or CCP bundles are
required beyond what importing the model modules themselves needs. The tests
certify that

(a) a zero shift reproduces the current outputs exactly (bitwise) in the SMM
    per-individual debt-penalty arrays, in the SMM debt-choice kernel, and in
    a solver kernel call;
(b) a nonzero shift changes exactly the loan-type-0 individuals' penalties by
    exactly the shift — through the accessor and through kernel-level
    invocations with hand-built inputs;
(c) the vector-size combinations (68/69/71/72) unpack correctly and the SMM
    tail-splitting helpers compose.
"""

import unittest

import numpy as np

import budget_shock as bs
import model_fitloans_dynamic as m
import model_solution_em as ms
from debt_limits import CONSUMPTION_FLOOR, get_debt_region_bounds
from fused_debt_search import (
    _flow_utility,
    get_maximum_loop_modified_resources_maxdebt,
)


SHIFT = -0.8
KAPPA = np.array([-0.5, -0.25, -0.1])


def _multicell_cells():
    return [bs.budget_education_cell_code(education, year)
            for education, year in bs.BUDGET_EDUCATION_CELLS]


def _multicell_vector(cells, with_kappa=False, with_shift=False):
    blocks = []
    for index in range(len(cells)):
        blocks.extend([1000.0 + index] * 4 + [50.0 + index, 0.0])
    vector = np.asarray(
        blocks + [2.0, 2.1, 2.2, 2.3] + [-1.0, -1.1, -1.2, -1.3]
    )
    if with_kappa:
        vector = np.concatenate((vector, KAPPA))
    if with_shift:
        vector = np.concatenate((vector, [SHIFT]))
    return vector


def _unpack(cells, **kwargs):
    return bs.unpack_parental_income_multicell_estimation_vector(
        _multicell_vector(cells, **kwargs), cells, index_kind="education_cell"
    )


def _tiny_spec(shift, debt_pen_baseline=-1.0e-6):
    """Minimal validated one-cell specification with a controllable penalty."""
    return bs.validate({
        "periods": np.array([201]),
        "index_kind": "education_cell",
        "education_year_grouping": bs.BUDGET_EDUCATION_YEAR_GROUPING,
        "mu_blocks": np.zeros((1, bs.N_MEAN_PARAMETERS)),
        "sigma_e": np.array([1.0]),
        "risk_aversion": np.full(4, 1.5),
        "debt_pen_parinc": np.array([debt_pen_baseline, 0.0, 0.0, 0.0]),
        "debt_pen_parameterization": bs.DEBT_PENALTY_PARAMETERIZATION,
        "debt_penalty_loan_type_shift": shift,
    })


class VectorSizeCombinationTests(unittest.TestCase):
    """(c) flags -> size -> tail order; all four layouts unpack correctly."""

    def test_flag_size_table(self):
        # flags (kappa, shift) -> size; the shift is always the LAST entry.
        self.assertEqual(bs.estimation_vector_size_multicell(10), 68)
        self.assertEqual(
            bs.estimation_vector_size_multicell(
                10, include_loan_type_debt_penalty=True
            ),
            69,
        )
        self.assertEqual(
            bs.estimation_vector_size_multicell(10, include_new_borrowing=True),
            71,
        )
        self.assertEqual(
            bs.estimation_vector_size_multicell(
                10, include_new_borrowing=True,
                include_loan_type_debt_penalty=True,
            ),
            72,
        )

    def test_all_layouts_unpack_with_correct_tails(self):
        cells = _multicell_cells()
        legacy = _unpack(cells)
        for with_kappa, with_shift in (
            (False, False), (False, True), (True, False), (True, True),
        ):
            spec = _unpack(cells, with_kappa=with_kappa, with_shift=with_shift)
            self.assertEqual(
                spec["debt_penalty_loan_type_shift"],
                SHIFT if with_shift else 0.0,
            )
            np.testing.assert_array_equal(
                spec["new_borrow_cost_entry_by_loan_type"],
                KAPPA[:2] if with_kappa else np.zeros(2),
            )
            self.assertEqual(
                spec["new_borrow_cost_continuation"],
                KAPPA[2] if with_kappa else 0.0,
            )
            # The tail must never disturb the core parameter mapping.
            np.testing.assert_array_equal(
                spec["mu_blocks"], legacy["mu_blocks"]
            )
            np.testing.assert_array_equal(spec["sigma_e"], legacy["sigma_e"])
            np.testing.assert_array_equal(
                spec["risk_aversion"], legacy["risk_aversion"]
            )
            np.testing.assert_array_equal(
                spec["debt_pen_parinc"], legacy["debt_pen_parinc"]
            )

    def test_tail_splitting_helpers_compose(self):
        cells = _multicell_cells()
        n_cells = len(cells)
        base = _multicell_vector(cells)
        # Flag off: identity, shift is None (exact current behavior).
        core, shift = m._split_loan_type_debt_penalty_tail(base, False)
        self.assertIsNone(shift)
        np.testing.assert_array_equal(core, base)
        # 69 = base + shift: strip the shift, then legacy tail slicing.
        core, shift = m._split_loan_type_debt_penalty_tail(
            _multicell_vector(cells, with_shift=True), True
        )
        self.assertEqual(shift, SHIFT)
        np.testing.assert_array_equal(core, base)
        risk, debt, kappa = m._split_multicell_shared_tail(
            core, n_cells, False
        )
        np.testing.assert_array_equal(risk, [2.0, 2.1, 2.2, 2.3])
        np.testing.assert_array_equal(debt, [-1.0, -1.1, -1.2, -1.3])
        self.assertIsNone(kappa)
        # 72 = base + kappa + shift: strip the shift, then the kappa tail.
        core, shift = m._split_loan_type_debt_penalty_tail(
            _multicell_vector(cells, with_kappa=True, with_shift=True), True
        )
        self.assertEqual(shift, SHIFT)
        risk, debt, kappa = m._split_multicell_shared_tail(
            core, n_cells, True
        )
        np.testing.assert_array_equal(risk, [2.0, 2.1, 2.2, 2.3])
        np.testing.assert_array_equal(debt, [-1.0, -1.1, -1.2, -1.3])
        np.testing.assert_array_equal(kappa, KAPPA)

    def test_flag_and_bounds_defaults(self):
        self.assertFalse(m.ESTIMATE_LOAN_TYPE_DEBT_PENALTY)
        self.assertEqual(m.LOAN_TYPE_DEBT_PENALTY_BOUNDS, (-3.0, 0.0))
        # The kappa flag keeps its own default.
        self.assertFalse(m.ESTIMATE_NEW_BORROWING_COST)


class AccessorTests(unittest.TestCase):
    """(b) the accessor shifts exactly the loan-type-0 penalties by SHIFT."""

    def setUp(self):
        cells = _multicell_cells()
        self.spec_zero = _unpack(cells)
        self.spec_shift = _unpack(cells, with_shift=True)
        self.x1 = np.column_stack(
            (np.tile(np.arange(1, 5), 2), np.ones(8))
        ).astype(np.int64)
        self.loan_type = np.array([0, 1, 0, 1, 1, 0, 1, 0])

    def test_zero_shift_is_bitwise_identity(self):
        base = bs.debt_penalty(self.spec_zero, self.x1)
        np.testing.assert_array_equal(
            bs.debt_penalty_by_loan_type(
                self.spec_zero, self.x1, self.loan_type
            ),
            base,
        )

    def test_none_loan_type_applies_no_shift(self):
        base = bs.debt_penalty(self.spec_shift, self.x1)
        np.testing.assert_array_equal(
            bs.debt_penalty_by_loan_type(self.spec_shift, self.x1), base
        )

    def test_nonzero_shift_moves_low_type_only(self):
        base = bs.debt_penalty(self.spec_shift, self.x1)
        effective = bs.debt_penalty_by_loan_type(
            self.spec_shift, self.x1, self.loan_type
        )
        low = self.loan_type == bs.DEBT_PENALTY_SHIFT_LOAN_TYPE
        np.testing.assert_array_equal(effective[~low], base[~low])
        np.testing.assert_array_equal(effective[low], base[low] + SHIFT)

    def test_invalid_loan_type_rejected(self):
        with self.assertRaises(ValueError):
            bs.debt_penalty_by_loan_type(
                self.spec_shift, self.x1, np.full(8, 2)
            )


class SmmKernelPathTests(unittest.TestCase):
    """(a)/(b) in the SMM path: per-individual debtpen arrays and the kernel."""

    PERIOD = 3

    def setUp(self):
        cells = _multicell_cells()
        self.spec_zero = _unpack(cells)
        self.spec_shift = _unpack(cells, with_shift=True)
        rng = np.random.default_rng(20260724)
        n = 24
        self.x1 = np.column_stack((
            rng.integers(1, 5, size=n), rng.integers(1, 5, size=n),
            np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64),
        ))
        self.loan_type = rng.integers(0, 2, size=n)

    def test_zero_shift_debtpen_arrays_bitwise_identical(self):
        baseline = m.discounted_explicit_horizon_debt_penalty(
            self.spec_zero, self.x1, self.PERIOD
        )
        with_loan_type = m.discounted_explicit_horizon_debt_penalty(
            self.spec_zero, self.x1, self.PERIOD, loan_type=self.loan_type
        )
        np.testing.assert_array_equal(with_loan_type, baseline)
        # A shifted spec without loan types applies no shift either.
        np.testing.assert_array_equal(
            m.discounted_explicit_horizon_debt_penalty(
                self.spec_shift, self.x1, self.PERIOD
            ),
            baseline,
        )

    def test_nonzero_shift_moves_low_type_debtpen_by_exactly_shift(self):
        baseline = m.discounted_explicit_horizon_debt_penalty(
            self.spec_shift, self.x1, self.PERIOD
        )
        shifted = m.discounted_explicit_horizon_debt_penalty(
            self.spec_shift, self.x1, self.PERIOD, loan_type=self.loan_type
        )
        low = self.loan_type == bs.DEBT_PENALTY_SHIFT_LOAN_TYPE
        np.testing.assert_array_equal(shifted[~low], baseline[~low])
        multiplier = bs.explicit_debt_penalty_multiplier(
            self.PERIOD, beta=m.beta, terminal_period=m.T
        )
        # Bitwise: the implementation computes (base + SHIFT) * multiplier.
        expected_low = np.ascontiguousarray(
            (bs.debt_penalty(self.spec_shift, self.x1)[low] + SHIFT)
            * multiplier,
            dtype=np.float64,
        )
        np.testing.assert_array_equal(shifted[low], expected_low)

    def _kernel_choice(self, spec, loan_type):
        """Hand-built one-draw kernel problem where the shift is decisive.

        Model period 9 has a discounted-horizon multiplier of exactly one, so
        the kernel's per-individual penalty IS the per-period penalty.
        Utility rises with candidate debt (more consumption, zero
        continuation), so with a negligible baseline penalty everyone
        borrows to the maximum; a large negative shift must flip exactly the
        loan-type-0 individuals to zero debt.
        """
        debt_grid = np.array([0.0, 5000.0, 15000.0])
        n = len(loan_type)
        debtpen_i = m.discounted_explicit_horizon_debt_penalty(
            spec, self.x1[:n], 9, loan_type=loan_type
        )
        zeros = np.zeros(n, dtype=np.float64)
        return m.solve_one_draw_debt_idx_terminal_only(
            budget=np.full(n, 20000.0),
            e=np.zeros(n),
            debt_grid=debt_grid,
            sigma_i=np.full(n, 1.5),
            debtpen_i=debtpen_i,
            ccp_path_row=np.zeros((n, debt_grid.size)),
            terminal_row=np.zeros((n, debt_grid.size)),
            b_idx=np.zeros(n, dtype=np.int64),
            max_idx=np.full(n, debt_grid.size - 1, dtype=np.int64),
            beta_term=1.0,
            kappa_entry_i=zeros,
            kappa_cont_i=zeros,
            prev_debt_i=zeros,
            c_floor=CONSUMPTION_FLOOR,
            fallback_idx=debt_grid.size - 1,
        )

    def test_kernel_zero_shift_bitwise_identical(self):
        loan_type = np.array([0, 1, 0, 1])
        spec = _tiny_spec(0.0)
        observed = self._kernel_choice(spec, loan_type)
        # Same problem through the pre-shift path (no loan types at all).
        debt_grid = np.array([0.0, 5000.0, 15000.0])
        n = len(loan_type)
        zeros = np.zeros(n, dtype=np.float64)
        baseline = m.solve_one_draw_debt_idx_terminal_only(
            budget=np.full(n, 20000.0),
            e=np.zeros(n),
            debt_grid=debt_grid,
            sigma_i=np.full(n, 1.5),
            debtpen_i=m.discounted_explicit_horizon_debt_penalty(
                spec, self.x1[:n], 9
            ),
            ccp_path_row=np.zeros((n, debt_grid.size)),
            terminal_row=np.zeros((n, debt_grid.size)),
            b_idx=np.zeros(n, dtype=np.int64),
            max_idx=np.full(n, debt_grid.size - 1, dtype=np.int64),
            beta_term=1.0,
            kappa_entry_i=zeros,
            kappa_cont_i=zeros,
            prev_debt_i=zeros,
            c_floor=CONSUMPTION_FLOOR,
            fallback_idx=debt_grid.size - 1,
        )
        np.testing.assert_array_equal(observed, baseline)
        # Everyone borrows to the maximum with a negligible penalty.
        np.testing.assert_array_equal(observed, np.full(n, 2))

    def test_kernel_nonzero_shift_flips_low_type_only(self):
        loan_type = np.array([0, 1, 0, 1])
        choices = self._kernel_choice(_tiny_spec(-50.0), loan_type)
        # Loan-type-0 individuals are deterred from any positive debt; the
        # loan-type-1 individuals are untouched.
        np.testing.assert_array_equal(choices, np.array([0, 2, 0, 2]))


class SolverPathTests(unittest.TestCase):
    """(a)/(b) in the solver: per-task shift resolution and a kernel call."""

    GRID = np.array(
        [0.0, 300.0, 1500.0, 3000.0, 6000.0, 10000.0,
         20000.0, 35000.0, 50000.0, 69000.0, 90000.0]
    )

    def test_resolver_semantics(self):
        self.assertEqual(ms.resolve_task_debt_penalty_shift(None, 0), 0.0)
        self.assertEqual(
            ms.resolve_task_debt_penalty_shift({}, 0), 0.0
        )
        spec_zero = _tiny_spec(0.0)
        spec_shift = _tiny_spec(SHIFT)
        self.assertEqual(
            ms.resolve_task_debt_penalty_shift(spec_zero, 0), 0.0
        )
        self.assertEqual(
            ms.resolve_task_debt_penalty_shift(spec_shift, 0), SHIFT
        )
        # The high-borrowing type (loan type 1) never receives the shift.
        self.assertEqual(
            ms.resolve_task_debt_penalty_shift(spec_shift, 1), 0.0
        )

    def test_module_default_task_shift_is_zero(self):
        # With the production (or absent) bundle the per-task global must be
        # exactly zero, keeping every guarded add inactive.
        self.assertEqual(ms.task_debt_pen_shift, 0.0)

    def _solver_kernel_payoffs(self, debt_pen):
        """One fused-kernel call built exactly as ``get_conditional`` does:
        the (possibly shifted) scalar penalty is attached to the candidate
        next-period debt through the continuation column."""
        choice = np.array([0, 1, 2], dtype=np.int64)
        x2 = np.array([0, 0, 0], dtype=np.int64)
        n = self.GRID.size
        resources = np.full(n, 25000.0)
        continuation = np.zeros((n, 1))
        continuation[:, 0] += debt_pen * (self.GRID > 0.0)
        return get_maximum_loop_modified_resources_maxdebt(
            1.4, self.GRID, self.GRID, resources, continuation, choice, x2,
        )

    def _reference_payoffs(self, debt_pen):
        """Plain-numpy optimum; exact because the <=11-point grid keeps the
        kernel's local window covering the full feasible interval."""
        choice = np.array([0, 1, 2], dtype=np.int64)
        x2 = np.array([0, 0, 0], dtype=np.int64)
        n = self.GRID.size
        lo_idx, hi_idx, cap_start = get_debt_region_bounds(
            self.GRID, x2, choice
        )
        payoff = np.empty(n)
        for it in range(n):
            if it >= cap_start:
                candidates = [int(lo_idx[it])]
            else:
                candidates = [
                    k for k in range(int(lo_idx[it]), int(hi_idx[it]) + 1)
                    if 25000.0 + self.GRID[k] >= CONSUMPTION_FLOOR
                ]
            values = [
                _flow_utility(1.4, 25000.0 + self.GRID[k])
                + debt_pen * (self.GRID[k] > 0.0)
                for k in candidates
            ]
            payoff[it] = max(values)
        return payoff

    def test_solver_kernel_zero_shift_bitwise_identical(self):
        spec_zero = _tiny_spec(0.0, debt_pen_baseline=-0.001)
        debt_pen = float(
            bs.debt_penalty_design_vector(spec_zero)[0]
        )
        # The per-task resolution with a zero shift performs NO operation on
        # the scalar penalty (the guard skips the add entirely).
        for loan_type in (0, 1):
            task_shift = ms.resolve_task_debt_penalty_shift(
                spec_zero, loan_type
            )
            debt_pen_task = debt_pen
            if task_shift != 0.0:
                debt_pen_task += task_shift
            np.testing.assert_array_equal(
                self._solver_kernel_payoffs(debt_pen_task),
                self._solver_kernel_payoffs(debt_pen),
            )

    def test_solver_kernel_shift_binds_for_low_type_only(self):
        spec_shift = _tiny_spec(-50.0, debt_pen_baseline=-0.001)
        debt_pen = float(bs.debt_penalty_design_vector(spec_shift)[0])
        baseline = self._solver_kernel_payoffs(debt_pen)
        np.testing.assert_array_equal(
            baseline, self._reference_payoffs(debt_pen)
        )
        # Loan type 1: no shift, bitwise identical payoffs.
        shift_high = ms.resolve_task_debt_penalty_shift(spec_shift, 1)
        self.assertEqual(shift_high, 0.0)
        np.testing.assert_array_equal(
            self._solver_kernel_payoffs(debt_pen), baseline
        )
        # Loan type 0: the shifted penalty changes the payoffs and matches
        # the exact reference with the shifted penalty.
        shift_low = ms.resolve_task_debt_penalty_shift(spec_shift, 0)
        self.assertEqual(shift_low, -50.0)
        shifted = self._solver_kernel_payoffs(debt_pen + shift_low)
        np.testing.assert_array_equal(
            shifted, self._reference_payoffs(debt_pen + shift_low)
        )
        self.assertTrue(np.any(shifted != baseline))


if __name__ == "__main__":
    unittest.main()
