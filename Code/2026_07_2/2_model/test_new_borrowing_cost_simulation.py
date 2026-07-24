# -*- coding: utf-8 -*-
"""Synthetic tests for the forward-simulation new-borrowing cost (kappa)
and the debt-penalty convention switch in ``model_simulation_em``.

These tests exercise ``get_utility_agents`` directly with small synthetic
inputs and a synthetic budget-shock specification injected into the module
global ``budget_params``. No server files are required beyond what the
module already loads at import time.
"""

import unittest

import numpy as np

import budget_shock as bs
import model_simulation_em as sim


BIGNEG = -100000.0


def make_spec(kappa_entry=(0.0, 0.0), kappa_continuation=0.0):
    """Minimal synthetic budget-shock spec for the utility function."""
    return {
        # Four parental-income group levels (legacy length-four layout).
        "debt_pen_parinc": np.array([-10.0, -20.0, -30.0, -40.0]),
        "new_borrow_cost_entry_by_loan_type": np.asarray(
            kappa_entry, dtype=np.float64
        ),
        "new_borrow_cost_continuation": float(kappa_continuation),
    }


def brute_force_utility(spec, sigma_u, x1, x2, b, b1, financial_help,
                        budget_psi, wage_psi, j, penalty_multiplier,
                        loan_types=None):
    """Loop-based reference of the simulator's education-choice utility.

    Reproduces the exact operation order of ``get_utility_agents`` under
    ``maxdebt=False``: CRRA flow utility, debt penalty on candidate
    ``b1 > 0`` scaled by ``penalty_multiplier``, the one-shot kappa event
    cost on new borrowing (chosen ``b1`` above accrued current debt), and
    finally the maxdebt=False debt rules (consumption floor, no
    repayment below the current debt index).
    """
    r = 0.05
    debt_range = sim.get_debt_range()
    n = np.shape(x1)[0]
    B = np.shape(b1)[0]
    x1_new = sim.get_x1_new(x1)
    w = sim.wage0(x1_new, x2)
    kappa_entry = np.asarray(spec["new_borrow_cost_entry_by_loan_type"])
    kappa_continuation = float(spec["new_borrow_cost_continuation"])
    charge_kappa = np.any(kappa_entry != 0.0) or kappa_continuation != 0.0
    u = np.zeros((n, B))
    for i in range(n):
        wage_shock = (
            np.exp(w[i, 0] + wage_psi[i]) * j[i, 2] * 1 / 2 * (40 * 52)
        )
        tuition = {0: 0, 1: 4000, 2: 8000, 3: 14000}[int(j[i, 1])]
        current_debt = debt_range[int(b[i])]
        base_c = (
            financial_help[i] + wage_shock + budget_psi[i]
            - (1 + r) * current_debt - tuition
        )
        penalty = (
            spec["debt_pen_parinc"][int(x1[i, 0]) - 1] * penalty_multiplier
        )
        for k in range(B):
            c = base_c + b1[k]
            scaled_c = 0.00001 * max(c, sim.CONSUMPTION_FLOOR)
            s = sigma_u[i]
            if abs(s - 1.0) < 1e-8:
                value = 0.1 * np.log(scaled_c)
            else:
                value = 0.1 * scaled_c ** (1.0 - s) / (1.0 - s)
            if b1[k] > 0.0:
                value = value + penalty
            if charge_kappa and b1[k] > (1 + r) * current_debt:
                if current_debt == 0.0:
                    loan = 0 if loan_types is None else int(loan_types[i])
                    value = value + kappa_entry[loan]
                else:
                    value = value + kappa_continuation
            if c < sim.CONSUMPTION_FLOOR:
                value = BIGNEG
            u[i, k] = value
        u[i, : int(b[i])] = BIGNEG
    return u


class TestNewBorrowingCostSimulation(unittest.TestCase):

    def setUp(self):
        self._saved_params = sim.budget_params
        self._saved_convention = sim.SIM_DEBT_PENALTY_CONVENTION
        # Three synthetic students: two with zero current debt (one per
        # loan type) and one with strictly positive current debt.
        self.x1 = np.array(
            [[1, 2, 0, 0], [2, 3, 0, 1], [3, 1, 1, 0]], dtype=np.int64
        )
        self.x2 = np.zeros((3, 10), dtype=np.float64)
        self.b = np.array([0, 0, 10], dtype=np.int64)
        self.b1 = sim.get_debt_range()
        self.loan_types = np.array([0, 1, 0], dtype=np.int64)
        self.sigma_u = np.array([2.0, 2.0, 2.0])
        # Large resources keep every candidate above the consumption
        # floor, so no cell is masked by the debt rules.
        self.financial_help = np.array([50000.0, 60000.0, 70000.0])
        self.budget_psi = np.array([100.0, -200.0, 300.0])
        self.wage_psi = np.zeros(3)
        # Four-year enrollment, no labor supply.
        self.j = np.array([[1, 2, 0]] * 3, dtype=np.int64)
        self.period = 3

    def tearDown(self):
        sim.budget_params = self._saved_params
        sim.SIM_DEBT_PENALTY_CONVENTION = self._saved_convention

    def call_utility(self, spec):
        sim.budget_params = spec
        return sim.get_utility_agents(
            self.sigma_u, self.x1, self.x2, self.b, self.b1,
            self.financial_help, self.budget_psi, self.wage_psi,
            self.j, self.period, 0, False, loan_types=self.loan_types,
        )

    def test_zero_kappa_legacy_matches_brute_force(self):
        """Zero kappas + legacy convention reproduce the current formula."""
        spec = make_spec()
        u = self.call_utility(spec)
        multiplier = bs.explicit_debt_penalty_multiplier(
            self.period, beta=sim.beta, terminal_period=sim.T
        )
        reference = brute_force_utility(
            spec, self.sigma_u, self.x1, self.x2, self.b, self.b1,
            self.financial_help, self.budget_psi, self.wage_psi,
            self.j, multiplier, loan_types=self.loan_types,
        )
        np.testing.assert_array_equal(u, reference)

    def test_kappa_charges_exactly_the_new_borrowing_cells(self):
        """Kappa hits entry/continuation rows correctly, rollover exempt."""
        kappa_spec = make_spec(
            kappa_entry=(-500.0, -250.0), kappa_continuation=-75.0
        )
        u_kappa = self.call_utility(kappa_spec)
        u_base = self.call_utility(make_spec())
        difference = u_kappa - u_base
        r = 0.05
        debt_range = sim.get_debt_range()
        accrued = (1 + r) * debt_range[self.b]
        expected = np.zeros_like(difference)
        # Row 0: no current debt, loan type 0 -> entry cost kappa0[0].
        expected[0, self.b1 > accrued[0]] = -500.0
        # Row 1: no current debt, loan type 1 -> entry cost kappa0[1].
        expected[1, self.b1 > accrued[1]] = -250.0
        # Row 2: positive current debt -> continuation cost kappa1.
        expected[2, self.b1 > accrued[2]] = -75.0
        # The subtraction of two large utilities reintroduces float
        # rounding of order 1e-13; the exact-equality check against the
        # brute-force reference below is the bitwise test.
        np.testing.assert_allclose(difference, expected, rtol=0.0, atol=1e-9)
        # Rollover column: keeping the current grid point implies
        # b1 = debt_range[b] <= (1+r)*debt_range[b]; no charge.
        self.assertEqual(difference[2, self.b[2]], 0.0)
        # No-borrowing column b1 = 0 is never charged.
        self.assertEqual(difference[0, 0], 0.0)
        self.assertEqual(difference[1, 0], 0.0)
        # Full brute-force cross-check with kappas active.
        multiplier = bs.explicit_debt_penalty_multiplier(
            self.period, beta=sim.beta, terminal_period=sim.T
        )
        reference = brute_force_utility(
            kappa_spec, self.sigma_u, self.x1, self.x2, self.b, self.b1,
            self.financial_help, self.budget_psi, self.wage_psi,
            self.j, multiplier, loan_types=self.loan_types,
        )
        np.testing.assert_array_equal(u_kappa, reference)

    def test_single_flow_convention_charges_penalty_once(self):
        """The single_flow switch drops the discounted multiplier."""
        spec = make_spec()
        sim.SIM_DEBT_PENALTY_CONVENTION = "single_flow"
        u_single = self.call_utility(spec)
        reference = brute_force_utility(
            spec, self.sigma_u, self.x1, self.x2, self.b, self.b1,
            self.financial_help, self.budget_psi, self.wage_psi,
            self.j, 1.0, loan_types=self.loan_types,
        )
        np.testing.assert_array_equal(u_single, reference)
        # The two conventions must differ wherever candidate debt is
        # positive (the multiplier at period 3 with beta=0.98 is > 1).
        sim.SIM_DEBT_PENALTY_CONVENTION = "legacy_multiplier"
        u_legacy = self.call_utility(spec)
        # Cells masked by the debt rules (columns below the current debt
        # index) are BIGNEG under both conventions; exclude them.
        unmasked = u_legacy != BIGNEG
        positive_debt_columns = (self.b1 > 0.0)[None, :] & unmasked
        self.assertTrue(
            np.all(u_single[positive_debt_columns]
                   != u_legacy[positive_debt_columns])
        )
        zero_debt_columns = (self.b1 == 0.0)[None, :] & unmasked
        np.testing.assert_array_equal(
            u_single[zero_debt_columns], u_legacy[zero_debt_columns]
        )

    def test_unknown_convention_raises(self):
        sim.SIM_DEBT_PENALTY_CONVENTION = "typo"
        with self.assertRaises(ValueError):
            self.call_utility(make_spec())


if __name__ == "__main__":
    unittest.main()
