"""Tests for the epsilon-inversion core (extract_budget_shock_epsilons).

All tests run on synthetic arrays -- no server data. They verify the
inversion against brute-force scans of the kernel argmax: interval
endpoints, the one-sided zero/cap bounds, the infeasible and degenerate
flags, and the monotonicity property the bisection relies on.
"""

import unittest

import numpy as np

import extract_budget_shock_epsilons as ex
import model_fitloans_dynamic as mfd


def brute_force_interval(budget, continuation, b_idx, max_idx, j_obs, sigma,
                         eps_low=-2.0e5, eps_high=2.0e5, step=25.0):
    """Scan the kernel argmax over an epsilon grid for ONE observation."""
    grid_size = mfd.debt_range.size
    columns = np.arange(grid_size)[None, :]
    window = (columns >= b_idx) & (columns <= max_idx)
    matches = []
    for eps in np.arange(eps_low, eps_high, step):
        choice = ex.kernel_argmax(
            np.asarray([eps]), np.asarray([budget]), continuation[None, :],
            window, np.asarray([max_idx]), sigma,
        )[0]
        if choice == j_obs:
            matches.append(eps)
    if not matches:
        return None
    return min(matches), max(matches)


class InversionCoreTests(unittest.TestCase):
    def setUp(self):
        self.sigma = 1.5
        self.grid = mfd.debt_range
        self.B = self.grid.size
        # A mildly decreasing continuation in debt keeps interior choices
        # optimal over bands of epsilon (borrow only when resources are low).
        self.continuation = -1.0e-5 * self.grid.astype(np.float64)
        # Budget low enough that borrowing matters, comfortably above ruin.
        self.budget = float(mfd.CONSUMPTION_FLOOR) + 1500.0

    def _invert_single(self, j_obs, b_idx=0, max_idx=None):
        max_idx = self.B - 1 if max_idx is None else max_idx
        lo, hi, status = ex.invert_cell_type(
            np.asarray([self.budget]),
            self.continuation[None, :].copy(),
            np.asarray([b_idx]), np.asarray([max_idx]),
            np.asarray([j_obs]), self.sigma, tolerance=0.5,
        )
        return lo[0], hi[0], status[0]

    def test_interior_interval_matches_brute_force(self):
        j_obs = 2
        lo, hi, status = self._invert_single(j_obs)
        self.assertEqual(status, ex.STATUS_INTERIOR)
        scan = brute_force_interval(
            self.budget, self.continuation, 0, self.B - 1, j_obs, self.sigma,
        )
        self.assertIsNotNone(scan)
        scan_lo, scan_hi = scan
        # Brute-force step is 25 dollars; endpoints must agree to that.
        self.assertLess(abs(lo - scan_lo), 30.0)
        if np.isfinite(hi):
            self.assertLess(abs(hi - scan_hi), 30.0)

    def test_zero_row_reports_lower_bound_only(self):
        lo, hi, status = self._invert_single(j_obs=0)
        self.assertEqual(status, ex.STATUS_ZERO)
        self.assertTrue(np.isfinite(lo))
        self.assertTrue(np.isinf(hi) and hi > 0)
        # Just above the bound the kernel must pick the zero point; just
        # below it must not.
        window = np.ones((1, self.B), dtype=bool)
        above = ex.kernel_argmax(
            np.asarray([lo + 5.0]), np.asarray([self.budget]),
            self.continuation[None, :], window,
            np.asarray([self.B - 1]), self.sigma,
        )[0]
        below = ex.kernel_argmax(
            np.asarray([lo - 5.0]), np.asarray([self.budget]),
            self.continuation[None, :], window,
            np.asarray([self.B - 1]), self.sigma,
        )[0]
        self.assertEqual(above, 0)
        self.assertGreater(below, 0)

    def test_cap_row_reports_upper_bound_only(self):
        j_cap = self.B - 1
        lo, hi, status = self._invert_single(j_obs=j_cap)
        self.assertEqual(status, ex.STATUS_CAP)
        self.assertTrue(np.isinf(lo) and lo < 0)
        self.assertTrue(np.isfinite(hi))
        window = np.ones((1, self.B), dtype=bool)
        below = ex.kernel_argmax(
            np.asarray([hi - 5.0]), np.asarray([self.budget]),
            self.continuation[None, :], window,
            np.asarray([j_cap]), self.sigma,
        )[0]
        above = ex.kernel_argmax(
            np.asarray([hi + 5.0]), np.asarray([self.budget]),
            self.continuation[None, :], window,
            np.asarray([j_cap]), self.sigma,
        )[0]
        self.assertEqual(below, j_cap)
        self.assertLess(above, j_cap)

    def test_out_of_window_is_infeasible(self):
        lo, hi, status = self._invert_single(j_obs=4, b_idx=0, max_idx=3)
        self.assertEqual(status, ex.STATUS_INFEASIBLE)
        self.assertTrue(np.isnan(lo) and np.isnan(hi))

    def test_degenerate_window_is_flagged(self):
        lo, hi, status = self._invert_single(j_obs=3, b_idx=3, max_idx=3)
        self.assertEqual(status, ex.STATUS_DEGENERATE)

    def test_skipped_choice_is_infeasible(self):
        # Make index 1 strictly dominated for every epsilon: its
        # continuation sits far below the line of its neighbors, so the
        # argmax steps over it.
        continuation = self.continuation.copy()
        continuation[1] -= 10.0
        lo, hi, status = ex.invert_cell_type(
            np.asarray([self.budget]), continuation[None, :],
            np.asarray([0]), np.asarray([self.B - 1]),
            np.asarray([1]), self.sigma, tolerance=0.5,
        )
        self.assertEqual(status[0], ex.STATUS_INFEASIBLE)

    def test_argmax_is_nonincreasing_in_epsilon(self):
        rng = np.random.default_rng(3)
        continuation = np.sort(rng.normal(0.0, 0.05, size=self.B))[::-1]
        window = np.ones((1, self.B), dtype=bool)
        previous = self.B
        for eps in np.linspace(-5.0e4, 5.0e4, 401):
            choice = ex.kernel_argmax(
                np.asarray([eps]), np.asarray([self.budget]),
                continuation[None, :].copy(), window,
                np.asarray([self.B - 1]), self.sigma,
            )[0]
            self.assertLessEqual(choice, previous)
            previous = choice

    def test_utility_matches_kernel_formula(self):
        c = np.asarray([mfd.CONSUMPTION_FLOOR, 25000.0, 60000.0])
        for sigma in (0.5, 1.0, 2.5):
            expected = np.where(
                abs(sigma - 1.0) < 1e-8,
                0.1 * np.log(0.00001 * c),
                0.1 * ((0.00001 * c) ** (1.0 - sigma)) / (1.0 - sigma),
            )
            np.testing.assert_allclose(
                ex.kernel_utility(c, sigma), expected, rtol=1e-12,
            )
        # Below the floor is infeasible.
        self.assertEqual(
            ex.kernel_utility(
                np.asarray([mfd.CONSUMPTION_FLOOR - 1.0]), 1.5
            )[0],
            -np.inf,
        )


if __name__ == "__main__":
    unittest.main()
