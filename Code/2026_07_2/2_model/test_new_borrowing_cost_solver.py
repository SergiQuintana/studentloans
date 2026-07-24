# -*- coding: utf-8 -*-
"""Tests for the one-shot new-borrowing event cost in the debt maximizers.

Synthetic inputs only; no server estimate files are read (the solver modules
import their local first-stage inputs at module load, but every array used by
these tests is built here). Three kernels are under test:

  * the reference guarded maximizer
    ``model_solution_em.get_maximum_loop_modified_c_maxdebt``;
  * the fused kernel
    ``fused_debt_search.get_maximum_loop_modified_resources_maxdebt``;
  * the grouped kernel ``model_solution_fast._grouped_education_search``.

Convention verified throughout: the row cost is the entry kappa0 when the
row's current debt is exactly zero and the continuation kappa1 otherwise; a
candidate ``k`` is charged iff ``k > lo_idx[it]``, where ``lo_idx`` snaps
accrued debt UP to the grid, so the rollover candidate is never charged even
when its grid value strictly exceeds accrued debt.

Grids with at most 11 points make the +-10 local window cover the full
feasible interval on every row, so the windowed search coincides with the
brute-force optimum and exact (bitwise) equality can be asserted.
"""

import unittest

import numpy as np

from debt_limits import (
    CONSUMPTION_FLOOR,
    INTEREST_RATE,
    get_debt_region_bounds,
)
from fused_debt_search import get_maximum_loop_modified_resources_maxdebt
import model_solution_em as ms
import model_solution_fast as msf


def small_grid():
    """Eleven points: zero debt, a snap-up gap everywhere, and rows whose
    accrued debt exceeds the undergraduate lifetime cap (cap region)."""
    return np.array(
        [0.0, 300.0, 1500.0, 3000.0, 6000.0, 10000.0,
         20000.0, 35000.0, 50000.0, 69000.0, 90000.0]
    )


def power_utility(sigma_u, consumption):
    consumption = np.maximum(consumption, CONSUMPTION_FLOOR)
    return 0.1 * (
        (0.00001 * consumption) ** (1.0 - sigma_u)
        / (1.0 - sigma_u)
    )


def brute_force_reference(sigma_u, debt_grid, next_debt_grid, resources,
                          continuation, choice, x2, kappa0, kappa1):
    """Plain-numpy optimum under the new-borrowing cost convention.

    Exact reference for the kernels only when the local window covers the
    full feasible interval (grids with <= 11 points). Returns payoffs, chosen
    indices, and the number of rows whose optimum is tied (must be zero for
    the index comparison to be meaningful).
    """
    cont = continuation[:, 0]
    n = cont.shape[0]
    quadrature = resources.shape[0] // n
    lo_idx, hi_idx, cap_start = get_debt_region_bounds(debt_grid, x2, choice)
    payoff = np.zeros(resources.shape[0])
    chosen = np.zeros(resources.shape[0], dtype=np.int64)
    ties = 0

    for shock_index in range(quadrature):
        for it in range(n):
            row = it + n * shock_index
            res = resources[row]
            kappa_row = kappa0 if debt_grid[it] == 0.0 else kappa1
            lo = int(lo_idx[it])
            hi = int(hi_idx[it])

            if it >= cap_start:
                # Mechanical rollover: never a new loan, never charged.
                payoff[row] = (
                    power_utility(sigma_u, res + next_debt_grid[lo]) + cont[lo]
                )
                chosen[row] = lo
                continue

            candidates = [
                k for k in range(lo, hi + 1)
                if res + next_debt_grid[k] >= CONSUMPTION_FLOOR
            ]
            if not candidates:
                # Forced max admissible debt at the consumption floor; a new
                # loan (and hence a charge) whenever hi > lo.
                value = (
                    power_utility(sigma_u, res + next_debt_grid[hi]) + cont[hi]
                )
                if kappa_row != 0.0 and hi > lo:
                    value = value + kappa_row
                payoff[row] = value
                chosen[row] = hi
                continue

            best_k = candidates[0]
            best_value = -np.inf
            n_best = 0
            for k in candidates:
                value = (
                    power_utility(sigma_u, res + next_debt_grid[k]) + cont[k]
                )
                if kappa_row != 0.0 and k > lo:
                    value = value + kappa_row
                if value > best_value:
                    best_value = value
                    best_k = k
                    n_best = 1
                elif value == best_value:
                    n_best += 1
            ties += int(n_best > 1)
            payoff[row] = best_value
            chosen[row] = best_k

    return payoff, chosen, ties


class NewBorrowingCostKernelTests(unittest.TestCase):

    def _run_all_kernels(self, sigma_u, debt_grid, next_debt_grid, resources,
                         continuation, choice, x2, kappa0, kappa1):
        """Return (fused, reference-maximizer, grouped) payoff vectors."""
        fused = get_maximum_loop_modified_resources_maxdebt(
            sigma_u, debt_grid, next_debt_grid, resources,
            continuation.copy(), choice, x2, kappa0, kappa1,
        )
        c = resources[..., np.newaxis] + next_debt_grid
        em = ms.get_maximum_loop_modified_c_maxdebt(
            sigma_u, debt_grid, c, continuation.copy(), choice, x2,
            kappa0, kappa1,
        )
        lo_idx, hi_idx, cap_start = get_debt_region_bounds(
            debt_grid, x2, choice
        )
        continuations = np.ascontiguousarray(continuation[:, 0][None, :])
        payoffs = np.zeros((1, resources.shape[0]))
        msf._grouped_education_search(
            sigma_u, next_debt_grid, resources, continuations,
            lo_idx, hi_idx, cap_start, payoffs,
            debt_grid, kappa0, kappa1,
        )
        return fused, em, payoffs[0]

    def _random_cases(self):
        rng = np.random.default_rng(20260723)
        grid = small_grid()
        n = grid.shape[0]
        cases = []
        stages = ((1, 0, 0), (1, 2, 0), (2, 0, 2), (3, 0, 0))
        for education, two_exp, four_exp in stages:
            choice = np.array([0, education, 2], dtype=np.int64)
            x2 = np.array([0, two_exp, four_exp], dtype=np.int64)
            for sigma_u in (0.6, 1.4, 3.0):
                for quadrature in (1, 3):
                    resources = rng.uniform(
                        -9000.0, 40000.0, size=quadrature * n
                    )
                    continuation = (
                        rng.uniform(-8.0, 0.0, size=n)
                        - 0.3 * np.linspace(0.0, 5.0, n)
                    )[:, None]
                    label = (
                        f"educ={education},two={two_exp},four={four_exp},"
                        f"sigma={sigma_u},q={quadrature}"
                    )
                    cases.append(
                        (label, sigma_u, grid, resources, continuation,
                         choice, x2)
                    )
        # Steeply decreasing continuation: many rows choose pure rollover.
        choice = np.array([0, 2, 2], dtype=np.int64)
        x2 = np.array([0, 0, 1], dtype=np.int64)
        resources = rng.uniform(0.0, 30000.0, size=2 * n)
        continuation = -np.linspace(0.0, 50.0, n)[:, None]
        cases.append(
            ("rollover_heavy", 1.4, grid, resources, continuation, choice, x2)
        )
        return cases

    def test_zero_kappa_matches_brute_force_and_legacy_call(self):
        """Zero kappas: every kernel equals the no-kappa brute force, whether
        the kappa arguments are passed explicitly or omitted (defaults)."""
        for case in self._random_cases():
            label, sigma_u, grid, resources, continuation, choice, x2 = case
            with self.subTest(label=label):
                expected, _, _ = brute_force_reference(
                    sigma_u, grid, grid, resources, continuation, choice, x2,
                    0.0, 0.0,
                )
                fused, em, grouped = self._run_all_kernels(
                    sigma_u, grid, grid, resources, continuation, choice, x2,
                    0.0, 0.0,
                )
                self.assertTrue(np.array_equal(fused, expected))
                self.assertTrue(np.array_equal(em, expected))
                self.assertTrue(np.array_equal(grouped, expected))

                # Legacy argument lists (kappa omitted) are the exact same
                # execution path.
                fused_legacy = get_maximum_loop_modified_resources_maxdebt(
                    sigma_u, grid, grid, resources, continuation.copy(),
                    choice, x2,
                )
                c = resources[..., np.newaxis] + grid
                em_legacy = ms.get_maximum_loop_modified_c_maxdebt(
                    sigma_u, grid, c, continuation.copy(), choice, x2,
                )
                lo_idx, hi_idx, cap_start = get_debt_region_bounds(
                    grid, x2, choice
                )
                continuations = np.ascontiguousarray(
                    continuation[:, 0][None, :]
                )
                payoffs = np.zeros((1, resources.shape[0]))
                msf._grouped_education_search(
                    sigma_u, grid, resources, continuations,
                    lo_idx, hi_idx, cap_start, payoffs,
                )
                self.assertTrue(np.array_equal(fused_legacy, expected))
                self.assertTrue(np.array_equal(em_legacy, expected))
                self.assertTrue(np.array_equal(payoffs[0], expected))

    def test_nonzero_kappa_matches_brute_force(self):
        """Nonzero kappas: all kernels equal the brute-force optimum on rows
        with zero debt (entry), positive debt (continuation), cap-region rows,
        and rollover-optimal rows. Payoff equality with a tie-free reference
        pins the chosen index as well."""
        kappa_pairs = ((-800.0, -150.0), (-75.0, -500.0), (-2500.0, 0.0))
        for case in self._random_cases():
            label, sigma_u, grid, resources, continuation, choice, x2 = case
            for kappa0, kappa1 in kappa_pairs:
                with self.subTest(label=label, kappa0=kappa0, kappa1=kappa1):
                    expected, chosen, ties = brute_force_reference(
                        sigma_u, grid, grid, resources, continuation,
                        choice, x2, kappa0, kappa1,
                    )
                    # A tie-free reference means payoff equality identifies
                    # the chosen debt index uniquely.
                    self.assertEqual(ties, 0)
                    fused, em, grouped = self._run_all_kernels(
                        sigma_u, grid, grid, resources, continuation,
                        choice, x2, kappa0, kappa1,
                    )
                    self.assertTrue(np.array_equal(fused, expected))
                    self.assertTrue(np.array_equal(em, expected))
                    self.assertTrue(np.array_equal(grouped, expected))

    def test_rollover_optimum_is_selected_and_uncharged(self):
        """With a prohibitive event cost, every feasible row rolls over at
        ``lo_idx`` and pays nothing."""
        grid = small_grid()
        n = grid.shape[0]
        choice = np.array([0, 2, 2], dtype=np.int64)
        x2 = np.array([0, 0, 1], dtype=np.int64)
        sigma_u = 1.4
        resources = np.full(n, 20000.0)
        continuation = np.zeros((n, 1))
        kappa0, kappa1 = -1.0e6, -1.0e6

        lo_idx, hi_idx, cap_start = get_debt_region_bounds(grid, x2, choice)
        expected, chosen, _ = brute_force_reference(
            sigma_u, grid, grid, resources, continuation, choice, x2,
            kappa0, kappa1,
        )
        fused, em, grouped = self._run_all_kernels(
            sigma_u, grid, grid, resources, continuation, choice, x2,
            kappa0, kappa1,
        )
        self.assertTrue(np.array_equal(fused, expected))
        self.assertTrue(np.array_equal(em, expected))
        self.assertTrue(np.array_equal(grouped, expected))
        for it in range(n):
            lo = int(lo_idx[it])
            # Every row can afford rollover here, so the optimum is lo and
            # the payoff carries no kappa despite kappa being prohibitive.
            self.assertEqual(int(chosen[it]), lo)
            self.assertEqual(
                fused[it],
                float(power_utility(sigma_u, resources[it] + grid[lo])),
            )

    def test_snap_up_entry_convention(self):
        """A row whose accrued debt lies strictly between grid points: the
        snapped-up ``lo_idx`` candidate is a positive new loan in dollars but
        is NOT charged (it is the rollover candidate by convention)."""
        grid = small_grid()
        it = 1  # current debt 300 > 0: continuation cost applies to k > lo
        accrued = (1.0 + INTEREST_RATE) * grid[it]
        choice = np.array([0, 2, 2], dtype=np.int64)
        x2 = np.array([0, 0, 1], dtype=np.int64)
        lo_idx, _, _ = get_debt_region_bounds(grid, x2, choice)
        lo = int(lo_idx[it])
        # Confirm the snap-up scenario is real: grid[lo] strictly exceeds
        # accrued debt, i.e. choosing lo means borrowing a few new dollars.
        self.assertGreater(grid[lo], accrued)

        n = grid.shape[0]
        sigma_u = 1.4
        resources = np.full(n, 20000.0)
        continuation = np.zeros((n, 1))  # flat: utility favors max borrowing
        kappa0, kappa1 = -1.0e6, -1.0e6
        fused, em, grouped = self._run_all_kernels(
            sigma_u, grid, grid, resources, continuation, choice, x2,
            kappa0, kappa1,
        )
        uncharged_rollover = float(
            power_utility(sigma_u, resources[it] + grid[lo])
        )
        self.assertEqual(fused[it], uncharged_rollover)
        self.assertEqual(em[it], uncharged_rollover)
        self.assertEqual(grouped[it], uncharged_rollover)

    def test_window_exclusion_forces_rollover_candidate(self):
        """Production 100-point grid: the warm-started +-10 window sits far
        above ``lo_idx`` after an attractive high-debt row, so without the
        forced evaluation the uncharged rollover candidate would be missed."""
        grid = ms.get_debt_range()
        n = grid.shape[0]
        choice = np.array([0, 3, 0], dtype=np.int64)  # graduate: no cap rows
        x2 = np.array([0, 0, 0], dtype=np.int64)
        sigma_u = 1.4
        resources = np.full(n, 30000.0)
        # Large continuation bonus above index 60 pulls the row-0 argmax (and
        # with it the local window) far to the right.
        continuation = np.where(np.arange(n) >= 60, 6.0, 0.0)[:, None]
        kappa0, kappa1 = 0.0, -1.0e5  # entry free, continuation prohibitive

        lo_idx, hi_idx, cap_start = get_debt_region_bounds(grid, x2, choice)
        self.assertEqual(cap_start, n)  # no cap region in this setup
        fused, em, grouped = self._run_all_kernels(
            sigma_u, grid, grid, resources, continuation, choice, x2,
            kappa0, kappa1,
        )
        self.assertTrue(np.array_equal(fused, em))
        self.assertTrue(np.array_equal(fused, grouped))

        # Row 0 has zero debt and a zero entry cost: the interior optimum
        # with the continuation bonus is selected.
        hi0 = int(hi_idx[0])
        row0_values = (
            power_utility(sigma_u, resources[0] + grid[: hi0 + 1])
            + continuation[: hi0 + 1, 0]
        )
        self.assertEqual(fused[0], float(np.max(row0_values)))
        self.assertGreaterEqual(int(np.argmax(row0_values)), 60)

        # Rows 1..40: lo_idx is far below the window inherited from row 0 and
        # the continuation cost is prohibitive, so the exact payoff is the
        # uncharged rollover value. Without the forced lo evaluation these
        # rows would return a value near -1e5.
        for it in range(1, 41):
            lo = int(lo_idx[it])
            self.assertLess(lo, 50)
            rollover_value = float(
                power_utility(sigma_u, resources[it] + grid[lo])
                + continuation[lo, 0]
            )
            self.assertEqual(fused[it], rollover_value)

    def test_dispatcher_and_tracer_refuse_unsupported_paths(self):
        """The maxdebt=False maximizer and the debug tracer predate the event
        cost and must refuse to run with nonzero kappas."""
        grid = small_grid()
        n = grid.shape[0]
        choice = np.array([0, 2, 2], dtype=np.int64)
        x2 = np.array([0, 0, 1], dtype=np.int64)
        resources = np.full(n, 20000.0)
        continuation = np.zeros((n, 1))
        c = resources[..., np.newaxis] + grid
        with self.assertRaises(NotImplementedError):
            ms.get_maximum(
                1.4, c, continuation, None, grid, choice, x2, False,
                -100.0, 0.0,
            )
        # maxdebt=True dispatch with kappas equals the direct kernel call.
        via_dispatch = ms.get_maximum(
            1.4, c, continuation.copy(), None, grid, choice, x2, True,
            -100.0, -50.0,
        )
        direct = ms.get_maximum_loop_modified_c_maxdebt(
            1.4, grid, c, continuation.copy(), choice, x2, -100.0, -50.0,
        )
        self.assertTrue(np.array_equal(via_dispatch, direct))


if __name__ == "__main__":
    unittest.main(verbosity=2)
