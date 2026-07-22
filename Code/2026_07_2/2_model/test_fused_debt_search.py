# -*- coding: utf-8 -*-
"""Equivalence tests for the optional matrix-free Bellman search."""

import os
import unittest
from unittest.mock import patch

import numpy as np

from debt_limits import CONSUMPTION_FLOOR, INTEREST_RATE, get_debt_region_bounds
from fused_debt_search import (
    FUSED_SEARCH_ENV_VAR,
    fused_consumption_search_enabled,
    get_maximum_loop_modified_resources_maxdebt,
)


def production_debt_grid():
    return np.concatenate(
        (
            np.array([0, 300, 500, 620, 770, 950], dtype=np.float64),
            np.linspace(1166, 3500, 16),
            np.linspace(3720, 8800, 25),
            np.linspace(9200, 20000, 25),
            np.linspace(22700, 100000, 28),
        )
    )


def power_utility(sigma_u, consumption):
    consumption = np.maximum(consumption, CONSUMPTION_FLOOR)
    return 0.1 * (
        (0.00001 * consumption) ** (1.0 - sigma_u)
        / (1.0 - sigma_u)
    )


def legacy_matrix_reference(
    sigma_u, debt_grid, next_debt_grid, resources, continuation, choice, x2
):
    """Python transcription of get_maximum_loop_modified_c_maxdebt."""
    consumption = resources[:, None] + next_debt_grid
    continuation = continuation[:, 0]
    ncont = continuation.shape[0]
    quadrature = consumption.shape[0] // ncont
    payoff = np.zeros(consumption.shape[0])
    lo_idx, hi_idx, cap_start = get_debt_region_bounds(debt_grid, x2, choice)

    for shock_index in range(quadrature):
        amax_new = 0
        for it in range(ncont):
            row_idx = it + ncont * shock_index
            c2 = consumption[row_idx]

            if it >= cap_start:
                idx_use = int(lo_idx[it])
                payoff[row_idx] = (
                    power_utility(sigma_u, max(c2[idx_use], CONSUMPTION_FLOOR))
                    + continuation[idx_use]
                )
                continue

            lo = int(lo_idx[it])
            hi = int(hi_idx[it])

            if it == 0:
                c2new = c2[lo : hi + 1]
                c2new = c2new[c2new >= CONSUMPTION_FLOOR]
                if len(c2new) == 0:
                    idx_use = hi
                    payoff[row_idx] = (
                        power_utility(
                            sigma_u, max(c2[idx_use], CONSUMPTION_FLOOR)
                        )
                        + continuation[idx_use]
                    )
                    amax_new = idx_use
                else:
                    firstbound = hi + 1 - len(c2new)
                    final = (
                        power_utility(sigma_u, c2new)
                        + continuation[firstbound : hi + 1]
                    )
                    amax = int(np.argmax(final))
                    amax_new = amax + firstbound
                    payoff[row_idx] = final[amax]
                continue

            bound_left = max(amax_new - 10, lo)
            bound_left = max(bound_left, it)
            bound_right = min(bound_left + 20, hi + 1)
            if bound_right <= bound_left:
                bound_left = max(lo, it)
                bound_right = hi + 1

            if c2[bound_left] < CONSUMPTION_FLOOR:
                c2new = c2[lo : hi + 1]
                c2new = c2new[c2new >= CONSUMPTION_FLOOR]
                if len(c2new) == 0:
                    idx_use = hi
                    payoff[row_idx] = (
                        power_utility(
                            sigma_u, max(c2[idx_use], CONSUMPTION_FLOOR)
                        )
                        + continuation[idx_use]
                    )
                    amax_new = idx_use
                    continue

                bound_left = hi + 1 - len(c2new)
                bound_left = max(bound_left, lo)
                bound_left = max(bound_left, it)
                bound_right = hi + 1

            final = (
                power_utility(sigma_u, c2[bound_left:bound_right])
                + continuation[bound_left:bound_right]
            )
            amax = int(np.argmax(final))

            if amax != len(final) - 1:
                amax_old = amax_new
                amax_new = amax + bound_left

                if (amax_new - amax_old) > 9:
                    c2new = c2[lo : hi + 1]
                    c2new = c2new[c2new >= CONSUMPTION_FLOOR]
                    if len(c2new) == 0:
                        idx_use = hi
                        payoff[row_idx] = (
                            power_utility(
                                sigma_u, max(c2[idx_use], CONSUMPTION_FLOOR)
                            )
                            + continuation[idx_use]
                        )
                        amax_new = idx_use
                        continue

                    firstbound = hi + 1 - len(c2new)
                    firstbound = max(firstbound, lo)
                    firstbound = max(firstbound, it)
                    final = (
                        power_utility(sigma_u, c2[firstbound : hi + 1])
                        + continuation[firstbound : hi + 1]
                    )
                    amax = int(np.argmax(final))
                    amax_new = amax + firstbound

            payoff[row_idx] = final[amax]

    return payoff


class FusedDebtSearchTests(unittest.TestCase):
    def test_feature_flag_defaults_on_and_accepts_common_false_values(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(fused_consumption_search_enabled())

        for value in ("0", "false", "FALSE", "no", "off"):
            with self.subTest(value=value):
                with patch.dict(
                    os.environ, {FUSED_SEARCH_ENV_VAR: value}, clear=True
                ):
                    self.assertFalse(fused_consumption_search_enabled())

        with patch.dict(
            os.environ, {FUSED_SEARCH_ENV_VAR: "1"}, clear=True
        ):
            self.assertTrue(fused_consumption_search_enabled())

    def test_matches_legacy_matrix_search_across_production_cases(self):
        debt_grid = production_debt_grid()
        scaled_debt = debt_grid / debt_grid[-1]

        cases = []
        stages = (
            (1, 0, 0),
            (1, 2, 0),
            (2, 0, 2),
            (3, 0, 0),
        )
        for shock_nodes in (5, 25):
            shocks = np.linspace(-2.5, 2.5, shock_nodes) * 4500.0
            for education, two_exp, four_exp in stages:
                choice = np.array([0, education, 2], dtype=np.int64)
                x2 = np.array([0, two_exp, four_exp], dtype=np.int64)
                for cash_level in (-5000.0, 10000.0, 30000.0):
                    resources = (
                        cash_level
                        + shocks[:, None]
                        - (1.0 + INTEREST_RATE) * debt_grid[None, :]
                    ).reshape(-1)
                    for sigma_u in (0.6, 1.4, 3.0):
                        continuation = (
                            -0.25
                            - 1.35 * scaled_debt
                            - 0.45 * scaled_debt**2
                            + 0.22 * np.sin(8.0 * np.pi * scaled_debt)
                            - 0.55 * (debt_grid > 0.0)
                        )[:, None]
                        cases.append(
                            (
                                shock_nodes,
                                education,
                                two_exp,
                                four_exp,
                                cash_level,
                                sigma_u,
                                resources,
                                continuation,
                                choice,
                                x2,
                            )
                        )

        for case in cases:
            (
                shock_nodes,
                education,
                two_exp,
                four_exp,
                cash_level,
                sigma_u,
                resources,
                continuation,
                choice,
                x2,
            ) = case
            label = (
                f"q={shock_nodes},educ={education},two={two_exp},"
                f"four={four_exp},cash={cash_level},sigma={sigma_u}"
            )
            with self.subTest(label=label):
                expected = legacy_matrix_reference(
                    sigma_u,
                    debt_grid,
                    debt_grid,
                    resources,
                    continuation.copy(),
                    choice,
                    x2,
                )
                actual = get_maximum_loop_modified_resources_maxdebt(
                    sigma_u,
                    debt_grid,
                    debt_grid,
                    resources,
                    continuation.copy(),
                    choice,
                    x2,
                )
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

        # When Numba is installed, the calls above must have produced a genuine
        # nopython specialization rather than silently falling back to Python.
        if hasattr(
            get_maximum_loop_modified_resources_maxdebt,
            "nopython_signatures",
        ):
            self.assertGreater(
                len(
                    get_maximum_loop_modified_resources_maxdebt.nopython_signatures
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
