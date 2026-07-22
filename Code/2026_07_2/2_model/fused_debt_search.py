# -*- coding: utf-8 -*-
"""Matrix-free consumption/debt maximization for the Bellman solver.

The legacy solver first materializes ``consumption[row, next_debt]`` and then
searches that matrix.  This module evaluates the same consumption entries only
when the maximizer visits them.  Keeping this kernel separate makes the
optimization easy to disable without removing the legacy implementation.
"""

import os

import numpy as np

try:
    from numba import njit
except ImportError:  # Keep lightweight equivalence tests usable without Numba.
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(function):
            return function

        return decorator

from debt_limits import CONSUMPTION_FLOOR, get_debt_region_bounds


FUSED_SEARCH_ENV_VAR = "MODEL_FUSED_CONSUMPTION_SEARCH"


def fused_consumption_search_enabled():
    """Read the process-level feature flag (enabled by default)."""
    value = os.environ.get(FUSED_SEARCH_ENV_VAR)
    if value is None:
        return True
    return value.strip().lower() not in ("0", "false", "no", "off")


@njit(inline="always")
def _flow_utility(sigma_u, consumption):
    """Scalar equivalent of ``model_solution_em.get_power_utility``."""
    if consumption < CONSUMPTION_FLOOR:
        consumption = CONSUMPTION_FLOOR
    return 0.1 * (
        (0.00001 * consumption) ** (1.0 - sigma_u)
        / (1.0 - sigma_u)
    )


@njit
def get_maximum_loop_modified_resources_maxdebt(
    sigma_u, debt_grid, next_debt_grid, resources, continuation, choice, x2
):
    """Run the existing ``maxdebt=True`` search without building its matrix.

    ``resources[row]`` is consumption before adding candidate next-period debt,
    so the matrix entry formerly stored at ``c[row, k]`` is reconstructed as
    ``resources[row] + next_debt_grid[k]``.  The search windows, floor handling,
    cap-region rule, and previous-argmax heuristic intentionally match the
    legacy kernel.
    """
    ncont = continuation.shape[0]
    quadrature = int(resources.shape[0] / ncont)
    payoff = np.zeros(resources.shape[0])

    lo_idx, hi_idx, cap_start = get_debt_region_bounds(debt_grid, x2, choice)

    for shock_index in range(quadrature):
        amax_new = 0

        for it in range(ncont):
            row_idx = it + ncont * shock_index
            resource = resources[row_idx]

            # Once the lifetime cap binds, debt evolves mechanically.
            if it >= cap_start:
                idx_use = lo_idx[it]
                payoff[row_idx] = (
                    _flow_utility(
                        sigma_u, resource + next_debt_grid[idx_use]
                    )
                    + continuation[idx_use, 0]
                )
                continue

            lo = lo_idx[it]
            hi = hi_idx[it]

            if it == 0:
                feasible_count = 0
                for candidate in range(lo, hi + 1):
                    if resource + next_debt_grid[candidate] >= CONSUMPTION_FLOOR:
                        feasible_count += 1

                if feasible_count == 0:
                    idx_use = hi
                    payoff[row_idx] = (
                        _flow_utility(
                            sigma_u, resource + next_debt_grid[idx_use]
                        )
                        + continuation[idx_use, 0]
                    )
                    amax_new = idx_use
                    continue

                firstbound = hi + 1 - feasible_count
                best_index = firstbound
                best_value = (
                    _flow_utility(
                        sigma_u, resource + next_debt_grid[firstbound]
                    )
                    + continuation[firstbound, 0]
                )
                for candidate in range(firstbound + 1, hi + 1):
                    value = (
                        _flow_utility(
                            sigma_u, resource + next_debt_grid[candidate]
                        )
                        + continuation[candidate, 0]
                    )
                    if value > best_value:
                        best_value = value
                        best_index = candidate

                amax_new = best_index
                payoff[row_idx] = best_value
                continue

            # Local window around the previous argmax, clipped to feasibility.
            bound_left = max(amax_new - 10, lo)
            bound_left = max(bound_left, it)
            bound_right = min(bound_left + 20, hi + 1)

            if bound_right <= bound_left:
                bound_left = max(lo, it)
                bound_right = hi + 1

            if resource + next_debt_grid[bound_left] < CONSUMPTION_FLOOR:
                feasible_count = 0
                for candidate in range(lo, hi + 1):
                    if resource + next_debt_grid[candidate] >= CONSUMPTION_FLOOR:
                        feasible_count += 1

                if feasible_count == 0:
                    idx_use = hi
                    payoff[row_idx] = (
                        _flow_utility(
                            sigma_u, resource + next_debt_grid[idx_use]
                        )
                        + continuation[idx_use, 0]
                    )
                    amax_new = idx_use
                    continue

                bound_left = hi + 1 - feasible_count
                bound_left = max(bound_left, lo)
                bound_left = max(bound_left, it)
                bound_right = hi + 1

            best_index = bound_left
            best_value = (
                _flow_utility(
                    sigma_u, resource + next_debt_grid[bound_left]
                )
                + continuation[bound_left, 0]
            )
            for candidate in range(bound_left + 1, bound_right):
                value = (
                    _flow_utility(
                        sigma_u, resource + next_debt_grid[candidate]
                    )
                    + continuation[candidate, 0]
                )
                if value > best_value:
                    best_value = value
                    best_index = candidate

            # Match the legacy rule: a maximum at the window's right boundary
            # does not update the center used for the next current-debt state.
            if best_index != bound_right - 1:
                amax_old = amax_new
                amax_new = best_index

                if (amax_new - amax_old) > 9:
                    feasible_count = 0
                    for candidate in range(lo, hi + 1):
                        if (
                            resource + next_debt_grid[candidate]
                            >= CONSUMPTION_FLOOR
                        ):
                            feasible_count += 1

                    if feasible_count == 0:
                        idx_use = hi
                        payoff[row_idx] = (
                            _flow_utility(
                                sigma_u, resource + next_debt_grid[idx_use]
                            )
                            + continuation[idx_use, 0]
                        )
                        amax_new = idx_use
                        continue

                    firstbound = hi + 1 - feasible_count
                    firstbound = max(firstbound, lo)
                    firstbound = max(firstbound, it)

                    best_index = firstbound
                    best_value = (
                        _flow_utility(
                            sigma_u, resource + next_debt_grid[firstbound]
                        )
                        + continuation[firstbound, 0]
                    )
                    for candidate in range(firstbound + 1, hi + 1):
                        value = (
                            _flow_utility(
                                sigma_u, resource + next_debt_grid[candidate]
                            )
                            + continuation[candidate, 0]
                        )
                        if value > best_value:
                            best_value = value
                            best_index = candidate
                    amax_new = best_index

            payoff[row_idx] = best_value

    return payoff
