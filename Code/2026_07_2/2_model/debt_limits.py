# -*- coding: utf-8 -*-
"""Shared student-loan feasibility rules.

This module owns the economic limits and their debt-grid mappings.  Bellman,
SMM, and forward-simulation choice searches remain in their consumer modules,
but they all call these compiled primitives.
"""

import numpy as np

try:
    from numba import njit, prange
except ImportError:  # Keep validation tests usable in lightweight environments.
    prange = range

    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(function):
            return function

        return decorator


INTEREST_RATE = 0.05
CONSUMPTION_FLOOR = 2000.0


@njit(cache=True)
def get_annual_cap_by_stage(educ_choice, twoy_exp, foury_exp):
    """Return the annual borrowing cap for the chosen education stage."""
    if educ_choice == 1:
        if twoy_exp <= 0:
            return 8391.0
        if twoy_exp == 1:
            return 9309.0
        return 12581.0

    if educ_choice == 2:
        if foury_exp <= 0:
            return 8391.0
        if foury_exp == 1:
            return 9309.0
        return 12581.0

    if educ_choice == 3:
        return 23222.0
    return 0.0


@njit(cache=True)
def get_lifetime_cap_by_stage(educ_choice):
    """Return the lifetime cap for undergraduate or graduate borrowing."""
    if educ_choice == 3:
        return 150000.0
    return 70786.0


@njit(cache=True)
def lower_bound_index(grid, value):
    """First debt-grid index whose value is greater than or equal to value."""
    n = grid.shape[0]
    for k in range(n):
        if grid[k] >= value:
            return k
    return n - 1


@njit(cache=True)
def upper_bound_index(grid, value):
    """Last debt-grid index whose value is less than or equal to value."""
    idx = 0
    for k in range(grid.shape[0]):
        if grid[k] <= value:
            idx = k
        else:
            break
    return idx


@njit(cache=True)
def nearest_grid_index(grid, value):
    """Nearest debt-grid index, preserving the production SMM convention."""
    best_index = 0
    best_distance = (grid[0] - value) ** 2
    for k in range(1, grid.shape[0]):
        distance = (grid[k] - value) ** 2
        if distance < best_distance:
            best_distance = distance
            best_index = k
    return best_index


@njit(cache=True)
def get_debt_region_bounds(
    debt_grid, x2, choice, interest_rate=INTEREST_RATE,
):
    """Bounds for every current grid point used by the Bellman maximizer."""
    n = debt_grid.shape[0]
    lo_idx = np.empty(n, dtype=np.int64)
    hi_idx = np.empty(n, dtype=np.int64)
    educ_choice = int(choice[1])

    if educ_choice == 0:
        for current_index in range(n):
            accrued = (1.0 + interest_rate) * debt_grid[current_index]
            index = lower_bound_index(debt_grid, accrued)
            lo_idx[current_index] = index
            hi_idx[current_index] = index
        return lo_idx, hi_idx, 0

    annual_cap = get_annual_cap_by_stage(
        educ_choice, int(x2[1]), int(x2[2])
    )
    lifetime_cap = get_lifetime_cap_by_stage(educ_choice)
    cap_start = n

    for current_index in range(n):
        accrued = (1.0 + interest_rate) * debt_grid[current_index]
        if accrued >= lifetime_cap:
            index = lower_bound_index(debt_grid, accrued)
            lo_idx[current_index] = index
            hi_idx[current_index] = index
            if cap_start == n:
                cap_start = current_index
        else:
            maximum = accrued + annual_cap
            if maximum > lifetime_cap:
                maximum = lifetime_cap
            lo_idx[current_index] = lower_bound_index(debt_grid, accrued)
            hi_idx[current_index] = upper_bound_index(debt_grid, maximum)
            if hi_idx[current_index] < lo_idx[current_index]:
                hi_idx[current_index] = lo_idx[current_index]

    return lo_idx, hi_idx, cap_start


@njit(cache=True)
def get_simulation_bounds_indices(
    current_debt_indices, x2, choices, debt_grid,
    interest_rate=INTEREST_RATE,
):
    """Row-specific strict debt-grid bounds used in forward simulation."""
    n = current_debt_indices.shape[0]
    lo_idx = np.empty(n, dtype=np.int64)
    hi_idx = np.empty(n, dtype=np.int64)
    cap_region = np.zeros(n, dtype=np.int64)

    for i in range(n):
        accrued = (1.0 + interest_rate) * debt_grid[int(current_debt_indices[i])]
        educ_choice = int(choices[i, 1])
        if educ_choice == 0:
            index = lower_bound_index(debt_grid, accrued)
            lo_idx[i] = index
            hi_idx[i] = index
            cap_region[i] = 1
            continue

        annual_cap = get_annual_cap_by_stage(
            educ_choice, int(x2[i, 1]), int(x2[i, 2])
        )
        lifetime_cap = get_lifetime_cap_by_stage(educ_choice)
        if accrued >= lifetime_cap:
            index = lower_bound_index(debt_grid, accrued)
            lo_idx[i] = index
            hi_idx[i] = index
            cap_region[i] = 1
        else:
            maximum = accrued + annual_cap
            if maximum > lifetime_cap:
                maximum = lifetime_cap
            lo_idx[i] = lower_bound_index(debt_grid, accrued)
            hi_idx[i] = upper_bound_index(debt_grid, maximum)
            if hi_idx[i] < lo_idx[i]:
                hi_idx[i] = lo_idx[i]

    return lo_idx, hi_idx, cap_region


@njit(parallel=True, fastmath=True, cache=True)
def precompute_smm_bounds_indices(
    previous_debt, states, choices, debt_grid,
    interest_rate=INTEREST_RATE,
):
    """Production-SMM bounds using its retained nearest-grid convention."""
    n = previous_debt.shape[0]
    lo_idx = np.empty(n, dtype=np.int64)
    hi_idx = np.empty(n, dtype=np.int64)

    for i in prange(n):
        educ_choice = int(choices[i, 1])
        annual_cap = get_annual_cap_by_stage(
            educ_choice, int(states[i, 1]), int(states[i, 2])
        )
        lifetime_cap = get_lifetime_cap_by_stage(educ_choice)
        maximum = previous_debt[i] * (1.0 + interest_rate) + annual_cap
        if maximum > lifetime_cap:
            maximum = lifetime_cap
        lo_idx[i] = nearest_grid_index(debt_grid, previous_debt[i])
        hi_idx[i] = nearest_grid_index(debt_grid, maximum)

    return lo_idx, hi_idx
