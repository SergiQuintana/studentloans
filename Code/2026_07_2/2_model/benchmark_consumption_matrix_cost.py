"""One-off benchmark for the Bellman schooling-debt calculation.

Purpose
-------
Measure whether it is worth eliminating the full consumption matrix created
before the student-loan maximization in ``model_solution_em.py``.

This script is deliberately self-contained.  It does not modify, import, or
write any production model artifact.  It reproduces the production debt grid,
borrowing limits, utility normalization, consumption-floor convention, and the
``maxdebt=True`` local-window search.  It benchmarks four comparable paths:

1. current-style local search after constructing the full consumption matrix;
2. the same local search with consumption calculated on demand ("fused");
3. exhaustive feasible search after constructing the matrix;
4. the same exhaustive feasible search calculated on demand.

The matrix and fused versions of each search use identical decision logic and
are checked for numerical equality.  Several representative and stress cases
vary shock count, education stage, resources, risk aversion, continuation-value
shape, and debt-grid size.

Run from any directory with the Python environment used for the model::

    python benchmark_consumption_matrix_cost.py

Useful options::

    python benchmark_consumption_matrix_cost.py --quick
    python benchmark_consumption_matrix_cost.py --no-save
    python benchmark_consumption_matrix_cost.py --samples 9 --target-seconds 0.25

Default output (beside this script):

    benchmark_consumption_matrix_cost_results.csv
    benchmark_consumption_matrix_cost_report.json

Both output files and this script can be removed after the decision is made.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    import numba
    from numba import njit
except ImportError as exc:  # pragma: no cover - explicit run-time guidance
    raise SystemExit(
        "This benchmark requires numba. Run it with the same Python environment "
        "used for model_solution_em.py."
    ) from exc


INTEREST_RATE = 0.05
CONSUMPTION_FLOOR = 2000.0
MAX_CALIBRATION_REPEATS = 10_000


def production_debt_grid(points: int = 100) -> np.ndarray:
    """Return the production grid or an index-interpolated stress-test grid."""
    base = np.concatenate(
        (
            np.array([0, 300, 500, 620, 770, 950], dtype=np.float64),
            np.linspace(1166, 3500, 16),
            np.linspace(3720, 8800, 25),
            np.linspace(9200, 20000, 25),
            np.linspace(22700, 100000, 28),
        )
    )
    if points == base.size:
        return base
    old_index = np.arange(base.size, dtype=np.float64)
    new_index = np.linspace(0.0, float(base.size - 1), int(points))
    return np.interp(new_index, old_index, base)


@njit(cache=False)
def annual_cap(education: int, two_year_experience: int, four_year_experience: int) -> float:
    if education == 1:
        if two_year_experience <= 0:
            return 8391.0
        if two_year_experience == 1:
            return 9309.0
        return 12581.0
    if education == 2:
        if four_year_experience <= 0:
            return 8391.0
        if four_year_experience == 1:
            return 9309.0
        return 12581.0
    if education == 3:
        return 23222.0
    return 0.0


@njit(cache=False)
def lifetime_cap(education: int) -> float:
    return 150000.0 if education == 3 else 70786.0


@njit(cache=False)
def lower_bound_index(grid: np.ndarray, value: float) -> int:
    for index in range(grid.size):
        if grid[index] >= value:
            return index
    return grid.size - 1


@njit(cache=False)
def upper_bound_index(grid: np.ndarray, value: float) -> int:
    result = 0
    for index in range(grid.size):
        if grid[index] <= value:
            result = index
        else:
            break
    return result


@njit(cache=False)
def debt_bounds(
    debt_grid: np.ndarray,
    education: int,
    two_year_experience: int,
    four_year_experience: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Reproduce debt_limits.get_debt_region_bounds for schooling choices."""
    n = debt_grid.size
    lo = np.empty(n, dtype=np.int64)
    hi = np.empty(n, dtype=np.int64)
    flow_cap = annual_cap(education, two_year_experience, four_year_experience)
    stock_cap = lifetime_cap(education)
    cap_start = n

    for current_index in range(n):
        accrued = (1.0 + INTEREST_RATE) * debt_grid[current_index]
        if accrued >= stock_cap:
            index = lower_bound_index(debt_grid, accrued)
            lo[current_index] = index
            hi[current_index] = index
            if cap_start == n:
                cap_start = current_index
        else:
            maximum = min(accrued + flow_cap, stock_cap)
            lo[current_index] = lower_bound_index(debt_grid, accrued)
            hi[current_index] = upper_bound_index(debt_grid, maximum)
            if hi[current_index] < lo[current_index]:
                hi[current_index] = lo[current_index]
    return lo, hi, cap_start


@njit(cache=False, inline="always")
def utility_scalar(risk_aversion: float, consumption: float) -> float:
    consumption = max(consumption, CONSUMPTION_FLOOR)
    return 0.1 * ((0.00001 * consumption) ** (1.0 - risk_aversion)) / (
        1.0 - risk_aversion
    )


@njit(cache=False, inline="always")
def candidate_consumption(
    resources: np.ndarray,
    debt_grid: np.ndarray,
    consumption_matrix: np.ndarray,
    row: int,
    next_index: int,
    use_matrix: bool,
) -> float:
    if use_matrix:
        return consumption_matrix[row, next_index]
    return resources[row] + debt_grid[next_index]


@njit(cache=False, inline="always")
def first_floor_feasible(
    resources: np.ndarray,
    debt_grid: np.ndarray,
    consumption_matrix: np.ndarray,
    row: int,
    lo: int,
    hi: int,
    use_matrix: bool,
) -> int:
    for next_index in range(lo, hi + 1):
        consumption = candidate_consumption(
            resources, debt_grid, consumption_matrix, row, next_index, use_matrix
        )
        if consumption >= CONSUMPTION_FLOOR:
            return next_index
    return -1


@njit(cache=False)
def local_window_search(
    resources: np.ndarray,
    debt_grid: np.ndarray,
    continuation: np.ndarray,
    lo_index: np.ndarray,
    hi_index: np.ndarray,
    cap_start: int,
    risk_aversion: float,
    consumption_matrix: np.ndarray,
    use_matrix: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the maxdebt=True local-window decision logic.

    Passing ``use_matrix=True`` reads preconstructed consumption.  Passing
    ``False`` calculates exactly the same candidate consumption on demand.
    """
    n_current = debt_grid.size
    rows = resources.size
    quadrature = rows // n_current
    payoff = np.empty(rows, dtype=np.float64)
    policy = np.empty(rows, dtype=np.int64)

    for shock_index in range(quadrature):
        previous_argmax = 0
        for current_index in range(n_current):
            row = shock_index * n_current + current_index
            lo = int(lo_index[current_index])
            hi = int(hi_index[current_index])

            if current_index >= cap_start:
                use_index = lo
                consumption = candidate_consumption(
                    resources, debt_grid, consumption_matrix, row, use_index, use_matrix
                )
                payoff[row] = utility_scalar(risk_aversion, consumption) + continuation[use_index]
                policy[row] = use_index
                continue

            if current_index == 0:
                first = first_floor_feasible(
                    resources, debt_grid, consumption_matrix, row, lo, hi, use_matrix
                )
                if first < 0:
                    use_index = hi
                    consumption = candidate_consumption(
                        resources, debt_grid, consumption_matrix, row, use_index, use_matrix
                    )
                    payoff[row] = utility_scalar(risk_aversion, consumption) + continuation[use_index]
                    policy[row] = use_index
                    previous_argmax = use_index
                    continue

                best_index = first
                best_value = -np.inf
                for next_index in range(first, hi + 1):
                    consumption = candidate_consumption(
                        resources, debt_grid, consumption_matrix, row, next_index, use_matrix
                    )
                    value = utility_scalar(risk_aversion, consumption) + continuation[next_index]
                    if value > best_value:
                        best_value = value
                        best_index = next_index
                payoff[row] = best_value
                policy[row] = best_index
                previous_argmax = best_index
                continue

            bound_left = max(previous_argmax - 10, lo, current_index)
            bound_right = min(bound_left + 20, hi + 1)
            if bound_right <= bound_left:
                bound_left = max(lo, current_index)
                bound_right = hi + 1

            left_consumption = candidate_consumption(
                resources, debt_grid, consumption_matrix, row, bound_left, use_matrix
            )
            if left_consumption < CONSUMPTION_FLOOR:
                first = first_floor_feasible(
                    resources, debt_grid, consumption_matrix, row, lo, hi, use_matrix
                )
                if first < 0:
                    use_index = hi
                    consumption = candidate_consumption(
                        resources, debt_grid, consumption_matrix, row, use_index, use_matrix
                    )
                    payoff[row] = utility_scalar(risk_aversion, consumption) + continuation[use_index]
                    policy[row] = use_index
                    previous_argmax = use_index
                    continue
                bound_left = max(first, lo, current_index)
                bound_right = hi + 1

            best_index = bound_left
            best_value = -np.inf
            for next_index in range(bound_left, bound_right):
                consumption = candidate_consumption(
                    resources, debt_grid, consumption_matrix, row, next_index, use_matrix
                )
                value = utility_scalar(risk_aversion, consumption) + continuation[next_index]
                if value > best_value:
                    best_value = value
                    best_index = next_index

            # Match production: update and possibly redo the search only when
            # the local argmax is not the rightmost point of the local slice.
            if best_index != bound_right - 1:
                old_argmax = previous_argmax
                previous_argmax = best_index
                if previous_argmax - old_argmax > 9:
                    first = first_floor_feasible(
                        resources, debt_grid, consumption_matrix, row, lo, hi, use_matrix
                    )
                    if first < 0:
                        use_index = hi
                        consumption = candidate_consumption(
                            resources, debt_grid, consumption_matrix, row, use_index, use_matrix
                        )
                        payoff[row] = utility_scalar(risk_aversion, consumption) + continuation[use_index]
                        policy[row] = use_index
                        previous_argmax = use_index
                        continue
                    first = max(first, lo, current_index)
                    best_index = first
                    best_value = -np.inf
                    for next_index in range(first, hi + 1):
                        consumption = candidate_consumption(
                            resources,
                            debt_grid,
                            consumption_matrix,
                            row,
                            next_index,
                            use_matrix,
                        )
                        value = utility_scalar(risk_aversion, consumption) + continuation[next_index]
                        if value > best_value:
                            best_value = value
                            best_index = next_index
                    previous_argmax = best_index

            payoff[row] = best_value
            policy[row] = best_index

    return payoff, policy


@njit(cache=False)
def exhaustive_search(
    resources: np.ndarray,
    debt_grid: np.ndarray,
    continuation: np.ndarray,
    lo_index: np.ndarray,
    hi_index: np.ndarray,
    cap_start: int,
    risk_aversion: float,
    consumption_matrix: np.ndarray,
    use_matrix: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Search every legally feasible, floor-feasible next-debt candidate."""
    n_current = debt_grid.size
    rows = resources.size
    quadrature = rows // n_current
    payoff = np.empty(rows, dtype=np.float64)
    policy = np.empty(rows, dtype=np.int64)

    for shock_index in range(quadrature):
        for current_index in range(n_current):
            row = shock_index * n_current + current_index
            lo = int(lo_index[current_index])
            hi = int(hi_index[current_index])

            if current_index >= cap_start:
                use_index = lo
                consumption = candidate_consumption(
                    resources, debt_grid, consumption_matrix, row, use_index, use_matrix
                )
                payoff[row] = utility_scalar(risk_aversion, consumption) + continuation[use_index]
                policy[row] = use_index
                continue

            first = first_floor_feasible(
                resources, debt_grid, consumption_matrix, row, lo, hi, use_matrix
            )
            if first < 0:
                use_index = hi
                consumption = candidate_consumption(
                    resources, debt_grid, consumption_matrix, row, use_index, use_matrix
                )
                payoff[row] = utility_scalar(risk_aversion, consumption) + continuation[use_index]
                policy[row] = use_index
                continue

            best_index = first
            best_value = -np.inf
            for next_index in range(first, hi + 1):
                consumption = candidate_consumption(
                    resources, debt_grid, consumption_matrix, row, next_index, use_matrix
                )
                value = utility_scalar(risk_aversion, consumption) + continuation[next_index]
                if value > best_value:
                    best_value = value
                    best_index = next_index
            payoff[row] = best_value
            policy[row] = best_index

    return payoff, policy


def build_consumption_matrix(resources: np.ndarray, debt_grid: np.ndarray) -> np.ndarray:
    """The allocation/broadcasting operation whose cost is under study."""
    return resources[:, None] + debt_grid[None, :]


def matrix_local_once(case: "BenchmarkCase") -> tuple[np.ndarray, np.ndarray]:
    matrix = build_consumption_matrix(case.resources, case.debt_grid)
    return local_window_search(
        case.resources,
        case.debt_grid,
        case.continuation,
        case.lo,
        case.hi,
        case.cap_start,
        case.risk_aversion,
        matrix,
        True,
    )


def fused_local_once(case: "BenchmarkCase") -> tuple[np.ndarray, np.ndarray]:
    return local_window_search(
        case.resources,
        case.debt_grid,
        case.continuation,
        case.lo,
        case.hi,
        case.cap_start,
        case.risk_aversion,
        case.dummy_matrix,
        False,
    )


def matrix_exhaustive_once(case: "BenchmarkCase") -> tuple[np.ndarray, np.ndarray]:
    matrix = build_consumption_matrix(case.resources, case.debt_grid)
    return exhaustive_search(
        case.resources,
        case.debt_grid,
        case.continuation,
        case.lo,
        case.hi,
        case.cap_start,
        case.risk_aversion,
        matrix,
        True,
    )


def fused_exhaustive_once(case: "BenchmarkCase") -> tuple[np.ndarray, np.ndarray]:
    return exhaustive_search(
        case.resources,
        case.debt_grid,
        case.continuation,
        case.lo,
        case.hi,
        case.cap_start,
        case.risk_aversion,
        case.dummy_matrix,
        False,
    )


@dataclass
class CaseSpec:
    name: str
    debt_points: int
    shock_nodes: int
    education: int
    two_year_experience: int
    four_year_experience: int
    cash_level: float
    risk_aversion: float
    continuation_profile: str = "smooth"


@dataclass
class BenchmarkCase:
    spec: CaseSpec
    debt_grid: np.ndarray
    resources: np.ndarray
    continuation: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    cap_start: int
    risk_aversion: float
    dummy_matrix: np.ndarray


def make_case(spec: CaseSpec) -> BenchmarkCase:
    debt_grid = production_debt_grid(spec.debt_points)
    standardized_shocks = np.linspace(-2.5, 2.5, spec.shock_nodes)
    shock_dollars = 4500.0 * standardized_shocks
    resources = (
        spec.cash_level
        + shock_dollars[:, None]
        - (1.0 + INTEREST_RATE) * debt_grid[None, :]
    ).reshape(-1)

    scaled_debt = debt_grid / 100000.0
    continuation = -0.25 - 1.35 * scaled_debt - 0.45 * scaled_debt**2
    if spec.continuation_profile == "wavy":
        continuation = continuation + 0.22 * np.sin(8.0 * np.pi * scaled_debt)
    elif spec.continuation_profile == "flat":
        continuation = -0.25 - 0.25 * scaled_debt
    continuation = continuation - 0.55 * (debt_grid > 0.0)

    lo, hi, cap_start = debt_bounds(
        debt_grid,
        spec.education,
        spec.two_year_experience,
        spec.four_year_experience,
    )
    return BenchmarkCase(
        spec=spec,
        debt_grid=debt_grid,
        resources=np.asarray(resources, dtype=np.float64),
        continuation=np.asarray(continuation, dtype=np.float64),
        lo=lo,
        hi=hi,
        cap_start=int(cap_start),
        risk_aversion=float(spec.risk_aversion),
        dummy_matrix=np.empty((1, 1), dtype=np.float64),
    )


def case_specs(quick: bool) -> list[CaseSpec]:
    stages = [
        ("two_year_first", 1, 0, 0),
        ("two_year_later", 1, 2, 0),
        ("four_year_later", 2, 0, 2),
        ("graduate", 3, 0, 0),
    ]
    resources = [("tight", -5000.0), ("typical", 10000.0), ("ample", 30000.0)]
    shocks = [5, 25]
    specs: list[CaseSpec] = []

    if quick:
        stages = [stages[0], stages[-1]]
        resources = [resources[1]]

    for shock_nodes in shocks:
        for stage_name, education, two_exp, four_exp in stages:
            for resource_name, cash in resources:
                specs.append(
                    CaseSpec(
                        name=f"main_{stage_name}_{resource_name}_q{shock_nodes}",
                        debt_points=100,
                        shock_nodes=shock_nodes,
                        education=education,
                        two_year_experience=two_exp,
                        four_year_experience=four_exp,
                        cash_level=cash,
                        risk_aversion=1.4,
                    )
                )

    if not quick:
        # Risk-aversion and nonconcavity checks at production dimensions.
        for risk in (0.6, 2.0, 3.0):
            specs.append(
                CaseSpec(
                    name=f"risk_sigma{risk:g}",
                    debt_points=100,
                    shock_nodes=25,
                    education=2,
                    two_year_experience=0,
                    four_year_experience=2,
                    cash_level=10000.0,
                    risk_aversion=risk,
                )
            )
        for profile in ("flat", "wavy"):
            specs.append(
                CaseSpec(
                    name=f"continuation_{profile}",
                    debt_points=100,
                    shock_nodes=25,
                    education=3,
                    two_year_experience=0,
                    four_year_experience=0,
                    cash_level=10000.0,
                    risk_aversion=1.4,
                    continuation_profile=profile,
                )
            )

        # Stress tests show whether the conclusion changes with a finer grid.
        for debt_points in (200, 400):
            for stage_name, education, two_exp, four_exp in (stages[0], stages[-1]):
                specs.append(
                    CaseSpec(
                        name=f"stress_b{debt_points}_{stage_name}",
                        debt_points=debt_points,
                        shock_nodes=25,
                        education=education,
                        two_year_experience=two_exp,
                        four_year_experience=four_exp,
                        cash_level=10000.0,
                        risk_aversion=1.4,
                    )
                )
    return specs


def benchmark_callable(
    function,
    *args,
    samples: int,
    target_seconds: float,
) -> tuple[float, list[float], int]:
    """Return median seconds/call, all sample timings, and calls/sample."""
    # One untimed call warms Numba and all relevant allocation paths.
    result = function(*args)
    if isinstance(result, tuple):
        _ = float(np.asarray(result[0]).reshape(-1)[0])
    else:
        _ = float(np.asarray(result).reshape(-1)[0])

    start = time.perf_counter()
    function(*args)
    single = max(time.perf_counter() - start, 1e-9)
    calls = int(max(1, min(MAX_CALIBRATION_REPEATS, target_seconds / single)))

    timings: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _sample in range(samples):
            start = time.perf_counter()
            checksum = 0.0
            for _call in range(calls):
                result = function(*args)
                if isinstance(result, tuple):
                    checksum += float(np.asarray(result[0]).reshape(-1)[0])
                else:
                    checksum += float(np.asarray(result).reshape(-1)[0])
            elapsed = time.perf_counter() - start
            if not math.isfinite(checksum):
                raise RuntimeError("Non-finite benchmark checksum")
            timings.append(elapsed / calls)
    finally:
        if gc_was_enabled:
            gc.enable()
    return statistics.median(timings), timings, calls


def max_value_difference(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]) -> float:
    return float(np.max(np.abs(np.asarray(left[0]) - np.asarray(right[0]))))


def policy_difference_share(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]) -> float:
    return float(np.mean(np.asarray(left[1]) != np.asarray(right[1])))


def run_case(case: BenchmarkCase, samples: int, target_seconds: float) -> dict:
    # Compile and validate before timing.
    matrix_local_result = matrix_local_once(case)
    fused_local_result = fused_local_once(case)
    matrix_full_result = matrix_exhaustive_once(case)
    fused_full_result = fused_exhaustive_once(case)

    local_fusion_value_error = max_value_difference(matrix_local_result, fused_local_result)
    local_fusion_policy_error = policy_difference_share(matrix_local_result, fused_local_result)
    full_fusion_value_error = max_value_difference(matrix_full_result, fused_full_result)
    full_fusion_policy_error = policy_difference_share(matrix_full_result, fused_full_result)
    if local_fusion_value_error > 1e-12 or local_fusion_policy_error != 0.0:
        raise AssertionError(f"Local matrix/fused mismatch in {case.spec.name}")
    if full_fusion_value_error > 1e-12 or full_fusion_policy_error != 0.0:
        raise AssertionError(f"Full matrix/fused mismatch in {case.spec.name}")

    build_seconds, _, build_calls = benchmark_callable(
        build_consumption_matrix,
        case.resources,
        case.debt_grid,
        samples=samples,
        target_seconds=target_seconds,
    )
    matrix_local_seconds, _, matrix_local_calls = benchmark_callable(
        matrix_local_once,
        case,
        samples=samples,
        target_seconds=target_seconds,
    )
    fused_local_seconds, _, fused_local_calls = benchmark_callable(
        fused_local_once,
        case,
        samples=samples,
        target_seconds=target_seconds,
    )
    matrix_full_seconds, _, matrix_full_calls = benchmark_callable(
        matrix_exhaustive_once,
        case,
        samples=samples,
        target_seconds=target_seconds,
    )
    fused_full_seconds, _, fused_full_calls = benchmark_callable(
        fused_exhaustive_once,
        case,
        samples=samples,
        target_seconds=target_seconds,
    )

    candidate_counts = case.hi - case.lo + 1
    matrix_bytes = case.resources.size * case.debt_grid.size * 8
    local_vs_full_value_error = max_value_difference(matrix_local_result, matrix_full_result)
    local_vs_full_policy_error = policy_difference_share(matrix_local_result, matrix_full_result)

    return {
        **asdict(case.spec),
        "mean_feasible_candidates": float(np.mean(candidate_counts)),
        "median_feasible_candidates": float(np.median(candidate_counts)),
        "max_feasible_candidates": int(np.max(candidate_counts)),
        "matrix_bytes": int(matrix_bytes),
        "matrix_mib": matrix_bytes / (1024.0**2),
        "build_us": 1e6 * build_seconds,
        "matrix_local_total_us": 1e6 * matrix_local_seconds,
        "fused_local_total_us": 1e6 * fused_local_seconds,
        "matrix_full_total_us": 1e6 * matrix_full_seconds,
        "fused_full_total_us": 1e6 * fused_full_seconds,
        "local_fusion_speedup": matrix_local_seconds / fused_local_seconds,
        "full_fusion_speedup": matrix_full_seconds / fused_full_seconds,
        "build_share_of_matrix_local": min(1.0, build_seconds / matrix_local_seconds),
        "local_vs_full_max_value_difference": local_vs_full_value_error,
        "local_vs_full_policy_difference_share": local_vs_full_policy_error,
        "local_fusion_max_value_difference": local_fusion_value_error,
        "full_fusion_max_value_difference": full_fusion_value_error,
        "timing_calls_build": build_calls,
        "timing_calls_matrix_local": matrix_local_calls,
        "timing_calls_fused_local": fused_local_calls,
        "timing_calls_matrix_full": matrix_full_calls,
        "timing_calls_fused_full": fused_full_calls,
    }


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(rows: list[dict], projected_workers: int) -> dict:
    production = [row for row in rows if row["debt_points"] == 100]
    main = [row for row in production if row["name"].startswith("main_")]
    decision_rows = main or production
    speedups = [float(row["local_fusion_speedup"]) for row in decision_rows]
    full_speedups = [float(row["full_fusion_speedup"]) for row in decision_rows]
    build_shares = [float(row["build_share_of_matrix_local"]) for row in decision_rows]

    median_speedup = statistics.median(speedups)
    p25_speedup = percentile(speedups, 25)
    if median_speedup >= 1.25 and p25_speedup >= 1.10:
        recommendation = "IMPLEMENT_FUSION"
        explanation = (
            "Removing the matrix produces a material and reasonably consistent speedup "
            "at the current 100-point grid."
        )
    elif median_speedup >= 1.10:
        recommendation = "LOW_PRIORITY_OR_PROFILE_END_TO_END"
        explanation = (
            "The isolated kernel improves, but the likely full-solver gain is modest. "
            "Implement only if schooling debt calculations dominate end-to-end runtime."
        )
    else:
        recommendation = "DO_NOT_IMPLEMENT_FOR_SPEED"
        explanation = (
            "At the current grid size, eliminating the matrix does not provide enough "
            "isolated speedup to justify a production refactor."
        )

    amdahl = {}
    for schooling_share in (0.25, 0.50, 0.75):
        overall = 1.0 / ((1.0 - schooling_share) + schooling_share / median_speedup)
        amdahl[f"if_education_kernel_is_{int(100 * schooling_share)}pct_of_solver"] = overall

    q25_b100 = [
        row for row in decision_rows if row["shock_nodes"] == 25 and row["debt_points"] == 100
    ]
    representative_matrix_mib = statistics.median(
        [float(row["matrix_mib"]) for row in q25_b100 or decision_rows]
    )

    return {
        "recommendation": recommendation,
        "recommendation_explanation": explanation,
        "decision_case_count": len(decision_rows),
        "median_local_fusion_speedup": median_speedup,
        "p25_local_fusion_speedup": p25_speedup,
        "p75_local_fusion_speedup": percentile(speedups, 75),
        "median_exhaustive_fusion_speedup": statistics.median(full_speedups),
        "median_matrix_build_share_of_current_style_path": statistics.median(build_shares),
        "projected_total_solver_speedups": amdahl,
        "representative_q25_b100_matrix_mib_per_process": representative_matrix_mib,
        "representative_matrix_mib_at_projected_workers": (
            representative_matrix_mib * projected_workers
        ),
        "maximum_local_vs_exhaustive_policy_difference_share": max(
            float(row["local_vs_full_policy_difference_share"]) for row in rows
        ),
        "maximum_local_vs_exhaustive_value_difference": max(
            float(row["local_vs_full_max_value_difference"]) for row in rows
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller smoke benchmark instead of the generous default suite.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=7,
        help="Independent timing samples per implementation and case (default: 7).",
    )
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=0.15,
        help="Approximate time per timing sample after calibration (default: 0.15).",
    )
    parser.add_argument(
        "--projected-workers",
        type=int,
        default=60,
        help="Worker count used only for the concurrent-memory projection (default: 60).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print results without writing CSV and JSON output files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples < 3:
        raise SystemExit("--samples must be at least 3")
    if args.target_seconds <= 0:
        raise SystemExit("--target-seconds must be positive")

    specs = case_specs(args.quick)
    print(
        f"Running {len(specs)} cases with {args.samples} timing samples each. "
        "The first case includes Numba compilation and may pause before reporting.",
        flush=True,
    )
    started = time.perf_counter()
    rows: list[dict] = []
    for index, spec in enumerate(specs, start=1):
        case = make_case(spec)
        row = run_case(case, samples=args.samples, target_seconds=args.target_seconds)
        rows.append(row)
        print(
            f"[{index:02d}/{len(specs):02d}] {spec.name:<43} "
            f"matrix={row['matrix_local_total_us']:9.1f} us  "
            f"fused={row['fused_local_total_us']:9.1f} us  "
            f"speedup={row['local_fusion_speedup']:5.2f}x  "
            f"build-share={100.0 * row['build_share_of_matrix_local']:5.1f}%",
            flush=True,
        )

    summary = summarize(rows, projected_workers=args.projected_workers)
    environment = {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.perf_counter() - started,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
        "numba_version": numba.__version__,
        "quick": bool(args.quick),
        "samples": int(args.samples),
        "target_seconds": float(args.target_seconds),
    }
    report = {"environment": environment, "summary": summary, "cases": rows}

    print("\nDecision summary")
    print("----------------")
    print(f"Recommendation: {summary['recommendation']}")
    print(summary["recommendation_explanation"])
    print(
        "Median current-style fusion speedup: "
        f"{summary['median_local_fusion_speedup']:.3f}x "
        f"(p25={summary['p25_local_fusion_speedup']:.3f}x, "
        f"p75={summary['p75_local_fusion_speedup']:.3f}x)"
    )
    print(
        "Median matrix-build share of current-style path: "
        f"{100.0 * summary['median_matrix_build_share_of_current_style_path']:.1f}%"
    )
    print(
        "Median exhaustive-search fusion speedup: "
        f"{summary['median_exhaustive_fusion_speedup']:.3f}x"
    )
    print("Projected full-solver speedups from the median kernel result:")
    for label, value in summary["projected_total_solver_speedups"].items():
        print(f"  {label}: {value:.3f}x")
    print(
        f"Representative Q=25, B=100 matrix: "
        f"{summary['representative_q25_b100_matrix_mib_per_process']:.2f} MiB/process; "
        f"{summary['representative_matrix_mib_at_projected_workers']:.1f} MiB "
        f"across {args.projected_workers} simultaneous workers."
    )
    print(
        "Maximum diagnostic difference between the current local search and "
        "exhaustive search: policy share="
        f"{summary['maximum_local_vs_exhaustive_policy_difference_share']:.3%}, "
        f"value={summary['maximum_local_vs_exhaustive_value_difference']:.6g}."
    )

    if not args.no_save:
        directory = Path(__file__).resolve().parent
        csv_path = directory / "benchmark_consumption_matrix_cost_results.csv"
        json_path = directory / "benchmark_consumption_matrix_cost_report.json"
        write_csv(csv_path, rows)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved detailed cases: {csv_path}")
        print(f"Saved full report:    {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
