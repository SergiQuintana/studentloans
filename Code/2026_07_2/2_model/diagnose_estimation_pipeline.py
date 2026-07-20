"""Read-only, staged diagnostics for the structural-estimation pipeline.

The script traces nonfinite values from the auxiliary EM artifacts through
initial home-CCP prediction, the iteration-zero Bellman recursion, and the
prepared structural likelihood.  Production CCP, EVT, VJT, estimate, and
likelihood files are never modified.

Recommended server runs
-----------------------
Start with the exhaustive initial-CCP lineage check::

    python diagnose_estimation_pipeline.py --stage initial-ccp --workers 20

If needed, replay the complete iteration-zero Bellman recursion without
writing its outputs::

    python diagnose_estimation_pipeline.py --stage bellman --workers 20

Run every diagnostic stage::

    python diagnose_estimation_pipeline.py --stage full --workers 20

The Bellman stage is computationally comparable to one model solution.  Use
``--types`` and ``--x1-indices`` for a targeted run, for example::

    python diagnose_estimation_pipeline.py --stage bellman --types 1 --x1-indices 48
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

import diagnose_likelihood_inputs as likelihood_diagnostics
import model_predict_ccps as ccp_model
import model_solution_em as solution
from config import EST, LIK, OUT
from latent_types import TYPE_IDS, TYPE_NAMES, load_em_posteriors, type_components


DEFAULT_OUTPUT_DIRECTORY = Path(LIK("pipeline_diagnostics"))
DEFAULT_AUXILIARY_FILE = Path(EST("auxiliary_em_results.npz"))
DEFAULT_PARAMETER_FILE = Path(EST("param_g.npy"))
DEFAULT_WORKERS = min(20, max(1, os.cpu_count() or 1))
UNDERFLOW_LOG_THRESHOLD = math.log(np.nextafter(0.0, 1.0))

_WORKER_AUXILIARY_FILE: str | None = None
_WORKER_PARAMETER_VECTOR: np.ndarray | None = None


def shape_text(array):
    return "x".join(str(value) for value in np.shape(array))


def finite_min(array):
    array = np.asarray(array)
    finite = array[np.isfinite(array)]
    return float(np.min(finite)) if finite.size else np.nan


def finite_max(array):
    array = np.asarray(array)
    finite = array[np.isfinite(array)]
    return float(np.max(finite)) if finite.size else np.nan


def numeric_summary(name, array, **labels):
    array = np.asarray(array)
    result = {
        **labels,
        "array": name,
        "shape": shape_text(array),
        "dtype": str(array.dtype),
        "size": int(array.size),
    }
    if np.issubdtype(array.dtype, np.number):
        result.update(
            finite=int(np.count_nonzero(np.isfinite(array))),
            nan=int(np.count_nonzero(np.isnan(array))),
            positive_inf=int(np.count_nonzero(np.isposinf(array))),
            negative_inf=int(np.count_nonzero(np.isneginf(array))),
            finite_min=finite_min(array),
            finite_max=finite_max(array),
        )
    return result


def count_nonfinite(array):
    return int(np.count_nonzero(~np.isfinite(np.asarray(array))))


def state_text(state):
    return np.array2string(np.asarray(state, dtype=int), separator=" ")


def type_name(type_id):
    return str(TYPE_NAMES[int(type_id) - 1])


def parse_integer_selection(text, valid_values, label):
    valid_values = tuple(int(value) for value in valid_values)
    if text is None or str(text).strip().lower() in {"", "all"}:
        return valid_values
    selected = []
    for piece in str(text).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start_text, stop_text = piece.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            selected.extend(range(start, stop + 1))
        else:
            selected.append(int(piece))
    invalid = sorted(set(selected) - set(valid_values))
    if invalid:
        raise ValueError(f"Invalid {label}: {invalid}; valid values are {valid_values}.")
    return tuple(dict.fromkeys(selected))


def write_csv(rows, path, columns=None):
    frame = pd.DataFrame(rows)
    if frame.empty and columns is not None:
        frame = pd.DataFrame(columns=columns)
    frame.to_csv(path, index=False)
    print(f"Saved {path}", flush=True)
    return frame


def walk_numeric_arrays(value, prefix):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield from walk_numeric_arrays(nested, f"{prefix}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, nested in enumerate(value):
            yield from walk_numeric_arrays(nested, f"{prefix}[{index}]")
    else:
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number):
            yield prefix, array


def diagnose_auxiliary(auxiliary_file, parameter_file):
    """Summarize every numeric artifact feeding initial CCP prediction."""
    rows = []
    with np.load(auxiliary_file, allow_pickle=False) as archive:
        for key in archive.files:
            rows.append(numeric_summary(f"auxiliary_archive.{key}", archive[key]))

    q = load_em_posteriors(auxiliary_file)
    rows.append(numeric_summary("posterior_q", q))
    rows.append(numeric_summary("posterior_row_sums", np.sum(q, axis=1)))

    parameter_vector = np.asarray(
        np.load(parameter_file, allow_pickle=False), dtype=float
    )
    rows.append(numeric_summary("structural_param_g", parameter_vector))

    for type_id in TYPE_IDS:
        parameters = ccp_model.load_utility_parameters(
            int(type_id), results_file=str(auxiliary_file)
        )
        for name, array in walk_numeric_arrays(parameters, "initial_ccp_parameters"):
            rows.append(
                numeric_summary(
                    name,
                    array,
                    type_id=int(type_id),
                    type_name=type_name(type_id),
                )
            )
    return rows


def _initial_worker_initializer(auxiliary_file):
    global _WORKER_AUXILIARY_FILE
    _WORKER_AUXILIARY_FILE = str(auxiliary_file)


@lru_cache(maxsize=None)
def _worker_initial_parameters(type_id):
    if _WORKER_AUXILIARY_FILE is None:
        raise RuntimeError("Initial-CCP worker was not initialized.")
    return ccp_model.load_utility_parameters(
        int(type_id), results_file=_WORKER_AUXILIARY_FILE
    )


def _empty_initial_period_summary(period, type_id, x1_index, x1):
    return {
        "period": int(period),
        "type_id": int(type_id),
        "type_name": type_name(type_id),
        "x1_index": int(x1_index),
        "x1": state_text(x1),
        "states": 0,
        "debt_cells": 0,
        "g_nonfinite": 0,
        "expected_consumption_nonfinite": 0,
        "fresh_vjt_nonfinite": 0,
        "fresh_log_denom_nonfinite": 0,
        "fresh_home_log_ccp_nonfinite": 0,
        "fresh_ccp_zero": 0,
        "fresh_ccp_nonfinite": 0,
        "fresh_probability_underflow": 0,
        "stored_ccp_zero": 0,
        "stored_ccp_nonfinite": 0,
        "stored_fresh_mismatch": 0,
        "stored_bundle_missing": 0,
        "stored_key_missing": 0,
        "stored_shape_mismatch": 0,
    }


def _failure_reason_initial(
    g,
    expected_consumption,
    fresh_vjt,
    log_denom,
    home_log_ccp,
    fresh_ccp,
    stored_ccp,
    mismatch,
):
    reasons = []
    if count_nonfinite(g):
        reasons.append("g_nonfinite")
    if count_nonfinite(expected_consumption):
        reasons.append("expected_consumption_nonfinite")
    if count_nonfinite(fresh_vjt):
        reasons.append("fresh_vjt_nonfinite")
    if count_nonfinite(log_denom):
        reasons.append("fresh_log_denom_nonfinite")
    if count_nonfinite(home_log_ccp):
        reasons.append("fresh_home_log_ccp_nonfinite")
    if np.any((fresh_ccp == 0.0) & np.isfinite(home_log_ccp)):
        reasons.append("fresh_probability_underflow")
    if count_nonfinite(fresh_ccp):
        reasons.append("fresh_ccp_nonfinite")
    if stored_ccp is not None:
        if np.any(stored_ccp == 0.0):
            reasons.append("stored_ccp_zero")
        if count_nonfinite(stored_ccp):
            reasons.append("stored_ccp_nonfinite")
        if np.any(mismatch):
            reasons.append("stored_fresh_mismatch")
    return ";".join(reasons)


def scan_initial_ccp_task(task):
    """Scan one (joint type, invariant state) without writing CCP files."""
    type_id, x1_index, max_failures, atol, rtol = task
    parameters = _worker_initial_parameters(int(type_id))
    debt = np.asarray(solution.debt_range, dtype=float)
    x1_row = np.asarray(solution.invariant_states[int(x1_index)], dtype=int)
    inv = x1_row[None, :]
    x1_new = solution.get_x1_new(x1_row)
    summaries = []
    failures = []

    for period in ccp_model.INITIAL_CCP_PERIODS:
        summary = _empty_initial_period_summary(period, type_id, x1_index, x1_row)
        bundle_path = Path(
            ccp_model.initial_ccp_bundle_path(period, x1_row, type_id)
        )
        summary["stored_bundle_mtime"] = (
            bundle_path.stat().st_mtime if bundle_path.exists() else np.nan
        )
        bundle = None
        if bundle_path.exists():
            bundle = np.load(bundle_path, allow_pickle=False)
        else:
            summary["stored_bundle_missing"] = 1

        try:
            for x2_index, x2 in enumerate(solution.get_x2(period)):
                x2 = np.asarray(x2, dtype=int)
                choices = solution.get_possible_choices(x2)
                home = np.flatnonzero(np.all(choices == 0, axis=1))
                if home.size != 1:
                    raise ValueError(
                        f"Expected one home choice; found {home.size} for {x2}."
                    )
                base = int(home[0])
                g = np.asarray(
                    solution.get_all_g(
                        parameters["utility_parameters"],
                        inv,
                        x1_new,
                        x2,
                        choices,
                        period,
                    ),
                    dtype=float,
                )
                expected_consumption = np.asarray(
                    ccp_model.get_expected_consumption(
                        x1_new, x2, choices, type_id, parameters
                    ),
                    dtype=float,
                )
                fresh_vjt = np.asarray(
                    ccp_model.get_vjt_static(
                        parameters,
                        inv,
                        x1_new,
                        x2,
                        choices,
                        period,
                        debt,
                        type_id,
                    ),
                    dtype=float,
                )
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    log_denom = logsumexp(fresh_vjt, axis=1)
                    home_log_ccp = fresh_vjt[:, base] - log_denom
                    fresh_ccp = np.exp(home_log_ccp)

                key = f"ccp_t{period}_{inv}_{x2}"
                stored_ccp = None
                mismatch = np.zeros(fresh_ccp.shape, dtype=bool)
                if bundle is None:
                    summary["stored_key_missing"] += 1
                elif key not in bundle.files:
                    summary["stored_key_missing"] += 1
                else:
                    stored_ccp = np.asarray(bundle[key], dtype=float).reshape(-1)
                    if stored_ccp.shape != fresh_ccp.shape:
                        summary["stored_shape_mismatch"] += 1
                    else:
                        mismatch = ~np.isclose(
                            stored_ccp,
                            fresh_ccp,
                            atol=atol,
                            rtol=rtol,
                            equal_nan=True,
                        )

                summary["states"] += 1
                summary["debt_cells"] += int(fresh_ccp.size)
                summary["g_nonfinite"] += count_nonfinite(g)
                summary["expected_consumption_nonfinite"] += count_nonfinite(
                    expected_consumption
                )
                summary["fresh_vjt_nonfinite"] += count_nonfinite(fresh_vjt)
                summary["fresh_log_denom_nonfinite"] += count_nonfinite(log_denom)
                summary["fresh_home_log_ccp_nonfinite"] += count_nonfinite(
                    home_log_ccp
                )
                summary["fresh_ccp_zero"] += int(np.count_nonzero(fresh_ccp == 0.0))
                summary["fresh_ccp_nonfinite"] += count_nonfinite(fresh_ccp)
                summary["fresh_probability_underflow"] += int(
                    np.count_nonzero((fresh_ccp == 0.0) & np.isfinite(home_log_ccp))
                )
                if stored_ccp is not None and stored_ccp.shape == fresh_ccp.shape:
                    summary["stored_ccp_zero"] += int(
                        np.count_nonzero(stored_ccp == 0.0)
                    )
                    summary["stored_ccp_nonfinite"] += count_nonfinite(stored_ccp)
                    summary["stored_fresh_mismatch"] += int(np.count_nonzero(mismatch))

                reason = _failure_reason_initial(
                    g,
                    expected_consumption,
                    fresh_vjt,
                    log_denom,
                    home_log_ccp,
                    fresh_ccp,
                    stored_ccp,
                    mismatch,
                )
                if reason and len(failures) < max_failures:
                    candidate = np.zeros(fresh_ccp.shape, dtype=bool)
                    candidate |= ~np.isfinite(home_log_ccp)
                    candidate |= fresh_ccp == 0.0
                    if stored_ccp is not None and stored_ccp.shape == fresh_ccp.shape:
                        candidate |= ~np.isfinite(stored_ccp)
                        candidate |= stored_ccp == 0.0
                        candidate |= mismatch
                    debt_indices = np.flatnonzero(candidate)
                    debt_index = int(debt_indices[0]) if debt_indices.size else 0
                    failures.append(
                        {
                            "period": int(period),
                            "type_id": int(type_id),
                            "type_name": type_name(type_id),
                            "x1_index": int(x1_index),
                            "x1": state_text(x1_row),
                            "x2_index": int(x2_index),
                            "x2": state_text(x2),
                            "debt_index": debt_index,
                            "debt": float(debt[debt_index]),
                            "reason": reason,
                            "fresh_home_vjt": float(fresh_vjt[debt_index, base]),
                            "fresh_log_denom": float(log_denom[debt_index]),
                            "fresh_home_log_ccp": float(home_log_ccp[debt_index]),
                            "fresh_ccp": float(fresh_ccp[debt_index]),
                            "stored_ccp": (
                                float(stored_ccp[debt_index])
                                if stored_ccp is not None
                                and stored_ccp.shape == fresh_ccp.shape
                                else np.nan
                            ),
                            "home_choice_index": base,
                            "bundle": str(bundle_path),
                            "key": key,
                        }
                    )
        finally:
            if bundle is not None:
                bundle.close()
        summaries.append(summary)
    return {"summaries": summaries, "failures": failures}


def initial_failure_sort_key(row):
    return (
        -int(row["period"]),
        int(row["type_id"]),
        int(row["x1_index"]),
        int(row["x2_index"]),
        int(row["debt_index"]),
    )


def _fresh_initial_snapshot(failure, auxiliary_file):
    type_id = int(failure["type_id"])
    period = int(failure["period"])
    x1_index = int(failure["x1_index"])
    x2_index = int(failure["x2_index"])
    parameters = ccp_model.load_utility_parameters(
        type_id, results_file=str(auxiliary_file)
    )
    debt = np.asarray(solution.debt_range, dtype=float)
    x1_row = np.asarray(solution.invariant_states[x1_index], dtype=int)
    inv = x1_row[None, :]
    x1_new = solution.get_x1_new(x1_row)
    x2 = np.asarray(solution.get_x2(period)[x2_index], dtype=int)
    choices = solution.get_possible_choices(x2)
    home = int(np.flatnonzero(np.all(choices == 0, axis=1))[0])

    _, grant_type, transfer_type, _ = type_components(type_id)
    x1_design = np.asarray(x1_new, dtype=float).reshape(1, -1)
    state = np.asarray(x2, dtype=float).reshape(1, -1)
    expected_wage = ccp_model.predict_expected_wages(
        x1_design, state, choices, parameters["wage_parameters"]
    )[0]
    repeated_x1 = np.repeat(x1_design, len(choices), axis=0)
    expected_grant = ccp_model.expected_grants_vectorized(
        repeated_x1,
        choices[:, 1],
        choices[:, 2],
        parameters["financial_process"]["grant"],
        grant_type=grant_type,
    )
    expected_transfer = ccp_model.expected_transfers_vectorized(
        repeated_x1,
        choices[:, 1],
        choices[:, 2],
        parameters["financial_process"]["transfer"],
        transfer_type=transfer_type,
    )
    tuition = np.select(
        [choices[:, 1] == 1, choices[:, 1] == 2, choices[:, 1] == 3],
        [4000.0, 8000.0, 14000.0],
        default=0.0,
    )
    expected_consumption = expected_wage + expected_grant + expected_transfer - tuition
    expected_consumption[np.all(choices == 0, axis=1)] = 0.0
    g = solution.get_all_g(
        parameters["utility_parameters"], inv, x1_new, x2, choices, period
    )
    nonhome = np.any(choices != 0, axis=1).astype(float)
    consumption_term = (
        parameters["consumption_coefficient"]
        * expected_consumption[None, :]
        / ccp_model.MONEY_SCALE
    )
    debt_term = (
        parameters["debt_coefficient"]
        * debt[:, None]
        * nonhome[None, :]
        / ccp_model.MONEY_SCALE
    )
    fresh_vjt = g + consumption_term + debt_term
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        log_denom = logsumexp(fresh_vjt, axis=1)
        home_log_ccp = fresh_vjt[:, home] - log_denom
        fresh_ccp = np.exp(home_log_ccp)

    stored_ccp = np.full(debt.shape, np.nan)
    bundle_path = Path(failure["bundle"])
    if bundle_path.exists():
        with np.load(bundle_path, allow_pickle=False) as bundle:
            if failure["key"] in bundle.files:
                stored_ccp = np.asarray(bundle[failure["key"]], dtype=float)
    return {
        "period": np.asarray(period),
        "type_id": np.asarray(type_id),
        "x1_index": np.asarray(x1_index),
        "x1": x1_row,
        "x2_index": np.asarray(x2_index),
        "x2": x2,
        "debt_grid": debt,
        "choices": choices,
        "home_choice_index": np.asarray(home),
        "g": np.asarray(g),
        "expected_wage": np.asarray(expected_wage),
        "expected_grant": np.asarray(expected_grant),
        "expected_transfer": np.asarray(expected_transfer),
        "tuition": np.asarray(tuition),
        "expected_consumption": np.asarray(expected_consumption),
        "consumption_term": np.asarray(consumption_term),
        "debt_term": np.asarray(debt_term),
        "fresh_vjt": np.asarray(fresh_vjt),
        "fresh_log_denom": np.asarray(log_denom),
        "fresh_home_log_ccp": np.asarray(home_log_ccp),
        "fresh_ccp": np.asarray(fresh_ccp),
        "stored_ccp": np.asarray(stored_ccp),
    }


def _bellman_worker_initializer(parameter_file):
    global _WORKER_PARAMETER_VECTOR
    _WORKER_PARAMETER_VECTOR = np.asarray(
        np.load(parameter_file, allow_pickle=False), dtype=float
    )
    solution.reload_budgetshock_params()


def _dict_nonfinite_count(mapping):
    if not isinstance(mapping, dict):
        return count_nonfinite(mapping)
    return sum(count_nonfinite(value) for value in mapping.values())


def _empty_bellman_summary(period, type_id, x1_index, x1):
    return {
        "period": int(period),
        "type_id": int(type_id),
        "type_name": type_name(type_id),
        "x1_index": int(x1_index),
        "x1": state_text(x1),
        "states": 0,
        "incoming_evt_nonfinite": 0,
        "stored_ccp_zero": 0,
        "stored_ccp_nonfinite": 0,
        "choice_vjt_nan": 0,
        "choice_vjt_positive_inf": 0,
        "choice_vjt_negative_inf": 0,
        "g_nonfinite": 0,
        "utility_nan": 0,
        "utility_positive_inf": 0,
        "utility_negative_inf": 0,
        "outgoing_evt_nan": 0,
        "outgoing_evt_positive_inf": 0,
        "outgoing_evt_negative_inf": 0,
        "bundle_missing": 0,
        "key_missing": 0,
        "exception": "",
    }


def _array_problem_counts(array, prefix):
    array = np.asarray(array)
    return {
        f"{prefix}_nan": int(np.count_nonzero(np.isnan(array))),
        f"{prefix}_positive_inf": int(np.count_nonzero(np.isposinf(array))),
        f"{prefix}_negative_inf": int(np.count_nonzero(np.isneginf(array))),
    }


def scan_bellman_task(task):
    """Replay one iteration-zero Bellman task without persisting outputs."""
    type_id, x1_index, max_failures, capture_first = task
    if _WORKER_PARAMETER_VECTOR is None:
        raise RuntimeError("Bellman worker was not initialized.")
    type_id = int(type_id)
    x1_index = int(x1_index)
    x1_row = np.asarray(solution.invariant_states[x1_index], dtype=int)
    inv = x1_row[None, :]
    debt = np.asarray(solution.debt_range, dtype=float)
    utility_parameters = solution.build_param_g(type_id, _WORKER_PARAMETER_VECTOR)
    financial_parameters = solution.get_type_financial_parameters(type_id)
    sigma_u = float(solution.bs.risk_aversion(solution.budget_params, x1_row))
    evt_current = 0
    summaries = []
    failures = []
    snapshot = None

    for period in range(solution.T, 0, -1):
        summary = _empty_bellman_summary(period, type_id, x1_index, x1_row)
        summary["incoming_evt_nonfinite"] = _dict_nonfinite_count(evt_current)
        evt_next = {}
        bundle = None
        bundle_path = None
        if period < solution.T:
            bundle_path = Path(
                OUT(f"ccp/{period}/ccp_t{period}_[{x1_row}]_em{type_id}.npz")
            )
            if bundle_path.exists():
                bundle = np.load(bundle_path, allow_pickle=False)
            else:
                summary["bundle_missing"] = 1
        try:
            for x2_index, x2 in enumerate(solution.get_x2(period)):
                x2 = np.asarray(x2, dtype=int)
                key = f"ccp_t{period}_{inv}_{x2}"
                if period == solution.T:
                    terminal_evt = np.asarray(
                        solution.terminal_from_interp(inv, x2, sigma_u, debt),
                        dtype=float,
                    ).reshape(-1, 1)
                    evt_next[f"evt_t{period}_{inv}_{x2}"] = terminal_evt
                    summary["states"] += 1
                    counts = _array_problem_counts(terminal_evt, "outgoing_evt")
                    for name, value in counts.items():
                        summary[name] += value
                    reason = "terminal_evt_nonfinite" if count_nonfinite(terminal_evt) else ""
                    all_vjt = g = utility = ccp = None
                    outgoing_evt = terminal_evt
                else:
                    x1_new = solution.get_x1_new(x1_row)
                    x2_new = solution.get_x2_new(x2)
                    choices = solution.get_possible_choices(x2)
                    all_vjt = np.zeros((debt.size, choices.shape[0]), dtype=float)
                    for choice_index, choice in enumerate(choices):
                        all_vjt[:, choice_index] = solution.get_expected_conditional(
                            sigma_u,
                            inv,
                            x1_new,
                            x2,
                            x2_new,
                            debt,
                            debt,
                            choice,
                            period,
                            5,
                            evt_current,
                            0,
                            True,
                            financial_parameters,
                        )
                    g = np.asarray(
                        solution.get_all_g(
                            utility_parameters,
                            inv,
                            x1_new,
                            x2,
                            choices,
                            period,
                        ),
                        dtype=float,
                    )
                    utility = all_vjt + g
                    ccp = None
                    if bundle is None:
                        summary["key_missing"] += 1
                    elif key not in bundle.files:
                        summary["key_missing"] += 1
                    else:
                        ccp = np.asarray(bundle[key], dtype=float).reshape(-1)

                    home = np.flatnonzero(np.all(choices == 0, axis=1))
                    if home.size != 1:
                        raise ValueError(
                            f"Expected one home choice; found {home.size} for {x2}."
                        )
                    base = int(home[0])
                    if ccp is None:
                        outgoing_evt = np.full((debt.size, 1), np.nan)
                    else:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            outgoing_evt = (
                                all_vjt[:, base] - np.log(ccp) + solution.gamma
                            )[:, None]
                    evt_next[f"evt_t{period}_{inv}_{x2}"] = outgoing_evt
                    summary["states"] += 1
                    if ccp is not None:
                        summary["stored_ccp_zero"] += int(
                            np.count_nonzero(ccp == 0.0)
                        )
                        summary["stored_ccp_nonfinite"] += count_nonfinite(ccp)
                    for name, value in _array_problem_counts(
                        all_vjt, "choice_vjt"
                    ).items():
                        summary[name] += value
                    summary["g_nonfinite"] += count_nonfinite(g)
                    for name, value in _array_problem_counts(
                        utility, "utility"
                    ).items():
                        summary[name] += value
                    for name, value in _array_problem_counts(
                        outgoing_evt, "outgoing_evt"
                    ).items():
                        summary[name] += value

                    reasons = []
                    if summary["incoming_evt_nonfinite"]:
                        reasons.append("incoming_evt_nonfinite")
                    if ccp is None:
                        reasons.append("stored_ccp_missing")
                    else:
                        if np.any(ccp == 0.0):
                            reasons.append("stored_ccp_zero")
                        if count_nonfinite(ccp):
                            reasons.append("stored_ccp_nonfinite")
                    if count_nonfinite(all_vjt):
                        reasons.append("choice_vjt_nonfinite")
                    if count_nonfinite(g):
                        reasons.append("g_nonfinite")
                    if count_nonfinite(utility):
                        reasons.append("utility_nonfinite")
                    if count_nonfinite(outgoing_evt):
                        reasons.append("outgoing_evt_nonfinite")
                    reason = ";".join(reasons)

                if reason and len(failures) < max_failures:
                    candidate = np.flatnonzero(
                        ~np.isfinite(np.asarray(outgoing_evt).reshape(-1))
                    )
                    if ccp is not None:
                        candidate_ccp = np.flatnonzero(
                            (~np.isfinite(ccp)) | (ccp == 0.0)
                        )
                        if candidate_ccp.size:
                            candidate = candidate_ccp
                    debt_index = int(candidate[0]) if candidate.size else 0
                    failure = {
                        "period": int(period),
                        "type_id": type_id,
                        "type_name": type_name(type_id),
                        "x1_index": x1_index,
                        "x1": state_text(x1_row),
                        "x2_index": int(x2_index),
                        "x2": state_text(x2),
                        "debt_index": debt_index,
                        "debt": float(debt[debt_index]),
                        "reason": reason,
                        "incoming_evt_nonfinite": int(
                            summary["incoming_evt_nonfinite"]
                        ),
                        "stored_ccp": (
                            float(ccp[debt_index]) if ccp is not None else np.nan
                        ),
                        "outgoing_evt": float(
                            np.asarray(outgoing_evt).reshape(-1)[debt_index]
                        ),
                        "bundle": str(bundle_path) if bundle_path else "",
                        "key": key,
                    }
                    failures.append(failure)
                    if capture_first and snapshot is None:
                        snapshot = {
                            "period": np.asarray(period),
                            "type_id": np.asarray(type_id),
                            "x1_index": np.asarray(x1_index),
                            "x1": x1_row.copy(),
                            "x2_index": np.asarray(x2_index),
                            "x2": x2.copy(),
                            "debt_grid": debt.copy(),
                            "incoming_evt_nonfinite": np.asarray(
                                summary["incoming_evt_nonfinite"]
                            ),
                            "stored_ccp": (
                                ccp.copy()
                                if ccp is not None
                                else np.full(debt.shape, np.nan)
                            ),
                            "choice_vjt": (
                                np.asarray(all_vjt).copy()
                                if all_vjt is not None
                                else np.empty((0, 0))
                            ),
                            "g": (
                                np.asarray(g).copy()
                                if g is not None
                                else np.empty(0)
                            ),
                            "utility": (
                                np.asarray(utility).copy()
                                if utility is not None
                                else np.empty((0, 0))
                            ),
                            "outgoing_evt": np.asarray(outgoing_evt).copy(),
                        }
        except Exception as error:
            summary["exception"] = f"{type(error).__name__}: {error}"
            failures.append(
                {
                    "period": int(period),
                    "type_id": type_id,
                    "type_name": type_name(type_id),
                    "x1_index": x1_index,
                    "x1": state_text(x1_row),
                    "x2_index": -1,
                    "x2": "",
                    "debt_index": -1,
                    "debt": np.nan,
                    "reason": "exception",
                    "exception": traceback.format_exc(),
                }
            )
            summaries.append(summary)
            break
        finally:
            if bundle is not None:
                bundle.close()
        summaries.append(summary)
        evt_current = evt_next
    return {"summaries": summaries, "failures": failures, "snapshot": snapshot}


def bellman_failure_sort_key(row):
    return (
        -int(row["period"]),
        int(row["type_id"]),
        int(row["x1_index"]),
        int(row.get("x2_index", -1)),
        int(row.get("debt_index", -1)),
    )


def run_parallel_tasks(worker, tasks, workers, initializer, initargs, label):
    summaries = []
    failures = []
    started = time.perf_counter()
    completed = 0
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initializer, initargs=initargs
    ) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception:
                failures.append(
                    {
                        "period": -1,
                        "type_id": int(task[0]),
                        "type_name": type_name(task[0]),
                        "x1_index": int(task[1]),
                        "reason": "worker_exception",
                        "exception": traceback.format_exc(),
                    }
                )
            else:
                summaries.extend(result["summaries"])
                failures.extend(result["failures"])
            completed += 1
            if completed == 1 or completed % max(1, len(tasks) // 20) == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"[{label}] {completed}/{len(tasks)} tasks completed "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )
    return summaries, failures


def aggregate_period_type(frame, sum_columns):
    if frame.empty:
        return frame
    available = [column for column in sum_columns if column in frame.columns]
    return (
        frame.groupby(["period", "type_id", "type_name"], as_index=False)[available]
        .sum()
        .sort_values(["period", "type_id"])
    )


def run_likelihood_stage(output_directory, auxiliary_file, parameter_file):
    q = load_em_posteriors(auxiliary_file)
    x0 = np.asarray(np.load(parameter_file, allow_pickle=False), dtype=float)
    summary, details, arrays = likelihood_diagnostics.diagnose_likelihood_inputs(
        q, x0, worst_rows=20
    )
    summary.to_csv(output_directory / "likelihood_input_summary.csv", index=False)
    details.to_csv(output_directory / "likelihood_worst_rows.csv", index=False)
    arrays.to_csv(output_directory / "likelihood_array_summary.csv", index=False)
    budget = likelihood_diagnostics.diagnose_budget_parameters()
    budget.to_csv(output_directory / "budgetshock_parameter_summary.csv", index=False)
    return {
        "likelihood_cells": int(len(summary)),
        "likelihood_problem_cells": int(
            np.count_nonzero(
                (summary["utility_nan"] > 0)
                | (summary["utility_positive_inf"] > 0)
                | (summary["stable_log_probability_nonfinite"] > 0)
            )
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("auxiliary", "initial-ccp", "bellman", "likelihood", "full"),
        default="initial-ccp",
        help="Diagnostic stage to run (default: initial-ccp).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel workers for grid scans (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--types",
        default="all",
        help="Comma-separated type IDs/ranges, or 'all' (default: all).",
    )
    parser.add_argument(
        "--x1-indices",
        default="all",
        help="Comma-separated zero-based x1 indices/ranges, or 'all'.",
    )
    parser.add_argument(
        "--max-failures-per-task",
        type=int,
        default=5,
        help="Detailed failures retained per (type,x1) worker (default: 5).",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-12, help="Stored/fresh CCP absolute tolerance."
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-10, help="Stored/fresh CCP relative tolerance."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIRECTORY),
        help="Separate directory for diagnostic outputs.",
    )
    parser.add_argument(
        "--auxiliary-file",
        default=str(DEFAULT_AUXILIARY_FILE),
        help="Auxiliary EM results archive.",
    )
    parser.add_argument(
        "--parameter-file",
        default=str(DEFAULT_PARAMETER_FILE),
        help="Structural param_g vector used by iteration zero.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least one.")
    if args.max_failures_per_task < 1:
        raise ValueError("--max-failures-per-task must be at least one.")

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    auxiliary_file = Path(args.auxiliary_file)
    parameter_file = Path(args.parameter_file)
    selected_types = parse_integer_selection(args.types, TYPE_IDS, "type IDs")
    selected_x1 = parse_integer_selection(
        args.x1_indices,
        range(len(solution.invariant_states)),
        "x1 indices",
    )
    if args.stage == "full":
        stages = ("auxiliary", "initial-ccp", "bellman", "likelihood")
    elif args.stage == "initial-ccp":
        # The recommended first pass should always verify the EM artifacts that
        # feed the fresh CCP calculation before it compares any CCP arrays.
        stages = ("auxiliary", "initial-ccp")
    else:
        stages = (args.stage,)
    run_metadata = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "platform": platform.platform(),
        "stage": args.stage,
        "workers": args.workers,
        "types": list(selected_types),
        "x1_indices": list(selected_x1),
        "auxiliary_file": str(auxiliary_file),
        "parameter_file": str(parameter_file),
        "underflow_log_threshold": UNDERFLOW_LOG_THRESHOLD,
        "production_outputs_modified": False,
        "stage_results": {},
    }
    started = time.perf_counter()

    if "auxiliary" in stages:
        print("\n=== Auxiliary EM and parameter diagnostics ===", flush=True)
        rows = diagnose_auxiliary(auxiliary_file, parameter_file)
        frame = write_csv(rows, output_directory / "auxiliary_array_summary.csv")
        numeric_problem = (
            frame.get("nan", pd.Series(dtype=float)).fillna(0)
            + frame.get("positive_inf", pd.Series(dtype=float)).fillna(0)
            + frame.get("negative_inf", pd.Series(dtype=float)).fillna(0)
        )
        run_metadata["stage_results"]["auxiliary"] = {
            "arrays": int(len(frame)),
            "arrays_with_nonfinite": int(np.count_nonzero(numeric_problem > 0)),
        }

    if "initial-ccp" in stages:
        print("\n=== Fresh and stored initial-CCP diagnostics ===", flush=True)
        tasks = [
            (
                int(type_id),
                int(x1_index),
                args.max_failures_per_task,
                args.atol,
                args.rtol,
            )
            for type_id in selected_types
            for x1_index in selected_x1
        ]
        summaries, failures = run_parallel_tasks(
            scan_initial_ccp_task,
            tasks,
            min(args.workers, len(tasks)),
            _initial_worker_initializer,
            (str(auxiliary_file),),
            "initial CCP",
        )
        failures.sort(key=initial_failure_sort_key)
        summary_frame = write_csv(
            summaries, output_directory / "initial_ccp_task_summary.csv"
        )
        write_csv(failures, output_directory / "initial_ccp_failures.csv")
        aggregate = aggregate_period_type(
            summary_frame,
            [
                "states",
                "debt_cells",
                "g_nonfinite",
                "expected_consumption_nonfinite",
                "fresh_vjt_nonfinite",
                "fresh_log_denom_nonfinite",
                "fresh_home_log_ccp_nonfinite",
                "fresh_ccp_zero",
                "fresh_ccp_nonfinite",
                "fresh_probability_underflow",
                "stored_ccp_zero",
                "stored_ccp_nonfinite",
                "stored_fresh_mismatch",
                "stored_bundle_missing",
                "stored_key_missing",
                "stored_shape_mismatch",
            ],
        )
        aggregate.to_csv(
            output_directory / "initial_ccp_period_type_summary.csv", index=False
        )
        print(
            f"Saved {output_directory / 'initial_ccp_period_type_summary.csv'}",
            flush=True,
        )
        if failures:
            snapshot = _fresh_initial_snapshot(failures[0], auxiliary_file)
            snapshot_path = output_directory / "first_initial_ccp_failure_snapshot.npz"
            np.savez_compressed(snapshot_path, **snapshot)
            print(f"Saved {snapshot_path}", flush=True)
            print(
                "First initial-CCP failure: "
                f"period={failures[0]['period']}, type={failures[0]['type_id']}, "
                f"x1_index={failures[0]['x1_index']}, "
                f"x2_index={failures[0]['x2_index']}, "
                f"debt_index={failures[0]['debt_index']}, "
                f"reason={failures[0]['reason']}",
                flush=True,
            )
        run_metadata["stage_results"]["initial_ccp"] = {
            "tasks": len(tasks),
            "failure_examples": len(failures),
            "first_failure": failures[0] if failures else None,
        }

    if "bellman" in stages:
        print("\n=== Checked iteration-zero Bellman replay ===", flush=True)
        tasks = [
            (int(type_id), int(x1_index), args.max_failures_per_task, False)
            for type_id in selected_types
            for x1_index in selected_x1
        ]
        summaries, failures = run_parallel_tasks(
            scan_bellman_task,
            tasks,
            min(args.workers, len(tasks)),
            _bellman_worker_initializer,
            (str(parameter_file),),
            "Bellman",
        )
        failures.sort(key=bellman_failure_sort_key)
        summary_frame = write_csv(
            summaries, output_directory / "bellman_task_summary.csv"
        )
        write_csv(failures, output_directory / "bellman_failures.csv")
        aggregate = aggregate_period_type(
            summary_frame,
            [
                "states",
                "incoming_evt_nonfinite",
                "stored_ccp_zero",
                "stored_ccp_nonfinite",
                "choice_vjt_nan",
                "choice_vjt_positive_inf",
                "choice_vjt_negative_inf",
                "g_nonfinite",
                "utility_nan",
                "utility_positive_inf",
                "utility_negative_inf",
                "outgoing_evt_nan",
                "outgoing_evt_positive_inf",
                "outgoing_evt_negative_inf",
                "bundle_missing",
                "key_missing",
            ],
        )
        aggregate.to_csv(
            output_directory / "bellman_period_type_summary.csv", index=False
        )
        print(
            f"Saved {output_directory / 'bellman_period_type_summary.csv'}",
            flush=True,
        )
        if failures and failures[0].get("x2_index", -1) >= 0:
            _bellman_worker_initializer(str(parameter_file))
            captured = scan_bellman_task(
                (
                    int(failures[0]["type_id"]),
                    int(failures[0]["x1_index"]),
                    1,
                    True,
                )
            )
            if captured["snapshot"] is not None:
                snapshot_path = output_directory / "first_bellman_failure_snapshot.npz"
                np.savez_compressed(snapshot_path, **captured["snapshot"])
                print(f"Saved {snapshot_path}", flush=True)
            print(
                "First Bellman failure: "
                f"period={failures[0]['period']}, type={failures[0]['type_id']}, "
                f"x1_index={failures[0]['x1_index']}, "
                f"x2_index={failures[0].get('x2_index')}, "
                f"debt_index={failures[0].get('debt_index')}, "
                f"reason={failures[0]['reason']}",
                flush=True,
            )
        run_metadata["stage_results"]["bellman"] = {
            "tasks": len(tasks),
            "failure_examples": len(failures),
            "first_failure": failures[0] if failures else None,
        }

    if "likelihood" in stages:
        print("\n=== Existing prepared-likelihood diagnostics ===", flush=True)
        try:
            result = run_likelihood_stage(
                output_directory, auxiliary_file, parameter_file
            )
        except Exception as error:
            result = {
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            print(
                "Likelihood-input stage could not run. It is read-only and expects "
                "the prepared likelihood arrays to exist.",
                flush=True,
            )
            print(result["error"], flush=True)
        run_metadata["stage_results"]["likelihood"] = result

    run_metadata["elapsed_seconds"] = time.perf_counter() - started
    metadata_path = output_directory / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2, default=str)
    print(f"\nSaved {metadata_path}", flush=True)
    print(
        f"Diagnostics completed in {run_metadata['elapsed_seconds']:.1f}s. "
        "No production artifacts were modified.",
        flush=True,
    )


if __name__ == "__main__":
    main()
