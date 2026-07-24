# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 14:31:48 2025

@author: S.Quintana

SMM estimation algorithm to fit student loans distribution,
estimating a budget shock process.

ADJUSTED (as requested):
  A) Estimation uses a DEBT PENALTY that is:
        dp0 (constant baseline) + dp2*1{par=2} + dp3*1{par=3} + dp4*1{par=4}
     i.e. "constant + 3 deviations".

  B) For now we ONLY enforce this on the ESTIMATION side.

  C) Saved budgetshock_params.npy stores:
        "debt_pen_parinc": length-4 array [dp0, dp2, dp3, dp4]
"""

import numpy as np
import joblib
import os
import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from numba import get_num_threads, njit, prange, set_num_threads
from scipy.optimize import dual_annealing, minimize

from model_em_algorithm import (
    load_data_superfeasible,
    get_feasible,
    get_feasible_pubid,
    get_x1_new as expand_x1,
    _state_wage_design,
    load_fixed_wage_parameters,
)

from model_solution_em import (
    move_state_grad,
    get_x1_new,
    probability_graduation,
    get_debt_range,
    get_educ_level,
)

from model_simulation_em import tuition_agents
from model_interpolate_terminal import build_interpolator_dictionary

from config import DIR, OUT, EST, RDATA, ENSURE_DIR
import budget_shock as bs
from debt_limits import (
    CONSUMPTION_FLOOR,
    INTEREST_RATE,
    get_annual_cap_by_stage,
    get_lifetime_cap_by_stage,
    nearest_grid_index,
    precompute_smm_bounds_indices,
)
from financial_process import (
    load_auxiliary_financial_process,
    draw_grants_vectorized,
    draw_transfers_vectorized,
)
from latent_types import (
    N_TYPES,
    TYPE_IDS,
    TYPE_NAMES,
    TYPE_GRANT,
    TYPE_TRANSFER,
    TYPE_LOAN,
    validate_q,
    validate_saved_layout,
)

# Canonical roots from config (no chdir anywhere)
PATH_OUT        = DIR["MODEL_OUTPUT"]
PATH_EST        = DIR["MODEL_ESTIMATES"]
PATH_CONT_FINAL = DIR["MODEL_CONTINUATION_FINAL"]

T = 10
r = INTEREST_RATE
beta = 0.98

debt_range = get_debt_range()
_EM_POSTERIORS = None
_CELL_WORKER_CONTEXTS = None
_CELL_WORKER_MOMENT_SPEC = None
_CELL_WORKER_PRIMARY_WEIGHT = None
DEFAULT_CCP_WORKERS = max(1, min(16, os.cpu_count() or 1))
CCP_CACHE_MODES = ("off", "reuse", "rebuild")
CCP_CELL_CACHE_SCHEMA = 1
EDUCATION_CELL_SPECIFICATIONS = (
    "parental_income_basic", "parental_income_loan_type", "joint_type",
)
TYPE_INTEGRATION_MODES = ("sampled", "exact")
PARENTAL_INCOME_MOMENT_SPECS = (
    "fast_stock", "flow_stock", "fast_flow", "flow_plus_stock",
)
# Additional debt-status-split moment specification (see
# ``parental_income_split_moments``). It is deliberately NOT added to
# PARENTAL_INCOME_MOMENT_SPECS so every existing moment function keeps its
# exact current validation; the multicell objective branches on it instead.
SPLIT_MOMENT_SPEC = "flow_split_stock"
EXTENDED_PARENTAL_INCOME_MOMENT_SPECS = (
    PARENTAL_INCOME_MOMENT_SPECS + (SPLIT_MOMENT_SPEC,)
)
SPLIT_MOMENT_MINIMUM_GROUP_N = 30
STOCK_MOMENT_WEIGHT = 2.0
# Estimate the one-shot new-borrowing event cost (kappa) block. When False,
# the production SMM keeps its exact current behavior: a 68-entry multicell
# vector, unchanged moments, and kernels whose zero-kappa path is numerically
# identical to the pre-kappa code.
ESTIMATE_NEW_BORROWING_COST = False
NEW_BORROWING_COST_BOUNDS = (-2.0, 0.0)
EDUCATION_CELL_RESOURCE_MODES = ("simulated", "observed")
DEFAULT_PRIMARY_MOMENT_WEIGHT = 4.0
DEFAULT_EDUCATION_CELL_MAXITER = 5000
# Opt-in DFO-LS least-squares optimizer defaults (optimizer="dfols"; see
# Agents_Readme/Tasks/STUDENT_LOANS_FIT_MASTER_PLAN.md, section 5). All of
# these are inert on the default "hybrid"/"nelder-mead"/"dual-annealing"
# paths, whose behavior is unchanged.
DFOLS_WARM_START_MAXFUN = 300
DFOLS_COLD_MAXFUN_PER_PARAMETER = 35
DFOLS_MAXFUN_CAP = 3000
DFOLS_COLD_RHOBEG = 0.2
DFOLS_WARM_RHOBEG = 0.05
DFOLS_RESTART_PERTURBATION_SHARE = 0.1
UNCAPPED_EDUCATION_CELL_MAXITER = np.iinfo(np.int32).max


def load_full_em_posteriors():
    """Load posterior weights in the shared joint-type ordering."""
    global _EM_POSTERIORS
    if _EM_POSTERIORS is not None:
        return _EM_POSTERIORS

    with np.load(EST("auxiliary_em_results.npz"), allow_pickle=False) as results:
        validate_saved_layout(
            results["type_names"],
            results["type_school"],
            results["type_grant"],
            results["type_transfer"],
            results["type_loan"],
        )
        _EM_POSTERIORS = validate_q(results["q"])
    return _EM_POSTERIORS

# ==============================================================================
# Savers

def _ensure_est_dir():
    os.makedirs(PATH_EST, exist_ok=True)

def save_budgetshock_estimates(
    best_x: np.ndarray,
    periods: list[int],
    filename_prefix: str = "budgetshock",
    loan_heterogeneity: str = "homogeneous",
    index_kind: str = "model_period",
    estimation_parameterization: str = "joint_type",
    estimation_metadata=None,
):
    """
    Saves:
      - risk_aversion.npy        (ra_levels, length 4)
      - budgetshock_params.npy   (dict with periods, mu_blocks, sigma_e, debt_pen_parinc)
      - optional: budgetshock_bestx.npy (raw optimizer vector)

    debt_pen_parinc is length-4: [dp0, dp2, dp3, dp4]
    where dp0 is baseline and dp2/dp3/dp4 are deviations added when parinc==2/3/4.
    """
    _ensure_est_dir()

    best_x = np.asarray(best_x, dtype=np.float64)
    if estimation_parameterization == "parental_income_basic":
        budget_params = bs.unpack_parental_income_estimation_vector(
            best_x, periods, index_kind=index_kind
        )
    elif estimation_parameterization == "parental_income_loan_type":
        budget_params = bs.unpack_parental_income_loan_type_estimation_vector(
            best_x, periods, index_kind=index_kind
        )
    elif estimation_parameterization == "parental_income_basic_multicell_shared_risk_debt":
        budget_params = bs.unpack_parental_income_multicell_estimation_vector(
            best_x, periods, index_kind=index_kind
        )
    elif estimation_parameterization == "joint_type":
        budget_params = bs.unpack_estimation_vector(
            best_x,
            periods,
            loan_heterogeneity=loan_heterogeneity,
            index_kind=index_kind,
        )
    else:
        raise ValueError(
            "Unknown estimation_parameterization."
        )
    if estimation_metadata:
        budget_params.update(dict(estimation_metadata))
    bs.save(budget_params, raw_vector=best_x, filename_prefix=filename_prefix)

    if filename_prefix == "budgetshock":
        print(f"[saved] {EST('risk_aversion.npy')}")
        print(f"[saved] {EST('budgetshock_params.npy')}")
    else:
        print(f"[saved] {EST(f'{filename_prefix}_risk_aversion.npy')}")
        print(f"[saved] {EST(f'{filename_prefix}_params.npy')}")
    print(f"[saved] {EST(f'{filename_prefix}_bestx.npy')}")

# ==============================================================================
# Store interp_dict in cache memory

CACHE_DIR = ENSURE_DIR(OUT("cache"))

def get_interp_dict_cached(force_rebuild=False):
    cache_path = os.path.join(CACHE_DIR, "interp_dict.joblib")

    if (not force_rebuild) and os.path.exists(cache_path):
        print(f"[cache] Loading interp_dict from {cache_path}")
        interp_dict = joblib.load(cache_path)
        return interp_dict

    print("[cache] Building interp_dict (expensive)...")
    interp_dict, meta_dict, missing, context = build_interpolator_dictionary(
        pathcont=PATH_CONT_FINAL,
        debt_grid=debt_range,
        fields=8,
        lastschool_horizon=None,
        sex_filters=None,
        race_filters=None,
        verbose=False,
    )

    print(f"[cache] Saving interp_dict to {cache_path}")
    joblib.dump(interp_dict, cache_path, compress=3)
    return interp_dict

# ==============================================================================
# Continuation / CCP path

def _ccp_bundle_path(period, x1i, em_type):
    return Path(OUT(
        "evt_ccp", str(period + 1),
        f"evt_ccp_sequence_t{period+1}_{x1i}_em{em_type}.npz",
    ))


def _continuation_from_bundle(x1i, x2i, ji, period, bundle):
    """Extract the same individual continuation as the legacy loader."""
    graduation_possible = (
        ((x2i[1] >= 1) & (ji[1] == 1) & (x2i[4] == 0))
        | ((x2i[2] >= 3) & (ji[1] == 2) & (x2i[5] == 0))
        | (ji[1] == 3)
    )
    notgrad_x2 = move_state_grad(x2i, ji, period)
    nograd_key = f"evt_ccp_sequence_t{period+1}_{x1i}_{notgrad_x2}"
    evt_nograd = np.asarray(bundle[nograd_key], dtype=np.float64)
    if not graduation_possible:
        return evt_nograd

    grad_x2 = move_state_grad(x2i, ji, period, grad=1)
    grad_key = f"evt_ccp_sequence_t{period+1}_{x1i}_{grad_x2}"
    evt_grad = np.asarray(bundle[grad_key], dtype=np.float64)
    p_grad = probability_graduation(get_x1_new(x1i), x2i, ji)
    return p_grad * evt_grad + (1.0 - p_grad) * evt_nograd


def get_individual_continuation(i, x1, x2, j, period, em_type):
    """Compatibility wrapper for a single continuation lookup."""
    x1i = x1[i, :].astype(np.int64)
    x2i = x2[i, :].astype(np.int64)
    ji = j[i, :]
    with np.load(_ccp_bundle_path(period, x1i, em_type)) as bundle:
        return _continuation_from_bundle(x1i, x2i, ji, period, bundle)


def load_ccp_path(x1, state, choices, period, em_type):
    """Load one type's N x 100 path, opening each x1 archive only once.

    Results are numerically identical to the original individual loop. Within
    an invariant-state group, repeated (x2, choice) requests are also reused.
    """
    x1 = np.asarray(x1)
    state = np.asarray(state)
    choices = np.asarray(choices)
    sequence = np.zeros((x1.shape[0], debt_range.size), dtype=np.float64)
    if period == 9 or x1.shape[0] == 0:
        return sequence

    x1_integer = x1.astype(np.int64)
    unique_x1, group_index = np.unique(x1_integer, axis=0, return_inverse=True)
    for group, x1i in enumerate(unique_x1):
        rows = np.flatnonzero(group_index == group)
        continuation_cache = {}
        with np.load(_ccp_bundle_path(period, x1i, em_type)) as bundle:
            for row in rows:
                x2i = state[row].astype(np.int64)
                ji = choices[row]
                request = (
                    tuple(x2i.tolist()),
                    tuple(np.asarray(ji).tolist()),
                )
                value = continuation_cache.get(request)
                if value is None:
                    value = _continuation_from_bundle(x1i, x2i, ji, period, bundle)
                    continuation_cache[request] = value
                sequence[row] = value
    return sequence


def _load_ccp_type_worker(arguments):
    type_index, x1, state, choices, period, em_type = arguments
    return type_index, load_ccp_path(x1, state, choices, period, em_type)


def load_ccp_paths_parallel(x1, state, choices, period, workers=DEFAULT_CCP_WORKERS):
    """Build all 16 type paths, parallelizing independent type files."""
    workers = max(1, min(int(workers), len(TYPE_IDS)))
    arguments = [
        (type_index, x1, state, choices, period, em_type)
        for type_index, em_type in enumerate(TYPE_IDS)
    ]
    if workers == 1 or period == 9 or len(x1) == 0:
        results = map(_load_ccp_type_worker, arguments)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_load_ccp_type_worker, arguments, chunksize=1)
            results = list(results)
    ordered = [None] * len(TYPE_IDS)
    for type_index, path in results:
        ordered[type_index] = path
    return np.ascontiguousarray(np.stack(ordered, axis=0), dtype=np.float64)

# ==============================================================================
# Budget construction from observed data

def get_budget(income, debt, j):
    """
    Computes the budget without the budget shock using observed data.
    """
    w = income[:, 0]
    grants = income[:, 1]
    transfers = income[:, 2:].sum(axis=1)  # parental help + loans (your convention)

    full_part_time = j[:, 2].copy()
    wage = np.exp(w[:, None]) * full_part_time[:, None] * 0.5 * (40 * 52)

    budget = (grants + transfers + wage[:, 0] - (1 + r) * debt - tuition_agents(0, j))
    return budget.astype(np.float64)

# ==============================================================================
# Terminal interpolation helpers

def get_model(x1, x2final, interp_dict):
    sex = int(x1[2])
    eth = int(x1[3])
    educ = get_educ_level(x2final)
    major = int(x2final[8])
    lastschool = int(x2final[7])
    return interp_dict[(sex, eth, lastschool, educ, major)]

def move_states_T(x1i, x2i, ji, period, interp_dict):
    """
    Move state until terminal (T), using actual choice in current period and then home production.
    Return (model_grad, model_notgrad, p_grad).
    """
    home = np.array([0, 0, 0])
    graduated = 0

    for rem in range(period, T):
        if rem == period:
            if ((x2i[1] >= 1) & (ji[1] == 1) & (x2i[4] == 0)) | ((x2i[2] >= 3) & (ji[1] == 2) & (x2i[5] == 0)) | (ji[1] == 3):
                x2_next_grad    = move_state_grad(x2i, ji, period, grad=1)
                x2_next_notgrad = move_state_grad(x2i, ji, period, grad=0)
                x1_new = get_x1_new(x1i)
                p_grad = probability_graduation(x1_new, x2i, ji)
                graduated = 1
            else:
                p_grad = 0.0
                x2_next = move_state_grad(x2i, ji, period, grad=0)
        else:
            if graduated == 0:
                x2_next = move_state_grad(x2_next, home, period, grad=0)
            else:
                x2_next_grad    = move_state_grad(x2_next_grad, home, period, grad=0)
                x2_next_notgrad = move_state_grad(x2_next_notgrad, home, period, grad=0)

    if graduated == 0:
        model_grad = 0
        model_notgrad = get_model(x1i, x2_next, interp_dict)
    else:
        model_grad = get_model(x1i, x2_next_grad, interp_dict)
        model_notgrad = get_model(x1i, x2_next_notgrad, interp_dict)

    return model_grad, model_notgrad, float(p_grad)

def get_continuation(x1, state, choices, period, interp_dict):
    """
    Returns lists of interpolators and p_grad vector:
      [terminal_grad_list, terminal_notgrad_list, p_grad_array]
    """
    n = x1.shape[0]
    p_grad = np.zeros(n, dtype=np.float64)
    model_grad = []
    model_notgrad = []

    for i in range(n):
        mg, mn, pg = move_states_T(
            x1[i, :].astype("int"),
            state[i, :].astype("int"),
            choices[i, :],
            period,
            interp_dict,
        )
        model_grad.append(mg)
        model_notgrad.append(mn)
        p_grad[i] = pg

    return model_grad, model_notgrad, p_grad

# ==============================================================================
# Load fundamentals per period

def load_fundamentals(
    period, interp_dict, clear_data=False, ccp_workers=DEFAULT_CCP_WORKERS,
):
    """
    Loads and returns everything needed for estimation for a given period.
    """
    if clear_data:
        get_feasible()
        get_feasible_pubid()

    x1, state, debt, debtchoice, choices, income = load_data_superfeasible(period, return_income=True)
    types = load_full_em_posteriors()

    # keep only education choices
    mask = (choices[:, 1] > 0)
    x1 = x1[mask, :]
    state = state[mask, :]
    debt = debt[mask]
    debtchoice = debtchoice[mask]
    income = income[mask]
    types = types[mask]
    choices = choices[mask, :]

    # map debt indices to dollars
    debt = debt_range[debt.astype("int")]
    debtchoice = debt_range[debtchoice.astype("int")]

    # First axis follows the same type ordering as the posterior columns.
    ccp_path = load_ccp_paths_parallel(
        x1, state, choices, period, workers=ccp_workers
    )

    # budget w/o shock
    budget = get_budget(income, debt, choices)

    # terminal interpolators
    terminal_data = list(get_continuation(x1, state, choices, period, interp_dict))

    return x1, state, types, debt, debtchoice, choices, ccp_path, budget, terminal_data


def _hash_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _ccp_source_fingerprint(period, x1):
    """Fingerprint every general sequence bundle needed by this sample."""
    digest = hashlib.sha256()
    if period == 9:
        digest.update(b"terminal-period-zero-ccp")
        return digest.hexdigest()
    for x1i in np.unique(np.asarray(x1, dtype=np.int64), axis=0):
        for em_type in TYPE_IDS:
            path = _ccp_bundle_path(period, x1i, em_type)
            stat = path.stat()
            digest.update(str(path).encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _cell_ccp_cache_paths(period, education, program_year):
    directory = Path(OUT("cache", "fitloans_ccp"))
    stem = f"ccp_educ{education}_year{program_year}_t{period}"
    return directory / f"{stem}.npy", directory / f"{stem}.json"


def _write_cell_ccp_cache(array_path, metadata_path, ccp, metadata):
    array_path.parent.mkdir(parents=True, exist_ok=True)
    array_temp = array_path.with_suffix(array_path.suffix + ".tmp")
    metadata_temp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with array_temp.open("wb") as handle:
        np.save(handle, np.asarray(ccp, dtype=np.float64), allow_pickle=False)
    with metadata_temp.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(array_temp, array_path)
    os.replace(metadata_temp, metadata_path)


def load_education_cell_ccp(
    x1, state, choices, period, education, program_year,
    workers=DEFAULT_CCP_WORKERS, cache_mode="reuse",
):
    """Load/build the compact test cache for one education-program cell."""
    cache_mode = str(cache_mode).lower()
    if cache_mode not in CCP_CACHE_MODES:
        raise ValueError(f"ccp_cache_mode must be one of {CCP_CACHE_MODES}")
    if cache_mode == "off":
        return load_ccp_paths_parallel(x1, state, choices, period, workers=workers)

    array_path, metadata_path = _cell_ccp_cache_paths(
        period, education, program_year
    )
    expected = {
        "schema": CCP_CELL_CACHE_SCHEMA,
        "period": int(period),
        "education": int(education),
        "program_year": int(program_year),
        "education_year_grouping": bs.BUDGET_EDUCATION_YEAR_GROUPING,
        "type_ids": list(TYPE_IDS),
        "shape": [len(TYPE_IDS), int(len(x1)), int(debt_range.size)],
        "data_fingerprint": _hash_arrays(x1, state, choices),
        "source_fingerprint": _ccp_source_fingerprint(period, x1),
    }
    if cache_mode == "reuse" and array_path.exists() and metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                observed = json.load(handle)
            if observed == expected:
                cached = np.load(array_path, allow_pickle=False)
                if list(cached.shape) == expected["shape"]:
                    print(f"  [CCP cell cache] reuse {array_path}")
                    return np.ascontiguousarray(cached, dtype=np.float64)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        print(f"  [CCP cell cache] stale or invalid: {array_path}")

    print(
        f"  [CCP cell cache] build period={period} with {workers} workers "
        f"for {len(x1)} observations"
    )
    ccp = load_ccp_paths_parallel(x1, state, choices, period, workers=workers)
    _write_cell_ccp_cache(array_path, metadata_path, ccp, expected)
    print(f"  [CCP cell cache] saved {array_path}")
    return ccp


def load_education_cell(
    period, interp_dict, education=2, program_year=1,
    ccp_workers=DEFAULT_CCP_WORKERS, ccp_cache_mode="reuse",
):
    """Load an observed education cell while preserving model-period continuation.

    Program year is read from the pre-choice experience stored in ``state``.
    Annual loan flow is the cleaned disbursement measure; debt choice is the
    end-of-period stock.  The two are deliberately not reconstructed from one
    another in the data.
    """
    x1, state, debt_index, debtchoice_index, choices, income = (
        load_data_superfeasible(period, return_income=True)
    )
    q = load_full_em_posteriors()
    flow_path = RDATA(f"loanflow_superfeasible_t{period}.npy")
    if not os.path.exists(flow_path):
        raise FileNotFoundError(
            f"Missing {flow_path}. Rebuild superfeasible data with "
            "model_em_algorithm.get_data_superfeasible()."
        )
    loan_flow = np.asarray(np.load(flow_path), dtype=np.float64)
    lengths = {
        "x1": len(x1), "state": len(state), "debt": len(debt_index),
        "debtchoice": len(debtchoice_index), "choices": len(choices),
        "income": len(income), "q": len(q), "loan_flow": len(loan_flow),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Education-cell inputs are misaligned in period {period}: {lengths}")

    cell_code = bs.budget_education_cell_code(education, program_year)
    observed_code = bs.budget_education_cell_from_state(state, education)
    cell = (
        (choices[:, 1].astype(np.int64) == int(education))
        & (observed_code == cell_code)
    )
    individual_index = np.flatnonzero(cell).astype(np.int64)
    x1 = np.ascontiguousarray(x1[cell])
    state = np.ascontiguousarray(state[cell])
    choices = np.ascontiguousarray(choices[cell])
    income = np.ascontiguousarray(income[cell])
    debt = debt_range[np.asarray(debt_index[cell], dtype=np.int64)]
    debtchoice = debt_range[np.asarray(debtchoice_index[cell], dtype=np.int64)]
    q = validate_q(q[cell], n_individuals=int(cell.sum()))
    loan_flow = np.ascontiguousarray(loan_flow[cell], dtype=np.float64)
    # Per-observation annual borrowing cap for the chosen stage, used by the
    # optional flow_split_stock at-cap moment. The stage arguments follow the
    # shared debt_limits convention: education choice plus pre-choice two-year
    # and four-year experience.
    annual_cap = np.empty(len(state), dtype=np.float64)
    for row in range(len(state)):
        annual_cap[row] = get_annual_cap_by_stage(
            int(education), int(state[row, 1]), int(state[row, 2])
        )
    ccp = load_education_cell_ccp(
        x1, state, choices, period, education, program_year,
        workers=ccp_workers, cache_mode=ccp_cache_mode,
    )
    budget = get_budget(income, debt, choices)
    terminal_cell = list(get_continuation(x1, state, choices, period, interp_dict))
    return {
        "period": int(period),
        "cell_code": int(cell_code),
        "x1": x1,
        "state": state,
        "q": q,
        "individual_index": individual_index,
        "debt": np.ascontiguousarray(debt, dtype=np.float64),
        "debtchoice": np.ascontiguousarray(debtchoice, dtype=np.float64),
        "loan_flow": loan_flow,
        "annual_cap": np.ascontiguousarray(annual_cap, dtype=np.float64),
        "parinc": np.ascontiguousarray(x1[:, 0], dtype=np.int64),
        "choice": choices,
        "ccp_by_type": ccp,
        "observed_budget": np.ascontiguousarray(budget, dtype=np.float64),
        "terminal_data": terminal_cell,
    }


def posterior_loan_moments(parinc, flow_by_type, stock_by_type, q):
    """Loan-type/parental-income-group targets and diagnostics.

    Target ordering within each (loan type, parinc) cell is
    ``[mean positive annual flow, share with positive end stock]``.
    The returned diagnostic is the share with positive annual flow.
    """
    parinc = np.asarray(parinc, dtype=np.int64).reshape(-1)
    flow_by_type = np.asarray(flow_by_type, dtype=np.float64)
    stock_by_type = np.asarray(stock_by_type, dtype=np.float64)
    q = validate_q(q, n_individuals=len(parinc))
    expected_shape = (N_TYPES, len(parinc))
    if flow_by_type.shape != expected_shape or stock_by_type.shape != expected_shape:
        raise ValueError(f"Flow and stock arrays must both have shape {expected_shape}.")

    target, new_share, effective_weight, labels = [], [], [], []
    for loan_type in (0, 1):
        rows = np.flatnonzero(TYPE_LOAN == loan_type)
        weights = q[:, rows].T
        for parinc_level in range(1, 5):
            selected = parinc == parinc_level
            w = weights[:, selected].reshape(-1)
            flow = flow_by_type[rows][:, selected].reshape(-1)
            stock = stock_by_type[rows][:, selected].reshape(-1)
            total = w.sum()
            positive_flow = flow > 0.0
            positive_weight = w[positive_flow].sum()
            mean_flow = (
                np.sum(w[positive_flow] * flow[positive_flow]) / positive_weight
                if positive_weight > 0.0 else np.nan
            )
            stock_share = np.sum(w * (stock > 0.0)) / total if total > 0.0 else np.nan
            flow_share = positive_weight / total if total > 0.0 else np.nan
            target.extend((mean_flow, stock_share))
            new_share.append(flow_share)
            effective_weight.append(total)
            labels.append((loan_type, parinc_level))
    return (
        np.asarray(target), np.asarray(new_share),
        np.asarray(effective_weight), tuple(labels),
    )


def parental_income_loan_moments(parinc, flow_by_type, stock_by_type, q):
    """Existing flow/stock moments collapsed to the four parinc brackets.

    Target ordering within each parinc cell is
    ``[mean positive annual flow, share with positive end stock]``. All 16
    joint types remain integrated with their fixed posterior probabilities,
    but no loan-type-specific moment is targeted.
    """
    parinc = np.asarray(parinc, dtype=np.int64).reshape(-1)
    flow_by_type = np.asarray(flow_by_type, dtype=np.float64)
    stock_by_type = np.asarray(stock_by_type, dtype=np.float64)
    q = validate_q(q, n_individuals=len(parinc))
    expected_shape = (N_TYPES, len(parinc))
    if flow_by_type.shape != expected_shape or stock_by_type.shape != expected_shape:
        raise ValueError(f"Flow and stock arrays must both have shape {expected_shape}.")

    target, new_share, effective_weight, labels = [], [], [], []
    for parinc_level in range(1, 5):
        selected = parinc == parinc_level
        weights = q[selected].T.reshape(-1)
        flow = flow_by_type[:, selected].reshape(-1)
        stock = stock_by_type[:, selected].reshape(-1)
        total = weights.sum()
        positive_flow = flow > 0.0
        positive_weight = weights[positive_flow].sum()
        mean_flow = (
            np.sum(weights[positive_flow] * flow[positive_flow]) / positive_weight
            if positive_weight > 0.0 else np.nan
        )
        stock_share = (
            np.sum(weights * (stock > 0.0)) / total if total > 0.0 else np.nan
        )
        flow_share = positive_weight / total if total > 0.0 else np.nan
        target.extend((mean_flow, stock_share))
        new_share.append(flow_share)
        effective_weight.append(total)
        labels.append(parinc_level)
    return (
        np.asarray(target), np.asarray(new_share),
        np.asarray(effective_weight), tuple(labels),
    )


def parental_income_distribution_moments(
    parinc, flow, stock, moment_spec="fast_stock", q=None, eps=0.01,
):
    """Four fast-style distribution moments for each model parinc group.

    ``fast_stock`` reproduces the moment definitions in model_fitloans_fast:
    mean positive stock, share positive stock, standard deviation of positive
    stock, and its 80th percentile. ``flow_stock`` keeps the stock share but
    uses positive annual loan flow for the other three distribution moments.
    ``fast_flow`` uses annual loan flow for all four moments, including the
    share receiving a positive new loan. ``flow_plus_stock`` adds mean positive
    end-of-period stock and the end-of-period indebtedness share to those four
    flow moments.

    One-dimensional outcomes are ordinary simulated/data observations. Two-
    dimensional ``(16, N)`` outcomes require ``q`` and provide the retained
    exact-posterior validation mode.
    """
    if moment_spec not in PARENTAL_INCOME_MOMENT_SPECS:
        raise ValueError(f"moment_spec must be one of {PARENTAL_INCOME_MOMENT_SPECS}.")
    parinc = np.asarray(parinc, dtype=np.int64).reshape(-1)
    flow = np.asarray(flow, dtype=np.float64)
    stock = np.asarray(stock, dtype=np.float64)
    if flow.shape != stock.shape:
        raise ValueError("flow and stock must have the same shape.")

    if flow.ndim == 1:
        if flow.size != parinc.size:
            raise ValueError("One-dimensional outcomes must align with parinc.")
        flow_values, stock_values = flow, stock
        weights = np.ones(parinc.size, dtype=np.float64)
        groups = parinc
        unweighted = True
    elif flow.shape == (N_TYPES, parinc.size):
        q = validate_q(q, n_individuals=parinc.size)
        flow_values = flow.reshape(-1)
        stock_values = stock.reshape(-1)
        weights = q.T.reshape(-1)
        groups = np.tile(parinc, N_TYPES)
        unweighted = False
    else:
        raise ValueError(
            f"Outcomes must have shape {(parinc.size,)} or {(N_TYPES, parinc.size)}."
        )

    distribution_values = stock_values if moment_spec == "fast_stock" else flow_values
    output, flow_share, effective_weight, labels = [], [], [], []
    for level in range(1, 5):
        selected = groups == level
        w = weights[selected]
        values = distribution_values[selected]
        stocks = stock_values[selected]
        flows = flow_values[selected]
        total = w.sum()
        positive = values > 0.0
        positive_weight = w[positive].sum()
        participation = (
            flows if moment_spec in ("fast_flow", "flow_plus_stock")
            else stocks
        )
        share = (
            np.sum(w * (participation > 0.0)) / total
            if total > 0.0 else eps
        )
        if total <= 0.0 or positive_weight <= 0.0:
            mean, std, p80 = eps, eps, eps
            if moment_spec == "fast_stock":
                share = eps
        else:
            positive_values = values[positive]
            positive_weights = w[positive]
            mean = np.sum(positive_weights * positive_values) / positive_weight
            variance = (
                np.sum(positive_weights * (positive_values - mean) ** 2)
                / positive_weight
            )
            std = np.sqrt(max(float(variance), 0.0))
            p80 = (
                float(np.percentile(positive_values, 80))
                if unweighted
                else _weighted_quantile(positive_values, positive_weights, 0.80)
            )
        output.extend((mean, share, std, p80))
        if moment_spec == "flow_plus_stock":
            positive_stock = stocks > 0.0
            positive_stock_weight = w[positive_stock].sum()
            mean_positive_stock = (
                np.sum(w[positive_stock] * stocks[positive_stock])
                / positive_stock_weight
                if positive_stock_weight > 0.0 else eps
            )
            stock_share = (
                np.sum(w * positive_stock) / total if total > 0.0 else eps
            )
            output.extend((mean_positive_stock, stock_share))
        flow_share.append(np.sum(w * (flows > 0.0)) / total if total > 0.0 else np.nan)
        effective_weight.append(total)
        labels.append(level)
    return (
        np.asarray(output), np.asarray(flow_share),
        np.asarray(effective_weight), tuple(labels),
    )


def parental_income_split_moments(
    parinc, flow, begin_debt, annual_cap, eps=0.01, at_cap_fraction=0.99,
):
    """Five debt-status-split moments for each model parinc group.

    This is the ``flow_split_stock`` moment specification. For every
    parental-income group the target ordering is:

      1. share with a positive new-loan flow given beginning-of-period
         debt == 0 (entry rate);
      2. share with a positive new-loan flow given beginning debt > 0
         (continuation rate);
      3. mean positive flow given beginning debt == 0;
      4. mean positive flow given beginning debt > 0;
      5. share of positive flows at >= ``at_cap_fraction`` of the applicable
         annual borrowing cap (at-cap share, pooling both debt statuses).

    Beginning-of-period debt is an OBSERVED state, so the debt-status split is
    identical in data and simulation: the simulated flow is conditioned on the
    same observed beginning debt, and group membership never moves with the
    parameters. The flow itself must already remove interest accrual
    (``b_next - (1+r)*b_current`` in simulation; the cleaned disbursement
    measure ``loan_flow`` in the data). Empty groups receive the same ``eps``
    numerical floor as the existing distribution moments.
    """
    parinc = np.asarray(parinc, dtype=np.int64).reshape(-1)
    flow = np.asarray(flow, dtype=np.float64).reshape(-1)
    begin_debt = np.asarray(begin_debt, dtype=np.float64).reshape(-1)
    annual_cap = np.asarray(annual_cap, dtype=np.float64).reshape(-1)
    if not (flow.size == begin_debt.size == annual_cap.size == parinc.size):
        raise ValueError(
            "flow, begin_debt, and annual_cap must align with parinc."
        )
    if np.any(annual_cap <= 0.0):
        raise ValueError("annual_cap must be positive for every observation.")

    has_debt = begin_debt > 0.0
    positive_flow = flow > 0.0
    at_cap = positive_flow & (flow >= at_cap_fraction * annual_cap)
    output, flow_share, effective_weight, labels = [], [], [], []
    for level in range(1, 5):
        selected = parinc == level
        entry_group = selected & ~has_debt
        continuation_group = selected & has_debt
        entry_n = int(entry_group.sum())
        continuation_n = int(continuation_group.sum())
        entry_rate = (
            float(np.mean(positive_flow[entry_group])) if entry_n else eps
        )
        continuation_rate = (
            float(np.mean(positive_flow[continuation_group]))
            if continuation_n else eps
        )
        entry_positive = entry_group & positive_flow
        continuation_positive = continuation_group & positive_flow
        mean_entry_flow = (
            float(np.mean(flow[entry_positive]))
            if np.any(entry_positive) else eps
        )
        mean_continuation_flow = (
            float(np.mean(flow[continuation_positive]))
            if np.any(continuation_positive) else eps
        )
        group_positive = selected & positive_flow
        at_cap_share = (
            float(np.mean(at_cap[group_positive]))
            if np.any(group_positive) else eps
        )
        output.extend((
            entry_rate, continuation_rate, mean_entry_flow,
            mean_continuation_flow, at_cap_share,
        ))
        total = float(selected.sum())
        flow_share.append(
            float(np.mean(positive_flow[selected])) if total > 0.0 else np.nan
        )
        effective_weight.append(total)
        labels.append(level)
    return (
        np.asarray(output), np.asarray(flow_share),
        np.asarray(effective_weight), tuple(labels),
    )


def _warn_thin_split_groups(parinc, begin_debt, cell_label):
    """Warn once at data-load when a parinc-by-debt-status group is thin."""
    parinc = np.asarray(parinc, dtype=np.int64).reshape(-1)
    has_debt = np.asarray(begin_debt, dtype=np.float64).reshape(-1) > 0.0
    for level in range(1, 5):
        selected = parinc == level
        for status_mask, status_name in (
            (~has_debt, "beginning debt == 0"),
            (has_debt, "beginning debt > 0"),
        ):
            group_n = int(np.sum(selected & status_mask))
            if group_n < SPLIT_MOMENT_MINIMUM_GROUP_N:
                print(
                    f"WARNING [{cell_label}] {SPLIT_MOMENT_SPEC} group "
                    f"parinc={level}, {status_name}: N={group_n} < "
                    f"{SPLIT_MOMENT_MINIMUM_GROUP_N} observations."
                )


def parental_income_loan_type_distribution_moments(
    parinc, flow, stock, moment_spec="fast_stock", loan_type=None, q=None,
    eps=0.01,
):
    """Four moments for each loan-type by parental-income cell.

    One-dimensional outcomes use one persistent sampled loan type per person.
    ``(16, N)`` outcomes use the full posterior and the shared joint-type map;
    this is used for posterior-weighted data targets and exact validation.
    Ordering is low type parinc 1--4, then high type parinc 1--4.
    """
    if moment_spec not in PARENTAL_INCOME_MOMENT_SPECS:
        raise ValueError(f"moment_spec must be one of {PARENTAL_INCOME_MOMENT_SPECS}.")
    parinc = np.asarray(parinc, dtype=np.int64).reshape(-1)
    flow = np.asarray(flow, dtype=np.float64)
    stock = np.asarray(stock, dtype=np.float64)
    if flow.shape != stock.shape:
        raise ValueError("flow and stock must have the same shape.")

    sampled = flow.ndim == 1
    if sampled:
        loan_type = np.asarray(loan_type, dtype=np.int64).reshape(-1)
        if flow.shape != (parinc.size,) or loan_type.shape != (parinc.size,):
            raise ValueError("Sampled outcomes, parinc, and loan_type must align.")
        if not np.all(np.isin(loan_type, (0, 1))):
            raise ValueError("loan_type must contain only zero and one.")
    else:
        if flow.shape != (N_TYPES, parinc.size):
            raise ValueError(f"Exact outcomes must have shape {(N_TYPES, parinc.size)}.")
        q = validate_q(q, n_individuals=parinc.size)

    output, flow_share, effective_weight, labels = [], [], [], []
    for loan_level in (0, 1):
        for parinc_level in range(1, 5):
            selected_i = parinc == parinc_level
            if sampled:
                selected = selected_i & (loan_type == loan_level)
                weights = np.ones(np.sum(selected), dtype=np.float64)
                flow_values = flow[selected]
                stock_values = stock[selected]
            else:
                rows = np.flatnonzero(TYPE_LOAN == loan_level)
                weights = q[selected_i][:, rows].T.reshape(-1)
                flow_values = flow[rows][:, selected_i].reshape(-1)
                stock_values = stock[rows][:, selected_i].reshape(-1)

            distribution_values = (
                stock_values if moment_spec == "fast_stock" else flow_values
            )
            total = weights.sum()
            positive = distribution_values > 0.0
            positive_weight = weights[positive].sum()
            participation_values = (
                flow_values
                if moment_spec in ("fast_flow", "flow_plus_stock")
                else stock_values
            )
            share = (
                np.sum(weights * (participation_values > 0.0)) / total
                if total > 0.0 else eps
            )
            if total <= 0.0 or positive_weight <= 0.0:
                mean, std, p80 = eps, eps, eps
                if moment_spec == "fast_stock":
                    share = eps
            else:
                positive_values = distribution_values[positive]
                positive_weights = weights[positive]
                mean = np.sum(positive_weights * positive_values) / positive_weight
                variance = np.sum(
                    positive_weights * (positive_values - mean) ** 2
                ) / positive_weight
                std = np.sqrt(max(float(variance), 0.0))
                p80 = _weighted_quantile(positive_values, positive_weights, 0.80)
            output.extend((mean, share, std, p80))
            if moment_spec == "flow_plus_stock":
                positive_stock = stock_values > 0.0
                positive_stock_weight = weights[positive_stock].sum()
                mean_positive_stock = (
                    np.sum(
                        weights[positive_stock] * stock_values[positive_stock]
                    ) / positive_stock_weight
                    if positive_stock_weight > 0.0 else eps
                )
                stock_share = (
                    np.sum(weights * positive_stock) / total
                    if total > 0.0 else eps
                )
                output.extend((mean_positive_stock, stock_share))
            flow_share.append(
                np.sum(weights * (flow_values > 0.0)) / total
                if total > 0.0 else np.nan
            )
            effective_weight.append(total)
            labels.append((loan_level, parinc_level))
    return (
        np.asarray(output), np.asarray(flow_share),
        np.asarray(effective_weight), tuple(labels),
    )

# ==============================================================================
# Moments helpers

def _moments_one_group(debtchoice, nmoments=4, eps=0.01, ddof=0):
    debtchoice = np.asarray(debtchoice).ravel()
    anydebt = np.mean(debtchoice > 0)
    pos = debtchoice[debtchoice > 0]

    if pos.size == 0:
        meandebt = eps
        stdebt  = eps
        p80     = eps
        anydebt = eps
    else:
        meandebt = np.mean(pos)
        stdebt   = np.std(pos, ddof=ddof)
        p80      = np.percentile(pos, 80)

    all_moments = np.array([meandebt, anydebt, stdebt, p80], dtype=np.float64)
    return all_moments[:nmoments]

def get_moments_by_x1(x1, debtchoice, x1_col=0, nmoments=4, levels=None, eps=0.01, ddof=0):
    """
    Returns flattened moments by group: (G*nmoments,)
    """
    x1 = np.asarray(x1)
    debtchoice = np.asarray(debtchoice).ravel()
    g = x1[:, x1_col] if x1.ndim > 1 else x1.ravel()

    if levels is None:
        levels_out = np.sort(np.unique(g))
    else:
        levels_out = np.asarray(levels)

    M = np.zeros((levels_out.size, nmoments), dtype=np.float64)
    for i, lev in enumerate(levels_out):
        mask = (g == lev)
        M[i, :] = _moments_one_group(debtchoice[mask], nmoments=nmoments, eps=eps, ddof=ddof)

    return M.flatten()

def get_mean_share_by_group(x1, debtchoice, x1_col, eps=0.01):
    """
    Returns flattened array length 2*G:
      [mean_pos_g1, share_pos_g1, mean_pos_g2, share_pos_g2, ...]
    """
    x1 = np.asarray(x1)
    debtchoice = np.asarray(debtchoice).ravel()
    g = x1[:, x1_col].astype(int)
    levels = np.sort(np.unique(g))

    out = np.zeros((levels.size, 2), dtype=np.float64)
    for i, lev in enumerate(levels):
        mask = (g == lev)
        d = debtchoice[mask]
        share = np.mean(d > 0)
        pos = d[d > 0]
        mean_pos = np.mean(pos) if pos.size > 0 else eps
        if pos.size == 0:
            share = eps
        out[i, 0] = mean_pos
        out[i, 1] = share

    return out.flatten()


DEFAULT_POSTERIOR_MOMENT_BLOCKS = ("parental_income", "ability", "loan_type")


def _weighted_quantile(values, weights, probability):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[keep]
    weights = weights[keep]
    if values.size == 0:
        return np.nan
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = float(probability) * weights.sum()
    return float(values[min(np.searchsorted(np.cumsum(weights), cutoff), values.size - 1)])


def _weighted_debt_moments(values, weights, nmoments=4, eps=0.01):
    """Debt moments for posterior-weighted observations."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    values = values[keep]
    weights = weights[keep]
    total = weights.sum()
    if total <= 0.0:
        return np.full(nmoments, eps, dtype=np.float64)

    positive = values > 0.0
    positive_weight = weights[positive].sum()
    share = positive_weight / total
    if positive_weight <= 0.0:
        all_moments = np.array([eps, eps, eps, eps], dtype=np.float64)
    else:
        positive_values = values[positive]
        positive_weights = weights[positive]
        mean = np.sum(positive_weights * positive_values) / positive_weight
        variance = np.sum(positive_weights * (positive_values - mean) ** 2) / positive_weight
        p80 = _weighted_quantile(positive_values, positive_weights, 0.80)
        all_moments = np.array([mean, share, np.sqrt(variance), p80], dtype=np.float64)
    return all_moments[:nmoments]


def posterior_debt_moments(
    x1,
    debt_by_type,
    q,
    moment_blocks=DEFAULT_POSTERIOR_MOMENT_BLOCKS,
):
    """Construct flexible moments after integrating over all joint types.

    ``debt_by_type`` has shape (16, N). For data moments the same observed
    debt-choice stock is repeated over types; for simulated moments each row
    contains choices conditional on that joint type.
    """
    x1 = np.asarray(x1)
    debt_by_type = np.asarray(debt_by_type, dtype=np.float64)
    q = validate_q(q, n_individuals=x1.shape[0])
    if debt_by_type.shape != (N_TYPES, x1.shape[0]):
        raise ValueError(
            f"debt_by_type must have shape {(N_TYPES, x1.shape[0])}; "
            f"received {debt_by_type.shape}"
        )

    blocks = tuple(moment_blocks)
    unknown = set(blocks).difference(DEFAULT_POSTERIOR_MOMENT_BLOCKS)
    if unknown:
        raise ValueError(f"Unknown posterior moment blocks: {sorted(unknown)}")

    values = debt_by_type.reshape(-1)
    weights = q.T.reshape(-1)
    output = []
    for block in blocks:
        if block in ("parental_income", "ability"):
            column = 0 if block == "parental_income" else 1
            nmoments = 4 if block == "parental_income" else 2
            groups = np.tile(x1[:, column], N_TYPES)
            for level in (1, 2, 3, 4):
                selected = groups == level
                output.extend(
                    _weighted_debt_moments(
                        values[selected], weights[selected], nmoments=nmoments
                    )
                )
        elif block == "loan_type":
            # Ordering is [mean positive, share positive] for B=0, then B=1.
            for loan_type in (0, 1):
                type_rows = np.flatnonzero(TYPE_LOAN == loan_type)
                output.extend(
                    _weighted_debt_moments(
                        debt_by_type[type_rows].reshape(-1),
                        q[:, type_rows].T.reshape(-1),
                        nmoments=2,
                    )
                )
    return np.asarray(output, dtype=np.float64)

def dummies_234(g):
    g = np.asarray(g).astype(int)
    d2 = (g == 2).astype(np.float64)
    d3 = (g == 3).astype(np.float64)
    d4 = (g == 4).astype(np.float64)
    return d2, d3, d4

def print_moment_progress(m_data_flat, m_sim_flat, levels, nmoments=2, decimals=2, title=None):
    levels = np.asarray(levels)
    G = len(levels)

    m_data = np.asarray(m_data_flat, dtype=float).reshape(G, nmoments)
    m_sim  = np.asarray(m_sim_flat,  dtype=float).reshape(G, nmoments)
    m_diff = m_sim - m_data

    moment_names_all = ["MeanDebt>0", "Pr(debt>0)", "StdDebt>0", "P80Debt>0"]
    moment_names = moment_names_all[:nmoments]

    if title:
        print(title)

    print(f"{'Group':>5} | " + "  ".join([f"{name:^32}" for name in moment_names]))
    print(f"{'':>5} | " + "  ".join([f"{'data':>8} {'sim':>8} {'diff':>8}" for _ in moment_names]))

    for gi, lev in enumerate(levels):
        row = [f"{int(lev):>5} |"]
        for j in range(nmoments):
            row.append(f"{m_data[gi,j]:>8.{decimals}f} {m_sim[gi,j]:>8.{decimals}f} {m_diff[gi,j]:>8.{decimals}f}")
        print(" ".join(row))

# ==============================================================================
# CCP-type selection + sampling

@njit()
def get_ccp_type(ccp_by_type, type_index):
    """Select each sampled individual's CCP path by zero-based type index."""
    n = type_index.shape[0]
    ccp = np.empty((n, ccp_by_type.shape[2]), dtype=ccp_by_type.dtype)
    for i in range(n):
        ccp[i, :] = ccp_by_type[type_index[i], i, :]
    return ccp

def get_sample(x1, state, types, debt, choices, ccp_path, budget, terminal, n_sample=10000):
    """
    Bootstrap sample + type draw (kept consistent with your original version).

    Returns the sampled states and CCPs together with the permanent joint type
    and its loan component. The latter are retained now so the budget-shock
    distribution can be made loan-type-specific without changing the sampling
    interface again.
    """
    N = x1.shape[0]
    idx = np.random.choice(N, size=n_sample, replace=True)

    x1sample         = x1[idx, :]
    statesample      = state[idx, :]
    debtsample       = debt[idx]
    choicessample    = choices[idx, :]
    budgetsample     = budget[idx]

    # Draw the permanent joint type from each individual's full posterior.
    cumulative_q = np.cumsum(types[idx], axis=1)
    cumulative_q[:, -1] = 1.0
    type_index = np.sum(
        np.random.random(n_sample)[:, None] > cumulative_q,
        axis=1,
    ).astype(np.int64)
    type_id = type_index + 1
    loan_type = TYPE_LOAN[type_index].astype(np.int64)
    evt = get_ccp_type(ccp_path[:, idx, :], type_index)

    e = np.random.normal(loc=0.0, scale=1.0, size=n_sample).astype(np.float64)

    term_g, term_ng, p_grad = terminal
    terminal_sample = [
        [term_g[i]  for i in idx],
        [term_ng[i] for i in idx],
        p_grad[idx].copy(),
    ]

    return (
        x1sample,
        statesample,
        debtsample,
        choicessample,
        budgetsample,
        evt,
        e,
        terminal_sample,
        type_id,
        loan_type,
    )


def get_posterior_sample(
    x1, state, types, debt, choices, ccp_path, budget, terminal,
    n_sample=10000, rng=None,
):
    """Bootstrap observations while retaining every posterior type.

    Unlike ``get_sample``, this function never draws or classifies a latent
    type. The returned CCP tensor and posterior matrix remain aligned so the
    SMM objective can integrate over all sixteen joint types exactly.
    """
    rng = np.random.default_rng() if rng is None else rng
    types = validate_q(types, n_individuals=x1.shape[0])
    idx = rng.choice(x1.shape[0], size=int(n_sample), replace=True)
    term_g, term_ng, p_grad = terminal
    terminal_sample = [
        [term_g[i] for i in idx],
        [term_ng[i] for i in idx],
        p_grad[idx].copy(),
    ]
    return {
        "indices": idx,
        "x1": np.ascontiguousarray(x1[idx]),
        "state": np.ascontiguousarray(state[idx]),
        "debt": np.ascontiguousarray(debt[idx], dtype=np.float64),
        "choice": np.ascontiguousarray(choices[idx]),
        "budget": np.ascontiguousarray(budget[idx], dtype=np.float64),
        "ccp_by_type": np.ascontiguousarray(ccp_path[:, idx, :], dtype=np.float64),
        "q": np.ascontiguousarray(types[idx], dtype=np.float64),
        "terminal_data": terminal_sample,
    }

# ==============================================================================
# Debt rules (caps + monotone + consumption floor)

@njit()
def get_range_number(b, maxdebt):
    value = np.zeros(b.shape[0])
    value2 = np.zeros(b.shape[0])
    for i in range(b.shape[0]):
        value[i] = nearest_grid_index(debt_range, b[i])
        value2[i] = nearest_grid_index(debt_range, maxdebt[i])
    return value, value2

@njit()
def minimum_debt_maxdebt(u, b, maxdebt):
    b_idx, max_idx = get_range_number(b, maxdebt)
    for i in range(u.shape[0]):
        u[i, :int(b_idx[i])] = -100000000
        u[i, int(max_idx[i]) + 1:] = -100000000
    return u

@njit()
def get_debt_rules(c, vjt, previousdebt, state, choices):
    vjt[c < CONSUMPTION_FLOOR] = -100000000

    n = previousdebt.shape[0]
    maxdebt = np.empty(n, dtype=np.float64)

    for i in range(n):
        educ_choice = int(choices[i, 1])
        twoy_exp = int(state[i, 1])
        foury_exp = int(state[i, 2])

        annual_cap = get_annual_cap_by_stage(educ_choice, twoy_exp, foury_exp)
        lifetime_cap = get_lifetime_cap_by_stage(educ_choice)

        m = previousdebt[i] * (1 + r) + annual_cap
        if m > lifetime_cap:
            m = lifetime_cap
        maxdebt[i] = m

    vjt = minimum_debt_maxdebt(vjt, previousdebt, maxdebt)
    return vjt

# ==============================================================================
# Terminal evaluation caching

def get_relevant_terminal_subset_cached(terminal_data, sigma_risk, idx, cache):
    terminal_grad = terminal_data[0]
    terminal_notgrad = terminal_data[1]
    p_grad = terminal_data[2]

    out = np.zeros((len(idx), debt_range.size), dtype=np.float64)

    for k, i in enumerate(idx):
        key_n = (id(terminal_notgrad[i]), float(sigma_risk))
        if key_n not in cache:
            cache[key_n] = terminal_notgrad[i]((float(sigma_risk), debt_range))
        notv = cache[key_n]

        if p_grad[i] > 0:
            key_g = (id(terminal_grad[i]), float(sigma_risk))
            if key_g not in cache:
                cache[key_g] = terminal_grad[i]((float(sigma_risk), debt_range))
            gradv = cache[key_g]
            out[k, :] = p_grad[i] * gradv + (1 - p_grad[i]) * notv
        else:
            out[k, :] = notv

    return out

# ==============================================================================
# Bellman solver for one draw (terminal separated from CCP path)

@njit(parallel=True, fastmath=True)
def solve_one_draw_debt_idx_terminal_only(
    budget, e, debt_grid,
    sigma_i, debtpen_i,
    ccp_path_row, terminal_row,
    b_idx, max_idx,
    beta_term,
    kappa_entry_i, kappa_cont_i, prev_debt_i,
    c_floor=CONSUMPTION_FLOOR,
    fallback_idx=99,
):
    """
    v = u(c) + ccp_path + beta_term * terminal
    debtpen applies when debt_grid[j] > 0

    kappa is the one-shot new-borrowing event cost. It is charged when the
    candidate exceeds the accrued current debt, debt_grid[j] > (1+r)*prev_debt
    (the explicit SMM convention; the SMM lower bound is nearest-grid, not
    snap-up): kappa_entry_i[i] if prev_debt_i[i] <= 0, else kappa_cont_i[i].
    No discounting multiplier is ever applied (one-shot event timing). With
    zero kappa arrays the utility is unchanged.
    """
    n = budget.shape[0]
    B = debt_grid.size
    out = np.empty(n, dtype=np.int64)

    fb = fallback_idx
    if fb < 0: fb = 0
    if fb >= B: fb = B - 1

    for i in prange(n):
        lo = b_idx[i]
        hi = max_idx[i]
        if lo < 0: lo = 0
        if hi >= B: hi = B - 1
        if hi < lo: hi = lo

        sig = sigma_i[i]
        pen = debtpen_i[i]
        kappa = kappa_cont_i[i]
        if prev_debt_i[i] <= 0.0:
            kappa = kappa_entry_i[i]
        accrued_debt = (1.0 + r) * prev_debt_i[i]

        best_v = -1e30
        best_j = lo
        found_feasible = False

        for j in range(lo, hi + 1):
            c = budget[i] + e[i] + debt_grid[j]
            if c < c_floor:
                continue
            found_feasible = True

            if abs(sig - 1.0) < 1e-8:
                u = 0.1 * np.log(0.00001 * c)
            else:
                u = 0.1 * ((0.00001 * c) ** (1.0 - sig)) / (1.0 - sig)

            if debt_grid[j] > 0.0:
                u += pen

            if kappa != 0.0 and debt_grid[j] > accrued_debt:
                u += kappa

            v = u + ccp_path_row[i, j] + beta_term * terminal_row[i, j]
            if v > best_v:
                best_v = v
                best_j = j

        out[i] = hi if not found_feasible else best_j

    return out


def _zero_new_borrowing_kernel_arrays(n):
    """Zero kappa/previous-debt kernel inputs for paths without the new cost.

    The three returned views share one read-only buffer; with every kappa at
    zero the kernels are numerically identical to the pre-kappa code, so the
    previous-debt entries are irrelevant and may also be zero.
    """
    zeros = np.zeros(int(n), dtype=np.float64)
    return zeros, zeros, zeros


def _new_borrowing_kernel_arrays(spec, loan_type, prev_debt):
    """Per-individual kappa kernel inputs from the canonical specification.

    Entry costs are resolved by latent loan type; the continuation cost is
    shared. Both come back as float64 arrays aligned with ``prev_debt``.
    Specifications without the kappa block yield zero arrays.
    """
    entry_by_type, continuation = bs.new_borrowing_cost_parameters(spec)
    loan_type = np.asarray(loan_type, dtype=np.int64).reshape(-1)
    prev_debt = np.ascontiguousarray(prev_debt, dtype=np.float64).reshape(-1)
    if loan_type.size != prev_debt.size:
        raise ValueError("loan_type and prev_debt must align.")
    kappa_entry_i = np.ascontiguousarray(
        entry_by_type[loan_type], dtype=np.float64
    )
    kappa_cont_i = np.full(prev_debt.size, continuation, dtype=np.float64)
    return kappa_entry_i, kappa_cont_i, prev_debt

# ==============================================================================
# Precompute bounds indices (speed)

def precompute_bounds_indices(previousdebt, state, choices):
    return precompute_smm_bounds_indices(
        previousdebt, state, choices, debt_range, r
    )

# ==============================================================================
# Objective

EVAL_COUNTER = 0


def discounted_explicit_horizon_debt_penalty(spec, x1, model_period):
    """Present value of the common parinc flow penalty through period T-1.

    This is the SMM shortcut for the recursively repeated flow penalty in the
    Bellman solver. It deliberately excludes the terminal continuation at T.
    """
    flow_penalty = bs.debt_penalty(spec, x1).astype(np.float64)
    multiplier = bs.explicit_debt_penalty_multiplier(
        model_period, beta=beta, terminal_period=T
    )
    return np.ascontiguousarray(flow_penalty * multiplier, dtype=np.float64)

def minimize_distance_multi(params, moments_data, sample_by_period, periods):
    """
    params structure:
      - mu blocks: 7*P
      - sigma_e: P
      - ra_levels: 4
      - debt_pen_parinc: 4  [dp0, dp2, dp3, dp4]  (constant + 3 deviations)
    """
    global EVAL_COUNTER
    EVAL_COUNTER += 1

    P = len(periods)

    spec = bs.unpack_estimation_vector(params, periods)
    mu_blocks = spec["mu_blocks"]
    sigma_e_vec = spec["sigma_e"]
    ra_levels = spec["risk_aversion"]
    dp0, dp2, dp3, dp4 = spec["debt_pen_parinc"]

    sim_list = []
    per_period_loss = []

    for pi, p in enumerate(periods):
        pack = sample_by_period[p]
        x1 = pack["x1"]
        budget = pack["budget"]
        evt_ccp = pack["evt_ccp"]
        terminal_data = pack["terminal_data"]
        Z = pack["Z"]
        b_idx = pack["b_idx"]
        max_idx = pack["max_idx"]

        # period-specific mu params
        debtpen_i = discounted_explicit_horizon_debt_penalty(spec, x1, p)
        sigma_i = bs.risk_aversion(spec, x1).astype(np.float64)
        gi = x1[:, 0].astype(int) - 1

        # terminal evaluated by ra group
        n = budget.shape[0]
        B = debt_range.size
        terminal = np.zeros((n, B), dtype=np.float64)

        cache = {}
        for k in range(4):
            idx = np.where(gi == k)[0]
            if idx.size > 0:
                terminal[idx, :] = get_relevant_terminal_subset_cached(terminal_data, ra_levels[k], idx, cache)

        beta_term = float(beta ** (T - p))
        kappa_entry_i, kappa_cont_i, prev_debt_i = (
            _zero_new_borrowing_kernel_arrays(n)
        )

        s = Z.shape[0]
        sim_draws = np.zeros((24, s), dtype=np.float64)

        for k in range(s):
            e = bs.realization(spec, x1, p, Z[k, :]).astype(np.float64)

            debt_idx = solve_one_draw_debt_idx_terminal_only(
                budget=budget,
                e=e,
                debt_grid=debt_range,
                sigma_i=sigma_i,
                debtpen_i=debtpen_i,
                ccp_path_row=evt_ccp,
                terminal_row=terminal,
                b_idx=b_idx,
                max_idx=max_idx,
                beta_term=beta_term,
                kappa_entry_i=kappa_entry_i,
                kappa_cont_i=kappa_cont_i,
                prev_debt_i=prev_debt_i,
                c_floor=CONSUMPTION_FLOOR,
                fallback_idx=99,
            )

            debtvalue = debt_range[debt_idx]
            mom_par_k = get_moments_by_x1(x1, debtvalue, x1_col=0, nmoments=4)
            mom_ab_k  = get_mean_share_by_group(x1, debtvalue, x1_col=1)
            sim_draws[:, k] = np.concatenate([mom_par_k, mom_ab_k])

        sim_p = sim_draws.mean(axis=1)
        sim_list.append(sim_p)

        data_p = moments_data[24*pi:24*(pi+1)]
        denom_p = np.maximum(np.abs(data_p), 1e-6)
        per_period_loss.append(float(np.sum(((data_p - sim_p) / denom_p) ** 2)))

    simumoments = np.concatenate(sim_list)
    denom = np.maximum(np.abs(moments_data), 1e-6)
    loss = float(np.sum(((moments_data - simumoments) / denom) ** 2))

    # PRINTS (rich, like your old version)
    if (EVAL_COUNTER % 10) == 0:
        print("\n" + "="*120)
        print(f"[eval {EVAL_COUNTER}] total loss={loss:.6f}")
        print(f"dp0,dp2,dp3,dp4 = {np.round([dp0, dp2, dp3, dp4], 4)}   (<=0 expected)")
        print(f"ra_levels(par1..4) = {np.round(ra_levels, 4)}")
        print("sigma_e first 10:", np.round(sigma_e_vec[:min(10, P)], 2))

        # Show first 3 periods' mu blocks (or fewer if P<3)
        n_mu_show = min(3, P)
        for kk in range(n_mu_show):
            mu0_, mu2_, mu3_, mu4_, mab2_, mab3_, mab4_ = mu_blocks[kk, :]
            print(f"mu_block(period={periods[kk]}): "
                  f"[mu0={mu0_:.1f}, mu_p2={mu2_:.1f}, mu_p3={mu3_:.1f}, mu_p4={mu4_:.1f}, "
                  f"mu_a2={mab2_:.1f}, mu_a3={mab3_:.1f}, mu_a4={mab4_:.1f}]")

        print("per-period loss:", {int(periods[i]): round(per_period_loss[i], 6) for i in range(P)})

        # Pretty tables for first N periods
        n_show = min(7, P)
        for show_pi in range(n_show):
            show_p = periods[show_pi]
            data_show = moments_data[24*show_pi:24*(show_pi+1)]
            sim_show  = simumoments[24*show_pi:24*(show_pi+1)]

            mom_par_data = data_show[:16]
            mom_ab_data  = data_show[16:]
            mom_par_sim  = sim_show[:16]
            mom_ab_sim   = sim_show[16:]

            print(f"\n--- Pretty moment tables (period={show_p}) ---")
            print_moment_progress(mom_par_data, mom_par_sim, levels=np.array([1,2,3,4]), nmoments=4, decimals=2,
                                  title="ParInc moments (data vs simulated)")
            print_moment_progress(mom_ab_data, mom_ab_sim, levels=np.array([1,2,3,4]), nmoments=2, decimals=2,
                                  title="Ability moments (data vs simulated)")

        print("="*120 + "\n")

    return loss


def minimize_distance_multi_posterior(
    params,
    moments_data_by_period,
    sample_by_period,
    periods,
    shock_heterogeneity,
    moment_blocks,
):
    """SMM objective that integrates over the complete fixed EM posterior."""
    global EVAL_COUNTER
    EVAL_COUNTER += 1
    spec = bs.unpack_estimation_vector(
        params, periods, loan_heterogeneity=shock_heterogeneity
    )
    total_loss = 0.0
    per_period_loss = {}

    for p in periods:
        pack = sample_by_period[p]
        x1 = pack["x1"]
        q = pack["q"]
        budget = pack["budget"]
        ccp_by_type = pack["ccp_by_type"]
        terminal_data = pack["terminal_data"]
        Z = pack["Z"]
        b_idx = pack["b_idx"]
        max_idx = pack["max_idx"]

        debtpen_i = discounted_explicit_horizon_debt_penalty(spec, x1, p)
        sigma_i = bs.risk_aversion(spec, x1).astype(np.float64)
        parental_group = x1[:, 0].astype(int) - 1
        terminal = np.zeros((x1.shape[0], debt_range.size), dtype=np.float64)
        cache = {}
        for group_index in range(4):
            idx = np.where(parental_group == group_index)[0]
            if idx.size:
                terminal[idx] = get_relevant_terminal_subset_cached(
                    terminal_data, spec["risk_aversion"][group_index], idx, cache
                )

        beta_term = float(beta ** (T - p))
        kappa_entry_i, kappa_cont_i, prev_debt_i = (
            _zero_new_borrowing_kernel_arrays(x1.shape[0])
        )
        sim_draws = np.zeros((len(moments_data_by_period[p]), Z.shape[0]))
        for draw_index in range(Z.shape[0]):
            debt_by_type = np.empty((N_TYPES, x1.shape[0]), dtype=np.float64)
            for type_index in range(N_TYPES):
                shock = bs.realization(
                    spec,
                    x1,
                    p,
                    Z[draw_index],
                    loan_type=int(TYPE_LOAN[type_index]),
                ).astype(np.float64)
                debt_index = solve_one_draw_debt_idx_terminal_only(
                    budget=budget,
                    e=shock,
                    debt_grid=debt_range,
                    sigma_i=sigma_i,
                    debtpen_i=debtpen_i,
                    ccp_path_row=ccp_by_type[type_index],
                    terminal_row=terminal,
                    b_idx=b_idx,
                    max_idx=max_idx,
                    beta_term=beta_term,
                    kappa_entry_i=kappa_entry_i,
                    kappa_cont_i=kappa_cont_i,
                    prev_debt_i=prev_debt_i,
                    c_floor=CONSUMPTION_FLOOR,
                    fallback_idx=debt_range.size - 1,
                )
                debt_by_type[type_index] = debt_range[debt_index]
            sim_draws[:, draw_index] = posterior_debt_moments(
                x1, debt_by_type, q, moment_blocks=moment_blocks
            )

        simulated = sim_draws.mean(axis=1)
        data = moments_data_by_period[p]
        denominator = np.maximum(np.abs(data), 1.0e-6)
        period_loss = float(np.sum(((data - simulated) / denominator) ** 2))
        per_period_loss[int(p)] = period_loss
        total_loss += period_loss

    if EVAL_COUNTER % 10 == 0:
        print("\n" + "=" * 100)
        print(f"[posterior eval {EVAL_COUNTER}] total loss={total_loss:.6f}")
        print(f"loan heterogeneity={shock_heterogeneity}")
        print("loan mean shifts:", np.round(spec["loan_mean_shift"], 3))
        print("loan sigma ratios:", np.round(np.exp(spec["loan_log_sigma_ratio"]), 4))
        print("per-period loss:", per_period_loss)
        print("=" * 100 + "\n")
    return float(total_loss)

# ==============================================================================
# Multi-period estimator

def fit_budget_shock_multi(
    data_by_period,
    periods,
    max_outer=20,
    s=20,
    n_sample=10000,
    posterior_integration=False,
    shock_heterogeneity="homogeneous",
    moment_blocks=DEFAULT_POSTERIOR_MOMENT_BLOCKS,
):
    """
    params = [ mu_block_per_period (7*P), sigma_e_per_period (P), ra_levels (4), debt_pen_parinc (4) ]
    where debt_pen_parinc = [dp0, dp2, dp3, dp4] (constant + deviations).
    """
    P = len(periods)

    # Build data moments ONCE. The legacy branch is unchanged; the new branch
    # repeats observed debt over types and integrates using fixed q weights.
    moments_data_by_period = {}
    mom_data_list = []
    for p in periods:
        x1, state, types, debt, debtchoice, choices, ccp_path, budget, terminal_data = data_by_period[p]
        if posterior_integration:
            moments_data_by_period[p] = posterior_debt_moments(
                x1,
                np.broadcast_to(debtchoice, (N_TYPES, len(debtchoice))),
                types,
                moment_blocks=moment_blocks,
            )
        else:
            mom_par = get_moments_by_x1(x1, debtchoice, x1_col=0, nmoments=4).astype(np.float64)  # 16
            mom_ab  = get_mean_share_by_group(x1, debtchoice, x1_col=1).astype(np.float64)        # 8
            mom_data_list.append(np.concatenate([mom_par, mom_ab]))
    moments_data = (
        None if posterior_integration
        else np.concatenate(mom_data_list).astype(np.float64)
    )

    # Initial guess
    mu0_guess = 100
    mu_block0 = np.tile(np.array([mu0_guess, 100, 100, 100, 100, 100, 100], dtype=np.float64), P)

    sigma_e0 = np.full(P, 100.0, dtype=np.float64)
    ra0 = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)

    dp0_init = -1.0
    dp_parinc0 = np.array([dp0_init, -1.0, -1.0, -1.0], dtype=np.float64)  # [dp0, dp2, dp3, dp4]

    params0_parts = [mu_block0, sigma_e0, ra0, dp_parinc0]
    if shock_heterogeneity in ("mean", "both"):
        params0_parts.append(np.zeros(P, dtype=np.float64))
    if shock_heterogeneity in ("variance", "both"):
        params0_parts.append(np.zeros(P, dtype=np.float64))
    params0 = np.concatenate(params0_parts).astype(np.float64)
    # Validate the mode and vector layout before entering the optimizer.
    bs.unpack_estimation_vector(
        params0, periods, loan_heterogeneity=shock_heterogeneity
    )

    # Bounds
    MU_MIN, MU_MAX = -50000.0, 50000.0
    SIGE_MIN, SIGE_MAX = 1.0, 50000.0

    bounds = []
    for _ in range(7 * P):
        bounds.append((MU_MIN, MU_MAX))
    for _ in range(P):
        bounds.append((SIGE_MIN, SIGE_MAX))
    bounds += [(0.1001, 2.9999)] * 4
    bounds += [(-1e6, 0.0)] * 4  # dp0,dp2,dp3,dp4 <= 0
    if shock_heterogeneity in ("mean", "both"):
        bounds += [(MU_MIN, MU_MAX)] * P
    if shock_heterogeneity in ("variance", "both"):
        # sigma(B=1) = sigma(B=0) * exp(log ratio)
        bounds += [(-3.0, 3.0)] * P

    best_x = params0.copy()
    best_fun = 1e30
    tolr = 0.01
    it = 0

    while best_fun > tolr and it < max_outer:
        print(f"\n================ OUTER ITER {it} ================")
        np.random.seed(it)

        sample_by_period = {}

        for p in periods:
            x1, state, types, debt, debtchoice, choices, ccp_path, budget, terminal_data = data_by_period[p]

            if posterior_integration:
                sampled = get_posterior_sample(
                    x1, state, types, debt, choices, ccp_path, budget,
                    terminal_data, n_sample=n_sample,
                    rng=np.random.default_rng(100000 + 1000 * p + it),
                )
                x1s = sampled["x1"]
                states = sampled["state"]
                debts = sampled["debt"]
                choices_s = sampled["choice"]
                budget_s = sampled["budget"]
                terminal_s = sampled["terminal_data"]
            else:
                (
                    x1s,
                    states,
                    debts,
                    choices_s,
                    budget_s,
                    evt_ccp_s,
                    _,
                    terminal_s,
                    type_ids,
                    loan_types,
                ) = get_sample(
                    x1, state, types, debt, choices, ccp_path, budget,
                    terminal_data, n_sample=n_sample
                )

            rng = np.random.default_rng(12345 + 1000*p + it)
            Zp = rng.standard_normal((s, x1s.shape[0])).astype(np.float64)

            b_idx, max_idx = precompute_bounds_indices(
                debts.astype(np.float64),
                states.astype(np.int64),
                choices_s.astype(np.int64),
            )

            common_pack = {
                "x1": np.ascontiguousarray(x1s),
                "state": np.ascontiguousarray(states),
                "choice": np.ascontiguousarray(choices_s),
                "budget": np.ascontiguousarray(budget_s, dtype=np.float64),
                "evt_ccp": np.ascontiguousarray(evt_ccp_s, dtype=np.float64),
                "terminal_data": terminal_s,
                "Z": Zp,
                "b_idx": b_idx,
                "max_idx": max_idx,
            }
            if posterior_integration:
                common_pack["ccp_by_type"] = sampled["ccp_by_type"]
                common_pack["q"] = sampled["q"]
            else:
                common_pack["evt_ccp"] = np.ascontiguousarray(evt_ccp_s, dtype=np.float64)
                common_pack["type_id"] = np.ascontiguousarray(type_ids, dtype=np.int64)
                common_pack["loan_type"] = np.ascontiguousarray(loan_types, dtype=np.int64)
            sample_by_period[p] = common_pack

        if posterior_integration:
            objective = minimize_distance_multi_posterior
            args_obj = (
                moments_data_by_period,
                sample_by_period,
                periods,
                shock_heterogeneity,
                tuple(moment_blocks),
            )
        else:
            objective = minimize_distance_multi
            args_obj = (moments_data, sample_by_period, periods)

        print("---- Running minimize(..., method='Nelder-Mead') ----")
        res = minimize(
            objective,
            x0=best_x,
            args=args_obj,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": 10000, "disp": True},
        )

        print("\n[minimize done] success:", res.success)
        print("[minimize done] message:", res.message)
        print("[minimize done] fun:", float(res.fun))

        if float(res.fun) < best_fun:
            best_fun = float(res.fun)
            best_x = res.x.copy()

        it += 1
        if it == 10:
            tolr = 0.03
        if it == 15:
            tolr = 0.04

    print("\n================ FINAL (multi-period) =================")
    print("best_fun:", best_fun)
    print("best_x:", best_x)
    return best_x, best_fun


# ==============================================================================
# Education-cell pre-test (posterior integrated, simulated financial resources)

def assign_persistent_joint_types(packs, seed=12345):
    """Draw one joint EM type per balanced individual and reuse it everywhere."""
    if not packs:
        return []
    max_index = max(int(np.max(pack["individual_index"])) for pack in packs if len(pack["x1"]))
    rng = np.random.default_rng(int(seed) + 1_000_003)
    uniforms = rng.random(max_index + 1)
    assigned = []
    observed_q = {}
    sampled_by_individual = {}
    for pack in packs:
        indices = np.asarray(pack["individual_index"], dtype=np.int64)
        q = validate_q(pack["q"], n_individuals=len(indices))
        for individual, posterior in zip(indices, q):
            previous = observed_q.get(int(individual))
            if previous is not None and not np.allclose(previous, posterior, atol=1.0e-12):
                raise ValueError(
                    f"Posterior q differs across periods for individual index {individual}."
                )
            observed_q[int(individual)] = posterior
        cumulative = np.cumsum(q, axis=1)
        cumulative[:, -1] = 1.0
        type_index = np.sum(uniforms[indices, None] > cumulative, axis=1).astype(np.int64)
        for individual, latent_type in zip(indices, type_index):
            previous_type = sampled_by_individual.get(int(individual))
            if previous_type is not None and previous_type != int(latent_type):
                raise ValueError(
                    f"Sampled type changed across periods for individual index {individual}."
                )
            sampled_by_individual[int(individual)] = int(latent_type)
        out = dict(pack)
        out["sampled_type_index"] = np.ascontiguousarray(type_index)
        assigned.append(out)
    unique_individuals = np.asarray(sorted(sampled_by_individual), dtype=np.int64)
    sampled_unique = np.asarray(
        [sampled_by_individual[int(i)] for i in unique_individuals], dtype=np.int64
    )
    expected = np.sum([observed_q[int(i)] for i in unique_individuals], axis=0)
    observed = np.bincount(sampled_unique, minlength=N_TYPES).astype(np.float64)
    print(f"[sampled joint types] {len(unique_individuals)} unique individuals; seed={seed}")
    print("  type          observed share   posterior-expected share")
    for type_index, name in enumerate(TYPE_NAMES):
        print(
            f"  {type_index + 1:>2} {str(name):<10} "
            f"{observed[type_index] / len(unique_individuals):>12.4f} "
            f"{expected[type_index] / len(unique_individuals):>24.4f}"
        )
    return assigned


def _subset_cell_pack(pack, indices):
    indices = np.asarray(indices, dtype=np.int64)
    out = dict(pack)
    for name in (
        "x1", "state", "q", "debt", "debtchoice", "loan_flow",
        "parinc", "choice", "observed_budget", "individual_index",
    ):
        out[name] = np.ascontiguousarray(pack[name][indices])
    if "annual_cap" in pack:
        out["annual_cap"] = np.ascontiguousarray(pack["annual_cap"][indices])
    if "sampled_type_index" in pack:
        out["sampled_type_index"] = np.ascontiguousarray(
            pack["sampled_type_index"][indices], dtype=np.int64
        )
    out["ccp_by_type"] = np.ascontiguousarray(pack["ccp_by_type"][:, indices, :])
    term_g, term_ng, p_grad = pack["terminal_data"]
    out["terminal_data"] = [
        [term_g[i] for i in indices],
        [term_ng[i] for i in indices],
        p_grad[indices],
    ]
    return out


@njit(parallel=True, fastmath=True)
def solve_all_draws_debt_idx_pooled(
    budget_by_draw, shock_by_draw, debt_grid,
    sigma_i, debtpen_i, ccp_path_row, terminal_row,
    b_idx, max_idx, beta_term_i,
    kappa_entry_i, kappa_cont_i, prev_debt_i,
    c_floor=CONSUMPTION_FLOOR, fallback_idx=99,
):
    """Solve every sampled individual and simulation draw in one parallel call.

    Model-period differences are retained through ``beta_term_i`` and through
    each individual's already-selected CCP, terminal, and debt-bound rows.

    kappa is the one-shot new-borrowing event cost, charged when the
    candidate exceeds accrued current debt, debt_grid[j] > (1+r)*prev_debt
    (the explicit SMM convention; the SMM lower bound is nearest-grid, not
    snap-up): kappa_entry_i[i] if prev_debt_i[i] <= 0, else kappa_cont_i[i].
    No discounting multiplier is ever applied (one-shot event timing). With
    zero kappa arrays the utility is unchanged.
    """
    draws, n = budget_by_draw.shape
    B = debt_grid.size
    out = np.empty((draws, n), dtype=np.int64)

    fb = fallback_idx
    if fb < 0:
        fb = 0
    if fb >= B:
        fb = B - 1

    for flat_index in prange(draws * n):
        draw_index = flat_index // n
        i = flat_index - draw_index * n
        lo = b_idx[i]
        hi = max_idx[i]
        if lo < 0:
            lo = 0
        if hi >= B:
            hi = B - 1
        if hi < lo:
            hi = lo

        sig = sigma_i[i]
        pen = debtpen_i[i]
        beta_i = beta_term_i[i]
        kappa = kappa_cont_i[i]
        if prev_debt_i[i] <= 0.0:
            kappa = kappa_entry_i[i]
        accrued_debt = (1.0 + r) * prev_debt_i[i]
        best_v = -1e30
        best_j = lo
        found_feasible = False

        for j in range(lo, hi + 1):
            c = budget_by_draw[draw_index, i] + shock_by_draw[draw_index, i] + debt_grid[j]
            if c < c_floor:
                continue
            found_feasible = True
            if abs(sig - 1.0) < 1e-8:
                u = 0.1 * np.log(0.00001 * c)
            else:
                u = 0.1 * ((0.00001 * c) ** (1.0 - sig)) / (1.0 - sig)
            if debt_grid[j] > 0.0:
                u += pen
            if kappa != 0.0 and debt_grid[j] > accrued_debt:
                u += kappa
            v = u + ccp_path_row[i, j] + beta_i * terminal_row[i, j]
            if v > best_v:
                best_v = v
                best_j = j

        out[draw_index, i] = hi if not found_feasible else best_j
    return out


def prepare_education_cell_crns(
    packs, draws=20, seed=12345, type_integration="sampled",
    resource_mode="simulated",
):
    """Pre-draw every resource shock once and reuse it in all SMM evaluations."""
    if type_integration not in TYPE_INTEGRATION_MODES:
        raise ValueError(f"type_integration must be one of {TYPE_INTEGRATION_MODES}.")
    if resource_mode not in EDUCATION_CELL_RESOURCE_MODES:
        raise ValueError(f"resource_mode must be one of {EDUCATION_CELL_RESOURCE_MODES}.")
    wage_parameters = (
        load_fixed_wage_parameters() if resource_mode == "simulated" else None
    )
    financial = (
        load_auxiliary_financial_process(EST("auxiliary_em_results.npz"))
        if resource_mode == "simulated" else None
    )
    prepared = {}
    for pack in packs:
        period = pack["period"]
        x1 = pack["x1"]
        state = pack["state"]
        choice = pack["choice"]
        n = len(x1)
        rng = np.random.default_rng(seed + 1000 * period)
        crn = {
            "wage_z": rng.standard_normal((draws, n)),
            "grant_u": rng.random((draws, n)),
            "grant_z": rng.standard_normal((draws, n)),
            "transfer_u": rng.random((draws, n)),
            "transfer_z": rng.standard_normal((draws, n)),
            "budget_z": rng.standard_normal((draws, n)),
        }

        x1_design = expand_x1(x1)
        education = choice[:, 1].astype(np.int64)
        work = choice[:, 2].astype(np.int64)
        tuition = np.asarray(tuition_agents(0, choice), dtype=np.float64).reshape(-1)
        if type_integration == "sampled":
            if "sampled_type_index" not in pack:
                raise ValueError("Sampled integration requires persistent joint-type draws.")
            sampled_type = np.asarray(pack["sampled_type_index"], dtype=np.int64)
            if sampled_type.shape != (n,) or not np.all((0 <= sampled_type) & (sampled_type < N_TYPES)):
                raise ValueError("sampled_type_index is invalid or misaligned.")
            base_budget = np.empty((draws, n), dtype=np.float64)
        else:
            base_budget = np.empty((draws, N_TYPES, n), dtype=np.float64)

        if resource_mode == "observed":
            observed_budget = np.asarray(pack["observed_budget"], dtype=np.float64)
            if observed_budget.shape != (n,) or not np.all(np.isfinite(observed_budget)):
                raise ValueError(
                    f"Observed pre-choice budget is invalid in model period {period}."
                )
            if type_integration == "sampled":
                base_budget[:] = observed_budget[None, :]
            else:
                base_budget[:] = observed_budget[None, None, :]
        else:
            state_design = _state_wage_design(state)
            wage_mu = (
                np.column_stack((x1_design, state_design[:, :-1]))
                @ wage_parameters["school"]
            )
            wage_sigma = float(wage_parameters["sigmas"][0])
            for draw_index in range(draws):
                wage = (
                    np.exp(np.clip(
                        wage_mu + wage_sigma * crn["wage_z"][draw_index], -20.0, 20.0
                    ))
                    * work * 0.5 * (40 * 52)
                )
                if type_integration == "sampled":
                    grant = np.zeros(n, dtype=np.float64)
                    transfer = np.zeros(n, dtype=np.float64)
                    sampled_grant = TYPE_GRANT[sampled_type]
                    sampled_transfer = TYPE_TRANSFER[sampled_type]
                    for financial_type in (0, 1):
                        idx = np.flatnonzero(sampled_grant == financial_type)
                        if idx.size:
                            grant[idx] = draw_grants_vectorized(
                                x1_design[idx], education[idx], work[idx], financial["grant"],
                                grant_type=financial_type,
                                receipt_uniform=crn["grant_u"][draw_index, idx],
                                amount_standard_normal=crn["grant_z"][draw_index, idx],
                            )
                        idx = np.flatnonzero(sampled_transfer == financial_type)
                        if idx.size:
                            transfer[idx] = draw_transfers_vectorized(
                                x1_design[idx], education[idx], work[idx], financial["transfer"],
                                transfer_type=financial_type,
                                receipt_uniform=crn["transfer_u"][draw_index, idx],
                                amount_standard_normal=crn["transfer_z"][draw_index, idx],
                            )
                    base_budget[draw_index] = (
                        wage + grant + transfer - tuition - (1.0 + r) * pack["debt"]
                    )
                else:
                    for type_index in range(N_TYPES):
                        grant = draw_grants_vectorized(
                            x1_design, education, work, financial["grant"],
                            grant_type=int(TYPE_GRANT[type_index]),
                            receipt_uniform=crn["grant_u"][draw_index],
                            amount_standard_normal=crn["grant_z"][draw_index],
                        )
                        transfer = draw_transfers_vectorized(
                            x1_design, education, work, financial["transfer"],
                            transfer_type=int(TYPE_TRANSFER[type_index]),
                            receipt_uniform=crn["transfer_u"][draw_index],
                            amount_standard_normal=crn["transfer_z"][draw_index],
                        )
                        base_budget[draw_index, type_index] = (
                            wage + grant + transfer - tuition - (1.0 + r) * pack["debt"]
                        )

        prepared_pack = dict(pack)
        prepared_pack["base_budget_crn"] = np.ascontiguousarray(base_budget)
        prepared_pack["budget_z"] = np.ascontiguousarray(crn["budget_z"])
        prepared_pack["type_integration"] = type_integration
        prepared_pack["resource_mode"] = resource_mode
        if type_integration == "sampled":
            prepared_pack["ccp_sampled"] = np.ascontiguousarray(
                pack["ccp_by_type"][sampled_type, np.arange(n)], dtype=np.float64
            )
            prepared_pack["sampled_loan_type"] = np.ascontiguousarray(
                TYPE_LOAN[sampled_type], dtype=np.int64
            )
        prepared_pack["b_idx"], prepared_pack["max_idx"] = precompute_bounds_indices(
            pack["debt"].astype(np.float64),
            pack["state"].astype(np.int64),
            pack["choice"].astype(np.int64),
        )
        prepared[period] = prepared_pack
    if type_integration == "sampled" and prepared:
        packs_in_order = list(prepared.values())
        pooled_static = {
            "x1": np.ascontiguousarray(
                np.concatenate([pack["x1"] for pack in packs_in_order])
            ),
            "parinc": np.ascontiguousarray(
                np.concatenate([pack["parinc"] for pack in packs_in_order]),
                dtype=np.int64,
            ),
            "debt": np.ascontiguousarray(
                np.concatenate([pack["debt"] for pack in packs_in_order]),
                dtype=np.float64,
            ),
            "model_period": np.ascontiguousarray(
                np.concatenate([
                    np.full(len(pack["x1"]), int(pack["period"]), dtype=np.int64)
                    for pack in packs_in_order
                ]),
                dtype=np.int64,
            ),
            "loan_type": np.ascontiguousarray(
                np.concatenate([pack["sampled_loan_type"] for pack in packs_in_order]),
                dtype=np.int64,
            ),
            "base_budget": np.ascontiguousarray(
                np.concatenate([pack["base_budget_crn"] for pack in packs_in_order], axis=1),
                dtype=np.float64,
            ),
            "budget_z": np.ascontiguousarray(
                np.concatenate([pack["budget_z"] for pack in packs_in_order], axis=1),
                dtype=np.float64,
            ),
            "ccp": np.ascontiguousarray(
                np.concatenate([pack["ccp_sampled"] for pack in packs_in_order]),
                dtype=np.float64,
            ),
            "b_idx": np.ascontiguousarray(
                np.concatenate([pack["b_idx"] for pack in packs_in_order]),
                dtype=np.int64,
            ),
            "max_idx": np.ascontiguousarray(
                np.concatenate([pack["max_idx"] for pack in packs_in_order]),
                dtype=np.int64,
            ),
            "beta_term": np.ascontiguousarray(
                np.concatenate([
                    np.full(
                        len(pack["x1"]), float(beta ** (T - pack["period"])),
                        dtype=np.float64,
                    )
                    for pack in packs_in_order
                ]),
                dtype=np.float64,
            ),
        }
        # Per-observation annual caps back the optional flow_split_stock
        # at-cap moment; older synthetic packs without the key remain usable.
        if all("annual_cap" in pack for pack in packs_in_order):
            pooled_static["annual_cap"] = np.ascontiguousarray(
                np.concatenate([
                    pack["annual_cap"] for pack in packs_in_order
                ]),
                dtype=np.float64,
            )
        next(iter(prepared.values()))["pooled_sampled_static"] = pooled_static
    return prepared


def _pooled_observed_cell_moments(
    packs, specification="joint_type", moment_spec="fast_stock",
):
    parinc = np.concatenate([pack["parinc"] for pack in packs])
    q = np.concatenate([pack["q"] for pack in packs], axis=0)
    flow = np.concatenate([pack["loan_flow"] for pack in packs])
    stock = np.concatenate([pack["debtchoice"] for pack in packs])
    if specification not in EDUCATION_CELL_SPECIFICATIONS:
        raise ValueError(f"specification must be one of {EDUCATION_CELL_SPECIFICATIONS}.")
    if moment_spec == SPLIT_MOMENT_SPEC:
        if specification != "parental_income_basic":
            raise ValueError(
                f"The {SPLIT_MOMENT_SPEC} moments require the "
                "parental_income_basic specification."
            )
        begin_debt = np.concatenate([pack["debt"] for pack in packs])
        annual_cap = np.concatenate([pack["annual_cap"] for pack in packs])
        _warn_thin_split_groups(
            parinc, begin_debt,
            f"cell {int(packs[0]['cell_code'])}",
        )
        return parental_income_split_moments(
            parinc, flow, begin_debt, annual_cap
        )
    if specification == "parental_income_basic":
        return parental_income_distribution_moments(
            parinc, flow, stock, moment_spec=moment_spec
        )
    if specification == "parental_income_loan_type":
        return parental_income_loan_type_distribution_moments(
            parinc,
            np.broadcast_to(flow, (N_TYPES, len(flow))),
            np.broadcast_to(stock, (N_TYPES, len(stock))),
            moment_spec=moment_spec,
            q=q,
        )
    return posterior_loan_moments(
        parinc, np.broadcast_to(flow, (N_TYPES, len(flow))),
        np.broadcast_to(stock, (N_TYPES, len(stock))), q,
    )


def _print_cell_fit(data, simulated, data_new_share, sim_new_share, weights, labels, loss):
    print("\n" + "=" * 112)
    print(f"[education-cell eval {EVAL_COUNTER}] loss={loss:.6f}")
    print(" loan  parinc | mean positive annual flow: data       sim       diff | "
          "share end-stock>0: data     sim     diff | new-flow share (diagnostic)")
    for row, (loan_type, parinc) in enumerate(labels):
        m = 2 * row
        print(
            f"  {loan_type:>2}     {parinc:>2}   | "
            f"{data[m]:>10.2f} {simulated[m]:>10.2f} {simulated[m]-data[m]:>10.2f} | "
            f"{data[m+1]:>7.4f} {simulated[m+1]:>7.4f} {simulated[m+1]-data[m+1]:>+8.4f} | "
            f"{data_new_share[row]:>7.4f} -> {sim_new_share[row]:>7.4f}  "
            f"(posterior weight={weights[row]:.1f})"
        )
    print("=" * 112 + "\n")


def parental_income_moment_weight_pattern(moment_spec, primary_moment_weight):
    """Within-parinc SMM weights in the exact moment-output ordering."""
    if moment_spec == SPLIT_MOMENT_SPEC:
        # [entry rate, continuation rate, mean entry flow,
        #  mean continuation flow, at-cap share] = [4, 4, 2, 2, 1] at the
        # defaults (primary=4, stock weight=2).
        return np.asarray(
            [
                primary_moment_weight, primary_moment_weight,
                STOCK_MOMENT_WEIGHT, STOCK_MOMENT_WEIGHT, 1.0,
            ],
            dtype=np.float64,
        )
    base = [primary_moment_weight, primary_moment_weight, 1.0, 1.0]
    if moment_spec == "flow_plus_stock":
        base.extend((STOCK_MOMENT_WEIGHT, STOCK_MOMENT_WEIGHT))
    return np.asarray(base, dtype=np.float64)


def _print_split_fit(
    data, simulated, data_new_share, sim_new_share, weights, labels, loss,
    primary_moment_weight,
):
    """Fit table for the flow_split_stock moments; used only for that spec."""
    print("\n" + "=" * 132)
    print(f"[education-cell eval {EVAL_COUNTER}] weighted standardized loss={loss:.6f}")
    print(
        " loss weights per parinc group: "
        f"entry rate={primary_moment_weight:g}, continuation "
        f"rate={primary_moment_weight:g}, mean entry "
        f"flow={STOCK_MOMENT_WEIGHT:g}, mean continuation "
        f"flow={STOCK_MOMENT_WEIGHT:g}, at-cap share=1"
    )
    print(
        " parinc | entry rate (debt==0): data/sim/diff | continuation rate "
        "(debt>0): data/sim/diff | mean flow>0 (debt==0): data/sim/diff | "
        "mean flow>0 (debt>0): data/sim/diff | at-cap share: data/sim/diff"
    )
    for row, parinc in enumerate(labels):
        m = 5 * row
        pieces = []
        for offset, numeric_format in (
            (0, "6.4f"), (1, "6.4f"), (2, "8.2f"), (3, "8.2f"), (4, "6.4f"),
        ):
            pieces.append(
                f"{data[m+offset]:{numeric_format}}/"
                f"{simulated[m+offset]:{numeric_format}}/"
                f"{simulated[m+offset]-data[m+offset]:+8.2f}"
            )
        print(f"   {parinc:>2}   | " + " | ".join(pieces))
    print(" positive-flow share diagnostic (data -> simulation): " + ", ".join(
        f"parinc {level}: {data_new_share[row]:.4f}->{sim_new_share[row]:.4f}"
        for row, level in enumerate(labels)
    ))
    print(" cell observation weights: " + ", ".join(
        f"parinc {level}: {weights[row]:.1f}" for row, level in enumerate(labels)
    ))
    print("=" * 132 + "\n")


def _print_parental_income_fit(
    data, simulated, data_new_share, sim_new_share, weights, labels, loss,
    moment_spec, primary_moment_weight,
):
    print("\n" + "=" * 132)
    print(f"[education-cell eval {EVAL_COUNTER}] weighted standardized loss={loss:.6f}")
    if moment_spec == "flow_plus_stock":
        print(
            " loss weights per parinc group: "
            f"flow mean={primary_moment_weight:g}, flow receipt "
            f"share={primary_moment_weight:g}, flow std=1, flow p80=1, "
            f"stock mean={STOCK_MOMENT_WEIGHT:g}, stock indebted "
            f"share={STOCK_MOMENT_WEIGHT:g}"
        )
        print(
            " parinc | mean positive flow: data/sim/diff | share new-flow>0: "
            "data/sim/diff | std positive flow: data/sim/diff | "
            "p80 positive flow: data/sim/diff | mean positive stock: "
            "data/sim/diff | share end-stock>0: data/sim/diff"
        )
        for row, parinc in enumerate(labels):
            m = 6 * row
            pieces = []
            for offset, numeric_format in (
                (0, "8.2f"), (1, "6.4f"), (2, "8.2f"),
                (3, "8.2f"), (4, "8.2f"), (5, "6.4f"),
            ):
                pieces.append(
                    f"{data[m+offset]:{numeric_format}}/"
                    f"{simulated[m+offset]:{numeric_format}}/"
                    f"{simulated[m+offset]-data[m+offset]:+8.2f}"
                )
            print(f"   {parinc:>2}   | " + " | ".join(pieces))
        print(" cell observation weights: " + ", ".join(
            f"parinc {level}: {weights[row]:.1f}"
            for row, level in enumerate(labels)
        ))
        print("=" * 132 + "\n")
        return
    share_name = (
        "share new-flow>0" if moment_spec == "fast_flow"
        else "share end-stock>0"
    )
    share_weight_name = (
        "share receiving new loans" if moment_spec == "fast_flow"
        else "share indebted"
    )
    print(
        " loss weights per parinc group: "
        f"mean positive={primary_moment_weight:g}, "
        f"{share_weight_name}={primary_moment_weight:g}, std=1, p80=1"
    )
    source = "stock" if moment_spec == "fast_stock" else "annual flow"
    print(
        f" parinc | mean positive {source}: data/sim/diff | {share_name}: "
        f"data/sim/diff | std positive {source}: data/sim/diff | "
        f"p80 positive {source}: data/sim/diff"
    )
    for row, parinc in enumerate(labels):
        m = 4 * row
        print(
            f"   {parinc:>2}   | "
            f"{data[m]:>8.2f}/{simulated[m]:>8.2f}/{simulated[m]-data[m]:>+8.2f} | "
            f"{data[m+1]:>6.4f}/{simulated[m+1]:>6.4f}/"
            f"{simulated[m+1]-data[m+1]:>+7.4f} | "
            f"{data[m+2]:>8.2f}/{simulated[m+2]:>8.2f}/"
            f"{simulated[m+2]-data[m+2]:>+8.2f} | "
            f"{data[m+3]:>8.2f}/{simulated[m+3]:>8.2f}/"
            f"{simulated[m+3]-data[m+3]:>+8.2f}"
        )
    print(" positive-flow share diagnostic (data -> simulation): " + ", ".join(
        f"parinc {level}: {data_new_share[row]:.4f}->{sim_new_share[row]:.4f}"
        for row, level in enumerate(labels)
    ))
    print(" cell observation weights: " + ", ".join(
        f"parinc {level}: {weights[row]:.1f}" for row, level in enumerate(labels)
    ))
    print("=" * 132 + "\n")


def _print_parental_income_loan_type_fit(
    data, simulated, data_new_share, sim_new_share, weights, labels, loss,
    moment_spec, primary_moment_weight,
):
    print("\n" + "=" * 144)
    print(f"[education-cell eval {EVAL_COUNTER}] weighted standardized loss={loss:.6f}")
    if moment_spec == "flow_plus_stock":
        print(
            " TARGETED LOAN-TYPE MOMENTS; weights: "
            f"flow mean={primary_moment_weight:g}, flow receipt "
            f"share={primary_moment_weight:g}, flow std=1, flow p80=1, "
            f"stock mean={STOCK_MOMENT_WEIGHT:g}, stock indebted "
            f"share={STOCK_MOMENT_WEIGHT:g}"
        )
        print(
            " loan type | parinc | flow mean | flow receipt share | flow std | "
            "flow p80 | stock mean | stock indebted share"
        )
        for row, (loan_type, parinc) in enumerate(labels):
            m = 6 * row
            type_name = "low" if loan_type == 0 else "high"
            pieces = [
                f"{data[m+k]:.4f}/{simulated[m+k]:.4f}/"
                f"{simulated[m+k]-data[m+k]:+.4f}"
                for k in range(6)
            ]
            print(
                f" {type_name:>9} |   {parinc:>2}   | "
                + " | ".join(pieces)
                + f" (weight={weights[row]:.1f})"
            )
        print("=" * 144 + "\n")
        return
    share_name = (
        "share new-flow>0" if moment_spec == "fast_flow"
        else "share end-stock>0"
    )
    share_weight_name = (
        "share receiving new loans" if moment_spec == "fast_flow"
        else "share indebted"
    )
    print(
        " TARGETED LOAN-TYPE MOMENTS; loss weights per cell: "
        f"mean positive={primary_moment_weight:g}, "
        f"{share_weight_name}={primary_moment_weight:g}, std=1, p80=1"
    )
    source = "stock" if moment_spec == "fast_stock" else "annual flow"
    print(
        f" loan type | parinc | mean positive {source}: data/sim/diff | "
        f"{share_name}: data/sim/diff | std: data/sim/diff | "
        "p80: data/sim/diff | positive-flow share"
    )
    for row, (loan_type, parinc) in enumerate(labels):
        m = 4 * row
        type_name = "low" if loan_type == 0 else "high"
        print(
            f" {type_name:>9} |   {parinc:>2}   | "
            f"{data[m]:>8.2f}/{simulated[m]:>8.2f}/{simulated[m]-data[m]:>+8.2f} | "
            f"{data[m+1]:>6.4f}/{simulated[m+1]:>6.4f}/"
            f"{simulated[m+1]-data[m+1]:>+7.4f} | "
            f"{data[m+2]:>8.2f}/{simulated[m+2]:>8.2f}/"
            f"{simulated[m+2]-data[m+2]:>+8.2f} | "
            f"{data[m+3]:>8.2f}/{simulated[m+3]:>8.2f}/"
            f"{simulated[m+3]-data[m+3]:>+8.2f} | "
            f"{data_new_share[row]:.4f}->{sim_new_share[row]:.4f} "
            f"(weight={weights[row]:.1f})"
        )
    print("=" * 144 + "\n")


def minimize_distance_education_cell(
    params, data_moments, data_new_share, data_weights, labels, sample_by_period,
    cell_code, education, program_year, shock_heterogeneity,
):
    """One education-cell SMM objective with exact integration over 16 types."""
    global EVAL_COUNTER
    EVAL_COUNTER += 1
    spec = bs.unpack_estimation_vector(
        params, [cell_code], loan_heterogeneity=shock_heterogeneity,
        index_kind="education_cell",
    )
    draws = next(iter(sample_by_period.values()))["budget_z"].shape[0]
    simulated_by_draw = []
    diagnostic_by_draw = []

    for draw_index in range(draws):
        pooled_flow, pooled_stock, pooled_parinc, pooled_q = [], [], [], []
        for period, pack in sample_by_period.items():
            x1 = pack["x1"]
            terminal = np.zeros((len(x1), debt_range.size), dtype=np.float64)
            cache = {}
            parental_group = x1[:, 0].astype(int) - 1
            for group_index in range(4):
                idx = np.flatnonzero(parental_group == group_index)
                if idx.size:
                    terminal[idx] = get_relevant_terminal_subset_cached(
                        pack["terminal_data"], spec["risk_aversion"][group_index], idx, cache
                    )
            sigma_i = bs.risk_aversion(spec, x1).astype(np.float64)
            debtpen_i = discounted_explicit_horizon_debt_penalty(
                spec, x1, period
            )
            kappa_entry_i, kappa_cont_i, prev_debt_i = (
                _zero_new_borrowing_kernel_arrays(len(x1))
            )
            stock_by_type = np.empty((N_TYPES, len(x1)), dtype=np.float64)
            for type_index in range(N_TYPES):
                shock = bs.realization(
                    spec, x1, None, pack["budget_z"][draw_index],
                    loan_type=int(TYPE_LOAN[type_index]), education=education,
                    program_year=program_year,
                ).astype(np.float64)
                debt_index = solve_one_draw_debt_idx_terminal_only(
                    budget=pack["base_budget_crn"][draw_index, type_index],
                    e=shock, debt_grid=debt_range, sigma_i=sigma_i,
                    debtpen_i=debtpen_i,
                    ccp_path_row=pack["ccp_by_type"][type_index],
                    terminal_row=terminal, b_idx=pack["b_idx"], max_idx=pack["max_idx"],
                    beta_term=float(beta ** (T - period)),
                    kappa_entry_i=kappa_entry_i,
                    kappa_cont_i=kappa_cont_i,
                    prev_debt_i=prev_debt_i,
                    c_floor=CONSUMPTION_FLOOR,
                    fallback_idx=debt_range.size - 1,
                )
                stock_by_type[type_index] = debt_range[debt_index]
            flow_by_type = stock_by_type - (1.0 + r) * pack["debt"][None, :]
            pooled_flow.append(flow_by_type)
            pooled_stock.append(stock_by_type)
            pooled_parinc.append(pack["parinc"])
            pooled_q.append(pack["q"])
        moments, new_share, _, _ = posterior_loan_moments(
            np.concatenate(pooled_parinc), np.concatenate(pooled_flow, axis=1),
            np.concatenate(pooled_stock, axis=1), np.concatenate(pooled_q, axis=0),
        )
        simulated_by_draw.append(moments)
        diagnostic_by_draw.append(new_share)

    simulated = np.nanmean(np.asarray(simulated_by_draw), axis=0)
    sim_new_share = np.nanmean(np.asarray(diagnostic_by_draw), axis=0)
    valid = np.isfinite(data_moments) & np.isfinite(simulated)
    scale = np.where(np.arange(len(data_moments)) % 2 == 0,
                     np.maximum(np.abs(data_moments), 100.0), 1.0)
    loss = float(np.sum(((simulated[valid] - data_moments[valid]) / scale[valid]) ** 2))
    if EVAL_COUNTER % 10 == 0:
        _print_cell_fit(
            data_moments, simulated, data_new_share, sim_new_share,
            data_weights, labels, loss,
        )
        print("loan mean shift:", np.round(spec["loan_mean_shift"], 3),
              "loan sigma ratio:", np.round(np.exp(spec["loan_log_sigma_ratio"]), 4))
    return loss


def _pool_sampled_education_cell_evaluation(
    sample_by_period, spec, education, program_year,
):
    """Pool period packs without discarding any period-specific model input."""
    first_pack = next(iter(sample_by_period.values()))
    if "pooled_sampled_static" not in first_pack:
        raise ValueError("Sampled education-cell static arrays were not prepared.")
    static = first_pack["pooled_sampled_static"]
    terminals = []
    curve_cache = {}
    for period, pack in sample_by_period.items():
        x1 = pack["x1"]
        terminal = np.zeros((len(x1), debt_range.size), dtype=np.float64)
        parental_group = x1[:, 0].astype(np.int64) - 1
        has_loan_risk = "risk_aversion_by_loan_type" in spec
        loan_levels = (0, 1) if has_loan_risk else (None,)
        for loan_level in loan_levels:
            for group_index in range(4):
                selected = parental_group == group_index
                if loan_level is not None:
                    selected &= pack["sampled_loan_type"] == loan_level
                idx = np.flatnonzero(selected)
                if idx.size:
                    risk_level = (
                        spec["risk_aversion_by_loan_type"][loan_level, group_index]
                        if loan_level is not None
                        else spec["risk_aversion"][group_index]
                    )
                    terminal[idx] = get_relevant_terminal_subset_cached(
                        pack["terminal_data"], risk_level, idx, curve_cache,
                    )
        terminals.append(terminal)

    pooled = dict(static)
    pooled["terminal"] = np.ascontiguousarray(
        np.concatenate(terminals), dtype=np.float64
    )
    x1 = pooled["x1"]
    pooled["sigma_i"] = np.ascontiguousarray(
        bs.risk_aversion(
            spec, x1,
            loan_type=(
                pooled["loan_type"]
                if "risk_aversion_by_loan_type" in spec else None
            ),
        ), dtype=np.float64
    )
    pooled["debtpen_i"] = discounted_explicit_horizon_debt_penalty(
        spec, x1, pooled["model_period"]
    )
    # One-shot new-borrowing costs, resolved per individual from the sampled
    # loan type and the observed beginning-of-period debt. Specifications
    # without the kappa block produce zero arrays, leaving the kernels'
    # numerical output unchanged.
    pooled["kappa_entry_i"], pooled["kappa_cont_i"], _ = (
        _new_borrowing_kernel_arrays(spec, pooled["loan_type"], pooled["debt"])
    )
    pooled["shock"] = np.ascontiguousarray(
        bs.realization(
            spec, x1, None, pooled["budget_z"],
            loan_type=(
                pooled["loan_type"]
                if spec.get("loan_heterogeneity") == "mean" else None
            ),
            education=education, program_year=program_year,
            pre_choice_resources=pooled["base_budget"],
        ),
        dtype=np.float64,
    )
    if pooled["base_budget"].shape != pooled["shock"].shape:
        raise ValueError("Pooled budget and shock arrays are misaligned.")
    return pooled


def parental_income_cell_loss_and_residuals(
    simulated, data_moments, moment_spec, primary_moment_weight,
):
    """One cell's weighted SMM loss and its exactly consistent residuals.

    The scalar loss is the historical in-line computation, unchanged:
    ``sum_k w_k * ((sim_k - data_k) / max(|data_k|, 1e-6))**2`` over the
    finite moments, with ``w_k`` the tiled within-parinc weight pattern.
    The residual vector stacks
    ``r_k = sqrt(w_k) * (sim_k - data_k) / max(|data_k|, 1e-6)``
    with zeros at non-finite moments, so ``sum(r**2) == loss`` exactly (up
    to floating point). Least-squares optimizers (DFO-LS) consume the
    residuals; every scalar optimizer keeps consuming the identical loss.
    """
    simulated = np.asarray(simulated, dtype=np.float64)
    data_moments = np.asarray(data_moments, dtype=np.float64)
    valid = np.isfinite(data_moments) & np.isfinite(simulated)
    scale = np.maximum(np.abs(data_moments), 1.0e-6)
    moment_weights = np.tile(
        parental_income_moment_weight_pattern(
            moment_spec, primary_moment_weight
        ),
        4,
    )
    standardized_error = (simulated - data_moments) / scale
    loss = float(np.sum(moment_weights[valid] * standardized_error[valid] ** 2))
    residuals = np.zeros(data_moments.size, dtype=np.float64)
    residuals[valid] = (
        np.sqrt(moment_weights[valid]) * standardized_error[valid]
    )
    return loss, residuals


def _evaluate_sampled_parental_income_cell(
    full_params, data_moments, sample_by_period, cell_code, education,
    program_year, moment_spec, primary_moment_weight,
    new_borrowing_costs=None,
):
    """Evaluate one cell's parental-income moments without a global count.

    ``new_borrowing_costs`` is the optional shared kappa tail
    ``[kappa0_low, kappa0_high, kappa1]`` estimated jointly across cells. It
    is injected into the per-cell specification after unpacking, because the
    14-entry per-cell vector deliberately does not carry the kappa block.
    """
    spec = bs.unpack_parental_income_estimation_vector(
        full_params, [cell_code], index_kind="education_cell"
    )
    if new_borrowing_costs is not None:
        new_borrowing_costs = np.asarray(
            new_borrowing_costs, dtype=np.float64
        ).reshape(-1)
        if new_borrowing_costs.size != bs.N_NEW_BORROWING_PARAMETERS:
            raise ValueError(
                "new_borrowing_costs must contain "
                f"{bs.N_NEW_BORROWING_PARAMETERS} entries."
            )
        spec["new_borrow_cost_entry_by_loan_type"] = (
            new_borrowing_costs[:2].copy()
        )
        spec["new_borrow_cost_continuation"] = float(new_borrowing_costs[2])
    pooled = _pool_sampled_education_cell_evaluation(
        sample_by_period, spec, education, program_year
    )
    debt_index_by_draw = solve_all_draws_debt_idx_pooled(
        budget_by_draw=pooled["base_budget"],
        shock_by_draw=pooled["shock"],
        debt_grid=debt_range,
        sigma_i=pooled["sigma_i"],
        debtpen_i=pooled["debtpen_i"],
        ccp_path_row=pooled["ccp"],
        terminal_row=pooled["terminal"],
        b_idx=pooled["b_idx"],
        max_idx=pooled["max_idx"],
        beta_term_i=pooled["beta_term"],
        kappa_entry_i=pooled["kappa_entry_i"],
        kappa_cont_i=pooled["kappa_cont_i"],
        prev_debt_i=pooled["debt"],
        c_floor=CONSUMPTION_FLOOR,
        fallback_idx=debt_range.size - 1,
    )
    stock_by_draw = debt_range[debt_index_by_draw]
    flow_by_draw = stock_by_draw - (1.0 + r) * pooled["debt"][None, :]
    simulated_by_draw = []
    diagnostic_by_draw = []
    for draw_index in range(stock_by_draw.shape[0]):
        if moment_spec == SPLIT_MOMENT_SPEC:
            moments, new_share, _, _ = parental_income_split_moments(
                pooled["parinc"], flow_by_draw[draw_index], pooled["debt"],
                pooled["annual_cap"],
            )
        else:
            moments, new_share, _, _ = parental_income_distribution_moments(
                pooled["parinc"], flow_by_draw[draw_index], stock_by_draw[draw_index],
                moment_spec=moment_spec,
            )
        simulated_by_draw.append(moments)
        diagnostic_by_draw.append(new_share)
    simulated = np.nanmean(np.asarray(simulated_by_draw), axis=0)
    sim_new_share = np.nanmean(np.asarray(diagnostic_by_draw), axis=0)
    loss, residuals = parental_income_cell_loss_and_residuals(
        simulated, data_moments, moment_spec, primary_moment_weight
    )
    return loss, simulated, sim_new_share, spec, residuals


def _initialize_cell_smm_worker(
    contexts, moment_spec, primary_moment_weight, numba_threads,
):
    """Keep prepared cell data resident in each persistent SMM worker."""
    global _CELL_WORKER_CONTEXTS
    global _CELL_WORKER_MOMENT_SPEC
    global _CELL_WORKER_PRIMARY_WEIGHT
    _CELL_WORKER_CONTEXTS = contexts
    _CELL_WORKER_MOMENT_SPEC = moment_spec
    _CELL_WORKER_PRIMARY_WEIGHT = float(primary_moment_weight)
    set_num_threads(int(numba_threads))


def _split_multicell_shared_tail(params, n_cells, include_new_borrowing):
    """Slice the flat multicell vector's shared tail after the cell blocks.

    Returns ``(shared_risk, shared_debt, new_borrowing)`` where the last item
    is the three-entry kappa tail ``[kappa0_low, kappa0_high, kappa1]`` when
    ``include_new_borrowing`` is set and ``None`` otherwise. The risk and
    debt-penalty slices are unchanged from the legacy 68-entry layout.
    """
    params = np.asarray(params, dtype=np.float64)
    block_size = bs.PARENTAL_INCOME_MULTICELL_PARAMETERS_PER_CELL
    tail_start = int(n_cells) * block_size
    shared_risk = params[tail_start:tail_start + 4]
    shared_debt = params[tail_start + 4:tail_start + 8]
    new_borrowing = None
    if include_new_borrowing:
        expected = tail_start + 8 + bs.N_NEW_BORROWING_PARAMETERS
        if params.size != expected:
            raise ValueError(
                f"Multicell vector has {params.size} entries; expected "
                f"{expected} with the new-borrowing cost block."
            )
        new_borrowing = params[tail_start + 8:tail_start + 11]
    return shared_risk, shared_debt, new_borrowing


def _evaluate_cell_smm_worker(task):
    """Evaluate one education cell for one common parameter proposal."""
    cell_index, block, shared_risk, shared_debt = task[:4]
    new_borrowing = task[4] if len(task) > 4 else None
    context = _CELL_WORKER_CONTEXTS[cell_index]
    full_params = np.concatenate(
        (block[0:5], shared_risk, shared_debt, block[5:6])
    )
    loss, simulated, sim_new_share, _, residuals = (
        _evaluate_sampled_parental_income_cell(
            full_params,
            context["data_moments"],
            context["sample_by_period"],
            context["cell_code"],
            int(context["education"]),
            int(context["program_year"]),
            _CELL_WORKER_MOMENT_SPEC,
            _CELL_WORKER_PRIMARY_WEIGHT,
            new_borrowing_costs=new_borrowing,
        )
    )
    return cell_index, loss, simulated, sim_new_share, residuals


def minimize_distance_education_cells_parental_income(
    params, contexts, moment_spec, primary_moment_weight, cell_pool=None,
    return_residuals=False,
):
    """Joint multi-cell loss with risk aversion shared across cells.

    With ``return_residuals=True`` (used only by the opt-in DFO-LS branch)
    the return value is the stacked weighted residual vector: for each cell,
    in the existing cell order, ``sqrt(cell_weight)`` times the cell's
    residuals from ``parental_income_cell_loss_and_residuals``. Its sum of
    squares equals the scalar loss returned by the default mode exactly.
    """
    global EVAL_COUNTER
    EVAL_COUNTER += 1
    params = np.asarray(params, dtype=np.float64)
    n_cells = len(contexts)
    block_size = bs.PARENTAL_INCOME_MULTICELL_PARAMETERS_PER_CELL
    shared_risk, shared_debt, shared_new_borrowing = (
        _split_multicell_shared_tail(
            params, n_cells, ESTIMATE_NEW_BORROWING_COST
        )
    )
    tasks = []
    for cell_index in range(n_cells):
        block = params[cell_index * block_size:(cell_index + 1) * block_size]
        if shared_new_borrowing is None:
            tasks.append((cell_index, block, shared_risk, shared_debt))
        else:
            tasks.append((
                cell_index, block, shared_risk, shared_debt,
                shared_new_borrowing,
            ))

    if cell_pool is None:
        results = []
        for task in tasks:
            cell_index, block = task[0], task[1]
            context = contexts[cell_index]
            full_params = np.concatenate(
                (block[0:5], task[2], task[3], block[5:6])
            )
            loss, simulated, sim_new_share, _, residuals = (
                _evaluate_sampled_parental_income_cell(
                    full_params,
                    context["data_moments"],
                    context["sample_by_period"],
                    context["cell_code"],
                    int(context["education"]),
                    int(context["program_year"]),
                    moment_spec,
                    primary_moment_weight,
                    new_borrowing_costs=(
                        task[4] if len(task) > 4 else None
                    ),
                )
            )
            results.append(
                (cell_index, loss, simulated, sim_new_share, residuals)
            )
    else:
        results = cell_pool.map(_evaluate_cell_smm_worker, tasks, chunksize=1)

    results.sort(key=lambda item: item[0])
    total_loss = float(sum(
        contexts[item[0]]["cell_weight"] * item[1] for item in results
    ))

    if EVAL_COUNTER % 10 == 0:
        print("\n" + "#" * 132)
        print(
            f"[multi-cell eval {EVAL_COUNTER}] total weighted standardized "
            f"loss={total_loss:.6f}; shared risk aversion="
            f"{np.round(shared_risk, 4)}"
        )
        print(f"shared debt penalties={np.round(shared_debt, 4)}")
        if shared_new_borrowing is not None:
            print(
                "shared new-borrowing costs (one-shot, no multiplier): "
                f"kappa0 low/high={np.round(shared_new_borrowing[:2], 4)}; "
                f"kappa1 continuation={round(float(shared_new_borrowing[2]), 4)}"
            )
        print("#" * 132)
        for context, evaluation in zip(contexts, results):
            education = int(context["education"])
            program_year = int(context["program_year"])
            cell_index, loss, simulated, sim_new_share = evaluation[:4]
            block = params[
                cell_index * block_size:(cell_index + 1) * block_size
            ]
            cell_weight = float(context["cell_weight"])
            weighted_loss = cell_weight * loss
            year_label = bs.budget_program_year_label(
                education, program_year
            )
            print(
                f"EDUCATION {education}, PROGRAM YEAR {year_label}; "
                f"N={int(context['cell_observations'])}; "
                f"cell weight={cell_weight:.4f}; raw loss={loss:.6f}; "
                f"weighted contribution={weighted_loss:.6f}"
            )
            if moment_spec == SPLIT_MOMENT_SPEC:
                _print_split_fit(
                    context["data_moments"], simulated,
                    context["data_new_share"], sim_new_share,
                    context["data_weights"], context["labels"], loss,
                    primary_moment_weight,
                )
            else:
                _print_parental_income_fit(
                    context["data_moments"], simulated,
                    context["data_new_share"], sim_new_share,
                    context["data_weights"], context["labels"], loss,
                    moment_spec, primary_moment_weight,
                )
            print("shock means by parinc:", np.round(block[0:4], 3),
                  "cell sigma:", round(float(block[4]), 3))
            print(
                "pre-choice-resource slope ($ shock per $10,000 resources):",
                round(float(block[5]), 4),
            )
            if float(block[4]) <= 1.000001:
                print("WARNING: cell budget-shock sigma is at its lower bound (1.0).")
    if return_residuals:
        return np.concatenate([
            np.sqrt(float(contexts[item[0]]["cell_weight"])) * item[4]
            for item in results
        ])
    return total_loss


def minimize_distance_education_cell_parental_income(
    params, data_moments, data_new_share, data_weights, labels, sample_by_period,
    cell_code, education, program_year, type_integration="sampled",
    moment_spec="fast_stock", primary_moment_weight=DEFAULT_PRIMARY_MOMENT_WEIGHT,
    specification="parental_income_basic", integrated_data=None,
):
    """Parinc SMM, optionally with loan-type-specific risk aversion."""
    global EVAL_COUNTER
    EVAL_COUNTER += 1
    if type_integration not in TYPE_INTEGRATION_MODES:
        raise ValueError(f"type_integration must be one of {TYPE_INTEGRATION_MODES}.")
    if moment_spec not in PARENTAL_INCOME_MOMENT_SPECS:
        raise ValueError(f"moment_spec must be one of {PARENTAL_INCOME_MOMENT_SPECS}.")
    if not np.isfinite(primary_moment_weight) or primary_moment_weight <= 0.0:
        raise ValueError("primary_moment_weight must be positive and finite.")
    if specification == "parental_income_basic":
        spec = bs.unpack_parental_income_estimation_vector(
            params, [cell_code], index_kind="education_cell"
        )
    elif specification == "parental_income_loan_type":
        spec = bs.unpack_parental_income_loan_type_estimation_vector(
            params, [cell_code], index_kind="education_cell"
        )
        if integrated_data is None:
            raise ValueError("integrated_data is required for the loan-type specification.")
    else:
        raise ValueError("Unsupported parental-income specification.")
    draws = next(iter(sample_by_period.values()))["budget_z"].shape[0]
    simulated_by_draw = []
    diagnostic_by_draw = []
    integrated_by_draw = []
    integrated_diagnostic_by_draw = []

    if type_integration == "sampled":
        pooled = _pool_sampled_education_cell_evaluation(
            sample_by_period, spec, education, program_year
        )
        debt_index_by_draw = solve_all_draws_debt_idx_pooled(
            budget_by_draw=pooled["base_budget"],
            shock_by_draw=pooled["shock"],
            debt_grid=debt_range,
            sigma_i=pooled["sigma_i"],
            debtpen_i=pooled["debtpen_i"],
            ccp_path_row=pooled["ccp"],
            terminal_row=pooled["terminal"],
            b_idx=pooled["b_idx"],
            max_idx=pooled["max_idx"],
            beta_term_i=pooled["beta_term"],
            kappa_entry_i=pooled["kappa_entry_i"],
            kappa_cont_i=pooled["kappa_cont_i"],
            prev_debt_i=pooled["debt"],
            c_floor=CONSUMPTION_FLOOR,
            fallback_idx=debt_range.size - 1,
        )
        stock_by_draw = debt_range[debt_index_by_draw]
        flow_by_draw = stock_by_draw - (1.0 + r) * pooled["debt"][None, :]
        for draw_index in range(draws):
            if specification == "parental_income_loan_type":
                moments, new_share, _, _ = (
                    parental_income_loan_type_distribution_moments(
                        pooled["parinc"], flow_by_draw[draw_index],
                        stock_by_draw[draw_index], moment_spec=moment_spec,
                        loan_type=pooled["loan_type"],
                    )
                )
                integrated, integrated_new_share, _, _ = (
                    parental_income_distribution_moments(
                        pooled["parinc"], flow_by_draw[draw_index],
                        stock_by_draw[draw_index], moment_spec=moment_spec,
                    )
                )
                integrated_by_draw.append(integrated)
                integrated_diagnostic_by_draw.append(integrated_new_share)
            else:
                moments, new_share, _, _ = parental_income_distribution_moments(
                    pooled["parinc"], flow_by_draw[draw_index],
                    stock_by_draw[draw_index], moment_spec=moment_spec,
                )
            simulated_by_draw.append(moments)
            diagnostic_by_draw.append(new_share)
    else:
        # The exact validation mode retains its explicit type loop. Terminal
        # values are still constructed only once per period and evaluation.
        evaluation_by_period = {}
        for period, pack in sample_by_period.items():
            x1 = pack["x1"]
            cache = {}
            parental_group = x1[:, 0].astype(int) - 1
            if "risk_aversion_by_loan_type" in spec:
                terminal = np.zeros(
                    (2, len(x1), debt_range.size), dtype=np.float64
                )
                sigma_i = np.empty((2, len(x1)), dtype=np.float64)
                for loan_level in (0, 1):
                    sigma_i[loan_level] = bs.risk_aversion(
                        spec, x1, loan_type=loan_level
                    )
                    for group_index in range(4):
                        idx = np.flatnonzero(parental_group == group_index)
                        if idx.size:
                            terminal[loan_level, idx] = (
                                get_relevant_terminal_subset_cached(
                                    pack["terminal_data"],
                                    spec["risk_aversion_by_loan_type"][
                                        loan_level, group_index
                                    ],
                                    idx, cache,
                                )
                            )
            else:
                terminal = np.zeros(
                    (len(x1), debt_range.size), dtype=np.float64
                )
                for group_index in range(4):
                    idx = np.flatnonzero(parental_group == group_index)
                    if idx.size:
                        terminal[idx] = get_relevant_terminal_subset_cached(
                            pack["terminal_data"],
                            spec["risk_aversion"][group_index], idx, cache,
                        )
                sigma_i = bs.risk_aversion(spec, x1).astype(np.float64)
            zero_kappa_entry, zero_kappa_cont, zero_prev_debt = (
                _zero_new_borrowing_kernel_arrays(len(x1))
            )
            evaluation_by_period[period] = {
                "terminal": terminal,
                "sigma_i": sigma_i,
                "debtpen_i": discounted_explicit_horizon_debt_penalty(
                    spec, x1, period
                ),
                "kappa_entry_i": zero_kappa_entry,
                "kappa_cont_i": zero_kappa_cont,
                "prev_debt_i": zero_prev_debt,
            }

        for draw_index in range(draws):
            pooled_flow, pooled_stock, pooled_parinc, pooled_q = [], [], [], []
            for period, pack in sample_by_period.items():
                x1 = pack["x1"]
                evaluation = evaluation_by_period[period]
                stock_by_type = np.empty((N_TYPES, len(x1)), dtype=np.float64)
                for type_index in range(N_TYPES):
                    loan_level = int(TYPE_LOAN[type_index])
                    type_specific_risk = "risk_aversion_by_loan_type" in spec
                    shock_type = bs.realization(
                        spec, x1, None, pack["budget_z"][draw_index],
                        loan_type=(
                            int(TYPE_LOAN[type_index])
                            if specification == "parental_income_loan_type" else None
                        ),
                        education=education, program_year=program_year,
                        pre_choice_resources=(
                            pack["base_budget_crn"][draw_index, type_index]
                        ),
                    ).astype(np.float64)
                    debt_index = solve_one_draw_debt_idx_terminal_only(
                        budget=pack["base_budget_crn"][draw_index, type_index],
                        e=shock_type, debt_grid=debt_range,
                        sigma_i=(
                            evaluation["sigma_i"][loan_level]
                            if type_specific_risk else evaluation["sigma_i"]
                        ),
                        debtpen_i=evaluation["debtpen_i"],
                        ccp_path_row=pack["ccp_by_type"][type_index],
                        terminal_row=(
                            evaluation["terminal"][loan_level]
                            if type_specific_risk else evaluation["terminal"]
                        ),
                        b_idx=pack["b_idx"], max_idx=pack["max_idx"],
                        beta_term=float(beta ** (T - period)),
                        kappa_entry_i=evaluation["kappa_entry_i"],
                        kappa_cont_i=evaluation["kappa_cont_i"],
                        prev_debt_i=evaluation["prev_debt_i"],
                        c_floor=CONSUMPTION_FLOOR,
                        fallback_idx=debt_range.size - 1,
                    )
                    stock_by_type[type_index] = debt_range[debt_index]
                pooled_stock.append(stock_by_type)
                pooled_flow.append(stock_by_type - (1.0 + r) * pack["debt"][None, :])
                pooled_q.append(pack["q"])
                pooled_parinc.append(pack["parinc"])
            all_parinc = np.concatenate(pooled_parinc)
            all_flow = np.concatenate(pooled_flow, axis=1)
            all_stock = np.concatenate(pooled_stock, axis=1)
            all_q = np.concatenate(pooled_q, axis=0)
            if specification == "parental_income_loan_type":
                moments, new_share, _, _ = (
                    parental_income_loan_type_distribution_moments(
                        all_parinc, all_flow, all_stock,
                        moment_spec=moment_spec, q=all_q,
                    )
                )
                integrated, integrated_new_share, _, _ = (
                    parental_income_distribution_moments(
                        all_parinc, all_flow, all_stock,
                        moment_spec=moment_spec, q=all_q,
                    )
                )
                integrated_by_draw.append(integrated)
                integrated_diagnostic_by_draw.append(integrated_new_share)
            else:
                moments, new_share, _, _ = parental_income_distribution_moments(
                    all_parinc, all_flow, all_stock,
                    moment_spec=moment_spec, q=all_q,
                )
            simulated_by_draw.append(moments)
            diagnostic_by_draw.append(new_share)

    simulated = np.nanmean(np.asarray(simulated_by_draw), axis=0)
    sim_new_share = np.nanmean(np.asarray(diagnostic_by_draw), axis=0)
    valid = np.isfinite(data_moments) & np.isfinite(simulated)
    scale = np.maximum(np.abs(data_moments), 1.0e-6)
    # Each parinc group contributes the same four standardized residuals,
    # regardless of its sample size. Moment weights are defined within each
    # parental-income group and do not alter simulated choices.
    moment_weights = np.tile(
        parental_income_moment_weight_pattern(
            moment_spec, primary_moment_weight
        ),
        len(labels),
    )
    standardized_error = (simulated - data_moments) / scale
    loss = float(np.sum(moment_weights[valid] * standardized_error[valid] ** 2))
    if EVAL_COUNTER % 10 == 0:
        if specification == "parental_income_loan_type":
            _print_parental_income_loan_type_fit(
                data_moments, simulated, data_new_share, sim_new_share,
                data_weights, labels, loss, moment_spec, primary_moment_weight,
            )
            integrated_simulated = np.nanmean(np.asarray(integrated_by_draw), axis=0)
            integrated_sim_new_share = np.nanmean(
                np.asarray(integrated_diagnostic_by_draw), axis=0
            )
            print("INTEGRATED OVER LOAN TYPE (REPORTING DIAGNOSTIC; NOT IN LOSS)")
            _print_parental_income_fit(
                integrated_data[0], integrated_simulated,
                integrated_data[1], integrated_sim_new_share,
                integrated_data[2], integrated_data[3], loss,
                moment_spec, primary_moment_weight,
            )
        else:
            _print_parental_income_fit(
                data_moments, simulated, data_new_share, sim_new_share,
                data_weights, labels, loss, moment_spec, primary_moment_weight,
            )
        print("type integration:", type_integration, "moment specification:", moment_spec)
        print("shock means by parinc:", np.round(np.asarray(params)[0:4], 3),
              "common sigma:", round(float(np.asarray(params)[4]), 3))
        if specification == "parental_income_loan_type":
            print(
                "risk aversion low-loan type:",
                np.round(spec["risk_aversion_by_loan_type"][0], 4),
            )
            print(
                "risk aversion high-loan type:",
                np.round(spec["risk_aversion_by_loan_type"][1], 4),
            )
            print("debt penalties:", np.round(np.asarray(params)[9:13], 4))
        else:
            print("risk aversion:", np.round(spec["risk_aversion"], 4),
                  "debt penalties:", np.round(np.asarray(params)[9:13], 4))
        print(
            "common pre-choice-resource slope ($ shock per $10,000 resources):",
            round(float(spec["budget_resource_slope"][0]), 4),
        )
        if float(spec["sigma_e"][0]) <= 1.000001:
            print("WARNING: common budget-shock sigma is at its lower bound (1.0).")
    return loss


def minimize_distance_education_cell_differential(
    differential, fixed_common, *objective_args,
):
    """Estimate only loan-type differences after fixing the common benchmark."""
    full = np.concatenate((np.asarray(fixed_common, dtype=np.float64), differential))
    return minimize_distance_education_cell(full, *objective_args)


def fit_education_cell(
    packs, education=2, program_year=1, shock_heterogeneity="homogeneous",
    draws=20, n_sample=None, maxiter=DEFAULT_EDUCATION_CELL_MAXITER,
    seed=12345, initial=None,
    fixed_common=None, specification="parental_income_basic",
    type_integration="sampled", moment_spec="flow_plus_stock",
    primary_moment_weight=DEFAULT_PRIMARY_MOMENT_WEIGHT,
    resource_mode="simulated",
):
    """Estimate a single program-year cell without changing the dynamic solver."""
    if not packs:
        raise ValueError("No observations were found for the requested education cell.")
    if specification not in EDUCATION_CELL_SPECIFICATIONS:
        raise ValueError(f"specification must be one of {EDUCATION_CELL_SPECIFICATIONS}.")
    if type_integration not in TYPE_INTEGRATION_MODES:
        raise ValueError(f"type_integration must be one of {TYPE_INTEGRATION_MODES}.")
    if moment_spec not in PARENTAL_INCOME_MOMENT_SPECS:
        raise ValueError(f"moment_spec must be one of {PARENTAL_INCOME_MOMENT_SPECS}.")
    if not np.isfinite(primary_moment_weight) or primary_moment_weight <= 0.0:
        raise ValueError("primary_moment_weight must be positive and finite.")
    if resource_mode not in EDUCATION_CELL_RESOURCE_MODES:
        raise ValueError(f"resource_mode must be one of {EDUCATION_CELL_RESOURCE_MODES}.")
    if specification == "joint_type" and type_integration != "exact":
        raise ValueError("The retained joint_type specification requires exact integration.")
    data_moments, data_new_share, data_weights, labels = _pooled_observed_cell_moments(
        packs, specification=specification, moment_spec=moment_spec
    )
    integrated_data = None
    if specification == "parental_income_loan_type":
        integrated_data = _pooled_observed_cell_moments(
            packs, specification="parental_income_basic", moment_spec=moment_spec
        )
    if specification in (
        "parental_income_basic", "parental_income_loan_type",
    ) and type_integration == "sampled":
        packs = assign_persistent_joint_types(packs, seed=seed)
    rng = np.random.default_rng(seed)
    sampled = []
    for pack in packs:
        if n_sample is None or len(pack["x1"]) <= n_sample:
            sampled.append(pack)
        else:
            sampled.append(_subset_cell_pack(pack, rng.choice(len(pack["x1"]), n_sample, replace=True)))
    sample_by_period = prepare_education_cell_crns(
        sampled, draws=draws, seed=seed, type_integration=type_integration,
        resource_mode=resource_mode,
    )
    print(f"[current resources] {resource_mode}")
    if type_integration == "sampled":
        pooled_n = sum(len(pack["x1"]) for pack in sample_by_period.values())
        print(
            f"[parallel debt solver] {get_num_threads()} Numba threads; "
            f"{draws * pooled_n:,} draw-individual problems per objective call"
        )

    cell_code = bs.budget_education_cell_code(education, program_year)
    if specification in ("parental_income_basic", "parental_income_loan_type"):
        if shock_heterogeneity != "homogeneous":
            raise ValueError(
                "Parental-income specifications manage loan-type heterogeneity internally."
            )
        if fixed_common is not None:
            raise ValueError("fixed_common is only available for the joint_type specification.")
        if initial is None:
            initial = np.array(
                [5000.0] * 4 + [100.0] + [2.0] * 4 + [-2.0] * 4 + [0.0],
                dtype=np.float64,
            )
            if specification == "parental_income_loan_type":
                initial = np.concatenate((initial, initial[5:9].copy()))
        initial = np.asarray(initial, dtype=np.float64).reshape(-1)
        if initial.size == bs.LEGACY_PARENTAL_INCOME_ESTIMATION_VECTOR_SIZE:
            print("[initial] Appending a zero budget-resource slope to legacy 13 parameters.")
            initial = np.concatenate((initial, np.zeros(1, dtype=np.float64)))
        if specification == "parental_income_loan_type":
            if initial.size == bs.PARENTAL_INCOME_ESTIMATION_VECTOR_SIZE:
                print(
                    "[initial] Appending four high-loan-type risk-aversion "
                    "levels initialized at the low-type levels."
                )
                initial = np.concatenate((initial, initial[5:9].copy()))
            bs.unpack_parental_income_loan_type_estimation_vector(
                initial, [cell_code], index_kind="education_cell"
            )
        else:
            bs.unpack_parental_income_estimation_vector(
                initial, [cell_code], index_kind="education_cell"
            )
        bounds = (
            [(-50000.0, 50000.0)] * 4 + [(1.0, 50000.0)]
            + [(0.1001, 2.9999)] * 4 + [(-1.0e6, 0.0)] * 4
            + [(-50000.0, 50000.0)]
        )
        if specification == "parental_income_loan_type":
            bounds += [(0.1001, 2.9999)] * 4
        args = (
            data_moments, data_new_share, data_weights, labels, sample_by_period,
            cell_code, education, program_year, type_integration, moment_spec,
            primary_moment_weight, specification, integrated_data,
        )
        # SciPy gives a parameter initialized at zero an extremely small
        # default simplex step (0.00025). That would barely explore a slope
        # measured in dollars per $10,000. Preserve the usual Nelder-Mead
        # simplex for every existing parameter, but give the new slope a
        # modest $100 initial direction when it starts at the null value.
        initial_simplex = np.tile(initial, (initial.size + 1, 1))
        for parameter_index, value in enumerate(initial):
            initial_simplex[parameter_index + 1, parameter_index] = (
                1.05 * value if value != 0.0 else 0.00025
            )
        for special_index in range(13, initial.size):
            if initial[special_index] == 0.0:
                initial_simplex[special_index + 1, special_index] = 100.0
        result = minimize(
            minimize_distance_education_cell_parental_income,
            initial,
            args=args,
            method="Nelder-Mead",
            bounds=bounds,
            options={
                "maxiter": int(maxiter), "disp": True,
                "initial_simplex": initial_simplex,
            },
        )
        return result, (data_moments, data_new_share, data_weights, labels)

    if initial is None:
        parts = [
            np.tile(np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 1),
            np.array([1000.0]), np.full(4, 1.25), np.array([-1.0, -1.0, -1.0, -1.0]),
        ]
        if shock_heterogeneity in ("mean", "both"):
            parts.append(np.zeros(1))
        if shock_heterogeneity in ("variance", "both"):
            parts.append(np.zeros(1))
        initial = np.concatenate(parts)
    initial = np.asarray(initial, dtype=np.float64)
    bs.unpack_estimation_vector(
        initial, [cell_code], loan_heterogeneity=shock_heterogeneity,
        index_kind="education_cell",
    )
    bounds = (
        [(-50000.0, 50000.0)] * 7 + [(1.0, 50000.0)]
        + [(0.1001, 2.9999)] * 4 + [(-1.0e6, 0.0)] * 4
    )
    if shock_heterogeneity in ("mean", "both"):
        bounds += [(-50000.0, 50000.0)]
    if shock_heterogeneity in ("variance", "both"):
        bounds += [(-3.0, 3.0)]
    args = (
        data_moments, data_new_share, data_weights, labels, sample_by_period,
        cell_code, education, program_year, shock_heterogeneity,
    )
    if fixed_common is None:
        result = minimize(
            minimize_distance_education_cell, initial, args=args, method="Nelder-Mead",
            bounds=bounds, options={"maxiter": int(maxiter), "disp": True},
        )
    else:
        fixed_common = np.asarray(fixed_common, dtype=np.float64)
        if fixed_common.shape != (16,):
            raise ValueError("fixed_common must be the 16-parameter homogeneous estimate.")
        if shock_heterogeneity == "homogeneous":
            raise ValueError("fixed_common is only meaningful for a heterogeneous stage.")
        n_differential = int(shock_heterogeneity in ("mean", "both"))
        n_differential += int(shock_heterogeneity in ("variance", "both"))
        differential0 = initial[-n_differential:]
        differential_bounds = bounds[-n_differential:]
        result = minimize(
            minimize_distance_education_cell_differential,
            differential0,
            args=(fixed_common, *args),
            method="Nelder-Mead",
            bounds=differential_bounds,
            options={"maxiter": int(maxiter), "disp": True},
        )
        result.x_differential = result.x.copy()
        result.x = np.concatenate((fixed_common, result.x))
    return result, (data_moments, data_new_share, data_weights, labels)


class _DfolsShimResult:
    """SciPy-like result for the DFO-LS branch.

    Downstream code stores extra ``smm_*`` attributes on the result and reads
    ``.x`` (saved vector) and ``.fun``; this plain object supports both.
    """

    def __init__(self, x, fun, nfev, success, message):
        self.x = np.asarray(x, dtype=np.float64).copy()
        self.fun = float(fun)
        self.nfev = int(nfev)
        self.success = bool(success)
        self.message = str(message)


def solve_dfols_least_squares(
    residual_fun, x0, bounds, maxfun, restarts=0, seed=12345,
    rhobeg=DFOLS_COLD_RHOBEG, rhoend=1.0e-8,
):
    """Bounded DFO-LS solve with optional reproducible perturbation restarts.

    ``residual_fun`` maps a parameter vector to the stacked weighted residual
    vector whose sum of squares is the SMM loss. Each restart perturbs the
    best point so far by uniform noise of ``DFOLS_RESTART_PERTURBATION_SHARE``
    of each bound range (clipped to the bounds), drawn from
    ``np.random.default_rng(seed)`` so repeated calls are identical. Returns
    a ``_DfolsShimResult`` for the best solve; ``nfev`` counts evaluations
    across all solves.
    """
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    lower = np.asarray([pair[0] for pair in bounds], dtype=np.float64)
    upper = np.asarray([pair[1] for pair in bounds], dtype=np.float64)
    if lower.size != x0.size or upper.size != x0.size:
        raise ValueError("bounds must provide one (low, high) pair per parameter.")
    if int(maxfun) <= 0:
        raise ValueError("maxfun must be positive.")
    if int(restarts) < 0:
        raise ValueError("restarts must be non-negative.")
    try:
        import dfols
    except ImportError as error:
        raise ImportError(
            "optimizer='dfols' requires the DFO-LS package (import name "
            "'dfols'). Install it with: pip install DFO-LS"
        ) from error

    def run_solve(start, label):
        start = np.clip(np.asarray(start, dtype=np.float64), lower, upper)
        solution = dfols.solve(
            residual_fun,
            start,
            bounds=(lower, upper),
            maxfun=int(maxfun),
            rhobeg=float(rhobeg),
            rhoend=float(rhoend),
            objfun_has_noise=True,
            scaling_within_bounds=True,
        )
        # DFO-LS >= 1.6 exposes the objective as .obj; older versions as .f.
        objective = float(getattr(solution, "obj", getattr(solution, "f", np.nan)))
        print(
            f"[dfols {label} finished] loss={objective:.6f}; "
            f"evaluations={int(solution.nf)}"
        )
        return solution, objective

    best, best_objective = run_solve(x0, "solve")
    total_evaluations = int(best.nf)
    rng = np.random.default_rng(seed)
    for restart_index in range(int(restarts)):
        perturbed = best.x + rng.uniform(
            -1.0, 1.0, size=x0.size
        ) * DFOLS_RESTART_PERTURBATION_SHARE * (upper - lower)
        candidate, candidate_objective = run_solve(
            perturbed, f"restart {restart_index + 1}/{int(restarts)}"
        )
        total_evaluations += int(candidate.nf)
        if candidate_objective < best_objective:
            best, best_objective = candidate, candidate_objective
    return _DfolsShimResult(
        x=best.x,
        fun=best_objective,
        nfev=total_evaluations,
        success=(int(best.flag) == int(best.EXIT_SUCCESS)),
        message=str(best.msg),
    )


def fit_education_cells(
    packs_by_program_year, education=2, program_years=(1, 2, 3, 4),
    draws=20, n_sample=None, maxiter=DEFAULT_EDUCATION_CELL_MAXITER,
    seed=12345, initial=None, moment_spec="flow_plus_stock",
    primary_moment_weight=DEFAULT_PRIMARY_MOMENT_WEIGHT,
    resource_mode="simulated", optimizer="nelder-mead",
    annealing_maxfun=500, education_cells=None, cell_workers=None,
    cell_numba_threads=1, dfols_maxfun=None, dfols_restarts=0,
):
    """Jointly estimate several education cells with common risk aversion.

    Every selected parental-income moment from every cell enters the loss.
    Under ``flow_plus_stock`` this is 24 moments per education cell. Each cell
    has its own four means, sigma, and resource slope. Four parental-income
    risk-aversion levels and four debt penalties are estimated once and shared.
    """
    if education_cells is None:
        program_years = tuple(dict.fromkeys(
            int(bs.budget_program_year(education, year))
            for year in program_years
        ))
        cells = tuple((int(education), year) for year in program_years)
        packs_by_cell = {
            (int(education), year): packs_by_program_year[year]
            for year in program_years
        }
    else:
        cells = tuple(dict.fromkeys(
            (
                int(educ),
                int(bs.budget_program_year(educ, year)),
            )
            for educ, year in education_cells
        ))
        packs_by_cell = packs_by_program_year
    if len(cells) < 2 or len(set(cells)) != len(cells):
        raise ValueError("Multi-cell estimation requires at least two unique education cells.")
    if any(educ not in (1, 2, 3) or year < 1 for educ, year in cells):
        raise ValueError("Education cells must use education 1, 2, or 3 and a positive year.")
    if moment_spec not in EXTENDED_PARENTAL_INCOME_MOMENT_SPECS:
        raise ValueError(
            f"moment_spec must be one of {EXTENDED_PARENTAL_INCOME_MOMENT_SPECS}."
        )
    if resource_mode not in EDUCATION_CELL_RESOURCE_MODES:
        raise ValueError(f"resource_mode must be one of {EDUCATION_CELL_RESOURCE_MODES}.")
    if not np.isfinite(primary_moment_weight) or primary_moment_weight <= 0.0:
        raise ValueError("primary_moment_weight must be positive and finite.")
    optimizer = str(optimizer).lower()
    if optimizer not in ("nelder-mead", "dual-annealing", "hybrid", "dfols"):
        raise ValueError(
            "optimizer must be nelder-mead, dual-annealing, hybrid, or dfols."
        )
    if int(annealing_maxfun) <= 0:
        raise ValueError("annealing_maxfun must be positive.")
    if dfols_maxfun is not None and int(dfols_maxfun) <= 0:
        raise ValueError("dfols_maxfun must be positive when provided.")
    if int(dfols_restarts) < 0:
        raise ValueError("dfols_restarts must be non-negative.")
    # A caller-provided initial vector (the production restart machinery)
    # marks a warm start; DFO-LS then defaults to a small refinement budget.
    dfols_warm_start = initial is not None
    if int(cell_numba_threads) <= 0:
        raise ValueError("cell_numba_threads must be positive.")
    for cell in cells:
        if not packs_by_cell.get(cell):
            raise ValueError(f"No observations were found for education cell {cell}.")

    # Draw one posterior joint type per unique individual across the complete
    # multi-cell sample, so a person never changes latent type across years.
    combined = [
        pack for cell in cells for pack in packs_by_cell[cell]
    ]
    combined = assign_persistent_joint_types(combined, seed=seed)
    assigned_by_code = {
        bs.budget_education_cell_code(*cell): [] for cell in cells
    }
    for pack in combined:
        assigned_by_code[int(pack["cell_code"])].append(pack)

    rng = np.random.default_rng(seed)
    contexts = []
    data_summaries = {}
    for education_value, year in cells:
        cell_code = bs.budget_education_cell_code(education_value, year)
        packs = assigned_by_code[cell_code]
        summary = _pooled_observed_cell_moments(
            packs, specification="parental_income_basic", moment_spec=moment_spec
        )
        data_summaries[(education_value, year)] = summary
        sampled = []
        for pack in packs:
            if n_sample is None or len(pack["x1"]) <= n_sample:
                sampled.append(pack)
            else:
                idx = rng.choice(len(pack["x1"]), n_sample, replace=True)
                sampled.append(_subset_cell_pack(pack, idx))
        sample_by_period = prepare_education_cell_crns(
            sampled, draws=draws, seed=seed + 100000 * cell_code,
            type_integration="sampled", resource_mode=resource_mode,
        )
        observed_n = sum(len(pack["x1"]) for pack in packs)
        simulated_n = sum(
            len(pack["x1"]) for pack in sample_by_period.values()
        )
        data_moments, data_new_share, data_weights, labels = summary
        contexts.append({
            "cell_code": cell_code,
            "education": education_value,
            "program_year": year,
            "data_moments": data_moments,
            "data_new_share": data_new_share,
            "data_weights": data_weights,
            "labels": labels,
            "sample_by_period": sample_by_period,
            "cell_observations": int(observed_n),
        })
        year_label = bs.budget_program_year_label(education_value, year)
        print(
            f"[education {education_value}, program year {year_label}] "
            f"{observed_n:,} observed enrolled observations; "
            f"{simulated_n:,} simulated observations; "
            f"{draws * simulated_n:,} draw-individual problems per objective call"
        )
    mean_cell_observations = float(np.mean([
        context["cell_observations"] for context in contexts
    ]))
    for context in contexts:
        context["cell_weight"] = (
            context["cell_observations"] / mean_cell_observations
        )
    print("[education-cell loss weights: N_cell / mean(N)]")
    for context in contexts:
        year_label = bs.budget_program_year_label(
            context["education"], context["program_year"]
        )
        print(
            f"  education={context['education']} year={year_label}: "
            f"N={context['cell_observations']:,}, "
            f"weight={context['cell_weight']:.4f}"
        )
    print(f"[current resources] {resource_mode}")
    print(f"[parallel debt solver] {get_num_threads()} Numba threads")

    n_cells = len(cells)
    block_size = bs.PARENTAL_INCOME_MULTICELL_PARAMETERS_PER_CELL
    legacy_expected = bs.estimation_vector_size_multicell(n_cells)
    expected = bs.estimation_vector_size_multicell(
        n_cells, include_new_borrowing=ESTIMATE_NEW_BORROWING_COST
    )
    if initial is None:
        cell_block = np.array(
            [5000.0] * 4 + [100.0] + [0.0], dtype=np.float64
        )
        initial = np.concatenate(
            (np.tile(cell_block, n_cells), [2.0] * 4, [-2.0] * 4)
        )
    else:
        initial = np.asarray(initial, dtype=np.float64).reshape(-1)
        if initial.size == bs.PARENTAL_INCOME_ESTIMATION_VECTOR_SIZE:
            cell_block = np.concatenate((initial[0:5], initial[13:14]))
            initial = np.concatenate(
                (np.tile(cell_block, n_cells), initial[5:9], initial[9:13])
            )
            print("[initial] Replicated one-cell parameters across selected program years.")
        elif initial.size == n_cells * 10 + 4:
            old_blocks = initial[:n_cells * 10].reshape(n_cells, 10)
            cell_blocks = np.column_stack((old_blocks[:, 0:5], old_blocks[:, 9]))
            shared_debt = np.mean(old_blocks[:, 5:9], axis=0)
            initial = np.concatenate(
                (cell_blocks.reshape(-1), initial[-4:], shared_debt)
            )
            print(
                "[initial] Converted the previous multi-cell vector and "
                "initialized shared debt penalties at their across-year means."
            )
    if ESTIMATE_NEW_BORROWING_COST and initial.size == legacy_expected:
        initial = np.concatenate(
            (initial, np.zeros(bs.N_NEW_BORROWING_PARAMETERS))
        )
        print(
            "[initial] Appending three zero new-borrowing costs "
            "(kappa0 low/high, kappa1) to the legacy "
            f"{legacy_expected}-parameter vector."
        )
    if initial.size != expected:
        raise ValueError(
            f"Multi-cell initial vector has {initial.size} entries; expected {expected} "
            f"({block_size} per cell plus four shared risk-aversion and four "
            "shared debt-penalty levels"
            + (
                " and three shared new-borrowing costs)."
                if ESTIMATE_NEW_BORROWING_COST else ")."
            )
        )
    cell_codes = [bs.budget_education_cell_code(*cell) for cell in cells]
    bs.unpack_parental_income_multicell_estimation_vector(
        initial, cell_codes, index_kind="education_cell"
    )
    bounds = []
    for _ in cells:
        bounds.extend(
            [(-50000.0, 50000.0)] * 4 + [(1.0, 50000.0)]
            + [(-50000.0, 50000.0)]
        )
    bounds.extend([(0.1001, 2.9999)] * 4)
    bounds.extend([(-1.0e6, 0.0)] * 4)
    if ESTIMATE_NEW_BORROWING_COST:
        # kappas are flow-utility units; the CRRA flow utility here is
        # O(0.1-2), so a few utils already dominate the choice.
        bounds.extend(
            [NEW_BORROWING_COST_BOUNDS] * bs.N_NEW_BORROWING_PARAMETERS
        )

    def make_initial_simplex(center):
        center = np.asarray(center, dtype=np.float64)
        simplex = np.tile(center, (center.size + 1, 1))
        for parameter_index, value in enumerate(center):
            simplex[parameter_index + 1, parameter_index] = (
                1.05 * value if value != 0.0 else 0.00025
            )
        for cell_index in range(n_cells):
            slope_index = cell_index * block_size + 5
            if center[slope_index] == 0.0:
                simplex[slope_index + 1, slope_index] = 100.0
        if ESTIMATE_NEW_BORROWING_COST:
            # A zero-initialized kappa needs a negative in-bounds step of a
            # meaningful flow-utility size, not the tiny default 0.00025.
            for kappa_index in range(
                center.size - bs.N_NEW_BORROWING_PARAMETERS, center.size
            ):
                if center[kappa_index] == 0.0:
                    simplex[kappa_index + 1, kappa_index] = -0.25
        return simplex

    if cell_workers is None:
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cpus = os.cpu_count() or 1
        cell_workers = min(n_cells, available_cpus)
    cell_workers = max(1, min(int(cell_workers), n_cells))
    cell_pool = None
    if cell_workers > 1 and "fork" in mp.get_all_start_methods():
        fork_context = mp.get_context("fork")
        cell_pool = fork_context.Pool(
            processes=cell_workers,
            initializer=_initialize_cell_smm_worker,
            initargs=(
                contexts, moment_spec, primary_moment_weight,
                int(cell_numba_threads),
            ),
        )
        print(
            f"[parallel SMM cells] {cell_workers} persistent processes; "
            f"{int(cell_numba_threads)} Numba thread(s) per process"
        )
    else:
        print("[parallel SMM cells] disabled; using serial cell evaluation")

    objective_args = (
        contexts, moment_spec, primary_moment_weight, cell_pool,
    )
    print(f"[multi-cell optimizer] {optimizer}")
    try:
        if optimizer == "dfols":
            def dfols_residual_objective(params):
                return minimize_distance_education_cells_parental_income(
                    params, contexts, moment_spec, primary_moment_weight,
                    cell_pool, return_residuals=True,
                )

            if dfols_maxfun is not None:
                dfols_budget = int(dfols_maxfun)
            elif dfols_warm_start:
                dfols_budget = int(DFOLS_WARM_START_MAXFUN)
            else:
                dfols_budget = int(min(
                    DFOLS_COLD_MAXFUN_PER_PARAMETER * initial.size,
                    DFOLS_MAXFUN_CAP,
                ))
            dfols_rhobeg = (
                DFOLS_WARM_RHOBEG if dfols_warm_start else DFOLS_COLD_RHOBEG
            )
            print(
                f"[dfols] {'warm' if dfols_warm_start else 'cold'} start; "
                f"maxfun={dfols_budget} per solve; "
                f"restarts={int(dfols_restarts)}; rhobeg={dfols_rhobeg:g} "
                "(scaled bounds); rhoend=1e-08"
            )
            result = solve_dfols_least_squares(
                dfols_residual_objective,
                initial,
                bounds,
                maxfun=dfols_budget,
                restarts=int(dfols_restarts),
                seed=int(seed),
                rhobeg=dfols_rhobeg,
            )
            result.dfols_maxfun = int(dfols_budget)
            result.dfols_restarts = int(dfols_restarts)
            result.dfols_warm_start = bool(dfols_warm_start)
        annealing_result = None
        if optimizer in ("dual-annealing", "hybrid"):
            annealing_result = dual_annealing(
                minimize_distance_education_cells_parental_income,
                bounds=bounds,
                args=objective_args,
                x0=initial,
                maxiter=int(maxiter),
                maxfun=int(annealing_maxfun),
                seed=int(seed),
                no_local_search=True,
            )
            print(
                f"[annealing finished] loss={annealing_result.fun:.6f}; "
                f"evaluations={annealing_result.nfev}"
            )

        if optimizer == "dual-annealing":
            result = annealing_result
        elif optimizer != "dfols":
            local_start = (
                annealing_result.x if optimizer == "hybrid" else initial
            )
            result = minimize(
                minimize_distance_education_cells_parental_income,
                local_start,
                args=objective_args,
                method="Nelder-Mead",
                bounds=bounds,
                options={
                    "maxiter": int(maxiter), "disp": True,
                    "initial_simplex": make_initial_simplex(local_start),
                },
            )
            if annealing_result is not None:
                result.annealing_x = annealing_result.x.copy()
                result.annealing_fun = float(annealing_result.fun)
                result.annealing_nfev = int(annealing_result.nfev)
                result.total_nfev = int(annealing_result.nfev + result.nfev)
    finally:
        if cell_pool is not None:
            cell_pool.close()
            cell_pool.join()
    result.smm_optimizer = optimizer
    result.smm_cell_workers = int(cell_workers)
    result.smm_cell_numba_threads = int(cell_numba_threads)
    result.smm_cell_observations = np.asarray(
        [context["cell_observations"] for context in contexts], dtype=np.int64
    )
    result.smm_cell_weights = np.asarray(
        [context["cell_weight"] for context in contexts], dtype=np.float64
    )
    return result, data_summaries

# ==============================================================================
# Main entry point

def estimate_budget_shock():
    interp_dict = get_interp_dict_cached(force_rebuild=False)

    periods_to_fit = list(range(1, T))

    data_by_period = {}
    for p in periods_to_fit:
        print(f"\n[load fundamentals] period={p}")
        data_by_period[p] = load_fundamentals(p, interp_dict, clear_data=False)

    best_x, best_fun = fit_budget_shock_multi(
        data_by_period=data_by_period,
        periods=periods_to_fit,
        max_outer=20,
        s=20,
        n_sample=100000,
    )

    save_budgetshock_estimates(best_x=best_x, periods=periods_to_fit)
    return best_x, best_fun


def estimate_budget_shock_education_cell(
    education=2,
    program_year=1,
    shock_heterogeneity="homogeneous",
    specification="parental_income_basic",
    type_integration="sampled",
    moment_spec="flow_plus_stock",
    primary_moment_weight=DEFAULT_PRIMARY_MOMENT_WEIGHT,
    resource_mode="simulated",
    draws=20,
    n_sample=None,
    maxiter=DEFAULT_EDUCATION_CELL_MAXITER,
    seed=12345,
    save=False,
    initial=None,
    fixed_common=None,
    ccp_workers=DEFAULT_CCP_WORKERS,
    ccp_cache_mode="reuse",
):
    """Run the dynamic estimator as a one-program-year pre-test.

    The default is first-year four-year college. Observations are pooled across
    model periods, but each retains its period-specific CCP and terminal value.
    Set ``save=True`` only after validating the homogeneous benchmark.
    """
    program_year = int(bs.budget_program_year(education, program_year))
    interp_dict = get_interp_dict_cached(force_rebuild=False)
    packs = []
    for period in range(1, T):
        print(f"[load education cell] model period={period}")
        pack = load_education_cell(
            period, interp_dict, education=education, program_year=program_year,
            ccp_workers=ccp_workers, ccp_cache_mode=ccp_cache_mode,
        )
        if len(pack["x1"]):
            print(f"  retained {len(pack['x1'])} enrolled observations")
            packs.append(pack)
    result, data_summary = fit_education_cell(
        packs, education=education, program_year=program_year,
        shock_heterogeneity=shock_heterogeneity, draws=draws,
        n_sample=n_sample, maxiter=maxiter, seed=seed, initial=initial,
        fixed_common=fixed_common, specification=specification,
        type_integration=type_integration, moment_spec=moment_spec,
        primary_moment_weight=primary_moment_weight,
        resource_mode=resource_mode,
    )
    if save:
        cell_code = bs.budget_education_cell_code(education, program_year)
        if specification in ("parental_income_basic", "parental_income_loan_type"):
            prefix = (
                f"budgetshock_educ{education}_year{program_year}_{specification}_"
                f"{type_integration}_{moment_spec}"
                f"{'_observed_resources' if resource_mode == 'observed' else ''}"
            )
        else:
            prefix = (
                f"budgetshock_educ{education}_year{program_year}_{shock_heterogeneity}"
            )
        save_budgetshock_estimates(
            result.x, [cell_code], filename_prefix=prefix,
            loan_heterogeneity=shock_heterogeneity,
            index_kind="education_cell",
            estimation_parameterization=specification,
            estimation_metadata={
                "smm_type_integration": type_integration,
                "smm_moment_spec": moment_spec,
                "smm_primary_moment_weight": float(primary_moment_weight),
                "smm_within_parinc_moment_weights": (
                    parental_income_moment_weight_pattern(
                        moment_spec, primary_moment_weight
                    )
                ),
                "smm_resource_mode": resource_mode,
                "smm_draws": int(draws),
                "smm_seed": int(seed),
                "smm_n_sample": None if n_sample is None else int(n_sample),
            },
        )
    return result, data_summary


def estimate_budget_shock_education_cells(
    education=2,
    program_years=(1, 2, 3, 4),
    moment_spec="flow_plus_stock",
    primary_moment_weight=DEFAULT_PRIMARY_MOMENT_WEIGHT,
    resource_mode="simulated",
    draws=20,
    n_sample=None,
    maxiter=DEFAULT_EDUCATION_CELL_MAXITER,
    seed=12345,
    save=False,
    initial=None,
    optimizer="nelder-mead",
    annealing_maxfun=500,
    cell_workers=None,
    cell_numba_threads=1,
    ccp_workers=DEFAULT_CCP_WORKERS,
    ccp_cache_mode="reuse",
    dfols_maxfun=None,
    dfols_restarts=0,
):
    """Estimate multiple program-year cells with shared risk aversion."""
    program_years = tuple(dict.fromkeys(
        int(bs.budget_program_year(education, year))
        for year in program_years
    ))
    interp_dict = get_interp_dict_cached(force_rebuild=False)
    packs_by_program_year = {year: [] for year in program_years}
    for year in program_years:
        print(f"\n[load education cell] education={education}, program year={year}")
        for period in range(1, T):
            print(f"  model period={period}")
            pack = load_education_cell(
                period, interp_dict, education=education, program_year=year,
                ccp_workers=ccp_workers, ccp_cache_mode=ccp_cache_mode,
            )
            if len(pack["x1"]):
                print(f"    retained {len(pack['x1'])} enrolled observations")
                packs_by_program_year[year].append(pack)
    result, data_summary = fit_education_cells(
        packs_by_program_year,
        education=education,
        program_years=program_years,
        draws=draws,
        n_sample=n_sample,
        maxiter=maxiter,
        seed=seed,
        initial=initial,
        moment_spec=moment_spec,
        primary_moment_weight=primary_moment_weight,
        resource_mode=resource_mode,
        optimizer=optimizer,
        annealing_maxfun=annealing_maxfun,
        cell_workers=cell_workers,
        cell_numba_threads=cell_numba_threads,
        dfols_maxfun=dfols_maxfun,
        dfols_restarts=dfols_restarts,
    )
    if save:
        cell_codes = [
            bs.budget_education_cell_code(education, year)
            for year in program_years
        ]
        years_label = "-".join(str(year) for year in program_years)
        prefix = (
            f"budgetshock_educ{education}_years{years_label}_"
            f"parental_income_basic_multicell_shared_risk_debt_sampled_{moment_spec}"
            f"{'_observed_resources' if resource_mode == 'observed' else ''}"
        )
        save_budgetshock_estimates(
            result.x,
            cell_codes,
            filename_prefix=prefix,
            loan_heterogeneity="homogeneous",
            index_kind="education_cell",
            estimation_parameterization=(
                "parental_income_basic_multicell_shared_risk_debt"
            ),
            estimation_metadata={
                "smm_type_integration": "sampled",
                "smm_moment_spec": moment_spec,
                "smm_primary_moment_weight": float(primary_moment_weight),
                "smm_within_parinc_moment_weights": (
                    parental_income_moment_weight_pattern(
                        moment_spec, primary_moment_weight
                    )
                ),
                "smm_resource_mode": resource_mode,
                "smm_draws": int(draws),
                "smm_seed": int(seed),
                "smm_n_sample": None if n_sample is None else int(n_sample),
                "smm_program_years": np.asarray(program_years, dtype=np.int64),
                "smm_risk_aversion_shared_across_cells": True,
                "smm_debt_penalties_shared_across_cells": True,
                "smm_debt_penalty_timing": bs.DEBT_PENALTY_TIMING,
                "smm_optimizer": optimizer,
                "smm_annealing_maxfun": int(annealing_maxfun),
                "smm_cell_workers": int(result.smm_cell_workers),
                "smm_cell_numba_threads": int(result.smm_cell_numba_threads),
                "smm_cell_weighting": "enrolled_observations_over_mean_cell_size",
                "smm_cell_observations": result.smm_cell_observations.copy(),
                "smm_cell_weights": result.smm_cell_weights.copy(),
            },
        )
    return result, data_summary


def discover_observed_education_cells():
    """Return the fixed grouped education-year budget supports in model data."""
    cells = set()
    for period in range(1, T):
        _, state, _, _, choices, _ = load_data_superfeasible(
            period, return_income=True
        )
        education_choice = choices[:, 1].astype(np.int64)
        for education in (1, 2, 3):
            selected = education_choice == education
            if not np.any(selected):
                continue
            codes = np.unique(
                bs.budget_education_cell_from_state(
                    state[selected], education
                )
            )
            cells.update(
                (education, int(code - 100 * education)) for code in codes
            )
    expected = set(bs.BUDGET_EDUCATION_CELLS)
    missing = expected.difference(cells)
    if missing:
        raise ValueError(
            "The data do not contain all required grouped budget cells: "
            f"{tuple(sorted(missing))}"
        )
    return bs.BUDGET_EDUCATION_CELLS


def estimate_budget_shock_all_education(
    draws=100,
    n_sample=None,
    maxiter=1000,
    seed=12345,
    optimizer="hybrid",
    annealing_maxfun=500,
    moment_spec="flow_plus_stock",
    primary_moment_weight=DEFAULT_PRIMARY_MOMENT_WEIGHT,
    resource_mode="simulated",
    initial=None,
    restart=True,
    ccp_workers=DEFAULT_CCP_WORKERS,
    ccp_cache_mode="off",
    cell_workers=None,
    cell_numba_threads=1,
    dfols_maxfun=None,
    dfols_restarts=0,
):
    """Production SMM over every observed 2y, 4y, and graduate cell."""
    cells = discover_observed_education_cells()
    if not cells:
        raise ValueError("No observed education cells were found.")
    print("[production education cells]", cells)
    interp_dict = get_interp_dict_cached(force_rebuild=False)
    packs_by_cell = {cell: [] for cell in cells}
    for education, year in cells:
        print(f"\n[load production cell] education={education}, program year={year}")
        for period in range(1, T):
            pack = load_education_cell(
                period, interp_dict, education=education, program_year=year,
                ccp_workers=ccp_workers, ccp_cache_mode=ccp_cache_mode,
            )
            if len(pack["x1"]):
                print(
                    f"  model period={period}: retained "
                    f"{len(pack['x1'])} enrolled observations"
                )
                packs_by_cell[(education, year)].append(pack)

    cell_codes = [bs.budget_education_cell_code(*cell) for cell in cells]
    legacy_vector_size = bs.estimation_vector_size_multicell(len(cells))
    expected_size = bs.estimation_vector_size_multicell(
        len(cells), include_new_borrowing=ESTIMATE_NEW_BORROWING_COST
    )
    if initial is None and restart:
        raw_path = EST("budgetshock_bestx.npy")
        current = bs.load(raise_if_missing=False)
        if (
            current is not None
            and os.path.exists(raw_path)
            and current.get("estimation_parameterization")
            == "parental_income_basic_multicell_shared_risk_debt"
            and np.array_equal(current["periods"], np.asarray(cell_codes))
        ):
            candidate = np.asarray(np.load(raw_path), dtype=np.float64).reshape(-1)
            if candidate.size == expected_size:
                initial = candidate
                print(f"[restart] Using production estimate {raw_path}")
            elif (
                ESTIMATE_NEW_BORROWING_COST
                and candidate.size == legacy_vector_size
            ):
                # A saved legacy vector restarts the extended specification;
                # fit_education_cells appends three zero kappas so the first
                # evaluation reproduces the saved estimate exactly.
                initial = candidate
                print(
                    f"[restart] Using legacy {legacy_vector_size}-parameter "
                    f"estimate {raw_path}; the three new-borrowing costs "
                    "start at zero."
                )
        if initial is None and current is not None:
            print(
                "[restart] Existing production budget vector is incompatible "
                "with the grouped education-year support. Starting the new "
                f"{expected_size}-parameter specification; canonical files "
                "will be overwritten after successful estimation."
            )

    result, data_summary = fit_education_cells(
        packs_by_cell,
        education_cells=cells,
        draws=draws,
        n_sample=n_sample,
        maxiter=maxiter,
        seed=seed,
        initial=initial,
        moment_spec=moment_spec,
        primary_moment_weight=primary_moment_weight,
        resource_mode=resource_mode,
        optimizer=optimizer,
        annealing_maxfun=annealing_maxfun,
        cell_workers=cell_workers,
        cell_numba_threads=cell_numba_threads,
        dfols_maxfun=dfols_maxfun,
        dfols_restarts=dfols_restarts,
    )
    save_budgetshock_estimates(
        result.x,
        cell_codes,
        filename_prefix="budgetshock",
        loan_heterogeneity="homogeneous",
        index_kind="education_cell",
        estimation_parameterization=(
            "parental_income_basic_multicell_shared_risk_debt"
        ),
        estimation_metadata={
            "smm_scope": "all_observed_education_cells",
            "smm_education_cells": np.asarray(cells, dtype=np.int64),
            "smm_type_integration": "sampled",
            "smm_moment_spec": moment_spec,
            "smm_primary_moment_weight": float(primary_moment_weight),
            "smm_within_parinc_moment_weights": (
                parental_income_moment_weight_pattern(
                    moment_spec, primary_moment_weight
                )
            ),
            "smm_resource_mode": resource_mode,
            "smm_draws": int(draws),
            "smm_seed": int(seed),
            "smm_n_sample": None if n_sample is None else int(n_sample),
            "smm_risk_aversion_shared_across_cells": True,
            "smm_debt_penalties_shared_across_cells": True,
            "smm_debt_penalty_timing": bs.DEBT_PENALTY_TIMING,
            "smm_estimate_new_borrowing_cost": bool(ESTIMATE_NEW_BORROWING_COST),
            "smm_new_borrowing_cost_timing": bs.NEW_BORROWING_COST_TIMING,
            "smm_optimizer": optimizer,
            "smm_annealing_maxfun": int(annealing_maxfun),
            "smm_cell_workers": int(result.smm_cell_workers),
            "smm_cell_numba_threads": int(result.smm_cell_numba_threads),
            "smm_cell_weighting": "enrolled_observations_over_mean_cell_size",
            "smm_cell_observations": result.smm_cell_observations.copy(),
            "smm_cell_weights": result.smm_cell_weights.copy(),
        },
    )
    return result, data_summary


# If running as script:
if __name__ == "__main__":
    debt_range = get_debt_range()
    estimate_budget_shock()
