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

from numba import njit, prange
from scipy.optimize import minimize

from model_em_algorithm import (
    load_data_superfeasible,
    get_feasible,
    get_feasible_pubid,
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

from config import DIR, OUT, EST, ENSURE_DIR
import budget_shock as bs
from latent_types import TYPE_IDS, validate_q, validate_saved_layout

# Canonical roots from config (no chdir anywhere)
PATH_OUT        = DIR["MODEL_OUTPUT"]
PATH_EST        = DIR["MODEL_ESTIMATES"]
PATH_CONT_FINAL = DIR["MODEL_CONTINUATION_FINAL"]

T = 10
r = 0.05
beta = 0.98

debt_range = get_debt_range()
_EM_POSTERIORS = None


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
        )
        _EM_POSTERIORS = validate_q(results["q"])
    return _EM_POSTERIORS

# ==============================================================================
# Savers

def _ensure_est_dir():
    os.makedirs(PATH_EST, exist_ok=True)

def save_budgetshock_estimates(best_x: np.ndarray, periods: list[int], filename_prefix: str = "budgetshock"):
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
    budget_params = bs.unpack_estimation_vector(best_x, periods)
    bs.save(budget_params, raw_vector=best_x, filename_prefix=filename_prefix)

    print(f"[saved] {EST('risk_aversion.npy')}")
    print(f"[saved] {EST('budgetshock_params.npy')}")
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

def get_individual_continuation(i, x1, x2, j, period, em_type):
    """
    Loads sequence EVT (ccp-path) for period+1, taking graduation expectation if needed.
    """
    x1i = x1[i, :].astype("int")
    x2i = x2[i, :].astype("int")
    ji  = j[i, :]

    # graduation possible?
    if ((x2i[1] >= 1) & (ji[1] == 1) & (x2i[4] == 0)) | ((x2i[2] >= 3) & (ji[1] == 2) & (x2i[5] == 0)) | (ji[1] == 3):
        grad_x2    = move_state_grad(x2i, ji, period, grad=1)
        notgrad_x2 = move_state_grad(x2i, ji, period)

        evti = np.load(OUT("evt_ccp", str(period + 1), f"evt_ccp_sequence_t{period+1}_{x1i}_em{em_type}.npz"))
        evt_grad   = evti[f"evt_ccp_sequence_t{period+1}_{x1i}_{grad_x2}"]
        evt_nograd = evti[f"evt_ccp_sequence_t{period+1}_{x1i}_{notgrad_x2}"]

        x1_new = get_x1_new(x1i)
        p_grad = probability_graduation(x1_new, x2i, ji)

        vt = p_grad * evt_grad + (1 - p_grad) * evt_nograd
    else:
        evti = np.load(OUT("evt_ccp", str(period + 1), f"evt_ccp_sequence_t{period+1}_{x1i}_em{em_type}.npz"))
        notgrad_x2 = move_state_grad(x2i, ji, period)
        vt = evti[f"evt_ccp_sequence_t{period+1}_{x1i}_{notgrad_x2}"]

    return vt

def load_ccp_path(x1, state, choices, period, em_type):
    """
    Loads the EVT-ccp path matrix (N x 100) for this period.
    """
    sequence = np.zeros((x1.shape[0], 100), dtype=np.float64)
    if period == 9:
        return sequence

    for i in range(x1.shape[0]):
        sequence[i, :] = get_individual_continuation(i, x1, state, choices, period, em_type)

    return sequence

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

def load_fundamentals(period, interp_dict, clear_data=False):
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
    ccp_path = np.stack(
        [load_ccp_path(x1, state, choices, period, em_type)
         for em_type in TYPE_IDS],
        axis=0,
    )

    # budget w/o shock
    budget = get_budget(income, debt, choices)

    # terminal interpolators
    terminal_data = list(get_continuation(x1, state, choices, period, interp_dict))

    return x1, state, types, debt, debtchoice, choices, ccp_path, budget, terminal_data

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

    Returns:
      x1sample, statesample, debtsample, choicessample, budgetsample, evt, e, terminal_sample
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
    evt = get_ccp_type(ccp_path[:, idx, :], type_index)

    e = np.random.normal(loc=0.0, scale=1.0, size=n_sample).astype(np.float64)

    term_g, term_ng, p_grad = terminal
    terminal_sample = [
        [term_g[i]  for i in idx],
        [term_ng[i] for i in idx],
        p_grad[idx].copy(),
    ]

    return x1sample, statesample, debtsample, choicessample, budgetsample, evt, e, terminal_sample

# ==============================================================================
# Debt rules (caps + monotone + consumption floor)

@njit()
def get_range_number(b, maxdebt):
    value = np.zeros(b.shape[0])
    value2 = np.zeros(b.shape[0])
    for i in range(b.shape[0]):
        diff = (debt_range - b[i]) ** 2
        value[i] = np.argmin(diff)

        diff2 = (debt_range - maxdebt[i]) ** 2
        value2[i] = np.argmin(diff2)
    return value, value2

@njit()
def minimum_debt_maxdebt(u, b, maxdebt):
    b_idx, max_idx = get_range_number(b, maxdebt)
    for i in range(u.shape[0]):
        u[i, :int(b_idx[i])] = -100000000
        u[i, int(max_idx[i]) + 1:] = -100000000
    return u

@njit()
def get_annual_cap_by_stage(educ_choice, twoy_exp, foury_exp):
    cap = 0.0
    if educ_choice == 1:
        if twoy_exp <= 0:
            cap = 8391
        elif twoy_exp == 1:
            cap = 9309
        else:
            cap = 12581
    elif educ_choice == 2:
        if foury_exp <= 0:
            cap = 8391
        elif foury_exp == 1:
            cap = 9309
        else:
            cap = 12581
    elif educ_choice == 3:
        cap = 23222
    return cap

@njit()
def get_lifetime_cap_by_stage(educ_choice):
    if educ_choice == 3:
        return 150000
    else:
        return 70786

@njit()
def get_debt_rules(c, vjt, previousdebt, state, choices):
    vjt[c < 2000] = -100000000

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
    c_floor=2000.0,
    fallback_idx=99,
):
    """
    v = u(c) + ccp_path + beta_term * terminal
    debtpen applies when debt_grid[j] > 0
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

            v = u + ccp_path_row[i, j] + beta_term * terminal_row[i, j]
            if v > best_v:
                best_v = v
                best_j = j

        out[i] = hi if not found_feasible else best_j

    return out

# ==============================================================================
# Precompute bounds indices (speed)

@njit(parallel=True, fastmath=True)
def precompute_bounds_indices(previousdebt, state, choices):
    n = previousdebt.shape[0]
    maxdebt = np.empty(n, dtype=np.float64)

    for i in prange(n):
        educ_choice = int(choices[i, 1])
        twoy_exp = int(state[i, 1])
        foury_exp = int(state[i, 2])

        annual_cap = get_annual_cap_by_stage(educ_choice, twoy_exp, foury_exp)
        lifetime_cap = get_lifetime_cap_by_stage(educ_choice)

        m = previousdebt[i] * (1.0 + r) + annual_cap
        if m > lifetime_cap:
            m = lifetime_cap
        maxdebt[i] = m

    b_idx, max_idx = get_range_number(previousdebt, maxdebt)
    return b_idx.astype(np.int64), max_idx.astype(np.int64)

# ==============================================================================
# Objective

EVAL_COUNTER = 0

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
        debtpen_i = bs.debt_penalty(spec, x1).astype(np.float64)
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
                c_floor=2000.0,
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

# ==============================================================================
# Multi-period estimator

def fit_budget_shock_multi(data_by_period, periods, max_outer=20, s=20, n_sample=10000):
    """
    params = [ mu_block_per_period (7*P), sigma_e_per_period (P), ra_levels (4), debt_pen_parinc (4) ]
    where debt_pen_parinc = [dp0, dp2, dp3, dp4] (constant + deviations).
    """
    P = len(periods)

    # Build data moments ONCE
    mom_data_list = []
    for p in periods:
        x1, state, types, debt, debtchoice, choices, ccp_path, budget, terminal_data = data_by_period[p]
        mom_par = get_moments_by_x1(x1, debtchoice, x1_col=0, nmoments=4).astype(np.float64)  # 16
        mom_ab  = get_mean_share_by_group(x1, debtchoice, x1_col=1).astype(np.float64)        # 8
        mom_data_list.append(np.concatenate([mom_par, mom_ab]))
    moments_data = np.concatenate(mom_data_list).astype(np.float64)  # 24*P

    # Initial guess
    mu0_guess = 100
    mu_block0 = np.tile(np.array([mu0_guess, 100, 100, 100, 100, 100, 100], dtype=np.float64), P)

    sigma_e0 = np.full(P, 100.0, dtype=np.float64)
    ra0 = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)

    dp0_init = -1.0
    dp_parinc0 = np.array([dp0_init, -1.0, -1.0, -1.0], dtype=np.float64)  # [dp0, dp2, dp3, dp4]

    params0 = np.concatenate([mu_block0, sigma_e0, ra0, dp_parinc0]).astype(np.float64)

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

            x1s, states, debts, choices_s, budget_s, evt_ccp_s, _, terminal_s = get_sample(
                x1, state, types, debt, choices, ccp_path, budget, terminal_data, n_sample=n_sample
            )

            rng = np.random.default_rng(12345 + 1000*p + it)
            Zp = rng.standard_normal((s, x1s.shape[0])).astype(np.float64)

            b_idx, max_idx = precompute_bounds_indices(
                debts.astype(np.float64),
                states.astype(np.int64),
                choices_s.astype(np.int64),
            )

            sample_by_period[p] = {
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

        args_obj = (moments_data, sample_by_period, periods)

        print("---- Running minimize(..., method='Nelder-Mead') ----")
        res = minimize(
            minimize_distance_multi,
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


# If running as script:
if __name__ == "__main__":
    debt_range = get_debt_range()
    estimate_budget_shock()
