# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 14:31:48 2025

@author: S.Quintana

Single-period SMM estimation to fit student loans distribution
INCLUDING:
  - Parental-income moments (mean debt>0, Pr(debt>0), std debt>0, p80 debt>0) by par-inc group => 16 moments
  - Ability moments (mean debt>0, Pr(debt>0)) by ability group => 8 moments
  - Parameters:
      * mu block (7):  mu0 + par2/par3/par4 + ab2/ab3/ab4
      * sigma_e (1)
      * ra levels by parental income (4)
      * debt penalty block (7): dp0 + par2/par3/par4 + ab2/ab3/ab4
  => total params = 7 + 1 + 4 + 7 = 19

This file is COPY/PASTE READY.

NOTE: Your previous SyntaxError came from escaped quotes like f\"...\".
This version has NO escaped quotes inside f-strings.
"""

import numpy as np
import joblib
import os

os.chdir(r"C:\Users\S.Quintana\Dropbox\PhD\Projects\Papers\1_financial_constraints\Code\2026_02\2_model")

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

from config import DIR, OUT, EST

pathout = DIR["MODEL_OUTPUT"]
path_estimates = DIR["MODEL_ESTIMATES"]
pathcont = DIR["MODEL_CONTINUATION_FINAL"]

T = 10
r = 0.05
beta = 0.98
debt_range = get_debt_range()


# ==============================================================================
# Cache interpolator dictionary
# ==============================================================================

CACHE_DIR = os.path.join(pathout, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_interp_dict_cached(force_rebuild=False):
    cache_path = os.path.join(CACHE_DIR, "interp_dict.joblib")

    if (not force_rebuild) and os.path.exists(cache_path):
        print(f"[cache] Loading interp_dict from {cache_path}")
        return joblib.load(cache_path)

    print("[cache] Building interp_dict (expensive)...")
    interp_dict, meta_dict, missing, context = build_interpolator_dictionary(
        pathcont=pathcont,
        debt_grid=debt_range,
        fields=8,
        lastschool_horizon=None,
        sex_filters=None,
        race_filters=None,
        verbose=True
    )

    print(f"[cache] Saving interp_dict to {cache_path}")
    joblib.dump(interp_dict, cache_path, compress=3)
    return interp_dict


# ==============================================================================
# Continuation / CCP path helpers
# ==============================================================================

def get_individual_continuation(i, x1, x2, j, period, em_type):
    """
    Loads continuation values along the home-production path (stored as evt_ccp sequences).
    Takes expectation over graduation if the current choice implies possible graduation.
    """
    x1i = x1[i, :].astype(int)
    x2i = x2[i, :].astype(int)
    ji = j[i, :]

    could_grad = (
        ((x2i[1] >= 1) and (ji[1] == 1) and (x2i[4] == 0)) or
        ((x2i[2] >= 3) and (ji[1] == 2) and (x2i[5] == 0)) or
        (ji[1] == 3)
    )

    evti = np.load(
        f"{pathout}/evt_ccp/{period+1}/evt_ccp_sequence_t{period+1}_{x1i}_em{em_type}.npz"
    )

    if could_grad:
        grad_x2 = move_state_grad(x2i, ji, period, grad=1)
        notgrad_x2 = move_state_grad(x2i, ji, period, grad=0)

        evt_grad = evti[f"evt_ccp_sequence_t{period+1}_{x1i}_{grad_x2}"]
        evt_nograd = evti[f"evt_ccp_sequence_t{period+1}_{x1i}_{notgrad_x2}"]

        x1_new = get_x1_new(x1i)
        p_grad = probability_graduation(x1_new, x2i, ji)

        vt = p_grad * evt_grad + (1.0 - p_grad) * evt_nograd
    else:
        notgrad_x2 = move_state_grad(x2i, ji, period, grad=0)
        vt = evti[f"evt_ccp_sequence_t{period+1}_{x1i}_{notgrad_x2}"]

    return vt


def load_ccp_path(x1, state, choices, period, em_type):
    """
    Loads ccp path arrays (n x 100) for each individual, given an em_type.
    """
    sequence = np.zeros((x1.shape[0], 100), dtype=np.float64)

    if period == 9:
        return sequence

    for i in range(x1.shape[0]):
        sequence[i, :] = get_individual_continuation(i, x1, state, choices, period, em_type)

    return sequence


def get_budget(income, debt, j):
    """
    Budget without budget shock using observed data.
    debt is in dollars (already mapped to debt_range).
    """
    w = income[:, 0]
    grants = income[:, 1]
    transfers = income[:, 2:].sum(axis=1)  # includes parental help + loans
    full_part_time = j[:, 2].copy()

    wage = np.exp(w[:, None]) * full_part_time[:, None] * 0.5 * (40 * 52)
    budget = grants + transfers + wage[:, 0] - (1 + r) * debt - tuition_agents(0, j)
    return budget


def get_model(x1, x2final, interp_dict):
    sex = int(x1[2])
    eth = int(x1[3])
    educ = get_educ_level(x2final)
    major = int(x2final[8])
    lastschool = int(x2final[7])
    return interp_dict[(sex, eth, lastschool, educ, major)]


def move_states_T(x1i, x2i, ji, period, interp_dict):
    """
    Move state to terminal period using observed current choice then home production.
    Returns (model_grad, model_notgrad, p_grad).
    """
    home = np.array([0, 0, 0])
    graduated = 0

    for rem in range(period, T):
        if rem == period:
            could_grad = (
                ((x2i[1] >= 1) and (ji[1] == 1) and (x2i[4] == 0)) or
                ((x2i[2] >= 3) and (ji[1] == 2) and (x2i[5] == 0)) or
                (ji[1] == 3)
            )
            if could_grad:
                x2_next_grad = move_state_grad(x2i, ji, period, grad=1)
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
                x2_next_grad = move_state_grad(x2_next_grad, home, period, grad=0)
                x2_next_notgrad = move_state_grad(x2_next_notgrad, home, period, grad=0)

    if graduated == 0:
        model_grad = 0
        model_notgrad = get_model(x1i, x2_next, interp_dict)
    else:
        model_grad = get_model(x1i, x2_next_grad, interp_dict)
        model_notgrad = get_model(x1i, x2_next_notgrad, interp_dict)

    return model_grad, model_notgrad, p_grad


def get_continuation(x1, state, choices, period, interp_dict):
    n = x1.shape[0]
    p_grad = np.zeros(n, dtype=np.float64)
    model_grad = []
    model_notgrad = []

    for i in range(n):
        mg, mn, pg = move_states_T(
            x1[i, :].astype(int),
            state[i, :].astype(int),
            choices[i, :],
            period,
            interp_dict
        )
        model_grad.append(mg)
        model_notgrad.append(mn)
        p_grad[i] = pg

    return model_grad, model_notgrad, p_grad


def load_fundamentals(period, interp_dict, clear_data=False):
    """
    Load data and construct:
      x1, state, debt(prev), debtchoice(data), choices, ccp_path(type1/type2), budget, terminal_data
    """
    global debt_range

    if clear_data:
        get_feasible()
        get_feasible_pubid()

    x1, state, debt, debtchoice, choices, income = load_data_superfeasible(period, return_income=True)
    _ = np.load(f"{path_estimates}/em_q_typeff2.npy")  # keep (even if unused) to match original environment

    # Keep only education choices
    keep = choices[:, 1] > 0
    x1 = x1[keep, :]
    state = state[keep, :]
    debt = debt[keep]
    debtchoice = debtchoice[keep]
    income = income[keep]
    choices = choices[keep, :]

    # Map to dollars on debt grid
    debt = debt_range[debt.astype(int)]
    debtchoice = debt_range[debtchoice.astype(int)]

    # CCP paths
    ccp_path_type1 = load_ccp_path(x1, state, choices, period, 1)
    ccp_path_type2 = load_ccp_path(x1, state, choices, period, 2)
    ccp_path = [ccp_path_type1, ccp_path_type2]

    # Budget
    budget = get_budget(income, debt, choices)

    # Continuations
    model_grad, model_notgrad, p_grad = get_continuation(x1, state, choices, period, interp_dict)
    terminal_data = [model_grad, model_notgrad, p_grad]

    return x1, state, debt, debtchoice, choices, ccp_path, budget, terminal_data


@njit()
def get_ccp_type(ccp1, ccp2, eductype):
    ccp = np.zeros(ccp1.shape, dtype=np.float64)
    for i in range(eductype.shape[0]):
        if eductype[i] == 0:
            ccp[i, :] = ccp1[i, :]
        else:
            ccp[i, :] = ccp2[i, :]
    return ccp


# ==============================================================================
# Moments
# ==============================================================================

def _moments_one_group(debtchoice, nmoments=4, eps=0.01, ddof=0):
    debtchoice = np.asarray(debtchoice).ravel()

    anydebt = float(np.mean(debtchoice > 0))
    pos = debtchoice[debtchoice > 0]

    if pos.size == 0:
        meandebt = eps
        stdebt = eps
        p80 = eps
        anydebt = eps
    else:
        meandebt = float(np.mean(pos))
        stdebt = float(np.std(pos, ddof=ddof))
        p80 = float(np.percentile(pos, 80))

    all_m = np.array([meandebt, anydebt, stdebt, p80], dtype=np.float64)
    return all_m[:nmoments]


def get_moments_by_x1(x1, debtchoice, x1_col=0, nmoments=4, levels=None, eps=0.01, ddof=0):
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
      [mean_debt_pos_g1, share_pos_g1, mean_debt_pos_g2, share_pos_g2, ...]
    """
    x1 = np.asarray(x1)
    debtchoice = np.asarray(debtchoice).ravel()
    g = x1[:, x1_col].astype(int)

    levels = np.sort(np.unique(g))
    out = np.zeros((levels.size, 2), dtype=np.float64)

    for i, lev in enumerate(levels):
        mask = (g == lev)
        d = debtchoice[mask]
        share = float(np.mean(d > 0))
        pos = d[d > 0]
        mean_pos = float(np.mean(pos)) if pos.size > 0 else eps
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


# ==============================================================================
# Terminal cache (single-period version)
# ==============================================================================

def get_relevant_terminal_subset_cached(terminal_data, sigma_risk, idx, cache):
    terminal_grad = terminal_data[0]
    terminal_notgrad = terminal_data[1]
    p_grad = terminal_data[2]

    out = np.zeros((len(idx), debt_range.size), dtype=np.float64)

    for k, i in enumerate(idx):
        # notgrad
        key_n = (id(terminal_notgrad[i]), float(sigma_risk))
        if key_n not in cache:
            cache[key_n] = terminal_notgrad[i]((float(sigma_risk), debt_range))
        notv = cache[key_n]

        if p_grad[i] > 0:
            key_g = (id(terminal_grad[i]), float(sigma_risk))
            if key_g not in cache:
                cache[key_g] = terminal_grad[i]((float(sigma_risk), debt_range))
            gradv = cache[key_g]
            out[k, :] = p_grad[i] * gradv + (1.0 - p_grad[i]) * notv
        else:
            out[k, :] = notv

    return out


# ==============================================================================
# Debt bounds rules
# ==============================================================================

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
        u[i, :int(b_idx[i])] = -100000000.0
        u[i, int(max_idx[i]) + 1:] = -100000000.0

    return u


@njit()
def get_annual_cap_by_stage(educ_choice, twoy_exp, foury_exp):
    cap = 0.0

    if educ_choice == 1:
        if twoy_exp <= 0:
            cap = 8391.0
        elif twoy_exp == 1:
            cap = 9309.0
        else:
            cap = 12581.0

    elif educ_choice == 2:
        if foury_exp <= 0:
            cap = 8391.0
        elif foury_exp == 1:
            cap = 9309.0
        else:
            cap = 12581.0

    elif educ_choice == 3:
        cap = 23222.0

    return cap


@njit()
def get_lifetime_cap_by_stage(educ_choice):
    if educ_choice == 3:
        return 150000.0
    return 70786.0


def get_debt_rules(c, vjt, previousdebt, state, choices):
    """
    Enforce consumption floor + annual/lifetime caps on cumulative debt choice.
    """
    vjt[c < 2000.0] = -100000000.0

    n = previousdebt.shape[0]
    maxdebt = np.empty(n, dtype=np.float64)

    for i in range(n):
        educ_choice = int(choices[i, 1])
        twoy_exp = int(state[i, 1])
        foury_exp = int(state[i, 2])

        annual_cap = get_annual_cap_by_stage(educ_choice, twoy_exp, foury_exp)
        lifetime_cap = get_lifetime_cap_by_stage(educ_choice)

        m = previousdebt[i] * (1.0 + r) + annual_cap
        if m > lifetime_cap:
            m = lifetime_cap
        maxdebt[i] = m

    vjt = minimum_debt_maxdebt(vjt, previousdebt, maxdebt)
    return vjt


@njit()
def precompute_bounds_indices(previousdebt, state, choices):
    n = previousdebt.shape[0]
    maxdebt = np.empty(n, dtype=np.float64)

    for i in range(n):
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
# Utility and solver
# ==============================================================================

@njit()
def get_power_utility(sigma_u, c):
    c = np.maximum(c, 2000.0)
    return 0.1 * ((0.00001 * c) ** (1.0 - sigma_u) / (1.0 - sigma_u))


@njit(parallel=True, fastmath=True)
def solve_one_draw_debt_idx(
    budget, e, debt_grid, sigma_i, debtpen_i,
    ccp_path_row, terminal_row,
    b_idx, max_idx, beta_term,
    c_floor=2000.0, fallback_idx=99
):
    n = budget.shape[0]
    B = debt_grid.size
    out = np.empty(n, dtype=np.int64)

    fb = fallback_idx
    if fb < 0:
        fb = 0
    if fb >= B:
        fb = B - 1

    for i in prange(n):
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
# Pretty printer (FIXED: no escaped quotes)
# ==============================================================================

EVAL_COUNTER = 0

def print_moment_progress(m_data_flat, m_sim_flat, levels, nmoments=2, decimals=2, title=None):
    """
    Pretty print moment fit by group.
    Assumes flattened moments are ordered by group then by moment.
    """
    levels = np.asarray(levels)
    G = len(levels)

    m_data = np.asarray(m_data_flat, dtype=float).reshape(G, nmoments)
    m_sim = np.asarray(m_sim_flat, dtype=float).reshape(G, nmoments)
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
            row.append(f"{m_data[gi, j]:>8.{decimals}f} {m_sim[gi, j]:>8.{decimals}f} {m_diff[gi, j]:>8.{decimals}f}")
        print(" ".join(row))


# ==============================================================================
# Sampling
# ==============================================================================

def get_sample(x1, state, debt, choices, ccp_path, budget, terminal):
    n = 10000

    random_indices = np.random.choice(x1.shape[0], size=n, replace=True)

    x1sample = x1[random_indices, :]
    statesample = state[random_indices, :]
    debtsample = debt[random_indices]
    choicessample = choices[random_indices, :]
    budgetsample = budget[random_indices]

    ccp_path_type1 = ccp_path[0]
    ccp_path_type2 = ccp_path[1]
    ccp_1_sample = ccp_path_type1[random_indices, :]
    ccp_2_sample = ccp_path_type2[random_indices, :]

    terminal1 = terminal[0]
    terminal2 = terminal[1]
    terminal3 = terminal[2]
    terminal1_sample = [terminal1[i] for i in random_indices]
    terminal2_sample = [terminal2[i] for i in random_indices]
    terminal3_sample = [terminal3[i] for i in random_indices]
    terminal_sample = [terminal1_sample, terminal2_sample, terminal3_sample]

    # draw education type
    eductype = np.random.binomial(1, 0.5, x1sample.shape[0])
    evt = get_ccp_type(ccp_1_sample, ccp_2_sample, eductype)

    return x1sample, statesample, debtsample, choicessample, budgetsample, evt, terminal_sample


# ==============================================================================
# Objective with 7 mu + 7 dp + ability moments (single-period)
# ==============================================================================

def minimize_distance_singleperiod(params, moments_data, budget, terminal_data, previousdebt,
                                   evt_ccp, x1, state, choice, period, Z, b_idx, max_idx):
    """
    params:
      mu block (7): [mu0, mu_par2, mu_par3, mu_par4, mu_ab2, mu_ab3, mu_ab4]
      sigma_e (1)
      ra_levels (4) by parental income group
      dp block (7): [dp0, dp_par2, dp_par3, dp_par4, dp_ab2, dp_ab3, dp_ab4]
    """
    global EVAL_COUNTER
    EVAL_COUNTER += 1

    # unpack
    mu0, mu_par2, mu_par3, mu_par4, mu_ab2, mu_ab3, mu_ab4 = params[0:7]
    sigma_e = float(params[7])
    ra_levels = params[8:12]
    dp0, dp_par2, dp_par3, dp_par4, dp_ab2, dp_ab3, dp_ab4 = params[12:19]

    # groups
    par = x1[:, 0].astype(int)  # 1..4
    ab = x1[:, 1].astype(int)   # 1..4
    par2, par3, par4 = dummies_234(par)
    ab2_, ab3_, ab4_ = dummies_234(ab)

    # individual mu and debt penalty
    mu_i = (mu0
            + mu_par2 * par2 + mu_par3 * par3 + mu_par4 * par4
            + mu_ab2 * ab2_ + mu_ab3 * ab3_ + mu_ab4 * ab4_).astype(np.float64)

    debtpen_i = (dp0
                 + dp_par2 * par2 + dp_par3 * par3 + dp_par4 * par4
                 + dp_ab2 * ab2_ + dp_ab3 * ab3_ + dp_ab4 * ab4_).astype(np.float64)

    gi = (par - 1).astype(np.int64)  # 0..3
    sigma_i = ra_levels[gi].astype(np.float64)

    # terminal depends on ra_levels (by parental income)
    n = budget.shape[0]
    B = debt_range.size
    terminal = np.zeros((n, B), dtype=np.float64)

    cache = {}
    for k in range(4):
        idx = np.where(gi == k)[0]
        if idx.size > 0:
            terminal[idx, :] = get_relevant_terminal_subset_cached(terminal_data, float(ra_levels[k]), idx, cache)

    beta_term = float(beta ** (T - period))

    # simulate moments over draws
    s = Z.shape[0]
    sim_draws = np.zeros((24, s), dtype=np.float64)

    for k in range(s):
        e = mu_i + sigma_e * Z[k, :]

        debt_idx = solve_one_draw_debt_idx(
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
            fallback_idx=99
        )

        debtvalue = debt_range[debt_idx]
        mom_par = get_moments_by_x1(x1, debtvalue, x1_col=0, nmoments=4)     # 16
        mom_ab = get_mean_share_by_group(x1, debtvalue, x1_col=1)           # 8
        sim_draws[:, k] = np.concatenate([mom_par, mom_ab])

    sim_mom = sim_draws.mean(axis=1)

    denom = np.maximum(np.abs(moments_data), 1e-6)
    loss = float(np.sum(((moments_data - sim_mom) / denom) ** 2))

    if (EVAL_COUNTER % 10) == 0:
        print("\n" + "=" * 100)
        print(f"[eval {EVAL_COUNTER}] loss={loss:.6f}")
        print(f"sigma_e={sigma_e:.2f} (var={sigma_e**2:.2f})")
        print("ra_levels (par 1..4) =", np.round(ra_levels, 3))
        print("dp0, dp_par2, dp_par3, dp_par4, dp_ab2, dp_ab3, dp_ab4 =",
              np.round([dp0, dp_par2, dp_par3, dp_par4, dp_ab2, dp_ab3, dp_ab4], 3))
        print("mu0, mu_par2, mu_par3, mu_par4, mu_ab2, mu_ab3, mu_ab4 =",
              np.round([mu0, mu_par2, mu_par3, mu_par4, mu_ab2, mu_ab3, mu_ab4], 2))

        mom_par_data = moments_data[:16]
        mom_ab_data = moments_data[16:]
        mom_par_sim = sim_mom[:16]
        mom_ab_sim = sim_mom[16:]

        print("\n--- Pretty moment tables (period=%d) ---" % int(period))

        print_moment_progress(
            mom_par_data, mom_par_sim,
            levels=np.array([1, 2, 3, 4]),
            nmoments=4,
            decimals=2,
            title="ParInc moments (data vs simulated)"
        )

        print_moment_progress(
            mom_ab_data, mom_ab_sim,
            levels=np.array([1, 2, 3, 4]),
            nmoments=2,
            decimals=2,
            title="Ability moments (data vs simulated)"
        )
        print("=" * 100 + "\n")

    return loss


# ==============================================================================
# Estimation driver (single period)
# ==============================================================================

def fit_budget_shock_singleperiod_with_ability(
    x1, state, debt, debtchoice, choices, ccp_path, budget, terminal_data, period
):
    """
    Single-period SMM with:
      - 24 moments: parinc(16) + ability(8)
      - 19 params: mu(7) + sigma_e(1) + ra(4) + debtpen(7)
    Uses Powell with bounds and an outer-loop bootstrap resample like your original debug flow.
    """

    # data moments
    mom_par = get_moments_by_x1(x1, debtchoice, x1_col=0, nmoments=4).astype(np.float64)  # 16
    mom_ab = get_mean_share_by_group(x1, debtchoice, x1_col=1).astype(np.float64)        # 8
    moments_data = np.concatenate([mom_par, mom_ab]).astype(np.float64)                  # 24

    # initial params
    mu0_guess = 9000.0
    mu_block0 = np.array([mu0_guess, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    sigma_e0 = 2000.0
    ra0 = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    dp0 = np.array([-1.0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    params0 = np.concatenate([mu_block0, np.array([sigma_e0]), ra0, dp0]).astype(np.float64)  # 19

    # bounds
    MU_MIN, MU_MAX = -50000.0, 50000.0
    SIGE_MIN, SIGE_MAX = 1.0, 50000.0

    bounds = []
    bounds += [(MU_MIN, MU_MAX)] * 7            # mu block
    bounds += [(SIGE_MIN, SIGE_MAX)]            # sigma_e
    bounds += [(0.1001, 2.9999)] * 4            # ra
    bounds += [(-1e6, 0.0)] * 7                 # dp block <= 0

    tolr = 0.01
    best_fun = 1e30
    best_x = params0.copy()

    it = 0
    while best_fun > tolr:
        print(f"\n================ OUTER ITER {it} ================")
        np.random.seed(it)

        x1s, states, debts, choices_s, budget_s, evt_ccp, terminal_s = get_sample(
            x1, state, debt, choices, ccp_path, budget, terminal_data
        )

        # CRN shocks
        s = 20
        rng = np.random.default_rng(12345)
        Z = rng.standard_normal((s, x1s.shape[0])).astype(np.float64)

        # bounds indices
        b_idx, max_idx = precompute_bounds_indices(
            debts.astype(np.float64),
            states.astype(np.int64),
            choices_s.astype(np.int64)
        )

        # contiguous arrays for numba
        budget_s = np.ascontiguousarray(budget_s, dtype=np.float64)
        evt_ccp = np.ascontiguousarray(evt_ccp, dtype=np.float64)
        x1s = np.ascontiguousarray(x1s)
        states = np.ascontiguousarray(states)
        choices_s = np.ascontiguousarray(choices_s)

        args_obj = (
            moments_data,
            budget_s,
            terminal_s,
            debts.astype(np.float64),
            evt_ccp,
            x1s,
            states,
            choices_s,
            period,
            Z,
            b_idx,
            max_idx
        )

        print("---- Running minimize(..., method='Powell') ----")
        res = minimize(
            minimize_distance_singleperiod,
            x0=best_x,
            args=args_obj,
            method="Powell",
            bounds=bounds,
            options={"maxiter": 200, "disp": True}
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
        if it == 20:
            break

    print("\n================ FINAL (single period) =================")
    print("best_fun:", best_fun)
    print("best_x:", best_x)
    return best_x, best_fun


def estimation(period=1):
    interp_dict = get_interp_dict_cached(force_rebuild=False)

    x1, state, debt, debtchoice, choices, ccp_path, budget, terminal_data = load_fundamentals(
        period, interp_dict, clear_data=False
    )

    fit_budget_shock_singleperiod_with_ability(
        x1, state, debt, debtchoice, choices, ccp_path, budget, terminal_data, period
    )


# ==============================================================================
# Run
# ==============================================================================

if __name__ == "__main__":
    period = 1
    estimation(period=period)