# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 14:31:48 2025

@author: S.Quintana

Static SMM estimation of student loans distribution.

ADJUSTMENT:
- Risk aversion is NOT estimated.
- It is fixed at RA_FIXED and can be changed easily below.

MOMENTS USED:
1. Mean debt among indebted
2. Share indebted
3. Share at the individual legal maximum debt
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

from config import DIR, OUT, EST

pathout = DIR["MODEL_OUTPUT"]
path_estimates = DIR["MODEL_ESTIMATES"]
pathcont = DIR["MODEL_CONTINUATION_FINAL"]

T = 10
r = 0.05
beta = 0.98
debt_range = get_debt_range()

# ============================================================
# FIXED RISK AVERSION
RA_FIXED = 0.5
# ============================================================

# ============================================================
# Cache interpolator dictionary

CACHE_DIR = os.path.join(pathout, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_interp_dict_cached(force_rebuild=False):
    cache_path = os.path.join(CACHE_DIR, "interp_dict.joblib")

    if (not force_rebuild) and os.path.exists(cache_path):
        print(f"[cache] Loading interp_dict from {cache_path}")
        interp_dict = joblib.load(cache_path)
        return interp_dict

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

# ============================================================
# CCP path

def get_individual_continuation(i, x1, x2, j, period, em_type):
    x1i = x1[i, :].astype(int)
    x2i = x2[i, :].astype(int)
    ji = j[i, :]

    if ((x2i[1] >= 1) & (ji[1] == 1) & (x2i[4] == 0)) | ((x2i[2] >= 3) & (ji[1] == 2) & (x2i[5] == 0)) | (ji[1] == 3):
        grad_x2 = move_state_grad(x2i, ji, period, grad=1)
        notgrad_x2 = move_state_grad(x2i, ji, period)

        evti = np.load(f"{pathout}/evt_ccp/{period+1}/evt_ccp_sequence_t{period+1}_{x1i}_em{em_type}.npz")
        evt_grad = evti[f"evt_ccp_sequence_t{period+1}_{x1i}_{grad_x2}"]
        evt_nograd = evti[f"evt_ccp_sequence_t{period+1}_{x1i}_{notgrad_x2}"]

        x1_new = get_x1_new(x1i)
        p_grad = probability_graduation(x1_new, x2i, ji)
        vt = p_grad * evt_grad + (1 - p_grad) * evt_nograd
    else:
        evti = np.load(f"{pathout}/evt_ccp/{period+1}/evt_ccp_sequence_t{period+1}_{x1i}_em{em_type}.npz")
        notgrad_x2 = move_state_grad(x2i, ji, period)
        vt = evti[f"evt_ccp_sequence_t{period+1}_{x1i}_{notgrad_x2}"]

    return vt

def load_ccp_path(x1, state, choices, period, em_type):
    sequence = np.zeros((np.shape(x1)[0], 100))

    if period == 9:
        return sequence
    else:
        for i in range(np.shape(x1)[0]):
            sequence[i, :] = get_individual_continuation(i, x1, state, choices, period, em_type)
        return sequence

# ============================================================
# Budget

def get_budget(income, debt, j):
    w = income[:, 0]
    grants = income[:, 1]
    transfers = income[:, 2:].sum(axis=1)

    full_part_time = j[:, 2].copy()
    wage = np.exp(w[..., np.newaxis]) * full_part_time[..., None] * 0.5 * (40 * 52)

    budget = grants + transfers + wage[:, 0] - (1 + r) * debt - tuition_agents(0, j)
    return budget

# ============================================================
# Terminal continuation

def get_model(x1, x2final, interp_dict):
    sex = x1[2]
    eth = x1[3]
    educ = get_educ_level(x2final)
    major = x2final[8]
    lastschool = x2final[7]
    return interp_dict[(sex, eth, lastschool, educ, major)]

def move_states_T(x1i, x2i, ji, period, interp_dict):
    home = np.array([0, 0, 0])
    graduated = 0

    for rem in range(period, T):
        if rem == period:
            if ((x2i[1] >= 1) & (ji[1] == 1) & (x2i[4] == 0)) | ((x2i[2] >= 3) & (ji[1] == 2) & (x2i[5] == 0)) | (ji[1] == 3):
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
    n = np.shape(x1)[0]
    p_grad = np.zeros(n)
    model_grad = []
    model_notgrad = []

    for i in range(n):
        model_gradi, model_notgradi, p_gradi = move_states_T(
            x1[i, :].astype(int),
            state[i, :].astype(int),
            choices[i, :],
            period,
            interp_dict
        )
        model_grad.append(model_gradi)
        model_notgrad.append(model_notgradi)
        p_grad[i] = p_gradi

    return model_grad, model_notgrad, p_grad

# ============================================================
# Load data

def load_fundamentals(period, interp_dict, clear_data=False):
    global debt_range

    if clear_data:
        get_feasible()
        get_feasible_pubid()

    x1, state, debt, debtchoice, choices, income = load_data_superfeasible(period, return_income=True)

    x1 = x1[choices[:, 1] > 0, :]
    state = state[choices[:, 1] > 0, :]
    debt = debt[choices[:, 1] > 0]
    debtchoice = debtchoice[choices[:, 1] > 0]
    income = income[choices[:, 1] > 0]
    choices = choices[choices[:, 1] > 0, :]

    debt = debt_range[debt.astype(int)]
    debtchoice = debt_range[debtchoice.astype(int)]

    ccp_path_type1 = load_ccp_path(x1, state, choices, period, 1)
    ccp_path_type2 = load_ccp_path(x1, state, choices, period, 2)
    ccp_path = [ccp_path_type1, ccp_path_type2]

    budget = get_budget(income, debt, choices)

    model_grad, model_notgrad, p_grad = get_continuation(x1, state, choices, period, interp_dict)
    terminal_data = [model_grad, model_notgrad, p_grad]

    return x1, state, debt, debtchoice, choices, ccp_path, budget, terminal_data

# ============================================================
# Types

@njit()
def get_ccp_type(ccp1, ccp2, eductype):
    ccp = np.zeros(np.shape(ccp1))
    for i in range(np.shape(eductype)[0]):
        if eductype[i] == 0:
            ccp[i, :] = ccp1[i, :]
        else:
            ccp[i, :] = ccp2[i, :]
    return ccp

# ============================================================
# Debt choice restrictions and legal caps

@njit()
def get_range_number(b, maxdebt):
    value = np.zeros(np.shape(b)[0])
    value2 = np.zeros(np.shape(b)[0])

    for i in range(np.shape(b)[0]):
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

@njit()
def get_debt(debt_range, debt):
    debtvalues = np.zeros(np.shape(debt)[0])
    for i in range(np.shape(debt)[0]):
        debtvalues[i] = debt_range[int(debt[i])]
    return debtvalues

@njit()
def get_power_utility(sigma_u, c):
    c = np.maximum(c, 2000)
    return 0.1 * ((0.00001 * c) ** (1 - sigma_u) / (1 - sigma_u))

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

def compute_legal_maxdebt(previousdebt, state, choices):
    """
    Returns individual legal maximum debt in dollars.
    """
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

    return maxdebt

def map_to_grid(values):
    """
    Maps dollar values to nearest point in debt_range.
    """
    values = np.asarray(values).ravel()
    out = np.empty(values.shape[0], dtype=np.float64)

    for i in range(values.shape[0]):
        idx = np.argmin((debt_range - values[i]) ** 2)
        out[i] = debt_range[idx]

    return out

# ============================================================
# Moments

def _moments_one_group(debtchoice, maxdebt, nmoments=3, eps=0.01):
    """
    Moments by group:
      1. mean debt among indebted
      2. share indebted
      3. share at individual legal maximum debt
    """
    debtchoice = np.asarray(debtchoice).ravel()
    maxdebt = np.asarray(maxdebt).ravel()

    share_indebted = np.mean(debtchoice > 0)
    pos = debtchoice[debtchoice > 0]

    if pos.size == 0:
        mean_debt = eps
        share_indebted = eps
    else:
        mean_debt = np.mean(pos)

    share_max = np.mean(np.isclose(debtchoice, maxdebt))

    all_moments = np.array([mean_debt, share_indebted, share_max], dtype=float)
    return all_moments[:nmoments]

def get_moments_by_x1(x1, debtchoice, maxdebt, x1_col=0, nmoments=3, levels=None, eps=0.01):
    x1 = np.asarray(x1)
    debtchoice = np.asarray(debtchoice).ravel()
    maxdebt = np.asarray(maxdebt).ravel()

    g = x1[:, x1_col] if x1.ndim > 1 else x1.ravel()

    if levels is None:
        levels_out = np.sort(np.unique(g))
    else:
        levels_out = np.asarray(levels)

    M = np.zeros((levels_out.size, nmoments), dtype=float)

    for i, lev in enumerate(levels_out):
        mask = (g == lev)
        M[i, :] = _moments_one_group(
            debtchoice[mask],
            maxdebt[mask],
            nmoments=nmoments,
            eps=eps
        )

    return M.flatten()

def parinc_groups(x1, col=0, levels=None):
    g = x1[:, col].astype(int)
    if levels is None:
        levels = np.sort(np.unique(g))
    gi = np.searchsorted(levels, g)
    return g, gi, levels

# ============================================================
# Terminal evaluation

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

# ============================================================
# Debt solver

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

        if not found_feasible:
            out[i] = hi
        else:
            out[i] = best_j

    return out

# ============================================================
# Printing

EVAL_COUNTER = 0

def print_moment_progress(m_data_flat, m_sim_flat, levels, nmoments=3, decimals=2, title=None):
    G = len(levels)
    m_data = np.asarray(m_data_flat).reshape(G, nmoments)
    m_sim = np.asarray(m_sim_flat).reshape(G, nmoments)
    m_diff = m_sim - m_data

    moment_names_all = ["MeanDebt>0", "Pr(debt>0)", "Pr(legal max debt)"]
    moment_names = moment_names_all[:nmoments]

    if title:
        print(title)
    print(f"{'ParInc':>5} | " + "  ".join([f"{name:^32}" for name in moment_names]))
    print(f"{'':>5} | " + "  ".join([f"{'data':>8} {'sim':>8} {'diff':>8}" for _ in moment_names]))

    for gi, lev in enumerate(levels):
        row = [f"{int(lev):>5} |"]
        for j in range(nmoments):
            row.append(f"{m_data[gi,j]:>8.{decimals}f} {m_sim[gi,j]:>8.{decimals}f} {m_diff[gi,j]:>8.{decimals}f}")
        print(" ".join(row))

# ============================================================
# Objective with FIXED risk aversion

def minimize_distance(params, moments, budget, terminal_data, previousdebt, ccp_path,
                      x1, state, choice, period, Z, b_idx, max_idx):

    global EVAL_COUNTER
    EVAL_COUNTER += 1

    # params = [mu1, mu2, mu3, mu4, sigma_e, debtpen1, debtpen2, debtpen3, debtpen4]
    mu_levels = params[0:4]
    sigma_e = float(params[4])
    debt_levels = params[5:9]

    g, gi, levels = parinc_groups(x1, col=0)

    mu_i = mu_levels[gi].astype(np.float64)
    debtpen_i = debt_levels[gi].astype(np.float64)

    sigma_i = np.full(budget.shape[0], RA_FIXED, dtype=np.float64)

    n = budget.shape[0]
    B = debt_range.size
    terminal = np.zeros((n, B), dtype=np.float64)

    cache = {}
    all_idx = np.arange(n)
    terminal[:, :] = get_relevant_terminal_subset_cached(
        terminal_data, RA_FIXED, all_idx, cache
    )

    beta_term = float(beta ** (T - period))

    s = Z.shape[0]
    simumoments = np.zeros((moments.shape[0], s), dtype=np.float64)

    for k in range(s):
        e = mu_i + sigma_e * Z[k, :]

        debt_idx = solve_one_draw_debt_idx(
            budget=budget,
            e=e,
            debt_grid=debt_range,
            sigma_i=sigma_i,
            debtpen_i=debtpen_i,
            ccp_path_row=ccp_path,
            terminal_row=terminal,
            b_idx=b_idx,
            max_idx=max_idx,
            beta_term=beta_term,
            c_floor=2000.0,
            fallback_idx=99
        )

        debtvalue = debt_range[debt_idx]
        maxdebt_sim = debt_range[max_idx]

        simumoments[:, k] = get_moments_by_x1(
            x1, debtvalue, maxdebt_sim, nmoments=3
        )

    simumoments = simumoments.mean(axis=1)

    denom = np.maximum(np.abs(moments), 1e-6)
    loss = float(np.sum(((moments - simumoments) / denom) ** 2))

    if (EVAL_COUNTER % 10) == 0:
        print("\n" + "=" * 100)
        print(f"[eval {EVAL_COUNTER}] loss={loss:.6f}")
        print(f"RA_FIXED={RA_FIXED:.4f}")
        print(f"sigma_e={sigma_e:.2f}  (var={sigma_e**2:.2f})")
        print(f"parinc levels: {list(map(int, levels))}")
        print(f"mu(levels):    {np.round(mu_levels, 2)}")
        print(f"debtpen(levels): {np.round(debt_levels, 2)}")
        print_moment_progress(
            moments, simumoments,
            levels=levels,
            nmoments=3,
            decimals=2,
            title="Moments by parental-income group (data vs simulated)"
        )
        print("=" * 100 + "\n")

    return loss

# ============================================================
# Sampling

def get_sample(x1, state, debt, choices, ccp_path, budget, terminal):
    n = 100000

    random_indices = np.random.choice(np.shape(x1)[0], size=n, replace=True)

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

    eductype = np.random.binomial(1, 0.5, np.shape(x1sample)[0])
    evt = get_ccp_type(ccp_1_sample, ccp_2_sample, eductype)

    e = np.random.normal(loc=0, scale=1, size=n)

    return x1sample, statesample, debtsample, choicessample, budgetsample, evt, e, terminal_sample

# ============================================================
# Estimation

def fit_budget_shock(x1, state, debt, debtchoice, choices, ccp_path, budget, terminal_data, period):
    maxdebt_data = compute_legal_maxdebt(debt, state, choices)
    maxdebt_data = map_to_grid(maxdebt_data)

    moments = get_moments_by_x1(
        x1, debtchoice, maxdebt_data, nmoments=3
    ).astype(np.float64)

    params0 = np.array([
        5000.0, 5000.0, 5000.0, 5000.0,
        100.0,
        -2.0, -2.0, -2.0, -2.0
    ], dtype=np.float64)

    MU_MIN, MU_MAX = -50000.0, 50000.0
    SIGE_MIN, SIGE_MAX = 1.0, 50000.0

    bounds = [
        (MU_MIN, MU_MAX), (MU_MIN, MU_MAX), (MU_MIN, MU_MAX), (MU_MIN, MU_MAX),
        (SIGE_MIN, SIGE_MAX),
        (-1e6, 0.0), (-1e6, 0.0), (-1e6, 0.0), (-1e6, 0.0)
    ]

    tolr = 0.01
    best_fun = 1e30
    best_x = params0.copy()

    it = 0
    while best_fun > tolr:
        print(f"\n================ OUTER ITER {it} ================")
        np.random.seed(it)

        x1sample, statesample, debtsample, choicessample, budgetsample, evt_ccp, _, terminal_sample = get_sample(
            x1, state, debt, choices, ccp_path, budget, terminal_data
        )

        s = 20
        rng = np.random.default_rng(12345)
        Z = rng.standard_normal((s, x1sample.shape[0])).astype(np.float64)

        b_idx, max_idx = precompute_bounds_indices(
            debtsample.astype(np.float64),
            statesample.astype(np.int64),
            choicessample.astype(np.int64),
        )

        budgetsample = np.ascontiguousarray(budgetsample, dtype=np.float64)
        evt_ccp = np.ascontiguousarray(evt_ccp, dtype=np.float64)
        x1sample = np.ascontiguousarray(x1sample)

        args_obj = (
            moments,
            budgetsample.astype(np.float64),
            terminal_sample,
            debtsample.astype(np.float64),
            evt_ccp.astype(np.float64),
            x1sample,
            statesample,
            choicessample,
            period,
            Z,
            b_idx,
            max_idx
        )

        cb_counter = {"k": 0}
        def cb(xk):
            cb_counter["k"] += 1
            if cb_counter["k"] % 5 == 0:
                print(f"[Powell callback] iter={cb_counter['k']}  x0..4={np.round(xk[:5],2)}")

        print("---- Running minimize(..., method='Powell') ----")
        res = minimize(
            minimize_distance,
            x0=best_x,
            args=args_obj,
            method="Powell",
            bounds=bounds,
            callback=cb,
            options={
                "maxiter": 30,
                "disp": True,
            }
        )

        print("\n[Powell done] success:", res.success)
        print("[Powell done] message:", res.message)
        print("[Powell done] fun:", float(res.fun))
        print("[Powell done] x:", res.x)

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

    print("\n================ FINAL =================")
    print(f"RA_FIXED = {RA_FIXED}")
    print("best_x:", best_x)
    print("best_fun:", best_fun)
    return best_x, best_fun

# ============================================================
# Main

def estimation():
    interp_dict = get_interp_dict_cached(force_rebuild=False)

    x1, state, debt, debtchoice, choices, ccp_path, budget, terminal_data = load_fundamentals(
        period, interp_dict, clear_data=False
    )

    fit_budget_shock(x1, state, debt, debtchoice, choices, ccp_path, budget, terminal_data, period)

if __name__ == "__main__":
    debt_range = get_debt_range()
    period = 1
    estimation()