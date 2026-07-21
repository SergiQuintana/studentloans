# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 14:31:48 2025


@author: S.Quintana


This program performs the SMM estimation algorithm to fit student loans distribution.


I will try to borrow functions from the other programs. 


The structure is: 
    
    1st. Load Data. 
    2nd. Simulate individuals and draw types. 
    3rd. Get the CCPs. 
    4th. Estimation. 


"""


import numpy as np
import joblib 
import os


from numba import njit, prange
from scipy.optimize import minimize, differential_evolution

from model_em_algorithm import (load_all_arrays_feasible,
                               load_data_superfeasible,
                               get_feasible,
                               get_feasible_pubid,
                               get_vjt_static,
                               get_data_superfeasible)


from model_solution_em import (move_state_grad,
                               get_x1_new,
                               probability_graduation,
                               get_debt_range,
                               get_educ_level,
                               load_all_parameters)


from model_simulation_em import (tuition_agents)


from model_interpolate_terminal import build_interpolator_dictionary


from config import DIR, OUT, INP, FUN, RDATA, CONT, EST, LIK, ENSURE_DIR

# Canonical roots from config (no chdir anywhere)
PATH_OUT        = DIR["MODEL_OUTPUT"]
PATH_EST        = DIR["MODEL_ESTIMATES"]
PATH_CONT_FINAL = DIR["MODEL_CONTINUATION_FINAL"]

T = 10
r = 0.05
beta = 0.98
debt_range = get_debt_range()




def _ensure_dir_for_file(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def save_budgetshock_params(best_x, periods, K_mu=7, filename_prefix=""):
    """
    Saves:
      1) risk aversion only              -> EST("risk_aversion.npy")
      2) all other budget shock params   -> EST("budgetshock_params.npy")   (dict saved as npy)

    Parameters
    ----------
    best_x : np.ndarray
        Final parameter vector from your fit (the same 'best_x' you print at the end).
    periods : list[int]
        Periods used in estimation, in the SAME order as minimize_distance_multi uses.
    K_mu : int
        # of mu coefficients per period. Currently your code uses 7.
    filename_prefix : str
        Optional prefix to version outputs, e.g. "v1_" to write v1_risk_aversion.npy, etc.
    """
    periods = list(periods)
    P = len(periods)
    best_x = np.asarray(best_x, dtype=np.float64)

    # ---- unpack using your current layout:
    # [ mu_blocks (K_mu*P), sigma_e_vec (P), ra_levels (4), other (rest) ]
    mu_blocks = best_x[:K_mu * P].reshape(P, K_mu)
    sigma_e_vec = best_x[K_mu * P: K_mu * P + P].astype(np.float64)

    ra_start = K_mu * P + P
    ra_levels = best_x[ra_start: ra_start + 4].astype(np.float64)

    other = best_x[ra_start + 4:].astype(np.float64)

    # ---- 1) risk aversion only
    ra_path = EST(f"{filename_prefix}risk_aversion.npy")
    _ensure_dir_for_file(ra_path)
    np.save(ra_path, ra_levels)

    # ---- 2) budget shock pack (dict)
    pack = {
        "periods": np.array(periods, dtype=np.int64),
        "K_mu": np.int64(K_mu),
        "mu_blocks": mu_blocks,         # (P, K_mu)
        "sigma_e_vec": sigma_e_vec,     # (P,)
        "other": other,                 # e.g. debtpen block etc (whatever is left)
        "layout": "mu_blocks | sigma_e_vec | ra_levels | other",
    }

    pack_path = EST(f"{filename_prefix}budgetshock_params.npy")
    _ensure_dir_for_file(pack_path)
    np.save(pack_path, pack, allow_pickle=True)

    # ---- optional: convenient npz snapshot too
    snap_path = EST(f"{filename_prefix}budgetshock_snapshot.npz")
    _ensure_dir_for_file(snap_path)
    np.savez_compressed(
        snap_path,
        periods=np.array(periods, dtype=np.int64),
        K_mu=np.int64(K_mu),
        mu_blocks=mu_blocks,
        sigma_e_vec=sigma_e_vec,
        ra_levels=ra_levels,
        other=other,
        best_x=best_x,
    )

    print("\n[SAVED]")
    print("  risk aversion:", ra_path, "shape", ra_levels.shape)
    print("  budget shock pack:", pack_path)
    print("  snapshot:", snap_path)

#==============================================================================
# Store inter_dict in cache  memory

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
            verbose=False
        )

    print(f"[cache] Saving interp_dict to {cache_path}")
    joblib.dump(interp_dict, cache_path, compress=3)
    return interp_dict

#==============================================================================




def get_individual_continuation(i,x1,x2,j,period,em_type):
    
    """
    This function loops over all individuals and loads their continuation 
    value in sequence form, and takes expectation whenever is needed. 
    """
    
    x1i = x1[i,:].astype("int")
    x2i = x2[i,:].astype("int")
    ji = j[i,:]
    # See if the individual choice could imply a graduation chioce: 
    if ((x2i[1] >=1) & (ji[1] == 1) & (x2i[4] == 0)) | ((x2i[2]>=3) & (ji[1] == 2) & (x2i[5] == 0))  | (ji[1] == 3) :        
        # The choice could induce a graduation state. For this reason, take the expectation.
        grad_x2  =  move_state_grad(x2i,ji,period,grad=1)
        notgrad_x2 = move_state_grad(x2i,ji,period)
        #evt = np.load(f"evt/evt_t{period+1}_{x1}.npz")
        evti = np.load(
            OUT("evt_ccp", str(period+1),
                f"evt_ccp_sequence_t{period+1}_{x1i}_em{em_type}.npz")
        )
        evt_grad =  evti[f"evt_ccp_sequence_t{period+1}_{x1i}_{grad_x2}"]
        evt_nograd = evti[f"evt_ccp_sequence_t{period+1}_{x1i}_{notgrad_x2}"]
        x1_new = get_x1_new(x1i)
        p_grad = probability_graduation(x1_new,x2i,ji)
        vt = p_grad*evt_grad + (1-p_grad)*evt_nograd


    else: 
        evti = np.load(
            OUT("evt_ccp", str(period+1),
                f"evt_ccp_sequence_t{period+1}_{x1i}_em{em_type}.npz")
        )
        notgrad_x2 = move_state_grad(x2i,ji,period)
        vt = evti[f"evt_ccp_sequence_t{period+1}_{x1i}_{notgrad_x2}"]
    return vt       


          
         
    
    


def load_ccp_path(x1,state,choices,period,em_type):
    
    """
    This function computes the sum of the log of all the CCPs up to the last
    terminal value, assuming the agent always chooses home production. 
    
    Given the choice and period, move state space. 
    """
    
    # Initialize values
    
    sequence = np.zeros((np.shape(x1)[0],100))
    
    if period == 9 : 
        
        return sequence
    else: 
        # Loop over all individuals
        
        for i in range(np.shape(x1)[0]):
            
            sequence[i,:] = get_individual_continuation(i,x1,state,choices,period,em_type)
            
        return sequence
        


    
def get_budget(income,debt,j):
    
    
    """
    This function computes the budget without the budget shock using real observed data.
    """
    
    global debt_range
    
    w = income[:,0]
    grants = income[:,1]
    transfers = income[:,2:].sum(axis=1)  # sums parental help and loans
    
    
    full_part_time = j[:,2].copy()
    
    wage = np.exp((w[...,np.newaxis]))*full_part_time[...,None]*1/2*(40*52)
    
    budget = (grants + transfers + wage[:,0] -(1+r)*debt-tuition_agents(0,j))
    
    return budget



def get_model(x1, x2final, interp_dict):
    
    
    # get relevant data
    sex = x1[2]
    eth = x1[3]
    educ = get_educ_level(x2final)
    major = x2final[8]
    lastschool = x2final[7]
    
    model = interp_dict[(sex, eth, lastschool, educ, major)]
    
    #values = model((float(sigma), debt_range))
    
    return model

def move_states_T(x1i, x2i, ji, period, interp_dict):
    
    """
    This function moves the current state until the last period, first 
    using chioces and then using home production as a base category.
    """
    
    home = np.array([0,0,0])
    graduated = 0
    
    for rem in range(period,T):
        
        if rem == period: # use choices to move the state
            
            # First check if the choice could induce graduation
            if ((x2i[1] >=1) & (ji[1] == 1) & (x2i[4] == 0)) | ((x2i[2]>=3) & (ji[1] == 2) & (x2i[5] == 0))  | (ji[1] == 3) :
                
                x2_next_grad    = move_state_grad(x2i,ji,period,grad=1)
                x2_next_notgrad = move_state_grad(x2i,ji,period,grad=0)
                x1_new = get_x1_new(x1i)
                p_grad = probability_graduation(x1_new,x2i,ji)
                graduated = 1
            else:
                p_grad = 0
                x2_next   = move_state_grad(x2i,ji,period,grad=0)
        else:  # use home production as chioce
            
            # Check if the individual could have graduated
            if graduated == 0: 
                
                x2_next = move_state_grad(x2_next,home,period,grad=0)
        
            elif graduated == 1: 
                
                x2_next_grad    = move_state_grad(x2_next_grad,home,period,grad=0)
                x2_next_notgrad = move_state_grad(x2_next_notgrad,home,period,grad=0)
                
        
    
    
    # Finally, if the individual graduated take the expected future state: 
        
    if graduated == 0: 
        
        model_grad    = 0
        model_notgrad = get_model(x1i, x2_next, interp_dict)
        
    elif graduated == 1: 
        
        model_grad    = get_model(x1i, x2_next_grad, interp_dict)
        model_notgrad = get_model(x1i, x2_next_notgrad, interp_dict)
        
        
    return model_grad, model_notgrad, p_grad
        
    

def get_continuation(x1, state, choices, period, interp_dict):
    
    """
    This function loads the continuation value at perio 10 of an individual
    that is currently at state and period.  It does not load the value itself, but the
    interpolator that will be used once a sigma is guessed. For this reason,
    I load termianl with and without graduation, as well as graduation probabilities. 
    """
    
    # 1 --- Loop Over All Individuals
    
    n = np.shape(x1)[0]
    
    p_grad = np.zeros(n)
    model_grad = []
    model_notgrad = []
    
    for i in range(n):
        
    
        model_gradi, model_notgradi, p_gradi = move_states_T(x1[i,:].astype("int"), 
                                                          state[i,:].astype("int"),
                                                          choices[i,:], 
                                                          period,
                                                          interp_dict)
        model_grad.append(model_gradi)
        model_notgrad.append(model_notgradi)
        p_grad[i] = p_gradi
        
    return model_grad, model_notgrad, p_grad
    
    
    
def load_fundamentals(period, interp_dict, clear_data= False):
    
    """
    This function performs the execution of the code. 
    1st. Load data
    2nd. Get Education Choices only
    3rd. Build CCP path
    4th. 
    """
    
    global debt_range
    
    # 1 --- Load Data
    if clear_data == True:
        get_feasible()
        get_feasible_pubid()
        
    x1,state,debt, debtchoice, choices, income = load_data_superfeasible(period, return_income=True)
    types = np.load(EST("em_q_typeff2.npy"))
    # 2 -- Get Education Choices
    
    x1         = x1[choices[:,1]>0,:]
    state      = state[choices[:,1]>0,:]
    debt       = debt[choices[:,1]>0]
    debtchoice = debtchoice[choices[:,1]>0]
    income     = income[choices[:,1]>0]
    types     = types[choices[:,1]>0]
    choices    = choices[choices[:,1]>0,:]
    
    # 3  -- Map Debt to range
    
    debt = debt_range[debt.astype("int")]
    debtchoice = debt_range[debtchoice.astype("int")]
    
    # 4 --- Load CCP path
    
    ccp_path_type1  = load_ccp_path(x1,state,choices,period,1)
    ccp_path_type2  = load_ccp_path(x1,state,choices,period,2)
    
    ccp_path = [ccp_path_type1,ccp_path_type2]
    
    # 5 --- Get Possible Budget
    
    budget = get_budget(income,debt,choices)
    
    # 6 --- Get Possible Continuations
    
    model_grad, model_notgrad, p_grad = get_continuation(x1, state, choices, period, interp_dict)
    
    terminal_data = [model_grad, model_notgrad, p_grad]
    
    return x1, state, types, debt, debtchoice, choices, ccp_path, budget, terminal_data



@njit()
def get_ccp_type(ccp1,ccp2,eductype):
    
    ccp = np.zeros(np.shape(ccp1))
    
    for i in range(np.shape(eductype)[0]):
        
        if eductype[i] == 0:
        
            ccp[i,:] = ccp1[i,:]
        else:
            ccp[i,:] = ccp2[i,:]
            
    return ccp


def _moments_one_group(debtchoice, nmoments=4, eps=0.01, ddof=0):
    """
    debtchoice: 1D array for ONE group
    nmoments: 1..4  (1=mean, 2=+anydebt, 3=+std, 4=+p80)
    eps: fallback value when no one has debt > 0
    ddof: passed to np.std (0 population std, 1 sample std)
    """
    debtchoice = np.asarray(debtchoice).ravel()

    anydebt = np.mean(debtchoice > 0)  # share with positive debt
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

    all_moments = np.array([meandebt, anydebt, stdebt, p80], dtype=float)
    return all_moments[:nmoments]


def get_moments_by_x1(x1, debtchoice, x1_col=0, nmoments=4, levels=None, eps=0.01, ddof=0):
    """
    Computes requested moments of debtchoice by group defined in x1.

    x1: (N,) or (N,K) array. If (N,K), parental income is in column x1_col.
    debtchoice: (N,) array aligned with x1 rows.
    x1_col: which column in x1 contains the group variable (e.g., parental income)
    nmoments: 1..4
    levels: list/array of group values to report (default: sorted unique in data)
    Returns:
        levels_out: array of group values
        M: (G, nmoments) array of moments for each group
    """
    x1 = np.asarray(x1)
    debtchoice = np.asarray(debtchoice).ravel()

    g = x1[:, x1_col] if x1.ndim > 1 else x1.ravel()

    if levels is None:
        levels_out = np.sort(np.unique(g))
    else:
        levels_out = np.asarray(levels)

    M = np.zeros((levels_out.size, nmoments), dtype=float)

    for i, lev in enumerate(levels_out):
        mask = (g == lev)
        M[i, :] = _moments_one_group(debtchoice[mask], nmoments=nmoments, eps=eps, ddof=ddof)

    return M.flatten()


def parinc_design(x1, col=0):
    g = x1[:, col].astype(int)
    X = np.column_stack([
        np.ones(g.size),
        (g == 2).astype(float),
        (g == 3).astype(float),
        (g == 4).astype(float),
    ])
    return X, g


@njit()
def precompute_bounds_indices(previousdebt, state, choices):
    n = previousdebt.shape[0]
    maxdebt = np.empty(n, dtype=np.float64)

    for i in range(n):
        educ_choice = int(choices[i, 1])
        twoy_exp    = int(state[i, 1])
        foury_exp   = int(state[i, 2])

        annual_cap = get_annual_cap_by_stage(educ_choice, twoy_exp, foury_exp)
        lifetime_cap = get_lifetime_cap_by_stage(educ_choice)

        m = previousdebt[i] * (1.0 + r) + annual_cap
        if m > lifetime_cap:
            m = lifetime_cap
        maxdebt[i] = m

    b_idx, max_idx = get_range_number(previousdebt, maxdebt)
    return b_idx.astype(np.int64), max_idx.astype(np.int64)


def fit_budget_shock_multi(data_by_period, periods, max_outer=20, s=20, n_sample=10000):
    """
    Multi-period SMM:
      - Fits all periods in `periods` jointly
      - mu is period-specific (7 params per period)
      - sigma_e common
      - ra_by_parent common (4)
      - debtpen block common (7)

    params = [ mu_block_per_period (7*P), sigma_e (1), ra_levels (4), debtpen_block (7) ]
    moments = stack over periods: [ par(16) + ability(8) ] per period => 24*P moments
    """

    P = len(periods)

    # ---- Build data moments ONCE (stack periods)
    mom_data_list = []
    for p in periods:
        x1, state, types, debt, debtchoice, choices, ccp_path, budget, terminal_data = data_by_period[p]
        mom_par = get_moments_by_x1(x1, debtchoice, x1_col=0, nmoments=4).astype(np.float64)   # 16
        mom_ab  = get_mean_share_by_group(x1, debtchoice, x1_col=1).astype(np.float64)        # 8
        mom_data_list.append(np.concatenate([mom_par, mom_ab]))
    moments_data = np.concatenate(mom_data_list).astype(np.float64)  # length 24*P

    # ---- Initial guess
    # 7 mu params per period: [mu0, mu_par2, mu_par3, mu_par4, mu_ab2, mu_ab3, mu_ab4]
    mu0_guess = 9000.0
    mu_block0 = np.tile(np.array([mu0_guess, 0, 0, 0, 0, 0, 0], dtype=np.float64), P)

    sigma_e0 = np.full(P, 2000.0, dtype=np.float64)   # one per period
    ra0 = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    dp0 = np.array([-1.0, 0, 0, 0, 0, 0, 0], dtype=np.float64)

    params0 = np.concatenate([mu_block0, sigma_e0, ra0, dp0]).astype(np.float64)

    # ---- FINITE bounds (Powell can use them; DE not used here)
    MU_MIN, MU_MAX = -50000.0, 50000.0
    SIGE_MIN, SIGE_MAX = 1.0, 50000.0

    bounds = []
    # mu bounds for all periods
    for _ in range(7 * P):
        bounds.append((MU_MIN, MU_MAX))

    # sigma_e
    for _ in range(P):
        bounds.append((SIGE_MIN, SIGE_MAX))

    # ra levels (4)
    bounds += [(0.1001, 2.9999)] * 4

    # debtpen block (7) must be <=0
    bounds += [(-1e6, 0.0)] * 7

    # ---- Outer loop: resample each period + CRN shocks
    best_x = params0.copy()
    best_fun = 1e30
    tolr = 0.01

    it = 0
    while best_fun > tolr and it < max_outer:
        print(f"\n================ OUTER ITER {it} ================")
        np.random.seed(it)
        reset_terminal_cache()
        # For each period, draw a bootstrap sample and precompute indices + shocks
        sample_by_period = {}

        for p in periods:
            x1, state, types, debt, debtchoice, choices, ccp_path, budget, terminal_data = data_by_period[p]

            # ---- sample (your existing function)
            x1s, states, debts, choices_s, budget_s, evt_ccp_s, _, terminal_s = get_sample(
                x1, state, types, debt, choices, ccp_path, budget, terminal_data
            )

            # ---- shocks (CRN): Z_p is fixed within this outer iteration and period
            rng = np.random.default_rng(12345 + 1000*p + it)  # stable but different across p/it
            Zp = rng.standard_normal((s, x1s.shape[0])).astype(np.float64)

            # ---- bounds indices
            b_idx, max_idx = precompute_bounds_indices(
                debts.astype(np.float64),
                states.astype(np.int64),
                choices_s.astype(np.int64),
            )

            # ---- make contiguous / correct dtypes (important for numba speed)
            budget_s = np.ascontiguousarray(budget_s, dtype=np.float64)
            evt_ccp_s = np.ascontiguousarray(evt_ccp_s, dtype=np.float64)
            x1s = np.ascontiguousarray(x1s)
            states = np.ascontiguousarray(states)
            choices_s = np.ascontiguousarray(choices_s)

            # ---- precompute group membership ONCE for this period sample
            par = x1s[:, 0].astype(np.int64)
            gi = (par - 1).astype(np.int64)  # 0..3
            idx_by_group = [np.where(gi == k)[0] for k in range(4)]

            sample_by_period[p] = {
                "x1": x1s,
                "state": states,
                "choice": choices_s,
                "budget": budget_s,
                "evt_ccp": evt_ccp_s,
                "terminal_data": terminal_s,
                "Z": Zp,
                "b_idx": b_idx,
                "max_idx": max_idx,
                "gi": gi,
                "idx_by_group": idx_by_group,
                "period": p,
            }


        # Objective args for minimize
        args_obj = (moments_data, sample_by_period, periods)

        print("---- Running minimize(..., method='Powell') ----")
        res = minimize(
            minimize_distance_multi,
            x0=best_x,
            args=args_obj,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": 300000, "disp": True},
        )

        print("\n[Powell done] success:", res.success)
        print("[Powell done] message:", res.message)
        print("[Powell done] fun:", float(res.fun))

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




def get_mean_share_by_group(x1, debtchoice, x1_col, eps=0.01):
    """
    Returns flattened array of length 2*G:
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
        share = np.mean(d > 0)
        pos = d[d > 0]
        mean_pos = np.mean(pos) if pos.size > 0 else eps
        if pos.size == 0:
            share = eps
        out[i, 0] = mean_pos
        out[i, 1] = share

    return out.flatten()

def dummies_234(g):
    """
    g: array with values {1,2,3,4}
    returns three float arrays: d2,d3,d4
    """
    g = np.asarray(g).astype(int)
    d2 = (g == 2).astype(np.float64)
    d3 = (g == 3).astype(np.float64)
    d4 = (g == 4).astype(np.float64)
    return d2, d3, d4


def get_relevant_terminal_subset_cached_global(terminal_data, sigma_risk, idx, period):
    """
    Evaluate terminal interpolators only once per (period, interpolator, sigma).
    Reuses the evaluated debt-grid vector across all future objective evaluations.

    terminal_data = [terminal_grad_list, terminal_notgrad_list, p_grad]
      - terminal_grad_list[i] is an interpolator callable for person i (or 0)
      - terminal_notgrad_list[i] is interpolator callable
      - p_grad[i] is graduation probability

    idx: array of individual indices (subset)
    period: int, used to separate caches across periods
    """
    global TERMVAL_CACHE, TERMVAL_HITS, TERMVAL_MISSES, debt_range

    terminal_grad = terminal_data[0]
    terminal_notgrad = terminal_data[1]
    p_grad = terminal_data[2]

    sigk = _sigma_key(sigma_risk, nd=4)
    B = debt_range.size

    out = np.empty((len(idx), B), dtype=np.float64)

    for kk, i in enumerate(idx):
        # ---------- NOT GRAD ----------
        interp_n = terminal_notgrad[i]
        key_n = (int(period), 0, int(id(interp_n)), sigk)

        if key_n in TERMVAL_CACHE:
            TERMVAL_HITS += 1
            notv = TERMVAL_CACHE[key_n]
        else:
            TERMVAL_MISSES += 1
            notv = interp_n((float(sigma_risk), debt_range)).astype(np.float64, copy=False)
            TERMVAL_CACHE[key_n] = notv

        # ---------- GRAD if needed ----------
        if p_grad[i] > 0.0:
            interp_g = terminal_grad[i]
            # Some of your code sets model_grad=0 when not relevant
            if interp_g == 0 or interp_g is None:
                # If graduation is possible but grad model missing, fall back to notgrad
                out[kk, :] = notv
                continue

            key_g = (int(period), 1, int(id(interp_g)), sigk)

            if key_g in TERMVAL_CACHE:
                TERMVAL_HITS += 1
                gradv = TERMVAL_CACHE[key_g]
            else:
                TERMVAL_MISSES += 1
                gradv = interp_g((float(sigma_risk), debt_range)).astype(np.float64, copy=False)
                TERMVAL_CACHE[key_g] = gradv

            out[kk, :] = p_grad[i] * gradv + (1.0 - p_grad[i]) * notv
        else:
            out[kk, :] = notv

    return out

def get_relevant_terminal(terminal_data, sigma_risk):
    
    terminal_grad = terminal_data[0]
    termianl_notgrad = terminal_data[1]
    p_grad = terminal_data[2]
    terminal_i = np.zeros((np.shape(terminal_grad)[0],100))
    
    for i in range(np.shape(terminal_grad)[0]):
        
        if p_grad[i] > 0 : # gradtuation chances
        
            t_grad_i = terminal_grad[i]((float(sigma_risk),debt_range))
            t_notgrad_i = termianl_notgrad[i]((float(sigma_risk),debt_range))
            
            terminal_i[i,:] = p_grad[i]*t_grad_i + (1-p_grad[i])*t_notgrad_i
            
        else:
            terminal_i[i,:] = termianl_notgrad[i]((float(sigma_risk),debt_range))
            
    return terminal_i


@njit()
def get_range_number(b,maxdebt):
    
    value = np.zeros(np.shape(b)[0])
    value2 = np.zeros(np.shape(b)[0])
    
    for i in range(np.shape(b)[0]):
        
        diff = (debt_range - b[i])**2
            
        value[i] = np.argmin(diff)
        
        diff2 = (debt_range - maxdebt[i])**2
            
        value2[i] = np.argmin(diff2)
        
    return value, value2


@njit()
def minimum_debt_maxdebt(u, b, maxdebt):
    """
    Enforce:
      - cannot choose debt < previousdebt (monotone cumulative borrowing)
      - cannot choose debt > maxdebt (cap)
    b and maxdebt are in DOLLARS, mapped to nearest debt_range index.
    """
    b_idx, max_idx = get_range_number(b, maxdebt)

    for i in range(u.shape[0]):
        # lower bound: no repayment via choice (monotone)
        u[i, :int(b_idx[i])] = -100000000

        # upper bound: cannot exceed cap
        u[i, int(max_idx[i]) + 1:] = -100000000

    return u
        

@njit()
def get_annual_cap_by_stage(educ_choice, twoy_exp, foury_exp,
                            ):
    """
    educ_choice: 1=2yr, 2=4yr, 3=grad (your choices[:,1])
    twoy_exp, foury_exp: experience counters (assumed 0-based like your data)
    returns annual_cap in dollars
    """
    cap = 0.0

    if (educ_choice == 1): 
        # 4-year: year1 if exp==0, year2 if exp==1, else year3+
        if twoy_exp <= 0:
            cap = 8391
        elif twoy_exp == 1:
            cap = 9309
        else:
            cap = 12581

    elif (educ_choice == 2) : 
        # 4-year: year1 if exp==0, year2 if exp==1, else year3+
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
    """
    Lifetime cap depends on whether you're in UG or GRAD.
    UG: UG_AGG
    GRAD: TOTAL_AGG (UG+GRAD combined)
    """
    if educ_choice == 3:
        return  150000
    else:
        return  70786


def get_debt_rules(c, vjt, previousdebt, state, choices,
                   ):
    """
    Implements the SAME cap logic as the cleaned data:
      - consumption floor: c>=2000
      - annual cap depends on *education stage/experience*
      - lifetime cap depends on UG vs GRAD
      - enforce on the *cumulative debt decision* (monotone)
    """
    # consumption feasibility
    vjt[c < 2000] = -100000000

    # Build maxdebt per person
    n = previousdebt.shape[0]
    maxdebt = np.empty(n, dtype=np.float64)

    # ---- IMPORTANT: you must map these indices to your state layout ----
    # Here I assume:
    #   state[:,1] = twoyear_exp   (0-based)
    #   state[:,2] = fouryear_exp  (0-based)
    # and the schooling choice is choices[:,1] in {1,2,3}
    for i in range(n):
        educ_choice = int(choices[i, 1])
        twoy_exp    = int(state[i, 1])
        foury_exp   = int(state[i, 2])

        annual_cap = get_annual_cap_by_stage(
            educ_choice, twoy_exp, foury_exp,
        )

        lifetime_cap = get_lifetime_cap_by_stage(
            educ_choice
        )

        # max feasible debt is previous + annual cap, but cannot exceed lifetime cap
        m = previousdebt[i]*(1+r) + annual_cap
        if m > lifetime_cap:
            m = lifetime_cap
        maxdebt[i] = m

    # apply lower+upper bounds in the discrete grid
    vjt = minimum_debt_maxdebt(vjt, previousdebt, maxdebt)
    return vjt


@njit()
def get_debt(debt_range,debt):
    
    debtvalues = np.zeros(np.shape(debt)[0])
    
    for i in range(np.shape(debt)[0]):
        
        debtvalues[i] = debt_range[int(debt[i])]
        
    return debtvalues

@njit()
def get_power_utility(sigma_u,c):
    
    c = np.maximum(c,2000)
    
    return 0.1*((0.00001*c)**(1-sigma_u)/(1-sigma_u))


def get_optimal_debt_new(x1,budget,evt,e,previousdebt, sigma_risk):
    
    global debt_range
     
    
    c = (budget[...,None]+e[...,None]+debt_range)
        
    u = get_power_utility(sigma_risk,c)

    vjt = u + beta*evt
       
    vjt = get_debt_rules(c,vjt,previousdebt)
        
    debttheory = np.argmax(vjt,axis=1)
    
    debttheory[vjt[:,-1]== -100000000] = 99
    
    debtvalue = get_debt(debt_range,debttheory)
    
    moments = get_moments_by_x1(x1,debtvalue, nmoments=4)

    
    return moments

def get_optimal_debt_by_group(x1, budget, evt, e, previousdebt, g,
                              sigma_risk_levels, debt_penalty_levels, state, choice,period):
    global debt_range

    c = budget[:, None] + e[:, None] + debt_range[None, :]

    u = np.empty_like(c)
    debt_ind = (debt_range > 0).astype(np.float64)  # (100,)

    for lev, sig in sigma_risk_levels.items():
        mask = (g == lev)
        if np.any(mask):
            u[mask, :] = get_power_utility(sig, c[mask, :])

            # >>> ADD DEBT PENALTY HERE <<<
            # debt_penalty_levels[lev] should be negative if agents dislike debt
            u[mask, :] = u[mask, :] + debt_penalty_levels[lev] * debt_ind[None, :]

    vjt = u + beta**(T-period) * evt
    vjt = get_debt_rules(c, vjt, previousdebt,state, choice)

    debttheory = np.argmax(vjt, axis=1)
    #debttheory[vjt[:, -1] == -100000000] = 99

    debtvalue = get_debt(debt_range, debttheory)

    return get_moments_by_x1(x1, debtvalue, nmoments=4)


def parinc_groups(x1, col=0, levels=None):
    g = x1[:, col].astype(int)
    if levels is None:
        levels = np.sort(np.unique(g))  # e.g. [1,2,3,4]
    gi = np.searchsorted(levels, g)     # maps to 0..G-1
    return g, gi, levels


EVAL_COUNTER = 0

def print_moment_progress(m_data_flat, m_sim_flat, levels, nmoments=2, decimals=2, title=None):
    """
    Pretty print moment fit by group.
    Assumes flattened moments are ordered by group, then by moment.
    """

    levels = np.asarray(levels)
    G = len(levels)

    m_data = np.asarray(m_data_flat, dtype=float).reshape(G, nmoments)
    m_sim  = np.asarray(m_sim_flat,  dtype=float).reshape(G, nmoments)
    m_diff = m_sim - m_data

    moment_names_all = [
        "MeanDebt>0",
        "Pr(debt>0)",
        "StdDebt>0",
        "P80Debt>0"
    ]
    moment_names = moment_names_all[:nmoments]

    if title:
        print(title)

    print(f"{'Group':>5} | " + "  ".join([f"{name:^32}" for name in moment_names]))
    print(f"{'':>5} | " + "  ".join([f"{'data':>8} {'sim':>8} {'diff':>8}" for _ in moment_names]))

    for gi, lev in enumerate(levels):
        row = [f"{int(lev):>5} |"]
        for j in range(nmoments):
            row.append(
                f"{m_data[gi,j]:>8.{decimals}f} {m_sim[gi,j]:>8.{decimals}f} {m_diff[gi,j]:>8.{decimals}f}"
            )
        print(" ".join(row))


# Cache for evaluated terminal vectors on the debt grid
# Key: (period, gradflag, interp_id, sigma_key) -> np.ndarray shape (B,)
TERMVAL_CACHE = {}

# Optional: track cache stats
TERMVAL_HITS = 0
TERMVAL_MISSES = 0

def _sigma_key(sig, nd=4):
    """
    Convert continuous sigma to a stable cache key.
    Rounding is crucial: Powell will try close sigmas repeatedly.
    """
    return float(np.round(float(sig), nd))


def reset_terminal_cache():
    """Call this when you resample / start a new outer iteration."""
    global TERMVAL_CACHE, TERMVAL_HITS, TERMVAL_MISSES
    TERMVAL_CACHE = {}
    TERMVAL_HITS = 0
    TERMVAL_MISSES = 0


def terminal_cache_stats():
    global TERMVAL_HITS, TERMVAL_MISSES
    tot = TERMVAL_HITS + TERMVAL_MISSES
    hitrate = TERMVAL_HITS / tot if tot > 0 else 0.0
    return {"hits": TERMVAL_HITS, "misses": TERMVAL_MISSES, "hitrate": hitrate, "size": len(TERMVAL_CACHE)}
        
        
@njit(parallel=True, fastmath=True)
def solve_one_draw_debt_idx_terminal_only(
    budget, e, debt_grid,
    sigma_i, debtpen_i,
    ccp_path_row, terminal_row,
    b_idx, max_idx,
    beta_term,                 # = beta**(T - p)
    c_floor=2000.0,
    fallback_idx=99
):
    """
    Correct Bellman:
        v = u(c) + ccp_path + beta_term * terminal
    where ccp_path is *not* discounted here.
    """
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

            # power utility with stable log case
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


@njit(parallel=True, fastmath=True)
def solve_many_draws_debt_idx(
    budget, Z, mu_i, sigma_e,
    debt_grid, sigma_i, debtpen_i, evt,
    b_idx, max_idx, beta_pow,
    c_floor=2000.0
):
    """
    Returns debt_idx of shape (s, n) for all draws at once.
    Z: (s,n) standard normals
    """
    s = Z.shape[0]
    n = budget.shape[0]
    B = debt_grid.size
    out = np.empty((s, n), dtype=np.int64)

    for i in prange(n):
        lo = b_idx[i]
        hi = max_idx[i]
        if lo < 0: lo = 0
        if hi >= B: hi = B - 1
        if hi < lo: hi = lo

        sig = sigma_i[i]
        pen = debtpen_i[i]

        for k in range(s):
            e = mu_i[i] + sigma_e * Z[k, i]

            best_v = -1e30
            best_j = lo
            found_feasible = False

            for j in range(lo, hi + 1):
                c = budget[i] + e + debt_grid[j]
                if c < c_floor:
                    continue
                found_feasible = True

                # utility
                if abs(sig - 1.0) < 1e-8:
                    u = 0.1 * np.log(0.00001 * c)
                else:
                    u = 0.1 * ((0.00001 * c) ** (1.0 - sig)) / (1.0 - sig)

                if debt_grid[j] > 0.0:
                    u += pen

                v = u + beta_pow * evt[i, j]
                if v > best_v:
                    best_v = v
                    best_j = j

            # your “if no feasible consumption, take legal max” rule:
            out[k, i] = hi if not found_feasible else best_j

    return out

def minimize_distance_multi(params, moments_data, sample_by_period, periods):
    """
    Stacked multi-period objective.

    params structure:
      - mu blocks: 7*P  (period-specific)
          [mu0, mu_par2, mu_par3, mu_par4, mu_ab2, mu_ab3, mu_ab4] for each period in `periods` order
      - sigma_e: 1 (common)
      - ra_levels: 4 (common, by parental income quartile 1..4)
      - debtpen block: 7 (common)
          [dp0, dp_par2, dp_par3, dp_par4, dp_ab2, dp_ab3, dp_ab4]
    """

    global EVAL_COUNTER
    EVAL_COUNTER += 1

    P = len(periods)

    # ---- unpack
    mu_blocks = params[:7*P].reshape(P, 7)

    sigma_e_vec = params[7*P : 7*P + P].astype(np.float64)   # <-- period-specific
    
    ra_start = 7*P + P
    ra_levels = params[ra_start : ra_start + 4]
    
    dp_start = ra_start + 4
    dp_block = params[dp_start : dp_start + 7]
    dp0, dp_par2, dp_par3, dp_par4, dp_ab2, dp_ab3, dp_ab4 = dp_block

    # ---- build simulated moments stacked across periods
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
        mu0, mu_par2, mu_par3, mu_par4, mu_ab2, mu_ab3, mu_ab4 = mu_blocks[pi, :]
        sigma_e_p = float(sigma_e_vec[pi])
        # groups from x1
        par = x1[:, 0].astype(int)   # 1..4
        ab  = x1[:, 1].astype(int)   # 1..4
        par2, par3, par4 = dummies_234(par)
        ab2_, ab3_, ab4_ = dummies_234(ab)

        # per-individual mu and debt dislike (debt dislike common across periods)
        mu_i = (mu0
                + mu_par2*par2 + mu_par3*par3 + mu_par4*par4
                + mu_ab2*ab2_  + mu_ab3*ab3_  + mu_ab4*ab4_).astype(np.float64)

        debtpen_i = (dp0
                     + dp_par2*par2 + dp_par3*par3 + dp_par4*par4
                     + dp_ab2*ab2_  + dp_ab3*ab3_  + dp_ab4*ab4_).astype(np.float64)

        # risk aversion by parental income quartile (common)
        gi = pack["gi"]
        sigma_i = ra_levels[gi].astype(np.float64)

        # ---- terminal depends on ra_levels (so evaluate per group)
        n = budget.shape[0]
        B = debt_range.size
        terminal = np.zeros((n, B), dtype=np.float64)

        # ---- terminal depends on ra_levels: evaluate per group using GLOBAL cache
        n = budget.shape[0]
        B = debt_range.size
        terminal = np.zeros((n, B), dtype=np.float64)

        idx_by_group = pack["idx_by_group"]
        period_here = pack["period"]

        for k in range(4):
            idx = idx_by_group[k]
            if idx.size > 0:
                terminal[idx, :] = get_relevant_terminal_subset_cached_global(
                    terminal_data=terminal_data,
                    sigma_risk=float(ra_levels[k]),
                    idx=idx,
                    period=period_here
                )

       
        beta_term = float(beta ** (T - p))


        s = Z.shape[0]
        sim_draws = np.zeros((24, s), dtype=np.float64)

        for k in range(s):
            e = mu_i + sigma_e_p * Z[k, :]

            debt_idx = solve_one_draw_debt_idx_terminal_only(
                budget=budget,
                e=e.astype(np.float64),
                debt_grid=debt_range,
                sigma_i=sigma_i,
                debtpen_i=debtpen_i,
                ccp_path_row=evt_ccp,     # <-- pass CCP path separately
                terminal_row=terminal,    # <-- pass terminal separately
                b_idx=b_idx,
                max_idx=max_idx,
                beta_term=beta_term,      # <-- discount only terminal
                c_floor=2000.0,
                fallback_idx=99
            )

            debtvalue = debt_range[debt_idx]
            mom_par_k = get_moments_by_x1(x1, debtvalue, x1_col=0, nmoments=4)
            mom_ab_k  = get_mean_share_by_group(x1, debtvalue, x1_col=1)
            sim_draws[:, k] = np.concatenate([mom_par_k, mom_ab_k])

        sim_p = sim_draws.mean(axis=1)  # (24,)
        sim_list.append(sim_p)

        # period contribution (optional)
        data_p = moments_data[24*pi : 24*(pi+1)]
        denom_p = np.maximum(np.abs(data_p), 1e-6)
        per_period_loss.append(float(np.sum(((data_p - sim_p) / denom_p) ** 2)))

    simumoments = np.concatenate(sim_list)  # (24*P,)

    # ---- loss
    denom = np.maximum(np.abs(moments_data), 1e-6)
    loss = float(np.sum(((moments_data - simumoments) / denom) ** 2))

    # ---- printing every 10 evals: show summary + pretty tables for ONE period
    if (EVAL_COUNTER % 10) == 0:
        print("\n" + "="*100)
        print(f"[eval {EVAL_COUNTER}] total loss={loss:.6f}")
        print("sigma_e first 5 periods:", np.round(sigma_e_vec[:min(5,P)], 2))
        print(f"ra_levels (par 1..4) = {np.round(ra_levels, 2)}")
        print(f"debtpen block = {np.round(dp_block, 2)}")
        print("per-period loss:", {int(periods[i]): round(per_period_loss[i], 6) for i in range(P)})

        # show nice tables for first fitted period
        # show nice tables for the first 7 fitted periods (or fewer)
        n_show = min(7, P)

        for show_pi in range(n_show):
            show_p = periods[show_pi]
            data_show = moments_data[24*show_pi : 24*(show_pi+1)]
            sim_show  = simumoments[24*show_pi : 24*(show_pi+1)]

            mom_par_data = data_show[:16]
            mom_ab_data  = data_show[16:]
            mom_par_sim  = sim_show[:16]
            mom_ab_sim   = sim_show[16:]

            print(f"\n--- Pretty moment tables (period={show_p}) ---")

            print_moment_progress(
                mom_par_data, mom_par_sim,
                levels=np.array([1,2,3,4]),
                nmoments=4,
                decimals=2,
                title="ParInc moments (data vs simulated)"
            )

            print_moment_progress(
                mom_ab_data, mom_ab_sim,
                levels=np.array([1,2,3,4]),
                nmoments=2,
                decimals=2,
                title="Ability moments (data vs simulated)"
            )
        print("="*100 + "\n")

    return loss





def get_sample(x1, state, types, debt, choices,ccp_path, budget, terminal):
    
    
    n = 10000    # sample size 
    
    random_indices = np.random.choice(np.shape(x1)[0],  
                                  size=n,  
                                  replace=True) 
  
    x1sample         = x1[random_indices,:]
    statesample      = state[random_indices,:]
    debtsample       = debt[random_indices]
    choicessample    = choices[random_indices,:]
    budgetsample     = budget[random_indices]
    
    ccp_path_type1 = ccp_path[0]
    ccp_path_type2 = ccp_path[1]
    
    ccp_1_sample = ccp_path_type1[random_indices,:]
    ccp_2_sample = ccp_path_type2[random_indices,:]
    
    terminal1 = terminal[0]
    terminal2 = terminal[1]
    terminal3 = terminal[2]

    terminal1_sample = [terminal1[i] for i in random_indices]
    terminal2_sample = [terminal2[i] for i in random_indices]
    terminal3_sample = [terminal3[i] for i in random_indices]
    
    terminal_sample = [terminal1_sample, terminal2_sample, terminal3_sample]
    
    # draw education type
    #eductype  = np.random.binomial(1, sample[:,25], np.shape(sample)[0])
    eductype  = types - 1
    
    evt = get_ccp_type(ccp_1_sample,ccp_2_sample,eductype)

    
    e = np.random.normal(loc=0, scale=1, size=n)
    
    return x1sample, statesample, debtsample, choicessample, budgetsample,  evt, e,terminal_sample





def estimation():
    """
    Runs the full estimation and saves parameters for later use in the solution/simulation code.

    Outputs (in EST folder):
      - risk_aversion.npy
      - budgetshock_params.npy   (dict saved as .npy)
      - budgetshock_snapshot.npz (convenience bundle)
    """
    # 1) Load / build terminal interpolator dictionary (cached)
    interp_dict = get_interp_dict_cached(force_rebuild=False)

    # 2) Periods to fit (your existing choice)
    periods_to_fit = list(range(1, T))   # 1..9 if T=10

    # 3) Load fundamentals for each period (your existing loader)
    data_by_period = {}
    for p in periods_to_fit:
        print(f"\n[load fundamentals] period={p}")
        data_by_period[p] = load_fundamentals(p, interp_dict, clear_data=False)

    # 4) Fit model (your existing routine)
    best_x, best_fun = fit_budget_shock_multi(
        data_by_period=data_by_period,
        periods=periods_to_fit,
        max_outer=20,
        s=20,
        n_sample=10000
    )

    # 5) Save parameters for simulation/solution code
    # NOTE: requires the helper you added:
    #   save_budgetshock_params(best_x, periods, K_mu=7, filename_prefix="")
    save_budgetshock_params(best_x, periods_to_fit, K_mu=7, filename_prefix="")

    return best_x, best_fun
    

    

debt_range = get_debt_range()

estimation()
    
