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


from config import DIR, OUT, INP, FUN, RDATA, CONT, EST,LIK
pathfunctions = DIR["MODEL_FUNCOEF"]
path = DIR["MODEL_REALDATA"]
pathout = DIR["MODEL_OUTPUT"]
path_estimates  = DIR["MODEL_ESTIMATES"]
pathcont = DIR["MODEL_CONTINUATION_FINAL"]

T = 10
r = 0.05
beta = 0.98
debt_range = get_debt_range()




#==============================================================================
# Store inter_dict in cache  memory

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
        evti = np.load(f"{pathout}/evt_ccp/{period+1}/evt_ccp_sequence_t{period+1}_{x1i}_em{em_type}.npz")
        evt_grad =  evti[f"evt_ccp_sequence_t{period+1}_{x1i}_{grad_x2}"]
        evt_nograd = evti[f"evt_ccp_sequence_t{period+1}_{x1i}_{notgrad_x2}"]
        x1_new = get_x1_new(x1i)
        p_grad = probability_graduation(x1_new,x2i,ji)
        vt = p_grad*evt_grad + (1-p_grad)*evt_nograd


    else: 
        evti = np.load(f"{pathout}/evt_ccp/{period+1}/evt_ccp_sequence_t{period+1}_{x1i}_em{em_type}.npz")
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
    #types     = types[choices[:,1]>0]
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
    
    return x1, state, debt, debtchoice, choices, ccp_path, budget, terminal_data



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


def fit_budget_shock(x1, state, debt, debtchoice, choices, ccp_path, budget, terminal_data, period):
    """
    DEBUG VERSION (no differential_evolution):
      - Draw one sample per outer iteration
      - Precompute Z, bounds indices once per outer iteration
      - Use scipy.optimize.minimize with method="Powell" (bounded, derivative-free)
    """

    # ---- data moments (flattened by group)
    mom_par = get_moments_by_x1(x1, debtchoice, x1_col=0, nmoments=4).astype(np.float64)   # 16 moments
    mom_ab  = get_mean_share_by_group(x1, debtchoice, x1_col=1).astype(np.float64)        # 8 moments
    moments = np.concatenate([mom_par, mom_ab])

    # ---- initial guess
    params0 = np.array([
            # --- mu block (7)
            9000.0,   # mu0
            0.0, 0.0, 0.0,   # mu_par2, mu_par3, mu_par4
            0.0, 0.0, 0.0,   # mu_ab2,  mu_ab3,  mu_ab4
        
            # --- sigma_e (1)
            2000.0,
        
            # --- ra by parental income (4)
            2.0, 2.0, 2.0, 2.0,
        
            # --- debtpen block (7)  (<=0)
            -1.0,   # dp0
            0.0, 0.0, 0.0,   # dp_par2..4 (typically <=0, but let optimizer decide with bounds)
            0.0, 0.0, 0.0    # dp_ab2..4
        ], dtype=np.float64)

    # ---- FINITE bounds (safe for all methods)
    MU_MIN, MU_MAX = -50000.0, 50000.0
    SIGE_MIN, SIGE_MAX = 1.0, 50000.0
    
    bounds = [
        # --- mu block (7)
        (MU_MIN, MU_MAX), (MU_MIN, MU_MAX), (MU_MIN, MU_MAX), (MU_MIN, MU_MAX),
        (MU_MIN, MU_MAX), (MU_MIN, MU_MAX), (MU_MIN, MU_MAX),
    
        # --- sigma_e (1)
        (SIGE_MIN, SIGE_MAX),
    
        # --- ra (4)
        (0.1001, 2.9999), (0.1001, 2.9999), (0.1001, 2.9999), (0.1001, 2.9999),
    
        # --- debtpen block (7)  (must be <=0)
        (-1e6, 0.0), (-1e6, 0.0), (-1e6, 0.0), (-1e6, 0.0),
        (-1e6, 0.0), (-1e6, 0.0), (-1e6, 0.0)
    ]

    tolr = 0.01
    best_fun = 1e30
    best_x = params0.copy()

    it = 0
    while best_fun > tolr:
        print(f"\n================ OUTER ITER {it} ================")
        np.random.seed(it)

        # ---- sample once per outer iteration
        x1sample, statesample, debtsample, debtchoice_sample, choicessample, budgetsample, evt_ccp, _, terminal_sample = get_sample(
            x1, state, debt, debtchoice, choices, ccp_path, budget, terminal_data
        )

        # ---- fixed shocks for this outer loop (CRN)
        s = 20
        rng = np.random.default_rng(12345)
        Z = rng.standard_normal((s, x1sample.shape[0])).astype(np.float64)

        # ---- precompute bounds indices once for this sample
        b_idx, max_idx = precompute_bounds_indices(
            debtsample.astype(np.float64),
            statesample.astype(np.int64),
            choicessample.astype(np.int64),
        )
        
        # Get the types riht
        budgetsample = np.ascontiguousarray(budgetsample, dtype=np.float64)
        evt_ccp      = np.ascontiguousarray(evt_ccp, dtype=np.float64)
        x1sample     = np.ascontiguousarray(x1sample)

        # ---- objective args
        args_obj = (
            moments,
            budgetsample.astype(np.float64),
            terminal_sample,
            debtsample.astype(np.float64),             # previousdebt
            debtchoice_sample.astype(np.float64),      # <-- DATA outcome on same sample
            evt_ccp.astype(np.float64),
            x1sample,
            statesample,
            choicessample,
            period,
            Z,
            b_idx,
            max_idx
        )

        # ---- simple callback to see progress
        cb_counter = {"k": 0}
        def cb(xk):
            cb_counter["k"] += 1
            if cb_counter["k"] % 5 == 0:
                print(f"[Powell callback] iter={cb_counter['k']}  x0..4={np.round(xk[:5],2)}")

        print("---- Running minimize(..., method='Powell') ----")
        res = minimize(
            minimize_distance,
            x0=best_x,                 # start from best so far
            args=args_obj,
            method="Powell",
            bounds=bounds,
            callback=cb,
            options={
                "maxiter": 30,          # DEBUG: keep small so you see output quickly
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
    print("best_x:", best_x)
    print("best_fun:", best_fun)
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
            out[k, :] = p_grad[i]*gradv + (1-p_grad[i])*notv
        else:
            out[k, :] = notv

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
        
        
@njit(parallel=True, fastmath=True)
def solve_one_draw_debt_idx(
    budget, e, debt_grid, sigma_i, debtpen_i,
    ccp_path_row, terminal_row,   # <-- split inputs
    b_idx, max_idx,
    beta_term,                    # <-- beta^(T-period), applies ONLY to terminal
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

            # build conditional vf
            v = u + ccp_path_row[i, j] + beta_term * terminal_row[i, j]

            if v > best_v:
                best_v = v
                best_j = j

        out[i] = hi if not found_feasible else best_j

    return out


def compute_maxdebt(previousdebt, state, choice, r=0.05):
    """
    Person-specific max feasible cumulative debt in DOLLARS:
        maxdebt = min( previousdebt*(1+r) + annual_cap(educ,exp), lifetime_cap(educ) )
    Uses YOUR get_annual_cap_by_stage and get_lifetime_cap_by_stage.
    """
    previousdebt = np.asarray(previousdebt, dtype=np.float64)
    n = previousdebt.shape[0]
    maxdebt = np.empty(n, dtype=np.float64)

    for i in range(n):
        educ_choice = int(choice[i, 1])
        twoy_exp    = int(state[i, 1])
        foury_exp   = int(state[i, 2])

        annual_cap   = float(get_annual_cap_by_stage(educ_choice, twoy_exp, foury_exp))
        lifetime_cap = float(get_lifetime_cap_by_stage(educ_choice))

        m = previousdebt[i] * (1.0 + r) + annual_cap
        if m > lifetime_cap:
            m = lifetime_cap
        maxdebt[i] = m

    return maxdebt

def closest_grid_idx(grid, x):
    """Index of closest grid point to each x (debug; vectorized O(nB))."""
    grid = np.asarray(grid, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).ravel()
    diff = (grid[None, :] - x[:, None])**2
    return np.argmin(diff, axis=1).astype(np.int64)

def quantiles_2(x):
    """Rounded quantiles [min,p10,med,p90,max] to 2 decimals."""
    x = np.asarray(x, dtype=np.float64).ravel()
    q = np.quantile(x, [0.0, 0.1, 0.5, 0.9, 1.0])
    return np.round(q, 2)



def minimize_distance(params, moments, budget, terminal_data, previousdebt, debtchoice_data, ccp_path,
                      x1, state, choice, period, Z, b_idx, max_idx):

    global EVAL_COUNTER
    EVAL_COUNTER += 1

    # ---- unpack params
    mu0 = params[0]
    mu_par2, mu_par3, mu_par4 = params[1], params[2], params[3]
    mu_ab2,  mu_ab3,  mu_ab4  = params[4], params[5], params[6]

    sigma_e   = float(params[7])
    ra_levels = params[8:12]   # by parental income quartile (1..4)
    dp0 = params[12]
    dp_par2, dp_par3, dp_par4 = params[13], params[14], params[15]
    dp_ab2,  dp_ab3,  dp_ab4  = params[16], params[17], params[18]

    # groups
    par = x1[:, 0].astype(int)   # 1..4
    ab  = x1[:, 1].astype(int)   # 1..4
    par2, par3, par4 = dummies_234(par)
    ab2,  ab3,  ab4  = dummies_234(ab)

    # per-individual mu and debt dislike
    mu_i = (mu0
            + mu_par2*par2 + mu_par3*par3 + mu_par4*par4
            + mu_ab2*ab2   + mu_ab3*ab3   + mu_ab4*ab4).astype(np.float64)

    debtpen_i = (dp0
                 + dp_par2*par2 + dp_par3*par3 + dp_par4*par4
                 + dp_ab2*ab2   + dp_ab3*ab3   + dp_ab4*ab4).astype(np.float64)

    # risk aversion by parental income quartile
    gi = (par - 1).astype(int)   # 0..3
    sigma_i = ra_levels[gi].astype(np.float64)

    # ---- terminal depends on ra_levels
    n = budget.shape[0]
    B = debt_range.size
    terminal = np.zeros((n, B), dtype=np.float64)

    cache = {}  # cache per evaluation
    for k in range(4):
        idx = np.where(gi == k)[0]
        if idx.size > 0:
            terminal[idx, :] = get_relevant_terminal_subset_cached(
                terminal_data, ra_levels[k], idx, cache
            )

    beta_term = float(beta ** (T - period))

    # ---- simulate moments across draws using fixed Z
    s = Z.shape[0]
    sim_draws = np.zeros((moments.shape[0], s), dtype=np.float64)

    for k in range(s):
        e = mu_i + sigma_e * Z[k, :]
        debt_idx = solve_one_draw_debt_idx(
            budget=budget,
            e=e.astype(np.float64),
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
        debt_sim = debt_range[debt_idx]

        mom_par_k = get_moments_by_x1(x1, debt_sim, x1_col=0, nmoments=4)
        mom_ab_k  = get_mean_share_by_group(x1, debt_sim, x1_col=1)
        sim_draws[:, k] = np.concatenate([mom_par_k, mom_ab_k])

    sim_mom = sim_draws.mean(axis=1)

    # ---- loss
    denom = np.maximum(np.abs(moments), 1e-6)
    loss = float(np.sum(((moments - sim_mom) / denom) ** 2))

    # ---- PRINT DEBUG every 10 evals
    if (EVAL_COUNTER % 10) == 0:
        print("\n" + "="*100)
        print(f"[eval {EVAL_COUNTER}] period={period}  loss={loss:.6f}")
        print(f"sigma_e={sigma_e:.2f}")
        print("ra_levels(par 1..4)=", np.round(ra_levels, 2))
        print("debtpen block [dp0,dp_par2,dp_par3,dp_par4,dp_ab2,dp_ab3,dp_ab4] =",
              np.round(np.array([dp0,dp_par2,dp_par3,dp_par4,dp_ab2,dp_ab3,dp_ab4]), 2))

        # moment tables
        mom_par_data = moments[:16]
        mom_ab_data  = moments[16:]
        mom_par_sim  = sim_mom[:16]
        mom_ab_sim   = sim_mom[16:]

        print_moment_progress(
            mom_par_data, mom_par_sim,
            levels=np.array([1,2,3,4]),
            nmoments=4,
            decimals=2,
            title="ParInc moments (DATA vs SIM)"
        )
        print_moment_progress(
            mom_ab_data, mom_ab_sim,
            levels=np.array([1,2,3,4]),
            nmoments=2,
            decimals=2,
            title="Ability moments (DATA vs SIM)"
        )

        # --------- CAP / NEAR-CAP DIAGNOSTICS (DATA vs SIM) ----------
        maxdebt = compute_maxdebt(previousdebt, state, choice, r=r)

        # tolerance: half of median grid step
        dstep = np.diff(debt_range)
        step_med = float(np.median(dstep)) if dstep.size > 0 else 0.0
        tol = 0.5 * step_med

        # Pick ONE draw for interpretable comparison (k=0)
        k0 = 0
        e0 = mu_i + sigma_e * Z[k0, :]
        debt_idx0 = solve_one_draw_debt_idx(
            budget=budget,
            e=e0.astype(np.float64),
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
        debt_sim0 = debt_range[debt_idx0]

        # two definitions of "at cap"
        max_idx_grid  = closest_grid_idx(debt_range, maxdebt)
        data_idx_grid = closest_grid_idx(debt_range, debtchoice_data)
        sim_idx_grid  = closest_grid_idx(debt_range, debt_sim0)

        atcap_data_idx = (data_idx_grid == max_idx_grid)
        atcap_sim_idx  = (sim_idx_grid  == max_idx_grid)

        atcap_data_tol = (np.abs(debtchoice_data - maxdebt) <= tol)
        atcap_sim_tol  = (np.abs(debt_sim0      - maxdebt) <= tol)

        gap_data = maxdebt - debtchoice_data
        gap_sim  = maxdebt - debt_sim0

        # print summary (rounded to 2 decimals)
        print("\n---- CAP diagnostics (DATA vs SIM, draw k=0) ----")
        print(f"grid step median={step_med:.2f}  tol(half-step)={tol:.2f}")
        print(f"max(grid)={float(np.max(debt_range)):.2f}  max(maxdebt_i)={float(np.max(maxdebt)):.2f}")

        print("\nShares at/near cap:")
        print(f"  DATA at cap (idx-def)  = {atcap_data_idx.mean():.2f}")
        print(f"  SIM  at cap (idx-def)  = {atcap_sim_idx.mean():.2f}")
        print(f"  DATA near cap (|gap|<=tol) = {atcap_data_tol.mean():.2f}")
        print(f"  SIM  near cap (|gap|<=tol) = {atcap_sim_tol.mean():.2f}")

        print("\nMax borrowed:")
        print(f"  DATA max(debt) = {float(np.max(debtchoice_data)):.2f}")
        print(f"  SIM  max(debt) = {float(np.max(debt_sim0)):.2f}")

        print("\nGap to cap (cap - debt) quantiles [min,p10,med,p90,max]:")
        print("  DATA:", quantiles_2(gap_data))
        print("  SIM :", quantiles_2(gap_sim))

        print("="*100 + "\n")

    return loss






def get_sample(x1, state, debt, debtchoice, choices, ccp_path, budget, terminal):
    """
    Returns a bootstrap sample AND returns debtchoice_sample (DATA outcome)
    aligned with the sampled individuals.
    """
    n = 10000

    random_indices = np.random.choice(np.shape(x1)[0], size=n, replace=True)

    x1sample          = x1[random_indices, :]
    statesample       = state[random_indices, :]
    debtsample        = debt[random_indices]         # previous debt (dollars)
    debtchoice_sample = debtchoice[random_indices]   # DATA debt choice (dollars)
    choicessample     = choices[random_indices, :]
    budgetsample      = budget[random_indices]

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
    terminal_sample  = [terminal1_sample, terminal2_sample, terminal3_sample]

    # draw education type
    eductype = np.random.binomial(1, 0.5, np.shape(x1sample)[0])
    evt = get_ccp_type(ccp_1_sample, ccp_2_sample, eductype)

    e = np.random.normal(loc=0, scale=1, size=n)

    return (x1sample, statesample, debtsample, debtchoice_sample,
            choicessample, budgetsample, evt, e, terminal_sample)





def estimation():
    
    interp_dict = get_interp_dict_cached(force_rebuild=False)

    x1, state, debt, debtchoice, choices, ccp_path, budget, terminal_data = load_fundamentals(period, interp_dict, clear_data = False)

    fit_budget_shock(x1, state, debt, debtchoice, choices, ccp_path, budget, terminal_data, period)
    

    

debt_range = get_debt_range()

period = 2

estimation()
    
#ccp_path = load_fundamentals(1)
