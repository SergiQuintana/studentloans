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


os.chdir(r"C:\Users\S.Quintana\Dropbox\PhD\Projects\Papers\1_financial_constraints\Code\2026_01\2_model")



from numba import njit
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
                               get_educ_level)


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


# =============================================================================
# def get_ccps(period):
#     
#     
#     vjt = get_vjt_static_debt() # returns an Nx(debtrange) matrix, with the CCP prediction for each individual state x, at every possible debt period. 
#     
#     base = -1  # home production
#         
#     vjt = vjt - np.repeat(vjt[:,base][...,np.newaxis],np.shape(vjt)[1]).reshape(np.shape(vjt))
#     
#     # now get the exponent
#         
#     vjt = np.exp(vjt)
#         
#     # Get the sum
# 
#     vjt = vjt[:,base] / np.repeat(np.sum(vjt,axis=1)[...,None],np.shape(vjt)[1]).reshape(np.shape(vjt))
#     
#     return vjt
# =============================================================================





# =============================================================================
# def simulate_all_periods(params):
#     
#     
#     for period in periods:
#         
#         cont = get_continuation(period, x)
#         
#         bnew = get_loans(x,cont)
#         
#         xnew = move_state(x,bnew)
# 
#     
#     pass
# =============================================================================




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
    types = np.load(f"{path_estimates}/em_q_typeff2.npy")
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



def fit_budget_shock(x1, state, debt, debtchoice, choices, ccp_path, budget, terminal_data, period):

    moments = get_moments_by_x1(x1, debtchoice, nmoments=4)  # 12 moments if 4 groups

    # 9 parameters: mu1..mu4, sigma_e, ra1..ra4
    params0 = np.array([
        5000.0, 5000.0, 5000.0, 5000.0,  # mu1..mu4
        100.0,                                   # sigma_e
        2.0, 2.0, 2.0, 2.0,                        # ra1..ra4 in [0.1,3]
        -2.0, -2.0, -2.0, -2.0         # debtpen1..debtpen4 (should be <= 0 if “dislike”)
    ])

    # Bounds: mu unbounded, sigma_e > 0, ra in [0.1,3]
    bounds = [
        (None, None), (None, None), (None, None), (None, None),  # mu1..mu4
        (1.0, None),                                             # sigma_e >= 1
        (0.1001, 2.9999), (0.1001, 2.9999), (0.1001, 2.9999), (0.1001, 2.9999),  # ra
        (-1e6, 0.0), (-1e6, 0.0), (-1e6, 0.0), (-1e6, 0.0)       # debtpen <= 0
    ]
    tolr = 0.01
    newerror = 1e30
    it = 0

    while newerror > tolr:
        print(it)
        np.random.seed(it)

        x1sample, statesample, debtsample, choicessample, budgetsample, evt_ccp, _, terminal_sample = get_sample(
            x1, state, debt, choices, ccp_path, budget, terminal_data
        )
        
        
# =============================================================================
#         res = minimize(
#             minimize_distance,
#             params0,
#             args=(moments, budgetsample, terminal_sample, debtsample, evt_ccp, x1sample, statesample, choicessample, period),
#             method="Nelder-Mead",          # IMPORTANT: bounded + derivative-free
#             bounds=bounds,
#             options={"disp": True, "maxiter": 200}
#         )
#         
# =============================================================================
        res = differential_evolution(
            minimize_distance,
            bounds=bounds,
            args=(moments, budgetsample, terminal_sample, debtsample, evt_ccp, x1sample, statesample, choicessample, period),
            strategy="best1bin",          # IMPORTANT: bounded + derivative-free
            disp=True
        )

        if res.fun < newerror:
            paramshat = res.x
            funhat = res.fun

        newerror = res.fun
        params0 = paramshat 
        it += 1
        if it == 10: tolr = 0.03
        if it == 15: tolr = 0.04
        if it == 20: break

    print(paramshat, funhat)
    return paramshat, funhat

def get_relevant_terminal_subset(terminal_data, sigma_risk, idx):
    terminal_grad = terminal_data[0]
    terminal_notgrad = terminal_data[1]
    p_grad = terminal_data[2]

    out = np.zeros((len(idx), debt_range.size))

    for k, i in enumerate(idx):
        if p_grad[i] > 0:
            t_grad_i = terminal_grad[i]((float(sigma_risk), debt_range))
            t_not_i  = terminal_notgrad[i]((float(sigma_risk), debt_range))
            out[k, :] = p_grad[i] * t_grad_i + (1 - p_grad[i]) * t_not_i
        else:
            out[k, :] = terminal_notgrad[i]((float(sigma_risk), debt_range))

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

    if (educ_choice == 2) | (educ_choice == 1): 
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

def print_moment_progress(m_data_flat, m_sim_flat, levels, nmoments=3, decimals=2, title=None):
    """
    Pretty print moment fit by parental-income group.
    Assumes flattened moments are ordered by group, then by moment.
    """
    G = len(levels)
    m_data = np.asarray(m_data_flat).reshape(G, nmoments)
    m_sim  = np.asarray(m_sim_flat).reshape(G, nmoments)
    m_diff = m_sim - m_data

    # Names consistent with your _moments_one_group ordering when nmoments=3:
    # [mean_pos, anydebt, std_pos]
    moment_names_all = [
        "MeanDebt>0",
        "Pr(debt>0)",
        "StdDebt>0",
        "P80Debt>0"
    ]
    moment_names = moment_names_all[:nmoments]

    # Header
    if title:
        print(title)
    print(f"{'ParInc':>5} | " + "  ".join([f"{name:^32}" for name in moment_names]))
    print(f"{'':>5} | " + "  ".join([f"{'data':>8} {'sim':>8} {'diff':>8}" for _ in moment_names]))

    # Rows
    fmt = f"{{:>{decimals+6}.{decimals}f}}"
    for gi, lev in enumerate(levels):
        row = [f"{int(lev):>5} |"]
        for j in range(nmoments):
            row.append(
                f"{m_data[gi,j]:>8.{decimals}f} {m_sim[gi,j]:>8.{decimals}f} {m_diff[gi,j]:>8.{decimals}f}"
            )
        print(" ".join(row))

def minimize_distance(params, moments, budget, terminal_data, previousdebt, ccp_path, x1, state, choice,period):
    """
    params = [
        mu1, mu2, mu3, mu4,         # 0:4
        sigma_e,                    # 4
        ra1, ra2, ra3, ra4,         # 5:9  (bounded in [0.1,3])
        debtpen1, debtpen2, debtpen3, debtpen4  # 9:13  (typically <=0)
    ]

    moments: flattened (G*nmoments,) from data, here nmoments=4 so 16 if G=4.
    """

    global EVAL_COUNTER
    EVAL_COUNTER += 1

    # --- unpack parameters
    mu_levels   = params[0:4]    # mu1..mu4
    sigma_e     = params[4]      # scalar std dev (bounded >= 1 in optimizer)
    ra_levels   = params[5:9]    # ra1..ra4 (bounded in [0.1,3])
    debt_levels = params[9:13]   # debt penalty by group (often <= 0)

    # --- parental income grouping
    g, gi, levels = parinc_groups(x1, col=0)  # g are original values; gi in 0..G-1

    # individual-specific mean for normal drawA
    mu_i = mu_levels[gi]

    # dicts keyed by the actual parental-income labels in `levels`
    sigma_risk_levels = {int(levels[k]): float(ra_levels[k]) for k in range(len(levels))}
    debt_penalty_levels = {int(levels[k]): float(debt_levels[k]) for k in range(len(levels))}

    # --- terminal continuation depends on risk aversion: build by group ONCE per evaluation
    n = budget.shape[0]
    terminal = np.zeros((n, debt_range.size))

    for k, lev in enumerate(levels):
        idx = np.where(gi == k)[0]
        if idx.size > 0:
            terminal[idx, :] = get_relevant_terminal_subset(terminal_data, ra_levels[k], idx)

    # total continuation (terminal + CCP path)
    evt = terminal + ccp_path

    # --- simulation (CRN: fixed standard normals each evaluation)
    s = 20
    rng = np.random.default_rng(12345)
    Z = rng.standard_normal((s, n))

    simumoments = np.zeros((moments.shape[0], s))

    for k in range(s):
        # NORMAL budget shock draw
        enew = mu_i + sigma_e * Z[k, :]

        # optimal debt moments given group-specific RA + group-specific debt penalty
        simumoments[:, k] = get_optimal_debt_by_group(
            x1=x1,
            budget=budget,
            evt=evt,
            e=enew,
            previousdebt=previousdebt,
            g=g,
            sigma_risk_levels=sigma_risk_levels,
            debt_penalty_levels=debt_penalty_levels,
            state = state,
            choice = choice,
            period = period
        )

    simumoments = np.mean(simumoments, axis=1)

    # --- loss
    cost = (moments - simumoments) / moments
    loss = float(np.sum(cost ** 2))

    # --- readable output
    print("\n" + "="*100)
    print(f"[eval {EVAL_COUNTER}] loss={loss:.6f}")
    print(f"sigma_e={sigma_e:.2f}  (var={sigma_e**2:.2f})")
    print(f"parinc levels: {list(map(int, levels))}")
    print(f"mu(levels):    {np.round(mu_levels, 2)}")
    print(f"ra(levels):    {np.round(ra_levels, 2)}")
    print(f"debtpen(levels): {np.round(debt_levels, 2)}  (should be <=0 if 'dislike debt')")
    print_moment_progress(
        moments, simumoments,
        levels=levels,
        nmoments=4,
        decimals=2,
        title="Moments by parental-income group (data vs simulated)"
    )
    print("="*100 + "\n")

    return loss




def get_sample(x1, state, debt, choices,ccp_path, budget, terminal):
    
    
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
    eductype  = np.random.binomial(1, 0.5, np.shape(x1sample)[0])
    
    evt = get_ccp_type(ccp_1_sample,ccp_2_sample,eductype)

    
    e = np.random.normal(loc=0, scale=1, size=n)
    
    return x1sample, statesample, debtsample, choicessample, budgetsample,  evt, e,terminal_sample





def estimation():
    
    interp_dict = get_interp_dict_cached(force_rebuild=False)

    x1, state, debt, debtchoice, choices, ccp_path, budget, terminal_data = load_fundamentals(period, interp_dict, clear_data = False)

    fit_budget_shock(x1, state, debt, debtchoice, choices, ccp_path, budget, terminal_data, period)
    

    

debt_range = get_debt_range()

period = 1

estimation()
    
#ccp_path = load_fundamentals(1)
