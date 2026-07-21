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
from scipy.optimize import minimize

from model_em_algorithm import (load_all_arrays_feasible,
                               load_data_superfeasible,
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
    
    budget = (grants + transfers + wage[:,0] -(1+r)*debt_range[np.array(debt,dtype="int")]-tuition_agents(0,j))
    
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
    
    
    
def load_fundamentals(period, interp_dict):
    
    """
    This function performs the execution of the code. 
    1st. Load data
    2nd. Get Education Choices only
    3rd. Build CCP path
    4th. 
    """
    
    global debt_range
    
    # 1 --- Load Data
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
    

def fit_budget_shock(x1, state, debt, debtchoice, choices,ccp_path, budget, terminal_data):
    
    
    # First Get Moments
    moments = get_moments_by_x1(x1,debtchoice, nmoments=2)
    
    # guess a mean and a sigma
    
    #params = [1000,1000]  
    params = [-12000,12000,2]
    
    b1 = (-np.inf,np.inf)
    b2 = (1,np.inf)
    b3 = (0.1,3)
    bounds = (b1,b2,b3)
    
    
    tolr = 0.01
    newerror = 100000000000000000000000000000000000000
    it = 0
    while newerror > tolr:
        print(it) 
        # change random state
        np.random.seed(it)
        # Simulate the samples        
        x1sample, statesample, debtsample, choicessample, budgetsample,  evt, e, terminal_sample = get_sample(x1, state, debt, choices,ccp_path, budget, terminal_data)
        # Minimize
        res = minimize(minimize_distance,params,args=(moments,budgetsample,terminal_sample,debt,evt,x1sample),
                       bounds=bounds,method='Nelder-Mead',options={'disp':True, })
        if res.fun < newerror:
            
            paramshat = res.x
            funhat = res.fun
            
        newerror = res.fun
        
        it = it+1
        
        if it == 10:
            
            tolr = 0.03
            
        elif it == 15:
            
            tolr = 0.04
        
        if it == 20:
            newerror = 0

    print(paramshat,funhat)
    return paramshat, funhat



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

def get_debt_rules(c,vjt,previousdebt):
    
    global debt_range
    
    
    vjt[c<2000] = -100000000
    
    maxdebt = np.minimum(debt_range[-1],previousdebt + 22300)
        
    
    #vjt = minimum_debt_maxdebt(vjt,previousdebt,maxdebt)

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
    
    moments = get_moments_by_x1(x1,debtvalue, nmoments=2)

    
    return moments

def minimize_distance(params,moments,budget,terminal_data,previousdebt,ccp_path,x1):
    
    """
    This function fits the mean debt of the data with a normal distribution
    that takes parameters: 
        
        - constant, variance
        
    It simulates 10k individuals from the state distribution. 
    
    """
    
    print("Current Parameters", params)
    
    # Initialize parameters
    mu = params[0]
    sigma = params[1]
    sigma_risk = params[2]
    # Get the termianl continuation for the sigma of interest
    terminal = get_relevant_terminal(terminal_data,sigma_risk)
    
    # Sum the corresponding CCP path
    
    evt = terminal + ccp_path
        
    s = 200 
    
    simumoments = np.zeros((np.shape(moments)[0],s))
    
    for i in range(s):

        enew = np.random.normal(loc=mu, scale=sigma, size=np.shape(budget)[0]) 
        #enew =  np.random.lognormal(mean=mu, sigma=sigma, size=1000) 
        #enew = enew*0 
        simumoments[:,i] = get_optimal_debt_new(x1,budget,evt,enew,previousdebt,sigma_risk)
    
    simumoments = np.mean(simumoments,axis=1)
    print(simumoments)
    cost = (moments - simumoments)/moments
    
    #print("mean:",meandebt,meandebtsimu)
    #print("var:",vardebt,vardebtsimu)
    #print("anydebt:",anydebt,anydebtsimu)
    #print("p80:",p80simu,p80) 
    #print("Current loss:", np.sum(moments**2) )
    print(np.sum(cost**2))
    return np.sum(cost**2)



def get_sample(x1, state, debt, choices,ccp_path, budget, terminal):
    
    
    s = 100     # number of samples to generate 
    n = 1000    # sample size 
    
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

    x1, state, debt, debtchoice, choices, ccp_path, budget, terminal_data = load_fundamentals(period, interp_dict)

    fit_budget_shock(x1, state, debt, debtchoice, choices, ccp_path, budget, terminal_data)
    

    

debt_range = get_debt_range()

period = 1

estimation()
    
#ccp_path = load_fundamentals(1)
