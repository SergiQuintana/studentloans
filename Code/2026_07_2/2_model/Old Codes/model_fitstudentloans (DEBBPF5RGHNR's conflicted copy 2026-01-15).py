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
import os

os.chdir(r"C:\Users\S.Quintana\Dropbox\PhD\Projects\Papers\1_financial_constraints\Code\2026_01\2_model")



from model_em_algorithm import (load_all_arrays_feasible,
                               load_data_superfeasible,
                               get_vjt_static,
                               get_data_superfeasible)

from model_solution_em import (move_state_grad,
                               get_x1_new,
                               probability_graduation,
                               get_debt_range)

from model_simulation_em import (tuition_agents)

from model_interpolate_terminal import build_interpolator_dictionary

from config import DIR, OUT, INP, FUN, RDATA, CONT, EST,LIK
pathfunctions = DIR["MODEL_FUNCOEF"]
path = DIR["MODEL_REALDATA"]
pathout = DIR["MODEL_OUTPUT"]
path_estimates  = DIR["MODEL_ESTIMATES"]

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



def get_continuation(period,x):
    
    """
    This function builds the part that does not depend on the epsilon for
    the student loan decision sum ln CCPS + beta VTerminal.
    
    It reads CCPs and VTerminal that depend on any loans value and
    fixs it to the current level of loans. 
    """
    
    pass
    


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
        p_grad = probability_graduation(x1_new,x2,j)
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
    
    
    
def load_fundamentals(period):
    
    """
    This function performs the execution of the code. 
    1st. Load data
    2nd. Get Education Choices only
    3rd. Build CCP path
    4th. 
    """
    
    # 1 --- Load Data
    x1,state,debt,choices,income = load_data_superfeasible(period, return_income=True)
    types = np.load(f"{path_estimates}/em_q_typeff2.npy")
    # 2 -- Get Education Choices
    
    x1 = x1[choices[:,1]>0,:]
    state = state[choices[:,1]>0,:]
    debt = debt[choices[:,1]>0]
    income = income[choices[:,1]>0]
    #types = types[choices[:,1]>0]
    choices = choices[choices[:,1]>0,:]
    
    # 3 --- Load CCP path
    
    ccp_path  = load_ccp_path(x1,state,choices,period,1)
    
    # 4 --- Get Possible Budget
    
    budget = get_budget(income,debt,choices)
    
    return x1, state, debt, choices, ccp_path, budget



def estimation():
    
    # 1st. Load Data
    
    x1, state, debt, choices, ccp_path, budget = load_fundamentals(period)
    
    # 2nd.  Build Interpolated Continuation
    interp_dict, meta_dict, missing, context = build_interpolator_dictionary(
        pathcont=pathcont,
        debt_grid=debt_grid,
        fields=8,
        lastschool_horizon=None,
        sex_filters=None,
        race_filters=None,
        verbose=False
    )
    
    # 3rd. Generate Individuals and obtain their types
    
    types  = np.random.binomial(1, q[:,1], np.shape(q)[0]) +1
    
    
    

period = 1
    
ccp_path = load_fundamentals(1)
