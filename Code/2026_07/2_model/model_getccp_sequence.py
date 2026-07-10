# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 14:49:38 2026

@author: S.Quintana


This program estimates the model sequnce of CCPs until the terminal period,
but without including the terminal period. This will allow to search for risk
aversion coefficients when estimating the student loan distribution. 

The solution algorithm is analogous to that on model_solution_em.py

"""

import numpy as np
import multiprocessing 

from model_solution_em import (save_npz_here,
                               get_x2,
                               move_state_grad,
                               get_debt_range)

import model_solution_em as ms


from config import DIR

path_estimates  = DIR["MODEL_ESTIMATES"]
path = DIR["MODEL_REALDATA"]
path_out = DIR["MODEL_OUTPUT"]

T = 10
beta = 0.98


# =============================================================================
# def move_state_ccp(x1,x1_new,x2,j,ccps):
#     
#     """
#     This function moves the current state given your choice 
#     """
#     
#     if ((x2[1] >=1) & (j[1] == 1) & (x2[4] == 0)) | ((x2[2]>=3) & (j[1] == 2) & (x2[5] == 0))  | (j[1] == 3) :
#         # The choice could induce a graduation state. For this reason, take the expectation.
#         grad_x2  =  move_state_grad(x2,j,period,grad=1)
#         notgrad_x2 = move_state_grad(x2,j,period)
# 
#         ccp_grad =  ccps[f"evt_t{period+1}_{x1}_{grad_x2}"]
#         ccp_nograd = ccps[f"evt_t{period+1}_{x1}_{notgrad_x2}"]
#         p_grad = probability_graduation(x1_new,x2,j)
#         expected_ccp = p_grad*ccp_grad + (1-p_grad)*ccp_nograd
#         
#     else:
#         
#         return ccps[f"evt_t{period+1}_{x1}_{x2}"]
#     
#     
# =============================================================================

    
    

def get_ccp_continuation(x1,x2,b,period,ccps,evt):
    
    """
    This function computes what would be the continuation value of an individual
    that next period will have state (x2,period), without accounting for the terminal
    period, since this will me looped on the estimation to find the risk aversion
    coefficient. 
    
    Everything is using home production as a base category. Therefore, I will
    use the CCP of home production and I will move the state as if home production
    was the chioce
    """
    
    ccp_home =  ccps[f"ccp_t{period}_[{x1}]_{x2}"]
    
    homeproduction = np.array([0,0,0]) # impose agents choose homeproduction
    
    x2new = move_state_grad(x2,homeproduction,period)
    
    if period == 9:
        
        cont = 0
    else:
    
        cont = evt[f"evt_ccp_sequence_t{period+1}_{x1}_{x2new}"]
    
    evtnew = -np.log(ccp_home)  + beta*cont  
    
    return evtnew


def loop_over_x2(x1,b,period,ccps,evt):
    
    # Get set of states
    
    x2_set = get_x2(period)
    
    # Initialize values to store
    
    results_ccp = []
    names_ccp = []
    for x2 in x2_set:
        x2 = x2.astype("int")
        
        evt_new = get_ccp_continuation(x1,x2,b,period,ccps,evt)
        
        results_ccp.append(evt_new)
        names_ccp.append(f"evt_ccp_sequence_t{period}_{x1}_{x2.astype(int)}")
        
    return results_ccp, names_ccp
 
def get_ccp_sequence(i,x1,b,em_type):

    """This function computes CCP for choosing home production for every period and
    state using the auxiliary model."""
    
    print(f"Individual {x1[i].astype('int')}")
    
    evtnext = 0 
    
    # Generate x1
    inv = x1[i,:]
    
    results_ccp = []
    names_ccp = []
    
    for period in range(T-1,0,-1):
        
        # Load CCPs
        
        ccps = np.load(f"{path_out}/ccp/{period}/ccp_t{period}_[{inv}]_em{em_type}.npz")
        
        print(period)
        
        # Redefine continuation value
        
        evt = evtnext
        
        # Loop over all  x2 states
        
        results_ccp, names_ccp = loop_over_x2(inv,b,period,ccps,evt)

        # Save results
        
        save_npz_here(f"{path_out}/evt_ccp/{period}/evt_ccp_sequence_t{period}_{inv}_em{em_type}.npz",    names_ccp, results_ccp, compressed=True)
        
        # Obtain new continuation
        evtnext = dict(zip(names_ccp,results_ccp))
    
    return evtnext    

    

#-----------------------------------------------------------------------------#

# =============================================================================
# 
# if __name__ == '__main__':
# 
#     print("Estimation of the CCPs across periods")
# 
#     debt_range = ms.debt_range
#     
#     pool_obj = multiprocessing.Pool(processes=60)     
#     args = [(i,ms.invariant_states,ms.debt_range,1)
#             for i in range(np.shape(ms.invariant_states)[0])]
#     args.extend(
#         (i,ms.invariant_states,ms.debt_range,2)
#         for i in range(np.shape(ms.invariant_states)[0])
#     )
#      
#     results = pool_obj.starmap(get_ccp_sequence, args, chunksize=1)
#     pool_obj.close()
# 
# =============================================================================











