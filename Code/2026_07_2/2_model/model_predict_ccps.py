# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 16:13:32 2025

@author: S.Quintana
"""

import numpy as np
import multiprocessing 

from model_solution_em import (get_all_g,
                               save_npz_here,
                               get_possible_choices,
                               get_x1_new,
                               get_x2,
                               build_param_g)

from model_em_algorithm import (get_amount_educ)
import model_solution_em as ms


from config import DIR

path_estimates  = DIR["MODEL_ESTIMATES"]
path_out = DIR["MODEL_OUTPUT"]

T = 10
fields=8
occupations = 8
n_param_g_x1 = 9*(1+fields+occupations+1)
n_param_g_work = 2 + 2*fields+occupations+2
n_param_g_change = 1+ fields  + occupations
n_param_g_educ = get_amount_educ()
n_param_g_period = np.maximum((1+fields+occupations)*(T-1-1),0) + np.maximum((T-5-1),0)
n_param_g_period_work = np.maximum((T-1-1)*3,0)  + np.maximum((T-5-1)*2,0)
n_param_g_first = 1 + 4 + 1
n_param_g_exp = (fields-1)*4*6
n_param_g_types = 2 # one effect on associate and one on 4y schools
total_n = n_param_g_x1 + n_param_g_change + n_param_g_work + n_param_g_educ + n_param_g_period + n_param_g_period_work + n_param_g_first + n_param_g_exp + n_param_g_types
total_n_multi = total_n


def get_vjt_static(utility_parameters,x1,x1_new,x2,Jx,period,b):
    
    g = get_all_g(utility_parameters,x1,x1_new,x2,Jx,period)
    
    vjt_static = g + 0*b[...,np.newaxis]
    
    return vjt_static
    
    


def get_all_choices(x1,x1_new,x2,b,period,utility_parameters):
    
    # For each state x1 and x2 loop over all possible choices.
    
    Jx  = get_possible_choices(x2)
    # now loop over all the entrances in Jx
    # Create a matrix that will store the values
    
    all_vjt = get_vjt_static(utility_parameters,x1,x1_new,x2,Jx,period,b)
    
    return all_vjt
 
def get_all_ccps(i,x1,b,utility_parameters,em_type):

    """This function computes CCP for choosing home production for every period and
    state using the auxiliary model."""
    
    
    print(f"Individual {x1[i].astype('int')}")
    for period in range(T,0,-1):
        
        print(period)
        # Get set of states
        
        x2_set = get_x2(period)
        
        
        results_ccp = []
        names_ccp = []
        
        for x2 in x2_set:
            x2 = x2.astype("int")
            inv = x1[i,:]
            inv = inv[..., None].T
            # Generate x1
            x1_new = get_x1_new(inv[0])

            all_vjt = get_all_choices(x1,x1_new,x2,b,period,utility_parameters)
            
            base = -1  # set home production as the base category
            
            all_ccps = np.exp(all_vjt[:,base]) / (np.exp(all_vjt).sum(axis=1))
            results_ccp.append(all_ccps)
            names_ccp.append(f"ccp_t{period}_{inv}_{x2.astype(int)}")


        save_npz_here(f"{path_out}/ccp/{period}/ccp_t{period}_{inv}_em{em_type}.npz",    names_ccp, results_ccp, compressed=True)


def load_utility_parameters(em_type):
    
    global total_n_multi
    
    params = np.load(f"{path_estimates}/param_em_latest.npy")
    
    params = params[:total_n_multi]
    
    return build_param_g(em_type,params)



#-----------------------------------------------------------------------------#
# Try the functions


if __name__ == '__main__':

    print("Estimation of the CCPs across periods")
    utility_parameters1 = load_utility_parameters(1)
    utility_parameters2 = load_utility_parameters(2)
    debt_range = ms.debt_range
    
    pool_obj = multiprocessing.Pool(processes=10)     
    args = [(i,ms.invariant_states,ms.debt_range,utility_parameters1,1)
            for i in range(np.shape(ms.invariant_states)[0])]
    args.extend(
        (i,ms.invariant_states,ms.debt_range,utility_parameters2,2)
        for i in range(np.shape(ms.invariant_states)[0])
    )
     
    results = pool_obj.starmap(get_all_ccps, args, chunksize=1)
    pool_obj.close()












