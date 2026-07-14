# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:59:02 2023

@author: Sergi
"""

"""This code performs the Aguirregabiria and Mira 2001 estimation algorithm
to estimate the parameters of the model. The key idea of their algorithm is to guess
the CCPs , estimate the parameters, and use the model to create
new CCPs. Update until convergence. The steps in my case will be:

1. Estimate the model with CCPs. 
2. Update parameters and estimate all the continuation values. 
3. Repeat until convergence in parameters.
        
"""
import numpy as np
import scipy
import os
import time
import pandas as pd
#import numba
from numba import njit,jit,prange
import multiprocessing 
import multiprocessing as mp
from scipy.optimize import minimize
from os import environ
#print(environ['MKL_NUM_THREADS'])
#import mkl
import warnings
from pathlib import Path
import sys
import builtins
from numba.core.errors import NumbaPendingDeprecationWarning,NumbaDeprecationWarning

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    return builtins.print(*args, **kwargs)

warnings.simplefilter('ignore',category=NumbaDeprecationWarning)
warnings.simplefilter('ignore',category=NumbaPendingDeprecationWarning)

#mkl.set_num_threads(1)
#--------------------------#
# Manage Working Directoires

from config import DIR, OUT, ENSURE_DEFAULT_TREE, path_estimates, STATES

ENSURE_DEFAULT_TREE(T=10)

#-----------------------------------------------------------------------------------------------#
# Import model modules after setting working directory 
THIS_DIR = Path(__file__).resolve().parent

# Add this directory to the import path (so Python can find the modules)
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

# Now import without any chdir()
import model_solution_em as ms
import model_em_algorithm as me
import solve_many_continuations as mcf
import model_predict_ccps as mccp
import model_getccp_sequence as mgs
from model_interpolate_terminal import build_interp_cache 
from model_fitloans_dynamic import estimate_budget_shock 
from latent_types import TYPE_IDS
#-----------------------------------------------------------------------------------------------#

mu = 0
gamma = 0.57721
beta = 0.98
r = 0.05
T = 10

fields=8
occupations = 8 
n_param_g_x1 = 9*(1+fields+occupations+1)
n_param_g_work = 2 + 2*fields+occupations+2
n_param_g_change = 1+ fields  + occupations
n_param_g_educ = ms.get_amount_educ()
n_param_g_period = np.maximum((1+fields+occupations)*(T-1-1),0) + np.maximum((T-5-1),0)
n_param_g_period_work = np.maximum(3*(T-1-1),0) + np.maximum(2*(T-5-1),0)
n_param_g_first = 1 + 4 + 1
n_param_g_exp = (fields-1)*4*6
n_param_g_type = 2
total_n = n_param_g_x1 + n_param_g_change + n_param_g_work + n_param_g_educ + n_param_g_period + n_param_g_period_work + n_param_g_first + n_param_g_exp +  n_param_g_type

# param wage now depends on:
# ability
# parental income
# sex
# ethnicity
# experience before
# years 4y
# years 2y
# grad 4y
# grad 2y
# experience after
# major at graduation



solve_model = False
solve_continuation = False 
solve_qs = False
get_budget = False

if __name__ == '__main__':
    
    # Simulate all model states
    print("Simulating all model states...")
    ms.simulate_all_states(11)
    
    debt_range = ms.get_debt_range()


    # First check if continuation value should be estimated
    if solve_continuation == True:
    
        # Solve several continuation values
        mcf.solve_many_continuations()
    
        # Now interpolate them (build + store cache to disk)
        build_interp_cache(
            pathout=DIR["MODEL_OUTPUT"],
            pathcont=DIR["MODEL_CONTINUATION_FINAL"],
            debt_grid=debt_range,
            fields=fields,
            force_rebuild=True,   # set False after you have the cache
            verbose=False
        )
    
    else:
        print("Using already generated continuation values")
    
    
    # Now solve the model and simulate the data
    if solve_model == True:
    
        print("Solving the model...")
        solution_mode = 1
        ccp_real = []
        models = []
        utility_parameters = ms.load_param_g(real=0)
        ms.simulate_all_states(T)
    
        pool_obj = mp.Pool(
            processes=60,
            initializer=ms.init_worker_queue,
        )
    
        args = [
            (i, ms.invariant_states, debt_range, debt_range, ccp_real, sigma_u,
             utility_parameters, models, solution_mode, em_type)
            for i in range(np.shape(ms.invariant_states)[0])
            for em_type in range(1, 3)
        ]
    
        pool_obj = multiprocessing.Pool(60)
        results = pool_obj.starmap(ms.get_all_evt, args, chunksize=1)
        pool_obj.close()
        print("Simulating choices...")
    
    else:
        print("Using already generated data")
    
    
    if solve_qs == True:
    
        print("Solve for posterior probabilities EM-algorithm")
    
        me.get_feasible()
        me.get_feasible_pubid()
        me.get_superfeasible()
        me.get_x_g_superfeasible()
        pi, xnew, q = me.perform_em(2)
        q = np.load(f"{path_estimates}/em_q_typeff2.npy")
    
        print("Predict CCPs after auxiliary model")
    
        auxiliary_ccp_parameters = {
            type_id: mccp.load_utility_parameters(type_id)
            for type_id in TYPE_IDS
        }
        debt_range = ms.debt_range
    
        pool_obj = multiprocessing.Pool(processes=10)
        args = [
            (
                i,
                ms.invariant_states,
                ms.debt_range,
                auxiliary_ccp_parameters[type_id],
                type_id,
            )
            for type_id in TYPE_IDS
            for i in range(np.shape(ms.invariant_states)[0])
        ]
    
        results = pool_obj.starmap(mccp.get_all_ccps, args, chunksize=1)
        pool_obj.close()
    
    else:
        print("Using previously generated EM weights")
        q = np.load(f"{path_estimates}/em_q_typeff2.npy")
    
    
    # Generate the amount of aguirregabiria_mira iterations.
    iterations = 30
    solution_mode = 0
    
    for it in range(iterations):
    
        print("Iteration number : ", it)
    
        # Check which ccp estimation to use:
        if it != 0:
            ccp_real = 1
            utility_parameters1 = ms.build_param_g(1, x0)
            utility_parameters2 = ms.build_param_g(2, x0)
        else:
            ccp_real = 0
            # Initial guess:
            x0 = np.zeros((total_n,))
            x0 = np.load(f"{path_estimates}/param_g.npy")
            utility_parameters1 = ms.build_param_g(1, x0)
            utility_parameters2 = ms.build_param_g(2, x0)
    
            # Prepare the get_x matrix
            me.get_feasible()
            me.get_feasible_pubid()
            me.get_x_g_superfeasible()
    
        if get_budget == True:
            #--------------------------------------#
            # Build the sequence to estimate the budget shock
            print("Building Sequence to Estimate Budget Shock")
        
            pool_obj = multiprocessing.Pool(processes=60)
            args = [
                (i, ms.invariant_states, ms.debt_range, 1)
                for i in range(np.shape(ms.invariant_states)[0])
            ]
            args.extend(
                (i, ms.invariant_states, ms.debt_range, 2)
                for i in range(np.shape(ms.invariant_states)[0])
            )
        
            results = pool_obj.starmap(mgs.get_ccp_sequence, args, chunksize=1)
            pool_obj.close()
        
            #---------------------------------------#
            print("Estimation of the Budget Shock")
            
            estimate_budget_shock()
            ms.reload_budgetshock_params()
            
        #--------------------------------------#
        # Solve the model with ccps
        print("Solve the model and the ccps")
        models = 0
        conter = 0
        maxdebt = False
        pool_obj = multiprocessing.Pool(processes=60, initializer=ms.reload_budgetshock_params)
    
        args = [
            (i, ms.invariant_states, debt_range, debt_range, ccp_real,
             utility_parameters1, models, solution_mode, conter, 1, maxdebt)
            for i in range(np.shape(ms.invariant_states)[0])
        ]
        args.extend(
            (i, ms.invariant_states, debt_range, debt_range, ccp_real,
             utility_parameters2, models, solution_mode, conter, 2, maxdebt)
            for i in range(np.shape(ms.invariant_states)[0])
        )
    
        results = pool_obj.starmap(ms.get_all_evt, args, chunksize=1)
        pool_obj.close()
    
        #--------------------------------------#
        # Now prepare the data and evaluate the likelihood
        print("Preparing the data for the estimation")
    
        pool_obj = multiprocessing.Pool(T - 1)
        args = [period for period in range(1, T, 1)]
        results = pool_obj.map(me.prepare_vjt_feasible, args)
        pool_obj.close()
    
        choices_all, vjt_all_type1, vjt_all_type2, x1_new, choices_array_all, x_change, x_educ, x_first2, x_first4, x_firstgrad, x_exp = me.load_all_arrays_feasible()
    
        # Now optimize the likelihood
        print("Evaluating the likelihood")
        res = minimize(
            me.likelihood,
            x0,
            args=(choices_all, vjt_all_type1, vjt_all_type2, x1_new, choices_array_all,
                  x_change, x_educ, x_first2, x_first4, x_firstgrad, x_exp, q),
            jac=True,
            options={'disp': True},
            callback=me.store
        )
    
        # Get the updated parameters
        param_g = res.x
    
        # Update the initial guess:
        x0 = res.x * 0.7 + x0 * 0.3
    
        # Store the results
        np.save(f"{path_estimates}/estimates_it{it}_sigma_est.npy", param_g)
        np.save(f"{path_estimates}/param_g.npy", param_g)
    
        # Store standard errors
        se = np.diag(np.sqrt(res.hess_inv))
        np.save(f"{path_estimates}/se_it{it}_sigma_est.npy", se)
    
        # Store likelihood evaluation
        np.save(f"{path_estimates}/likelihood_it{it}_sigma_est.npy", np.array(res.fun))
            
        
        
            
        
        


