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
from numba.core.errors import NumbaPendingDeprecationWarning,NumbaDeprecationWarning

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

warnings.simplefilter('ignore',category=NumbaDeprecationWarning)
warnings.simplefilter('ignore',category=NumbaPendingDeprecationWarning)

#mkl.set_num_threads(1)
#--------------------------#
# Manage Working Directoires

from config import DIR, OUT, EST, ENSURE_DEFAULT_TREE, path_estimates, STATES

ENSURE_DEFAULT_TREE(T=10)

#-----------------------------------------------------------------------------------------------#
# Import model modules after setting working directory 
THIS_DIR = Path(__file__).resolve().parent

# Add this directory to the import path (so Python can find the modules)
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

# Now import without any chdir()
import model_solution_em as ms
import model_solution_fast as msf
import model_em_algorithm as me
import solve_many_continuations as mcf
import model_predict_ccps as mccp
import model_getccp_sequence as mgs
import model_getccp_sequence_fast as mgsf
from model_interpolate_terminal import build_interp_cache 
import model_fitloans_dynamic as mfd
from model_fitloans_dynamic import estimate_budget_shock_all_education
from latent_types import TYPE_IDS, load_em_posteriors, validate_q
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
solve_initial_ccps = False
get_budget = True

# Optional fast Bellman solver (Phase 1 of Agents_Readme/Tasks/SPEED_PLAN_FINAL.md).
# Exact drop-in for ms.get_all_evt: same signature, same artifacts, same numbers.
# Promoted to default 2026-07-23 after test_fast_solver_equivalence.py passed
# on the server: 20/20 tasks (types 1,2,5,9,16 x states 0,20,45,63, ccp_real 0
# and 1) bitwise-identical, ~2.1x faster. Set False to fall back to the
# original solver.
USE_FAST_SOLVER = True

# Fast CCP-sequence pipeline (2026-07-24, researcher-approved, flag-off;
# see Agents_Readme/Tasks/ESTIMATION_SPEED_ANALYSIS_2026_07_24.md and the
# EMIT_CCP_SEQUENCE note in model_solution_fast.py). When True:
#   * iteration 0 builds the terminal-free sequence with the vectorized
#     builder (model_getccp_sequence_fast), one dense matrix per period;
#   * every Bellman solve with updated CCPs (iterations >= 1) emits the
#     next iteration's sequence during its own backward pass, so the
#     standalone sequence stage is skipped from iteration 1 onward
#     (iteration 1 correctly reuses iteration 0's files: iteration 0's
#     solve does not update CCPs, so the legacy pipeline would rebuild
#     byte-identical sequences from the unchanged initial CCPs);
#   * the budget SMM reads the dense files.
# Continuation values are byte-identical to the legacy pipeline; gate
# before flipping: test_ccp_sequence_fast_equivalence.py (incl. --fused)
# must PASS on the server. Requires USE_FAST_SOLVER. Default False =
# exactly the previous pipeline.
USE_FAST_CCP_SEQUENCE = False

# Production loan-SMM controls. The auxiliary EM results are loaded above and
# remain fixed. A deliberately small annealing budget keeps each NPL iteration
# manageable; subsequent iterations restart from the saved production vector.
BUDGET_SMM_DRAWS = 100
BUDGET_SMM_ANNEALING_MAXFUN = 500
BUDGET_SMM_MAXITER = 1000
BUDGET_SMM_CCP_WORKERS = 60
BUDGET_SMM_CELL_WORKERS = None  # automatically one worker per cell, CPU permitting
# Profiling 2026-07-23: the pooled kernel is bitwise thread-count-invariant, so
# raising this to 6 uses all 60 server cores (10 cell workers x 6 threads) for
# a ~1.5x SMM speedup with no other pool active during the SMM phase.
# Promoted 1 -> 6 on 2026-07-24 (researcher approval; see
# Agents_Readme/Tasks/ESTIMATION_SPEED_ANALYSIS_2026_07_24.md, point 2).
BUDGET_SMM_CELL_NUMBA_THREADS = 6
# "dfols" is the benchmarked faster least-squares alternative (see
# Agents_Readme/Tasks/STUDENT_LOANS_FIT_MASTER_PLAN.md, section 5).
# Promoted "hybrid" -> "dfols" on 2026-07-24 (researcher approval; see
# ESTIMATION_SPEED_ANALYSIS_2026_07_24.md, point 1). Requires the DFO-LS
# package on the server (already in requirements.txt; reinstall deps once).
# Set back to "hybrid" to recover the previous optimizer exactly.
BUDGET_SMM_OPTIMIZER = "dfols"

# --- New-borrowing-cost (kappa) estimation switches (2026-07-23) -------------
# See Agents_Readme/Tasks/STUDENT_LOANS_FIT_MASTER_PLAN.md. Both default to the
# pre-kappa production behavior; flip BOTH to run the new specification:
#   ESTIMATE_NEW_BORROWING_COST = True  -> 71-parameter vector (kappa0_low,
#       kappa0_high, kappa1 appended; a saved 68-vector restart is upgraded
#       with zero kappas automatically).
#   BUDGET_SMM_MOMENT_SPEC = "flow_split_stock" -> entry/continuation moments
#       split by observed beginning-of-period debt + at-cap share.
ESTIMATE_NEW_BORROWING_COST = False
BUDGET_SMM_MOMENT_SPEC = "flow_split_stock"
# Spec B, heterogeneous debt aversion (loan-type debt-penalty shift, always the
# last vector entry; sizes 68/69/71/72): see the master plan.
ESTIMATE_LOAN_TYPE_DEBT_PENALTY = True

def _timing(stage, start, it=None):
    """Stage-timing print (added 2026-07-24): grep the run log for [TIMING]."""
    elapsed = time.perf_counter() - start
    prefix = f"it {it} | " if it is not None else ""
    print(
        f"[TIMING] {time.strftime('%Y-%m-%d %H:%M:%S')} | {prefix}{stage}: "
        f"{elapsed:,.1f} s",
        flush=True,
    )


if __name__ == '__main__':

    # Simulate all model states
    print("Simulating all model states...")
    ms.simulate_all_states(11)

    debt_range = ms.get_debt_range()

    # Select the Bellman solver. The fast solver is a validated drop-in; its
    # per-period static structure is prebuilt here so forked workers inherit
    # it instead of rebuilding it per process.
    if USE_FAST_SOLVER:
        print("Using FAST Bellman solver (model_solution_fast)")
        msf.build_all_period_statics()
        bellman_solver = msf.get_all_evt_fast
    else:
        bellman_solver = ms.get_all_evt

    if USE_FAST_CCP_SEQUENCE:
        if not USE_FAST_SOLVER:
            raise RuntimeError(
                "USE_FAST_CCP_SEQUENCE requires USE_FAST_SOLVER: only the "
                "fast solver emits the fused CCP sequence."
            )
        print("Using FAST CCP-sequence pipeline (dense format, fused solve)")
        # Module flags are set in the parent BEFORE any pool is created, so
        # forked workers (Linux, the production platform) inherit them.
        msf.EMIT_CCP_SEQUENCE = True
        mfd.CCP_SEQUENCE_FORMAT = "dense"


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
        utility_parameters = {
            em_type: ms.load_param_g(em_type, real=0)
            for em_type in TYPE_IDS
        }
        ms.simulate_all_states(T)

        conter = 0
        maxdebt = True
        args = [
            (i, ms.invariant_states, debt_range, debt_range, ccp_real,
             utility_parameters[em_type], models, solution_mode, conter,
             em_type, maxdebt)
            for em_type in TYPE_IDS
            for i in range(np.shape(ms.invariant_states)[0])
        ]

        pool_obj = multiprocessing.Pool(60)
        results = pool_obj.starmap(bellman_solver, args, chunksize=1)
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
        q = validate_q(q)
    
    else:
        print("Using previously generated EM weights")
        q = load_em_posteriors(EST("auxiliary_em_results.npz"))

    if solve_initial_ccps == True:
        print("Predicting initial CCPs for all joint types")

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
        print("Using previously generated initial CCPs")

    # The Bellman workers load one file for every type, invariant state, and
    # period 1,...,T-1. Verify that complete grid before starting the expensive
    # solve, and retry only incomplete type/state tasks.
    _stage_started = time.perf_counter()
    mccp.ensure_initial_ccps(
        ms.invariant_states,
        ms.debt_range,
        auxiliary_ccp_parameters
        if solve_initial_ccps
        else {
            type_id: mccp.load_utility_parameters(type_id)
            for type_id in TYPE_IDS
        },
    )
    
    
    _timing("ensure_initial_ccps", _stage_started)

    # Generate the amount of aguirregabiria_mira iterations.
    iterations = 30
    solution_mode = 0

    for it in range(iterations):

        print("Iteration number : ", it)
        _iteration_started = time.perf_counter()
    
        # Check which ccp estimation to use:
        if it != 0:
            ccp_real = 1
        else:
            ccp_real = 0
            # Initial guess:
            x0 = np.zeros((total_n,))
            x0 = np.load(f"{path_estimates}/param_g.npy")
    
            # Prepare the get_x matrix
            me.get_feasible()
            me.get_feasible_pubid()
            me.get_x_g_superfeasible()

        # Direct utility depends on the schooling component of the joint type;
        # financial components enter separately through the Bellman budget.
        utility_parameters = {
            em_type: ms.build_param_g(em_type, x0)
            for em_type in TYPE_IDS
        }
    
        if get_budget == True:
            #--------------------------------------#
            # Build the sequence to estimate the budget shock
            print("Building Sequence to Estimate Budget Shock")

            _stage_started = time.perf_counter()
            if (not USE_FAST_CCP_SEQUENCE) or it == 0:
                sequence_task = (
                    mgsf.get_ccp_sequence_task if USE_FAST_CCP_SEQUENCE
                    else mgs.get_ccp_sequence
                )
                pool_obj = multiprocessing.Pool(processes=60)
                args = [
                    (i, ms.invariant_states, ms.debt_range, em_type)
                    for em_type in TYPE_IDS
                    for i in range(np.shape(ms.invariant_states)[0])
                ]

                results = pool_obj.starmap(sequence_task, args, chunksize=1)
                pool_obj.close()
            else:
                print("Reusing dense CCP sequences emitted by the previous "
                      "Bellman solve")
            _timing("ccp-sequence build", _stage_started, it)

            #---------------------------------------#
            print("Estimation of the Budget Shock")
            _stage_started = time.perf_counter()

            mfd.ESTIMATE_NEW_BORROWING_COST = ESTIMATE_NEW_BORROWING_COST
            mfd.ESTIMATE_LOAN_TYPE_DEBT_PENALTY = ESTIMATE_LOAN_TYPE_DEBT_PENALTY
            estimate_budget_shock_all_education(
                draws=BUDGET_SMM_DRAWS,
                maxiter=BUDGET_SMM_MAXITER,
                optimizer=BUDGET_SMM_OPTIMIZER,
                annealing_maxfun=BUDGET_SMM_ANNEALING_MAXFUN,
                moment_spec=BUDGET_SMM_MOMENT_SPEC,
                resource_mode="simulated",
                restart=True,
                ccp_workers=BUDGET_SMM_CCP_WORKERS,
                cell_workers=BUDGET_SMM_CELL_WORKERS,
                cell_numba_threads=BUDGET_SMM_CELL_NUMBA_THREADS,
                # CCPs change at every NPL iteration. Production estimation
                # must therefore read the newly constructed sequences rather
                # than reuse the education-cell testing cache.
                ccp_cache_mode="off",
            )
            ms.reload_budgetshock_params()
            _timing("budget SMM", _stage_started, it)

        #--------------------------------------#
        # Solve the model with ccps
        print("Solve the model and the ccps")
        _stage_started = time.perf_counter()
        models = 0
        conter = 0
        maxdebt = True
        pool_obj = multiprocessing.Pool(processes=60, initializer=ms.reload_budgetshock_params)
    
        args = [
            (i, ms.invariant_states, debt_range, debt_range, ccp_real,
             utility_parameters[em_type], models, solution_mode, conter,
             em_type, maxdebt)
            for em_type in TYPE_IDS
            for i in range(np.shape(ms.invariant_states)[0])
        ]
    
        results = pool_obj.starmap(bellman_solver, args, chunksize=1)
        pool_obj.close()
        _timing("Bellman solve", _stage_started, it)

        #--------------------------------------#
        # Now prepare the data and evaluate the likelihood
        print("Preparing the data for the estimation")

        _stage_started = time.perf_counter()
        pool_obj = multiprocessing.Pool(T - 1)
        args = [period for period in range(1, T, 1)]
        results = pool_obj.map(me.prepare_vjt_feasible, args)
        pool_obj.close()
        _timing("prepare_vjt_feasible", _stage_started, it)

        _stage_started = time.perf_counter()
        choices_all, vjt_all_types, x1_new, choices_array_all, x_change, x_educ, x_first2, x_first4, x_firstgrad, x_exp = me.load_all_arrays_feasible()
        _timing("load_all_arrays_feasible", _stage_started, it)

        # Now optimize the likelihood
        print("Evaluating the likelihood")
        _stage_started = time.perf_counter()
        res = minimize(
            me.likelihood,
            x0,
            args=(choices_all, vjt_all_types, x1_new, choices_array_all,
                  x_change, x_educ, x_first2, x_first4, x_firstgrad, x_exp, q),
            jac=True,
            options={'disp': True},
            callback=me.store
        )
        _timing("likelihood optimization", _stage_started, it)

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

        _timing("FULL NPL ITERATION", _iteration_started, it)
            
        
        
            
        
        


