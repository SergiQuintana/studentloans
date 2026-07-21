# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:50:26 2024

@author: Sergi
"""

import numpy as np
import scipy
import os
import time
#import numba
from numba import njit,jit,prange
import multiprocessing 
import multiprocessing as mp
from scipy.optimize import minimize
from os import environ
#print(environ['MKL_NUM_THREADS'])
import warnings

from numba.core.errors import NumbaPendingDeprecationWarning,NumbaDeprecationWarning

warnings.simplefilter('ignore',category=NumbaDeprecationWarning)
warnings.simplefilter('ignore',category=NumbaPendingDeprecationWarning)

#mkl.set_num_threads(1)

#os.chdir(r"C:/Users/Sergi/Dropbox/PhD/Real Model")
mu = 0
gamma = 0.57721
beta = 0.98
r = 0.05
T = 10

# --------------------------------------------------------------------------- #
# Import the model components
# --------------------------------------------------------------------------- #
import model_solution_em as ms
import model_simulation_em as msim
#import model_solution_nodebt as msnodebt
#import model_simulation_nodebt as msimnodebt


#--------------------------#

from pathlib import Path
from config import DIR, OUT, INP, FUN, RDATA, CONT, EST, LIK

def _mk(p): Path(p).mkdir(parents=True, exist_ok=True)

# ensure runtime dirs exist (safe if they already exist)
for p in [
    # per-period VJT/EVT trees we read/write during sim:
    *[OUT("vjt", str(t)) for t in range(0, 11)],
    *[OUT("vjt_nog", str(t)) for t in range(0, 11)],
    *[OUT("vjt_conter", str(t)) for t in range(0, 11)],
    *[OUT("evt", str(t)) for t in range(0, 11)],
    *[OUT("evt_nog", str(t)) for t in range(0, 11)],
    *[OUT("evt_conter", str(t)) for t in range(0, 11)],
    # simulation outputs:
    OUT("choice"), OUT("state"), OUT("epsilon"), OUT("welfare"), OUT("grad_prob"), OUT("types"),
    # estimates, function coeffs & real data:
    DIR["MODEL_ESTIMATES"], DIR["MODEL_FUNCOEF"], DIR["MODEL_REALDATA"],
]:
    _mk(p)

# short aliases (mimic your earlier pattern)
pathfunctions   = DIR["MODEL_FUNCOEF"]
path_realdata   = DIR["MODEL_REALDATA"]
path_estimates  = DIR["MODEL_ESTIMATES"]

# helpers for vjt/evt container dirs
def _vjt_dir(period: int, conter: int = 0, maxdebt: bool = False) -> Path:
    if conter == 0:  # real
        return Path(OUT("vjt", str(period)))
    elif conter == 1:  # conter with/without maxdebt in filename
        return Path(OUT("vjt_conter", str(period)))
    else:  # conter_not
        return Path(OUT("vjt_conter_not", str(period)))

def _evt_dir(period: int, conter: int = 0, maxdebt: bool = False) -> Path:
    if conter == 0:
        return Path(OUT("evt", str(period)))
    elif conter == 1:
        return Path(OUT("evt_conter", str(period)))
    else:
        return Path(OUT("evt_nog", str(period)))  # you used evt_nog for “not” elsewhere

# single-place output roots
def _out_choice():  return Path(OUT("choice"))
def _out_state():   return Path(OUT("state"))
def _out_eps():     return Path(OUT("epsilon"))
def _out_welfare(): return Path(OUT("welfare"))
def _out_grad():    return Path(OUT("grad_prob"))


if __name__ == '__main__':
    
    # The first thing is to simulate the vjts with the parameters obtained during 
    # estimation
    
    debt_range = ms.get_debt_range()
        
    print("Solving the model...")
    solution_mode = 1
    ccp_real = []
    models = []
    sigma_u = 1.4
    conter = 0
    maxdebt = True
    utility_parameters1 = ms.load_param_g(1,real=1)
    utility_parameters2 = ms.load_param_g(2,real=1)
    uparams = [utility_parameters1,utility_parameters2]
    args = [(i,ms.invariant_states,debt_range,debt_range,ccp_real,utility_parameters1,models,solution_mode,conter,1,maxdebt) for i in range(np.shape(ms.invariant_states)[0])]
    args.extend(((i,ms.invariant_states,debt_range,debt_range,ccp_real,utility_parameters2,models,solution_mode,conter,2,maxdebt) for i in range(np.shape(ms.invariant_states)[0])))
    pool_obj = multiprocessing.Pool(60)
    results = pool_obj.starmap(ms.get_all_evt, args)
    pool_obj.close()
        
    # Now simulate choices. The idea is to simulate 10 times each individual
    
    print("Simulating model choices...")
    samples=30
    sigma_u =  1.4
    conterfactual = 0
    maxdebt = True 
    q = np.load(f"{path_estimates}/em_q_typeff2.npy")
    args = [(i,sigma_u,conterfactual,q,maxdebt,uparams) for i in range(1,samples+1)]
    pool_obj = multiprocessing.Pool(samples)
    results = pool_obj.starmap(msim.simulate_choices, args)
    pool_obj.close()
    
    #-------------------------------------------------------------------------#
    
    # Now generate the conterfactual vjts

    print("Simulating counterfactual terminal values...")


    
    print("Simulating counterfactual vjts...")
    conter = 1
    solution_mode = 1
    maxdebt = False
    args = [(i,ms.invariant_states,debt_range,debt_range,ccp_real,utility_parameters1,models,solution_mode,conter,1,maxdebt) for i in range(np.shape(ms.invariant_states)[0])]
    args.extend(((i,ms.invariant_states,debt_range,debt_range,ccp_real,utility_parameters2,models,solution_mode,conter,2,maxdebt) for i in range(np.shape(ms.invariant_states)[0])))
    #pool_obj = multiprocessing.Pool(60)
    #results = pool_obj.starmap(ms.get_all_evt, args)
    #pool_obj.close()
    
    print("Simulating conterfactual choices...")
    samples=30
    sigma_u =  1.4
    conterfactual = 1
    maxdebt = False
    args = [(i,sigma_u,conterfactual,q,maxdebt,uparams) for i in range(1,samples+1)]
    #pool_obj = multiprocessing.Pool(samples)
    #results = pool_obj.starmap(msim.simulate_choices, args)
    #pool_obj.close()
    
    #-------------------------------------------------------------------------#

    # Now generate the conterfactual vjts with maximum debt established
    print("Simulating counterfactual terminal values...")

   
    a  = False
    if a == True:
        debt_range = ms.get_debt_range()
        sigma_u = np.load(sigmas)

        #sigma_range = [for sigma in sigmas_u]
        conter = 1
        # load the data
        df = mcf.load_data()
        n = 100000
        args = [(df,initialdebt,lastperiod,sigma_u,n,conter) for lastperiod in range(0,T) for initialdebt in debt_range]
        pool_obj = multiprocessing.Pool(60)
        results = pool_obj.starmap(mcf.geteverything, args)
        pool_obj.close()
        # Now put everything together
            
        mcf.give_model_format_plan(debt_range,sigma_u,conter)
                
        # And finally give numpy format
                    
        mcf.data_to_numpy(sigma_u,conter)

        print("Simulating counterfactual vjts...")
        conter = 1
        solution_mode = 1
        maxdebt = True
        args = [(i,ms.invariant_states,debt_range,debt_range,ccp_real,utility_parameters1,models,solution_mode,conter,1,maxdebt) for i in range(np.shape(ms.invariant_states)[0])]
        args.extend(((i,ms.invariant_states,debt_range,debt_range,ccp_real,utility_parameters2,models,solution_mode,conter,2,maxdebt) for i in range(np.shape(ms.invariant_states)[0])))
        pool_obj = multiprocessing.Pool(60)
        results = pool_obj.starmap(ms.get_all_evt, args)
        pool_obj.close()
    
    print("Simulating conterfactual choices...")
    samples=30
    sigma_u =  1.4
    conterfactual = 1
    maxdebt = True
    args = [(i,sigma_u,conterfactual,q,maxdebt,uparams) for i in range(1,samples+1)]
    pool_obj = multiprocessing.Pool(samples)
    results = pool_obj.starmap(msim.simulate_choices, args)
    pool_obj.close()
    
    
    #-------------------------------------------------------------------------#
    
    # Now generate conterfactual without debt
    
    conter = 0
    solution_mode = 1
    maxdebt = True
    args = [(i,ms.invariant_states,debt_range,debt_range,ccp_real,utility_parameters1,models,solution_mode,conter,1,maxdebt) for i in range(np.shape(msnodebt.invariant_states)[0])]
    args.extend(((i,ms.invariant_states,debt_range,debt_range,ccp_real,utility_parameters2,models,solution_mode,conter,2,maxdebt) for i in range(np.shape(msnodebt.invariant_states)[0])))
    #pool_obj = multiprocessing.Pool(60)
    #results = pool_obj.starmap(msnodebt.get_all_evt, args)
    #pool_obj.close()
    
    
    
    print("Simulating conterfactual choices...")
    samples=30
    sigma_u =  1.4
    conterfactual = 0
    maxdebt = True
    args = [(i,sigma_u,conterfactual,q,maxdebt,uparams) for i in range(1,samples+1)]
    #pool_obj = multiprocessing.Pool(samples)
    #results = pool_obj.starmap(msimnodebt.simulate_choices, args)
    #pool_obj.close()
    
    
    #-------------------------------------------------------------------------#
    
    # Now generate conterfactual without debt but with grants
    
    conter = 1
    solution_mode = 1
    maxdebt = True
    args = [(i,ms.invariant_states,debt_range,debt_range,ccp_real,utility_parameters1,models,solution_mode,conter,1,maxdebt) for i in range(np.shape(msnodebt.invariant_states)[0])]
    args.extend(((i,ms.invariant_states,debt_range,debt_range,ccp_real,utility_parameters2,models,solution_mode,conter,2,maxdebt) for i in range(np.shape(msnodebt.invariant_states)[0])))
    #pool_obj = multiprocessing.Pool(60)
    #results = pool_obj.starmap(msnodebt.get_all_evt, args)
    #pool_obj.close()
    
    
    
    print("Simulating conterfactual choices...")
    samples=30
    sigma_u =  1.4
    conterfactual = 1
    maxdebt = True
    args = [(i,sigma_u,conterfactual,q,maxdebt,uparams) for i in range(1,samples+1)]
    #pool_obj = multiprocessing.Pool(samples)
    #results = pool_obj.starmap(msimnodebt.simulate_choices, args)
    #pool_obj.close()