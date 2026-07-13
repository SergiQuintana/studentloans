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
import mkl
import warnings

from numba.core.errors import NumbaPendingDeprecationWarning,NumbaDeprecationWarning

warnings.simplefilter('ignore',category=NumbaDeprecationWarning)
warnings.simplefilter('ignore',category=NumbaPendingDeprecationWarning)

#mkl.set_num_threads(1)

#os.chdir(r"C:/Users/Sergi/Dropbox/PhD/Real Model")
os.chdir(r"C:/Users/Sergi/Project/Real")
mu = 0
gamma = 0.57721
beta = 0.98
r = 0.05
T = 10


#--------------------------#

os.chdir(r"C:/Users/Sergi/Dropbox/PhD/Projects/Papers/1_financial_constraints/Code/2024_10/2_model/RealModel/codes")
import model_solution_em as ms
os.chdir(r"C:/Users/Sergi/Dropbox/PhD/Projects/Papers/1_financial_constraints/Code/2024_10/2_model/RealModel/codes")
import model_simulation_fitdebt_working_adusted as msim

os.chdir(r"C:/Users/Sergi/Project/Real")


if __name__ == '__main__':
    
    # The first thing is to simulate the vjts with the parameters obtained during 
    # estimation
    
    debt_range = ms.get_debt_range()
        
    #print("Solving the model...")
    solution_mode = 1
    ccp_real = []
    models = []
    sigma_u = 1.4
    conter = 0
    maxdebt = False
    utility_parameters1 = ms.load_param_g(1,real=1)
    utility_parameters2 = ms.load_param_g(2,real=1)
    #args = [(i,ms.invariant_states,debt_range,debt_range,ccp_real,sigma_u,utility_parameters1,models,solution_mode,conter,1,maxdebt) for i in range(np.shape(ms.invariant_states)[0])]
    #args.extend(((i,ms.invariant_states,debt_range,debt_range,ccp_real,sigma_u,utility_parameters2,models,solution_mode,conter,2,maxdebt) for i in range(np.shape(ms.invariant_states)[0])))
    #pool_obj = multiprocessing.Pool(60)
    #results = pool_obj.starmap(ms.get_all_evt, args)
    #pool_obj.close()
        
    # Now simulate choices. The idea is to simulate 10 times each individual
    
    print("Simulating model choices...")
    samples=30
    sigma_u =  1.4
    conterfactual = 0
    maxdebt = False 
    q = np.load("estimates/em_q_typeff2.npy")
    args = [(i,sigma_u,conterfactual,q,maxdebt) for i in range(1,samples+1)]
    pool_obj = multiprocessing.Pool(np.minimum(60,samples))
    results = pool_obj.starmap(msim.simulate_choices, args)
    pool_obj.close()
    
    #-------------------------------------------------------------------------#
    print("Simulating model choices...")
    samples=30
    sigma_u =  1.4
    conterfactual = 1
    maxdebt = True 
    q = np.load("estimates/em_q_typeff2.npy")
    args = [(i,sigma_u,conterfactual,q,maxdebt) for i in range(1,samples+1)]
    pool_obj = multiprocessing.Pool(np.minimum(60,samples))
    results = pool_obj.starmap(msim.simulate_choices, args)
    pool_obj.close()
   