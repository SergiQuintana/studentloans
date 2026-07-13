


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

warnings.simplefilter('ignore',category=NumbaDeprecationWarning)
warnings.simplefilter('ignore',category=NumbaPendingDeprecationWarning)

import model_continuation_final as mcf
import model_solution_em as ms

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



def solve_many_continuations():
    
    debt_range = ms.get_debt_range()
    sigma_range = np.round(np.arange(0.025,3.25,0.05),3)
    conter = 0
    # loop over several sigmas
        
    for sigma_u in sigma_range:
        
                
        print(f"Solving the continuation value for sigma {sigma_u}!")
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

# =============================================================================
# if __name__ == '__main__':
#     
# 
#     
#     debt_range = ms.get_debt_range()
#     sigma_range = np.round(np.arange(0.025,3.25,0.05),3)
#     
#     # loop over several sigmas
#     
#     for sigma_u in sigma_range:
#     
#             
#         print(f"Solving the continuation value for sigma {sigma_u}!")
#         # load the data
#         df = mcf.load_data()
#         n = 100000
#         args = [(df,initialdebt,lastperiod,sigma_u,n) for lastperiod in range(0,T) for initialdebt in debt_range]
#         pool_obj = multiprocessing.Pool(60)
#         results = pool_obj.starmap(mcf.geteverything, args)
#         pool_obj.close()
# 
#         # Now put everything together
#             
#         mcf.give_model_format(debt_range,sigma_u)
#             
#         # And finally give numpy format
#             
#         mcf.data_to_numpy(sigma_u)
# =============================================================================
