# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:46:43 2024

@author: Sergi
"""
import numpy as np
import scipy
import argparse
import os
import time
import pandas as pd
# Numba is optional. The auxiliary estimator remains correct without it, while
# server environments that provide Numba retain the legacy acceleration.
try:
    from numba import njit, jit, prange, get_num_threads, set_num_threads
    from numba.core.errors import NumbaPendingDeprecationWarning, NumbaDeprecationWarning
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*decorator_args, **decorator_kwargs):
        if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1:
            return decorator_args[0]
        return lambda function: function

    jit = njit
    prange = range

    def get_num_threads():
        return 1

    def set_num_threads(number_threads):
        if int(number_threads) != 1:
            raise RuntimeError("Numba is unavailable; only one Python thread can be used.")

    class NumbaPendingDeprecationWarning(Warning):
        pass

    class NumbaDeprecationWarning(Warning):
        pass
import multiprocessing 
import multiprocessing as mp
from scipy.optimize import minimize
from os import environ
#from optimparallel import minimize_parallel
#print(environ['MKL_NUM_THREADS'])
from scipy.optimize import check_grad
from scipy.optimize import approx_fprime
#import mkl
import warnings
from scipy.special import expit, logsumexp

warnings.simplefilter('ignore',category=NumbaDeprecationWarning)
warnings.simplefilter('ignore',category=NumbaPendingDeprecationWarning)

#mkl.set_num_threads(1)

#os.chdir(r"C:\Users\Sergi\Dropbox\PhD\Projects\Papers\1_financial_constraints\Model\Temp")
#os.chdir(r"C:/Users/Sergi/Project/Real")
mu = 0
sigma =0.4
gamma = 0.57721
beta = 0.98
r = 0.05
T = 10

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




from config import DIR, OUT, INP, FUN, RDATA, CONT, EST,LIK
from tables import write_auxiliary_em_tables
from latent_types import (
    TYPE_COMPONENTS,
    TYPE_NAMES,
    TYPE_IDS,
    TYPE_SCHOOL,
    TYPE_GRANT,
    TYPE_TRANSFER,
    TYPE_LOAN,
    N_TYPES as N_STRUCTURAL_TYPES,
    type_components,
    validate_q,
)
pathfunctions = DIR["MODEL_FUNCOEF"]
path = DIR["MODEL_REALDATA"]
pathout = DIR["MODEL_OUTPUT"]
path_estimates  = DIR["MODEL_ESTIMATES"]

# Representative joint IDs for routines that only vary the schooling component.
# They are derived from the shared layout rather than inferred from the ID value.
SCHOOL_REPRESENTATIVE_TYPE_IDS = tuple(
    next(
        type_id
        for type_id, school in zip(TYPE_IDS, TYPE_SCHOOL)
        if school == school_type
    )
    for school_type in (0, 1)
)
#-----------------------------------------------------------------------------#

def get_data_pubid(period,real_data=1):
    
    """This function loads the data of each corresponding period"""
    
    if real_data == 0 :
    
        pass
    
    else:
        
        debt_range = get_debt_range()
    
        state = np.load(f"{path}/state_t{period}.npy")

        choices = np.load(f"{path}/choice_t{period}.npy")
            
        x1 = np.load(f"{path}/invariant_state_t{period}.npy")

        debt = np.load(f"{path}/financial_t{period}.npy")  # only loans
        
        income = np.load(f"{path}/income_t{period}.npy")
        
        debtchoice = np.load(f"{path}/debtchoice_t{period}.npy")

        loanflow = np.load(f"{path}/loanflow_t{period}.npy")
        # map debt to the closer point on the grid
        
        value = np.zeros(np.shape(state)[0])
        value2 = np.zeros(np.shape(state)[0])
        
        for i in range(np.shape(state)[0]):
            
            diff = (debt_range - debt[i])**2
                
            value[i] = np.argmin(diff)
            
            diff2 = (debt_range - debtchoice[i])**2
                
            value2[i] = np.argmin(diff2)
            
        debt = value
        debtchoice = value2
    
    return x1,state,debt,choices, income, debtchoice, loanflow

def get_data_superfeasible():
    
    index = np.load(f"{path}/feasible_index.npy")
    
    for period in range(1,T):
        
        x1,state,debt,choices,income, debtchoice, loanflow = get_data_pubid(period)
        
        # check which pubids belong to the final list
        
        idx = np.isin(x1[:,0], index)
    
        x1_feasible  = x1[idx,:]
        state_feasible = state[idx,:]
        debt_feasible = debt[idx]
        choices_feasible = choices[idx,:]
        income_feasible = income[idx,:]
        debtchoice_feasible = debtchoice[idx]
        loanflow_feasible = loanflow[idx]
        
        np.save(f"{path}/state_superfeasible_t{period}.npy",state_feasible)
        np.save(f"{path}/invariant_state_superfeasible_t{period}.npy",x1_feasible)
        np.save(f"{path}/debt_superfeasible_t{period}.npy",debt_feasible)
        np.save(f"{path}/choice_superfeasible_t{period}.npy",choices_feasible)
        np.save(f"{path}/income_superfeasible_t{period}.npy",income_feasible)
        np.save(f"{path}/debtchoice_superfeasible_t{period}.npy",debtchoice_feasible)
        np.save(f"{path}/loanflow_superfeasible_t{period}.npy",loanflow_feasible)


def get_feasible_pubid():
    
    """This is a temporary function to solve data cleaning problems. 
    I will only get individuals that belong to a feasible possible 
    state. """
    
    for period in range(1,T):
        
        print(f"Current period is {period}")
        
        x1,state,debt,choices,income, debtchoice, loanflow = get_data_pubid(period)

        allstate = np.linspace(0,np.shape(state)[0]-1,np.shape(state)[0]).astype("int")
        
        feasible = np.load(f"{pathout}/states/states_t{period}.npy")
        state = state.astype('int')
        
        # First destroy illegal states
        idx = np.sort(np.where( (state==feasible[:,None]).all(-1) )[1]) # index of feasible elements
        idx2 = allstate[~(state==feasible[:,None]).all(2).any(0)]
        
        statenot = state[idx2,:]
        x1_feasible  = x1[idx,:]
        state_feasible = state[idx,:]
        debt_feasible = debt[idx]
        choices_feasible = choices[idx,:]
        income_feasible = income[idx,:]
        debtchoice_feasible = debtchoice[idx]
        # Then destroy illegal choices
        feasibleindex = check_choice_feasible(state_feasible,choices_feasible)
        feasiblepubid = x1_feasible[feasibleindex,0]
        
        
        np.save(f"{path}/feasible_pubid_t{period}.npy",feasiblepubid)
    # now get the final index
    get_superfeasible()
    
    # And finally get and store the data
    
    get_data_superfeasible()
    
    pass


def get_superfeasible():
    
    
    index = pd.DataFrame(np.load(f"{path}/feasible_pubid_t1.npy"),columns=["pubid"])
    index["re"] = 0 
    index = index.set_index('pubid')    
    for period in range(2,11):
        
        index2 = pd.DataFrame(np.load(f"{path}/feasible_pubid_t{period}.npy"),columns=["pubid"])
        index2[f're_{period}'] = 0 
        index2 = index2.set_index('pubid')
        index = index.merge(index2, how='outer', on = 'pubid')
        
    # Open measures
    
    measures = pd.read_stata(f"{path}/school_measures_clear.dta").rename(columns={'PUBID':'pubid'}).set_index('pubid')
    
    index = index.merge(measures, how='outer', on = 'pubid')
    
    index = index.dropna().reset_index()
    
    # Save the index
    
    np.save(f"{path}/feasible_index.npy",np.array(index['pubid']))
    
    # And save the measures
    
    measures_new = index[["pubid","late_school","summer_class","reason_summer"]]
    
    measures_new.to_csv(f"{path}/feasible_measures.csv",index=False)
        
        
        
def check_choice_feasible(state,choice):
    
    '''
    This function checks if a choice is legal
    '''
    
    total_n = np.shape(state)[0]
    
    feasible = np.zeros(total_n) <1

    for i in range(total_n):
        
        possible = get_possible_choices(state[i,:])
        
        idx = np.sort(np.where( (choice[i,:]==possible[:,None]).all(-1) )[1]) # index of feasible elements
        
        if idx == 0:
            
            feasible[i] = True
            
        else:
            feasible[i] = False
            
    return feasible
            
        


#-----------------------------------------------------------------------------#
# -------------------------- FUNCTIONS 
#-----------------------------------------------------------------------------#
def get_total_choices():
    
    choices_educ2 = np.array(np.meshgrid([12], [1], [0,1,2])).T.reshape(-1,3)
    choices_educ4 = np.array(np.meshgrid([0,1,2],[2], [1,2,3,4,5,6,7,8])).T.reshape(-1,3)
    choices_educ4[:, [2, 0]] = choices_educ4[:, [0, 2]]
    choices_no_educ = np.array(np.meshgrid([1,2], [0], [1,2,3,6,7,8,9,10])).T.reshape(-1,3)
    choices_no_educ[:, [2, 0]] = choices_no_educ[:, [0, 2]]
    choices_grad = np.array(np.meshgrid([13], [3], [0,1,2])).T.reshape(-1,3)
    home_production = np.array([[0,0,0]])
    # put both together
    choices = np.concatenate((choices_educ2,choices_educ4,choices_no_educ,choices_grad,home_production),axis=0)
    
    return choices


def get_choices_no_educ(x2):
    
    """
    This function returns the feasible choices for each graduation field. 
    """
    
    if x2[8] == 1: #Business
        choices_no_educ = np.array(np.meshgrid([1,2], [0], [1,9,10])).T.reshape(-1,3)
    elif x2[8] == 2: #STEM
        choices_no_educ = np.array(np.meshgrid([1,2], [0], [1,2,8,9,10])).T.reshape(-1,3)
    elif x2[8] == 4: #Education
        choices_no_educ = np.array(np.meshgrid([1,2], [0], [1,6,9,10])).T.reshape(-1,3)
    elif x2[8] == 5: #Social Sciences
        choices_no_educ = np.array(np.meshgrid([1,2], [0], [1,3,6,9,10])).T.reshape(-1,3)
    elif x2[8] == 6: #STEM
        choices_no_educ = np.array(np.meshgrid([1,2], [0], [1,3,6,7,9,10])).T.reshape(-1,3)
    elif x2[8] == 7: #Health
        choices_no_educ = np.array(np.meshgrid([1,2], [0], [1,6,8,9,10])).T.reshape(-1,3)
    elif x2[8] == 8: #Other Fields
        choices_no_educ = np.array(np.meshgrid([1,2], [0], [1,2,3,8,9,10])).T.reshape(-1,3)
    elif x2[8] == 12: #Associate Degree
        choices_no_educ = np.array(np.meshgrid([1,2], [0], [1,2,7,8,9,10])).T.reshape(-1,3)
        
    choices_no_educ[:, [2, 0]] = choices_no_educ[:, [0, 2]]

    return choices_no_educ
          
def get_possible_choices(x2):
    
    "Given a state get all possible choices"
    
    # The restriction here is that if an individual already has a 2y or a 4y degree
    # he can't choose to study that again. 
    
    # Chocies are: field, education, working. 
    # education -> 0  no education , 1 2y, 2 4y, 3 grad schol (only if 4y grad)
    # working -> 0 no working, 1 partime, 2 fulltime
    
    if (((x2[4] == 0) & (x2[5] == 1)) | ((x2[4] == 1) & (x2[5] == 1))) & (x2[6]==0) : # individuals with a 4y degree without a grad degree
        # They can't study a 4y anymore, only grad school
        
        choices_no_educ = get_choices_no_educ(x2)
        choices_grad = np.array(np.meshgrid([13], [3], [0,1,2])).T.reshape(-1,3)
        home_production = np.array([[0,0,0]])
        choices = np.concatenate((choices_no_educ,choices_grad,home_production),axis=0)
        return choices
    
    elif (x2[4] == 1) & (x2[5] == 0) : # individuals with a 2y degree
        # They can't choose 2y anymore, only 4y school
        
        if x2[2] < 2:
            choices_educ4 = np.array(np.meshgrid([0,1,2], [2], [1,2,3,4,5,6,7,8])).T.reshape(-1,3)
            choices_educ4[:, [2, 0]] = choices_educ4[:, [0, 2]]
        else:
            choices_educ4 = np.array(np.meshgrid([0,1,2], [2], [1,2,4,5,6,7,8])).T.reshape(-1,3)
            choices_educ4[:, [2, 0]] = choices_educ4[:, [0, 2]]
        
        choices_no_educ = get_choices_no_educ(x2)
        home_production = np.array([[0,0,0]])
        choices = np.concatenate((choices_educ4,choices_no_educ,home_production),axis=0)
        return choices
    
    elif x2[6]==1: # individuals with a graduate degree
    
        choices_no_educ = get_choices_no_educ(x2)
        home_production = np.array([[0,0,0]])
        choices = np.concatenate((choices_no_educ,home_production),axis=0)
        return choices
    
    else: 
        # Here all choices are possible (No! Individuals with more than 1 foryear exp can't choose 3y schools)
        if x2[2] < 2:
            choices_educ4 = np.array(np.meshgrid([0,1,2], [2], [1,2,3,4,5,6,7,8])).T.reshape(-1,3)
            choices_educ4[:, [2, 0]] = choices_educ4[:, [0, 2]]
        else:
            choices_educ4 = np.array(np.meshgrid([0,1,2], [2], [1,2,4,5,6,7,8])).T.reshape(-1,3)
            choices_educ4[:, [2, 0]] = choices_educ4[:, [0, 2]]
            
        choices_educ2 = np.array(np.meshgrid([12], [1], [0,1,2])).T.reshape(-1,3)
        choices_no_educ = np.array(np.meshgrid([1,2], [0], [1,2,3,6,7,8,9,10])).T.reshape(-1,3)
        choices_no_educ[:, [2, 0]] = choices_no_educ[:, [0, 2]]
        home_production = np.array([[0,0,0]])
        # put both together
        choices = np.concatenate((choices_educ2,choices_educ4,choices_no_educ,home_production),axis=0)

        return choices
    
def get_feasible():
    
    """This is a temporary function to solve data cleaning problems. 
    I will only get individuals that belong to a feasible possible 
    state. """
    
    for period in range(1,T):
        
        x1,state,debt,choices,income = get_data(period)
        
        feasible = np.load(f"{pathout}/states/states_t{period}.npy")
        
        state = state.astype('int')
                
        idx = np.sort(np.where( (state==feasible[:,None]).all(-1) )[1]) # index of feasible elements
        
        x1_feasible  = x1[idx,:]
        state_feasible = state[idx,:]
        debt_feasible = debt[idx]
        choices_feasible = choices[idx,:]
        income_feasible = income[idx,:]
        print("Saving!")
        np.save(f"{path}/state_feasible_t{period}.npy",state_feasible)
        np.save(f"{path}/invariant_state_feasible_t{period}.npy",x1_feasible)
        np.save(f"{path}/debt_feasible_t{period}.npy",debt_feasible)
        np.save(f"{path}/choice_feasible_t{period}.npy",choices_feasible)
        np.save(f"{path}/income_feasible_t{period}.npy",income_feasible)
        
        print("Percentage of destroied data", 1-np.shape(x1_feasible)[0]/np.shape(x1)[0])
        print("Available Data",np.shape(x1_feasible)[0])
        

def get_data_feasible(period, return_income = False):
    
    state = np.load(f"{path}/state_feasible_t{period}.npy")
    
    choices = np.load(f"{path}/choice_feasible_t{period}.npy")
        
    x1 = np.load(f"{path}/invariant_state_feasible_t{period}.npy")

    debt = np.load(f"{path}/debt_feasible_t{period}.npy")
    
    if return_income == True:
        
        income = np.load(f"{path}/income_feasible_t{period}.npy")
    
        return x1,state,debt,choices, income
    
    else:
        
        return x1,state,debt,choices

def load_data_superfeasible(period, return_income=False):
    
    
    state = np.load(f"{path}/state_superfeasible_t{period}.npy").astype('int')
    
    choices = np.load(f"{path}/choice_superfeasible_t{period}.npy")
        
    x1 = np.load(f"{path}/invariant_state_superfeasible_t{period}.npy")[:,1:]

    debt = np.load(f"{path}/debt_superfeasible_t{period}.npy")
    
    debtchoice = np.load(f"{path}/debtchoice_superfeasible_t{period}.npy")
    
    income = np.load(f"{path}/income_superfeasible_t{period}.npy")
    
    if return_income == False:
    
        return x1,state,debt,choices
    elif return_income == True: 
        
        return x1,state,debt, debtchoice, choices, income

def get_data_simulated(period,samples=50):
    
    # First choices
    choices = np.load(f"choice/choice_t{period}_s1.npy") 
    for sample in range(2,samples+1):
        temp = np.load(f"choice/choice_t{period}_s{sample}.npy") 
        choices = np.append(choices,temp)
        
    # Now for xs
    x = np.load(f"state/state_t{period}_s1.npy") 
    for sample in range(2,samples+1):
        temp = np.load(f"state/state_t{period}_s{sample}.npy") 
        x = np.concatenate((x,temp),axis=0)
    
    total_choices = get_total_choices()
        
    choices = total_choices[choices]
        
    x1 = x[:,:4]
      
    state = x[:,4:14]
        
    debt = x[:,14]
       
    return x1,state,debt,choices


def get_debt_range():
    
    debtrange1 = np.array([0,300,500,620,770,950])
    debtrange2 = np.linspace(1166,3500,16)
    debtrange3 = np.linspace(3720,8800,25)
    debtrange4 = np.linspace(9200,20000,25)
    debtrange5 = np.linspace(22700,100000,28)

    debt_range = np.concatenate((debtrange1,debtrange2,debtrange3,debtrange4,debtrange5))
    
    return debt_range

def get_data(period,real_data=1):
    
    """This function loads the data of each corresponding period"""
    
    if real_data == 0 :
    
        pass
    
    else:
        
        debt_range = get_debt_range()
    
        state = np.load(f"{path}/state_t{period}.npy")
        
        choices = np.load(f"{path}/choice_t{period}.npy")
            
        x1 = np.load(f"{path}/invariant_state_t{period}.npy")[:,1:]  # do not take PUBID

        debt = np.load(f"{path}/financial_t{period}.npy")  # only loans
        
        income = np.load(f"{path}/income_t{period}.npy")
        
        # map debt to the closer point on the grid
        
        value = np.zeros(np.shape(state)[0])
    
        for i in range(np.shape(state)[0]):
            
            diff = (debt_range - debt[i])**2
            
            value[i] = np.argmin(diff)
            
        debt = value
        
    
    return x1,state,debt,choices, income

@njit()
def map_vjt_columns(vjt_i,column_map,idx,total_choices,i):
    
    
    vjt_column = np.zeros((1,np.shape(total_choices)[0]))
    
    for choice in range(np.shape(total_choices)[0]):
        if choice in list(idx[:,0]):
            vjt_column[0,choice] = vjt_i[int(column_map[i,choice])]  # Get the corresponding chioce with the corresponding debt level.
        else:
            vjt_column[0,choice] =-np.inf
                
    return vjt_column
    
def prepare_vjt_feasible(period):
    """Build likelihood-ready VJTs while opening each compressed bundle once.

    Bellman output is stored in one ``.npz`` bundle for each period, permanent
    type, and invariant state. The legacy implementation reopened that same
    bundle for every observed individual. Here observations are grouped first
    by invariant state and then by dynamic state, so all required debt rows are
    selected in batches from a single open archive.

    The saved arrays retain the legacy layout: rows follow the estimation
    sample, columns follow ``get_total_choices()``, and dynamically infeasible
    choices contain ``-np.inf``.
    """

    print("Current period is", period)

    x1, state, debt, choices = load_data_superfeasible(
        period, return_income=False
    )
    del choices  # Choice realizations are not needed to prepare all alternatives.

    x1 = np.asarray(x1, dtype=np.int64)
    state = np.asarray(state, dtype=np.int64)
    debt = np.asarray(debt, dtype=np.int64).reshape(-1)
    total_choices = get_total_choices()
    number_observations = state.shape[0]
    number_choices = total_choices.shape[0]

    if x1.shape[0] != number_observations or debt.shape[0] != number_observations:
        raise ValueError(
            "Invariant states, dynamic states, and debt indices must have "
            "the same number of observations."
        )

    unique_x1, x1_group = np.unique(x1, axis=0, return_inverse=True)
    unique_state, state_group = np.unique(state, axis=0, return_inverse=True)

    # Choice feasibility depends only on the dynamic state, not on the latent
    # type or invariant state. Compute this mapping once and reuse it for all
    # sixteen type-specific arrays.
    choice_columns_by_state = []
    for x2i in unique_state:
        possible_choices = get_possible_choices(x2i)
        choice_columns = np.where(
            (total_choices == possible_choices[:, None]).all(-1)
        )[1]
        if choice_columns.shape[0] != possible_choices.shape[0]:
            raise ValueError(
                f"Could not map every feasible choice for dynamic state {x2i}."
            )
        choice_columns_by_state.append(choice_columns)

    rows_by_x1 = [
        np.flatnonzero(x1_group == group)
        for group in range(unique_x1.shape[0])
    ]

    # Keep the public one-based IDs used in the Bellman filenames, while taking
    # the number and ordering of types from the shared latent-type definition.
    for em_type in TYPE_IDS:
        vjt = np.full(
            (number_observations, number_choices), -np.inf, dtype=np.float64
        )

        for group, x1i in enumerate(unique_x1):
            group_rows = rows_by_x1[group]
            bundle_path = (
                f"{pathout}/vjt_nog/{period}/"
                f"vjt_t{period}_[{x1i}]_em{em_type}.npz"
            )

            with np.load(bundle_path, allow_pickle=False) as bundle:
                group_state_ids = state_group[group_rows]
                for state_id in np.unique(group_state_ids):
                    rows = group_rows[group_state_ids == state_id]
                    x2i = unique_state[state_id]
                    key = f"vjt_t{period}_[{x1i}]_{x2i}"
                    state_vjt = np.asarray(bundle[key])
                    choice_columns = choice_columns_by_state[state_id]

                    if state_vjt.ndim != 2:
                        raise ValueError(
                            f"Bellman array {key} in {bundle_path} must be "
                            f"two-dimensional; received {state_vjt.shape}."
                        )
                    if state_vjt.shape[1] != choice_columns.shape[0]:
                        raise ValueError(
                            f"Bellman array {key} has {state_vjt.shape[1]} "
                            f"choice columns; expected {choice_columns.shape[0]}."
                        )

                    selected = state_vjt[debt[rows], :]
                    vjt[np.ix_(rows, choice_columns)] = selected

        print(f"Period {period}, type {em_type} finished")
        np.save(
            f"{pathout}/likelihood/vjt_super_t{period}_em{em_type}.npy",
            vjt,
        )
    
    
def prepare_vjt(period):
    
    """"This function loads the vjts to avoid doing that at each iteration of the 
    likelihood function. I should parallelize this function at some point,
    it takes a lot of time..."""
    
    # At some point I could parallelize this!
        
    print("Current period is", period)
            
    x1,state,debt,choices = get_data_feasible(period)
        
    #x1,state,debt,choices = get_data_simulated(period)
            
    # Loop over all individuals
    total_choices = get_total_choices()  # This should load the array with all set of choices
    # Generate column map
    column_map = np.repeat(np.nan,np.shape(state)[0]*np.shape(total_choices)[0]).reshape(np.shape(state)[0],np.shape(total_choices)[0])  # This array will hold how to map from columns in payoff, to the corresponding choice. As safety I will create it with nans
            
    vjt = np.zeros((np.shape(state)[0],np.shape(total_choices)[0]))
            
    # Now loop over all individuals
            
    for i in range(np.shape(state)[0]):
            
        # Get individual i
            
        x2i = state[i,:].astype("int")
        x1i = x1[i,:].astype("int")
        bi = int(debt[i])
            
        #Check which choices are feasible
                
        # Identify the index of choices.
                
        Jx  = get_possible_choices(x2i)
                
        idx = np.where( (total_choices==Jx[:,None]).all(-1) )[1]  # This tells which index in tota_choices corresponds to each choice in Jx
    
        idx = np.concatenate((idx[...,np.newaxis],np.arange(0,np.shape(idx)[0]).reshape(np.shape(idx)[0],1)),axis=1)  # This just includes the index of Jx as a column in the array. 
    
        column_map[i,idx[:,0]] = idx[:,1]  # This performs the match to this individual of the corresponding mapping.
                
        # At some point I could load all the individuals with the same x1 at once
        # I should use the column map strategy:
            
        vjt_i = np.load(f"{pathout}/vjt_nog/vjt_t{period}_[{x1i}].npz")[f"vjt_t{period}_[{x1i}]_{x2i}"][bi,:]
               
        vjt[i,:] =  map_vjt_columns(vjt_i,column_map,idx,total_choices,i)
    
        # now save the data
    print(f"Period {period} finsihed")
    np.save(f"{pathout}/likelihood/vjt_t{period}.npy",vjt)
    
    

def get_x_change_i(x2,period,fields=8,occupations=8):
    
    """ This function computes the x vector for individual's previous choice. 
    Possibilities are: associate,fields,grad,occups,home. Notice that if the
    individual graduated this does not count as a change and I therefore
    will set it as 0"""

    x_change = np.zeros(1+fields+occupations+1)

    
    if period > 1: # Notice that otherwise nothing has changed. 

        # Check if last period was associate degree and no graduation
        
        if (x2[7] == (period -1)) & (x2[9]==12) & (x2[8]!=12):
            
            x_change[0] = 1
        
        # Check if last period was a bachelor period and no graduation
        
        elif (x2[7] == (period-1)) & (x2[9]!=12) & (x2[9]!=13) & (x2[8]!=x2[9]):
            
            
            x_change[int(x2[9])] = 1
            
        # Check if last period was an occupation period
        
        elif (x2[7]!= (period-1)) & (x2[9]!=0):
            
            if x2[9] < 4:
            
                x_change[int(fields+x2[9])] = 1
                
            else:
                
                x_change[int(fields+x2[9]-2)] = 1
        # Check if last period was a graduate school period
        
            # Nothing NOW here
        
        # Check if last period was home production
        
        elif (x2[7]!=period-1) & (x2[9]==0):
            
            x_change[-1] = 1        
    
    return x_change

def get_x_change(period,x2,fields=8,occupations=8):
    
    
    x_change = np.zeros((np.shape(x2)[0],1+fields+occupations+1))
    
    for i in range(np.shape(x2)[0]):
        
        x_change[i,:] = get_x_change_i(x2[i,:],period,fields,occupations)
        
    return x_change 

    
def get_x_educ_i(x2,period,fields=8):
    
    """This function computes the vector of education status for the indiviual
    Entries are:  associate, bachelor, master. No educ is base category"""
    
    x_educ = np.zeros(1+(fields-1)+1)
    
    
    if period > 1:
        
        # Check if the individual has an associate degree
        
        if (x2[8]==12):
            
            x_educ[0] = 1
            
        # Check if the individual has a bachelor degree
        
        elif (x2[5] == 1):
            
            if x2[8] < 3:
            
                x_educ[x2[8]] = 1
            else:
                x_educ[x2[8]-1] = 1
            
            # Now check if on top of that has a graduate degree
            
            if (x2[6]==1):
                
                x_educ[-1] = 1
    
    return x_educ


def get_x_educ(period,x2,fields=8):
    
    
    x_educ = np.zeros((np.shape(x2)[0],1+(fields-1)+1))
    
    for i in range(np.shape(x2)[0]):
        
        x_educ[i,:] = get_x_educ_i(x2[i,:],period,fields)
        
    return x_educ

def get_x_afqt_first(x1,x2,period,coltype):
    
    """
    This function returns a dummy for your afqt if it is your first time enrolled
    and 0 otherwise. 
    
    coltyp 1 -- associate degree
    coltyp 2 -- bachelor degree
    """
    x_afqt = np.zeros(4)
    
    #if x2[coltype] > 0: 
        
        #pass
    #else:
        #if coltype == 1:
            #x_afqt = np.ones(4)
        #else: 
            #x_afqt[int(x1[0,1]-1)] = 1
    
    if coltype == 1: 
        
        if period> 1:
            if (x2[7] == (period-1)) & (x2[9]==12):
                x_afqt = np.ones(4)
                
    
    if coltype == 2:
            
        if period > 1:
            if (x2[7] ==(period-1)) & (x2[9]<9):
                x_afqt[int(x1[1]-1)] = 1
                
    if coltype == 3:
            
        if period > 5:
            if (x2[7] ==(period-1)) & (x2[9]==13):
                x_afqt= 1 
            else:
                x_afqt = 0
        else:
            x_afqt = 0
       
    return x_afqt
    

def get_x_first(x1,x2,period,coltype):
    
    if coltype != 3: 
    
        x_first = np.zeros((np.shape(x2)[0],4))
            
    else: 
        
        x_first = np.zeros((np.shape(x2)[0],1))
        
    for i in range(np.shape(x2)[0]):
            
        x_first[i,:] =   get_x_afqt_first(x1[i,:],x2[i,:],period,coltype) 
        
    return x_first
    
def get_x_g():
    
    """This function generates all the xs matrices for the 
    g function to avoid doing that at each iteration of the likelihood. The
    ony remaining is the one corresponding to working full or part time which
    does not require a matrix."""
    
    total_choices = get_total_choices()
    
    # Now the x2_g matrix that depends on each period
    
    names = []
    result_change = []
    result_x1 = []
    result_educ = []
    result_first2 = []
    result_first4 = []
    
    for period in range(1,T):
        
        print(f"Preparing peroid {period}")
        
        x1,state,debt,choices = get_data_feasible(period)
        #x1,state,debt,choices = get_data_simulated(period)
        
        # Now get the matrix of changes
        
        x1_new = get_x1_new(x1)
        
        x_change = get_x_change(period,state)
        
        x_educ = get_x_educ(period,state)
        
        x_first2 = get_x_first(x1,state,period,1)
        
        x_first4 = get_x_first(x1,state,period,2)
        
        result_change.append(x_change)
        result_x1.append(x1_new)
        result_educ.append(x_educ)
        result_first2.append(x_first2)
        result_first4.append(x_first4)
        names.append(f"period{period}")
        
    np.savez_compressed(LIK("x_super_change.npz"), **{name: value for name, value in zip(names, result_change)})
    np.savez_compressed(LIK("x1_new.npz"),**{name:value for name,value in zip(names,result_x1)})
    np.savez_compressed(LIK("x_educ.npz"),**{name:value for name,value in zip(names,result_educ)})
    np.savez_compressed(LIK("x_first2.npz"),**{name:value for name,value in zip(names,result_first2)})
    np.savez_compressed(LIK("x_first4.npz"),**{name:value for name,value in zip(names,result_first4)}) 
        

@njit()
def get_x_exp(x1,x2):
    
    """
    This funciton returs the x_exp for the experience aftq effects.
    """
    
    x_exp = np.zeros(6*4)
    
    ability = x1[1] - 1
    
    exp = np.minimum(x2[2],5)
    
    x_exp[int(exp + ability*6)] = 1
    
    return x_exp

def get_x_exp_all(x1,x2):
    
    x_exp = np.zeros((np.shape(x2)[0],4*6))
    
    for i in range(np.shape(x2)[0]):
        
        x_exp[i,:] =   get_x_exp(x1[i,:],x2[i,:]) 
    
    return x_exp
    


def get_x_g_superfeasible():
    
    """This function generates all the xs matrices for the 
    g function to avoid doing that at each iteration of the likelihood. The
    ony remaining is the one corresponding to working full or part time which
    does not require a matrix."""
    
    total_choices = get_total_choices()
    
    # Now the x2_g matrix that depends on each period
    
    names = []
    result_change = []
    result_x1 = []
    result_educ = []
    result_first2 = []
    result_first4 = []
    result_firstgrad = []
    result_exp = []
    
    for period in range(1,T):
        
        print(f"Preparing peroid {period}")
        
        x1,state,debt,choices = load_data_superfeasible(period, return_income=False)
        #x1,state,debt,choices = get_data_simulated(period)
        
        # Now get the matrix of changes
        
        x1_new = get_x1_new(x1)
        
        x_change = get_x_change(period,state)
        
        x_educ = get_x_educ(period,state)
        
        x_first2 = get_x_first(x1,state,period,1)
        
        x_first4 = get_x_first(x1,state,period,2)
        
        x_firstgrad = get_x_first(x1,state,period,3)
        
        x_exp = get_x_exp_all(x1,state)
        
        result_change.append(x_change)
        result_x1.append(x1_new)
        result_educ.append(x_educ)
        result_first2.append(x_first2)
        result_first4.append(x_first4)
        result_firstgrad.append(x_firstgrad)
        result_exp.append(x_exp)
        names.append(f"period{period}")
        
    np.savez_compressed(LIK("x_super_change.npz"),**{name:value for name,value in zip(names,result_change)})
    np.savez_compressed(LIK("x1_super_new.npz"),**{name:value for name,value in zip(names,result_x1)})
    np.savez_compressed(LIK("x_super_educ.npz"),**{name:value for name,value in zip(names,result_educ)})
    np.savez_compressed(LIK("x_super_first2.npz"),**{name:value for name,value in zip(names,result_first2)})
    np.savez_compressed(LIK("x_super_first4.npz"),**{name:value for name,value in zip(names,result_first4)}) 
    np.savez_compressed(LIK("x_super_firstgrad.npz"),**{name:value for name,value in zip(names,result_firstgrad)}) 
    np.savez_compressed(LIK("x_super_exp.npz"),**{name:value for name,value in zip(names,result_exp)}) 

    pass
    
def get_x1_new(x1):
    
    # includes constant!
    
    x1_new = np.zeros(shape=(np.shape(x1)[0],9))
    for i in range(np.shape(x1)[0]):
        x1a = x1[i,:]
        parinc  = np.array([np.concatenate((np.zeros(int(x1a[0]-1)),np.array([1]),np.zeros(int(4-x1a[0]))),axis=0)],)
        ability = np.array([np.concatenate((np.zeros(int(x1a[1]-1)),np.array([1]),np.zeros(int(4-x1a[1]))),axis=0)],)
    
        x1_new[i,:] = np.append(1,np.array([np.concatenate((parinc[0,1:],ability[0,1:],x1a[2:4]))],))
        
    return x1_new
    
#@njit()
def get_all_g(utility_parameters,x1_new,x_change_p,x_educ_p,x_first2_p,x_first4_p,x_firstgrad_p,x_exp_p,period,em_type):
    
    """This function computes all g() values for a given individual"""
    
    # First get the parameters
    param_g = utility_parameters[0]
    param_g_work = utility_parameters[1]
    param_g_last = utility_parameters[2]
    param_g_educ = utility_parameters[3]
    param_g_period = utility_parameters[4]
    param_g_period_work = utility_parameters[5]
    param_g_first= utility_parameters[6]
    param_g_first_2 = param_g_first[0]
    param_g_first_4 = param_g_first[1]
    param_g_first_grad = param_g_first[2]
    param_g_exp = utility_parameters[7]
    param_type = utility_parameters[8]
    
    # Perform the different parts
    
    g_x1 = x1_new@param_g.T
      
    g_work  = param_g_work   
    
    g_change = x_change_p@param_g_last.T
    
    g_educ = x_educ_p@param_g_educ.T
    
    g_period = param_g_period[period-1,:]
    
    g_period_work = param_g_period_work[period-1,:]
    
    g_first2 = x_first2_p[:,0][...,None]@param_g_first_2[:,0][...,None].T  # solving the issue I created with unnecesarry columns
    
    g_first4 = x_first4_p@param_g_first_4.T
    
    g_firstgrad = x_firstgrad_p@param_g_first_grad.T
    
    g_exp = x_exp_p@param_g_exp.T
    
    # Sum all together
    
    # The joint type also contains grant and transfer components. Only its
    # schooling component enters direct utility; the financial components enter
    # the budget constraint when the model is solved.
    school_type, _, _, _ = type_components(em_type)
    if school_type == 0:
        g_type = 0
    else:
        g_type = param_type[:,0]
        
    g = g_x1  + g_work +g_change   + g_educ + g_period + g_period_work + g_first2 + g_first4 + g_firstgrad + g_exp + g_type

    return g

def get_choices_index(choices):
    
    total_choices = get_total_choices()
    
    idx = np.where( (total_choices==choices[:,None]).all(-1) )[1]
    
    return idx
    
def load_all_arrays():
    
    choices_all = []
    vjt_all = []
    choices_array_all = []
    
    for period in range(1,T):
        
        x1,state,debt,choices = get_data_feasible(period)
        
        #x1,state,debt,choices = get_data_simulated(period)
        
        vjt = np.load(f"likelihood/vjt_t{period}.npy")
        
        choices_index = get_choices_index(choices)
        
        choices_all.append(choices)
        vjt_all.append(vjt)
        choices_array_all.append(choices_index)
        
    x1_new = np.load("{pathout}/likelihood/x1_new.npz")
    x_change = np.load("{pathout}/likelihood/x_change.npz")
    x_educ = np.load("{pathout}/likelihood/x_educ.npz")
    x_first2 = np.load("{pathout}/likelihood/x_first2.npz")
    x_first4 = np.load("{pathout}/likelihood/x_first4.npz")
    
    return choices_all, vjt_all, x1_new, choices_array_all, x_change, x_educ, x_first2, x_first4


def load_all_arrays_feasible(auxiliar=0):
    
    choices_all = []
    # Outer index is the zero-based position in TYPE_IDS; inner index is period.
    # A list is retained because the number of observations may differ by period.
    vjt_all_types = [[] for _ in TYPE_IDS]
    choices_array_all = []
    
    for period in range(1,T):
        
        x1,state,debt,choices = load_data_superfeasible(period, return_income=False)
        
        #x1,state,debt,choices = get_data_simulated(period)
        if auxiliar == 0:
            for type_index, em_type in enumerate(TYPE_IDS):
                vjt_all_types[type_index].append(
                    np.load(
                        f"{pathout}/likelihood/"
                        f"vjt_super_t{period}_em{em_type}.npy"
                    )
                )
        
        choices_index = get_choices_index(choices)
        
        choices_all.append(choices)
        choices_array_all.append(choices_index)
        
    x1_new = np.load(f"{pathout}/likelihood/x1_super_new.npz")
    x_change = np.load(f"{pathout}/likelihood/x_super_change.npz")
    x_educ = np.load(f"{pathout}/likelihood/x_super_educ.npz")
    x_first2 = np.load(f"{pathout}/likelihood/x_super_first2.npz")
    x_first4 = np.load(f"{pathout}/likelihood/x_super_first4.npz")
    x_firstgrad = np.load(f"{pathout}/likelihood/x_super_firstgrad.npz")
    x_exp = np.load(f"{pathout}/likelihood/x_super_exp.npz")
    
    return choices_all, vjt_all_types, x1_new, choices_array_all, x_change, x_educ, x_first2, x_first4, x_firstgrad, x_exp

@njit()
def get_choices_array(choices):
    
    
    choices_array = np.zeros((np.shape(choices)[0],68))
    
    for i in range(np.shape(choices)[0]):
        
        choices_array[i,choices[i]] = 1
    
    return choices_array

@njit()
def get_column_choice(vjt,choices):
    
    vjt_column = np.zeros((np.shape(vjt)[0],1))
    for i in range(np.shape(vjt)[0]):
        vjt_column[i,0] = vjt[i,choices[i]]
        
    return vjt_column


@njit()
def numba_tile_new(x,reps):
    
    """This function performs the same as tile but in a 
    numba usable way"""
    shape_x = np.shape(x)[0]
    final  = np.zeros(shape_x*reps)
    for i in range(reps):
        
        final[i*shape_x:(i+1)*shape_x] = x
        
    return final


def build_param_x1(param_g_x1_temp,size,fields,occupation):
    """"This function builds the matrix of param_g_x1 considering the order in
    get_total_choices, which is:
    - 3 rows associate degree
    - 3 rows for each field
    - 2 rows for each occupation
    - 3 rows for grad school
    - row of zeros for home production"""
    
    param_g_x1 = np.zeros((size,9))

    # Associate degree    
    
    param_g_x1[0,:] = param_g_x1_temp[0,:]
    param_g_x1[1,:] = param_g_x1_temp[0,:]  
    param_g_x1[2,:] = param_g_x1_temp[0,:]
    
    # For fields
    
    for field in range(fields):
        
        param_g_x1[3+field*3,:] = param_g_x1_temp[field+1,:]
        param_g_x1[4+field*3,:] = param_g_x1_temp[field+1,:]
        param_g_x1[5+field*3,:] = param_g_x1_temp[field+1,:]
        
    # For occupations
    
    startocc = 3 + fields*3
    
    for occ in range(occupation):
        
        param_g_x1[startocc+occ*2,:]    = param_g_x1_temp[1+fields+occ,:]
        param_g_x1[startocc+1+occ*2,:]  = param_g_x1_temp[1+fields+occ,:]        
        
    # For grad school
    
    param_g_x1[3+fields*3+occupation*2,:]   = param_g_x1_temp[1+fields+occupation,:]
    param_g_x1[3+fields*3+occupation*2+1,:] = param_g_x1_temp[1+fields+occupation,:]
    param_g_x1[3+fields*3+occupation*2+2,:] = param_g_x1_temp[1+fields+occupation,:]    
    
    
    return param_g_x1

def build_param_period(param_g_period_temp,size,fields,occupation):
    """"This function builds the matrix of param_g_period considering the order in
    get_total_choices, which is:
    - 3 rows associate degree
    - 3 rows for each field
    - 2 rows for each occupation
    - 3 rows for grad school
    - row of zeros for home production"""
    
    # First reshape and store the grad school parameters
    param_g_grad = param_g_period_temp[(1+fields+occupation)*(T-2):]
    param_g_period_temp = param_g_period_temp[:(1+fields+occupation)*(T-2)].reshape((1+fields+occupation,T-2))
    
    param_g_period = np.zeros((size,T-1))

    # Associate degree    
    
    param_g_period[0,1:] = param_g_period_temp[0,:]
    param_g_period[1,1:] = param_g_period_temp[0,:]  
    param_g_period[2,1:] = param_g_period_temp[0,:]
    
    # For fields
    
    for field in range(fields):
        
        param_g_period[3+field*3,1:] = param_g_period_temp[field+1,:]
        param_g_period[4+field*3,1:] = param_g_period_temp[field+1,:]
        param_g_period[5+field*3,1:] = param_g_period_temp[field+1,:]
        
    # For occupations
    
    startocc = 3 + fields*3
    
    for occ in range(occupation):
        
        param_g_period[startocc+occ*2,1:]    = param_g_period_temp[1+fields+occ,:]
        param_g_period[startocc+1+occ*2,1:]  = param_g_period_temp[1+fields+occ,:]        
        
    # For grad school
    
    # add the 0s of the first 4 periods
    if T>5:
        temp  = np.concatenate((np.zeros(4),param_g_grad))
        
        param_g_period[3+fields*3+occupation*2,1:]   = temp
        param_g_period[3+fields*3+occupation*2+1,1:] = temp
        param_g_period[3+fields*3+occupation*2+2,1:] = temp   
    
    return param_g_period.T

def build_param_period_work(param_g_period_temp,size,fields,occupation):
    """"This function builds the matrix of param_g_period_work considering the order in
    get_total_choices, which is:
    - 3 rows associate degree (no work, part time, fulltime)
    - 3 rows for each field (no work, part time, fulltime)
    - 2 rows for each occupation (part time, fulltime)
    - 3 rows for grad school (no work, part time, fulltime)
    - row of zeros for home production"""
    
    # Define the object
    param_g_period = np.zeros((size,T-1))
    
    # First Get The Associate + Fields + Occupation Parameters
    
    param_g_period_nograd = param_g_period_temp[:(T-1-1)*3].reshape((3,T-1-1))
    
    param_g_period_grad = param_g_period_temp[(T-1-1)*3:(T-1-1)*3+(T-5-1)*2].reshape(2,T-5-1)
    
    
    # For Associate Degree and Fields
    
    for field in range(1+fields):
        
        param_g_period[1+field*3,1:] = param_g_period_nograd[0,:]
        param_g_period[2+field*3,1:] = param_g_period_nograd[1,:]
        
    # For occupations
    
    startocc = 3 + fields*3
    
    for occ in range(occupation):
        
        param_g_period[startocc+1+occ*2,1:]  = param_g_period_nograd[2,:]        
        
    # For grad school
    
    # add the 0s of the first 4 periods
    if T>5:
        
        param_g_period[3+fields*3+occupation*2+1,5:] = param_g_period_grad[0,:]
        param_g_period[3+fields*3+occupation*2+2,5:] = param_g_period_grad[1,:]
    
    return param_g_period.T

def build_param_g_work(param_work_temp,fields,occupation,size):
    
    """This function transform the aprameters of param g work into
    a matrix so that we can use it."""
    
    param_g_work = np.zeros((size))
    
    # first associate degree
    
    param_g_work[1] = param_work_temp[0]
    param_g_work[2] = param_work_temp[1]
    
    # now fields
    for field in range(fields):
        param_g_work[4+field*3] = param_work_temp[(1+field)*2]
        param_g_work[5+field*3] = param_work_temp[(1+field)*2 +1]
   
    # now for occupations
    
    startocc = 2 + fields*2 
    startocc2 = 3 + fields*3
    
    for occ in range(occupation):
        
        param_g_work[startocc2+1+2*occ] = param_work_temp[startocc+occ]
        
    # now for graduate school
    
    param_g_work[startocc2+2*occupation+1] = param_work_temp[startocc+occupation]
    param_g_work[startocc2+2*occupation+2] = param_work_temp[startocc+occupation+1]
    
    
    param_g_work = param_g_work[...,None].T
    
    return param_g_work


def build_param_g_last(param_g_last_temp,fields,occupation,size):
    
    """ This function builds the matrix of parameters for param_g last. It
    is a matrix Jx(associate,fields,grad,occupations,home) dimension.
    - Majors are affected by changes in majors. (also associate)
    - Occupations are affected by changes in occupations. 
    - If you come from home production you are also changing. 
    Notice that if last period an individual got graduated, that is not a change
    """
    total = 1+ fields + 1 + occupation
    param_g_last = np.zeros((size,total))
    
    # associate degree. It last choice was associate degree or occupation,
    # the individual should remain unafected
    
    param_g_last[0,1:fields+1] = np.repeat(param_g_last_temp[0],fields)
    param_g_last[1,1:fields+1] = np.repeat(param_g_last_temp[0],fields)
    param_g_last[2,1:fields+1] = np.repeat(param_g_last_temp[0],fields)
    
    # also if there was home production
    
    param_g_last[0,-1] = param_g_last_temp[0]
    param_g_last[1,-1] = param_g_last_temp[0]
    param_g_last[2,-1] = param_g_last_temp[0]
    
    # now for fields
    
    for field in range(fields):
        
        # set the parameter for changing into this field
        param_g_last[3+field*3,0:fields+1]   = np.repeat(param_g_last_temp[field+1],fields+1) # fields+ associate degree
        param_g_last[3+field*3+1,0:fields+1] = np.repeat(param_g_last_temp[field+1],fields+1) # fields+ associate degree
        param_g_last[3+field*3+2,0:fields+1] = np.repeat(param_g_last_temp[field+1],fields+1) # fields+ associate degree
        # set it to 0 if your last field was this one
        param_g_last[3+field*3,1+field]   = 0
        param_g_last[3+field*3+1,1+field] = 0
        param_g_last[3+field*3+2,1+field] = 0
        # set it as well if you are coming from home production
        param_g_last[3+field*3,-1]   = param_g_last_temp[field+1]
        param_g_last[3+field*3+1,-1] = param_g_last_temp[field+1]
        param_g_last[3+field*3+2,-1] = param_g_last_temp[field+1]
        
    # now for occupations: 
        
    startocc = 3+fields*3 
        
    for occ in range(occupation):
        
        # set the parameter for changing into this occupation
        param_g_last[startocc+occ*2,1+fields:1+fields+occupation]   = np.repeat(param_g_last_temp[1+fields+occ],occupation)
        param_g_last[startocc+occ*2+1,1+fields:1+fields+occupation] = np.repeat(param_g_last_temp[1+fields+occ],occupation)
        # set it to 0 if your last occupation was this one
        param_g_last[startocc+occ*2,fields+occ+1]   = 0
        param_g_last[startocc+occ*2+1,fields+occ+1] = 0
        # set it as well if you are coming from home production
        param_g_last[startocc+occ*2,-1]   = param_g_last_temp[1+fields+occ]
        param_g_last[startocc+occ*2+1,-1] = param_g_last_temp[1+fields+occ]
        
    return param_g_last

def get_param_educ(param_educ_temp,fields,occupation,size):
    """This function returns the parameters of your education effect on the 
    different fields. So for each of your education posibilities, there is
    an effect on each occupation. It is like x1 params, but the effect will 
    only be at occupations. The possible education categories are:
    no educ (base category), asociate, fields and grad: 1+fields+1. 
    
    Important: I am setting fields-1 since major 3 can't have any effect
    
    """
    
    param_educ = np.zeros((size,1+(fields-1)+1))
    occstart = (1+fields)*3  # associate + bachelor posibilities
    # Replace for each occupation choice: 
        
    # Business
    param_educ[occstart,:] = param_educ_temp[:9]  # They all have an effect
    param_educ[occstart+1,:] = param_educ_temp[:9]  # They all have an effect

    # STEM
    param_educ[occstart+2,0] = param_educ_temp[9]  #Associate Degree
    param_educ[occstart+2,2] = param_educ_temp[10]  #STEM
    param_educ[occstart+2,7] = param_educ_temp[11]  #Other Fields
    param_educ[occstart+2,8] = param_educ_temp[12]  #Grad Degree
    
    param_educ[occstart+3,0] = param_educ_temp[9]  #Associate Degree
    param_educ[occstart+3,2] = param_educ_temp[10]  #STEM
    param_educ[occstart+3,7] = param_educ_temp[11]  #Other Fields
    param_educ[occstart+3,8] = param_educ_temp[12]  #Grad Degree
    
    #Social Sciences
    param_educ[occstart+4,4] = param_educ_temp[13]  #Social Sciences
    param_educ[occstart+4,5] = param_educ_temp[14]  #Humanitites
    param_educ[occstart+4,7] = param_educ_temp[15]  #Other Fields
    param_educ[occstart+4,8] = param_educ_temp[16]  #Grad Degree
    
    param_educ[occstart+5,4] = param_educ_temp[13]  #Social Sciences
    param_educ[occstart+5,5] = param_educ_temp[14]  #Humanitites
    param_educ[occstart+5,7] = param_educ_temp[15]  #Other Fields
    param_educ[occstart+5,8] = param_educ_temp[16]  #Grad Degree
    
    #Education
    param_educ[occstart+6,3] = param_educ_temp[17]  #Education
    param_educ[occstart+6,4] = param_educ_temp[18]  #Social Sciences
    param_educ[occstart+6,5] = param_educ_temp[19]  #Humanities
    param_educ[occstart+6,6] = param_educ_temp[20]  #Health
    param_educ[occstart+6,8] = param_educ_temp[21]  #Grad Degree
    
    param_educ[occstart+7,3] = param_educ_temp[17]  #Education
    param_educ[occstart+7,4] = param_educ_temp[18]  #Social Sciences
    param_educ[occstart+7,5] = param_educ_temp[19]  #Humanities
    param_educ[occstart+7,6] = param_educ_temp[20]  #Health
    param_educ[occstart+7,8] = param_educ_temp[21]  #Grad Degree
    
    #Humanities
    param_educ[occstart+8,0] = param_educ_temp[22]  #Associate Degree
    param_educ[occstart+8,5] = param_educ_temp[23]  #Humanities
    param_educ[occstart+8,8] = param_educ_temp[24]  #Grad Degree
    
    param_educ[occstart+9,0] = param_educ_temp[22]  #Associate Degree
    param_educ[occstart+9,5] = param_educ_temp[23]  #Humanities
    param_educ[occstart+9,8] = param_educ_temp[24]  #Grad Degree
    
    #Health
    param_educ[occstart+10,0] = param_educ_temp[25]  #Associate Degree
    param_educ[occstart+10,2] = param_educ_temp[26]  #STEM
    param_educ[occstart+10,6] = param_educ_temp[27]  #Health
    param_educ[occstart+10,8] = param_educ_temp[28]  #Graduate Degere
    
    param_educ[occstart+11,0] = param_educ_temp[25]  #Associate Degree
    param_educ[occstart+11,2] = param_educ_temp[26]  #STEM
    param_educ[occstart+11,6] = param_educ_temp[27]  #Health
    param_educ[occstart+11,8] = param_educ_temp[28]  #Graduate Degere
    
    #Services
    param_educ[occstart+12,:] = param_educ_temp[29:38]
    
    param_educ[occstart+13,:] = param_educ_temp[29:38]
    
    #Production
    param_educ[occstart+14,:] = param_educ_temp[38:47]
    
    param_educ[occstart+15,:] = param_educ_temp[38:47]
        
    return param_educ

def build_param_first(param_first_temp,fields,occupations,size):
    """
    This function creates the parameters for the first time enrollment cost. 
    There are 5 parameters. One for associate degree, and one for each afqt for
    4y schools. I will build this function so that it multiplies afqt, which
    means it will be JX4.

    """
    
    param_first2 = np.zeros((size,4))
    param_first4 = np.zeros((size,4))
    param_firstgrad = np.zeros((size,1))
    
    # The first 3 rows are for associate degree
    
    param_first2[:3,:] = param_first_temp[0]
    
    # Now the others depend on afqt. Loop over fields
    
    param_first4[3:3+fields*3,:] = param_first_temp[1:-1]
    
    param_firstgrad[(3+fields*3+2*occupations):(3+fields*3+2*occupations)+3] = param_first_temp[-1]
    
        
    param_first = [param_first2,param_first4,param_firstgrad]
    
    return param_first

def build_param_type(param_type_temp,fields,size):
    
    """
    This function creates the paramers for the effect of the unobserved types
    that will be recovered with the EM-algorithm. Belonging to this type has
    an effect on associate degree and 4y schools. The effect is different, so 
    there are only two parameters.
    """
    
    param_type = np.zeros((size,1))
    
    # Replace the first 3 with the first parameter
    
    param_type[:3] = param_type_temp[0]
    
    # The next fields with the other
    
    param_type[3:(fields+1)*3] = param_type_temp[1]
    
    return param_type

def build_param_exp(param_g_exp_temp,fields,size):
    """"This function builds the matrix of experience ability
    effects on fields of study. The matrix will be
    Total Choices X (ExperienceXAFQTLevel)"""
    
    exp = 6 
    afqt = 4
    
    param_g_exp = np.zeros((size,exp*afqt))

    # I will assume parameters come by order of experience, aftq. 
    # So first exp 0 , afqt=1, the exp 1 afqt =1,....
    
    param_g_exp_temp = param_g_exp_temp.reshape((fields-1,exp*afqt))
    
    # Now replace at the fields
    
    for field in range(fields):
        
        if field < 2:
            param_g_exp[3+field*3,:] = param_g_exp_temp[field,:]
            param_g_exp[3+field*3+1,:] = param_g_exp_temp[field,:]
            param_g_exp[3+field*3+2,:] = param_g_exp_temp[field,:]
            
        if field > 2:
            
            param_g_exp[3+(field)*3,:] = param_g_exp_temp[field-1,:]
            param_g_exp[3+(field)*3+1,:] = param_g_exp_temp[field-1,:]
            param_g_exp[3+(field)*3+2,:] = param_g_exp_temp[field-1,:]
    
    return param_g_exp

def build_param_g(param_utility):
    
    """This function takes as inputs the vector of utility parameters and consutcs
    the different matrix with the parameters. Those are
    
    param_g_x1      ---   associated with preferences of individulas
    param_g_work    ---   full vs part time preference dislike of work
    param_g_last    ---   last choice effect on current choice
    param_g_educ    ---   effect of education level on occupations
    param_g_period  ---   period effect on choices
    param_g_first   ---   cost of first time enrollment 
    
    """
    
    size = np.shape(get_total_choices())[0]
     
    fields = 8
    occupation = 8
    
    # First get the x1 parameters. For this we have one for each category:
    # associate degree + number of fields + number of occupations + grad school
    # and notice that there are 9 parameters inside each category
    
    amount_x1 = 9*(1+fields+occupation+1)
    param_g_x1 = param_utility[:amount_x1].reshape((1+fields+occupation+1,9))
    
    param_g_x1 = build_param_x1(param_g_x1,size,fields,occupation)
    
    # Now get the parameters for working full or part time. Those are:
    # 2 for associate, 2 for each field, 1 for each occupation, 2 for grad school. 
    # Total = 2+ 2*fields + occupations + 2
    
    amount_work = (2+2*fields+occupation+2)
    param_g_work = param_utility[amount_x1:amount_x1+amount_work]
    
    param_g_work = build_param_g_work(param_g_work,fields,occupation,size)
    
    # Now get the parameters for last choice. The possible last choices are: 
    # associate + fields + grad + occupations +home production(but home prodution has no coefficient)
    # (and as off now, grad school has no coefficient either!)
    
    amount_last = 1+ fields  + occupation
    param_g_last =  param_utility[amount_x1+amount_work:amount_x1+amount_work+amount_last]
    param_g_last = build_param_g_last(param_g_last,fields,occupation,size)
    
    # Now get the parameters for the education status preferences effects
    # There are in total: no educ (base category), associate, fields, grad: 1+fields+1
    
    amount_educ = get_amount_educ()
    param_educ = param_utility[amount_x1+amount_work+amount_last:amount_x1+amount_work+amount_last+amount_educ]
    param_educ = get_param_educ(param_educ,fields,occupation,size)
    
    
    # Now get the parameters related with period effects
    # There are in total: occupation*(period-1) + field*(period-1) + period-4-1 (grad school)
    
    amount_period = np.maximum((1+fields+occupation)*(T-1-1),0) + np.maximum((T-5-1),0)
    param_period = param_utility[amount_x1+amount_work+amount_last+amount_educ:amount_x1+amount_work+amount_last+amount_educ+amount_period]
    param_period = build_param_period(param_period,size,fields,occupation)
    
    # Now get the parameters related with the period/working effects
    # There are in total: occupation*(period-1) + field*(period-1)*2 + period-4-1 (grad school)
    
    amount_period_work = np.maximum((T-1-1)*3,0)  + np.maximum((T-5-1)*2,0)
    param_period_work = param_utility[amount_x1+amount_work+amount_last+amount_educ+amount_period:amount_x1+amount_work+amount_last+amount_educ+amount_period+amount_period_work]
    param_period_work = build_param_period_work(param_period_work,size,fields,occupation)
    
    # Now get parameters related with  first time enrollment effects
    # In total there are 1 for associates and for each afqt one for 4y school. = 5
    
    amount_first = 1 + 4 + 1
    past = amount_x1+amount_work+amount_last+amount_educ+amount_period + amount_period_work
    param_first = param_utility[past:past+amount_first]
    param_first = build_param_first(param_first,fields,occupations,size)
    
    # Now get parameters related with the experience ability effects
    # In total there are: fieldsX6X4 parameters
    
    amount_exp = (fields-1)*6*4
    past = amount_x1+amount_work+amount_last+amount_educ+amount_period+amount_period_work + amount_first
    param_exp = param_utility[past:past+amount_exp]
    param_exp = build_param_exp(param_exp,fields,size)
    
    # Now for the type parameters. There is one parameter for the effect in
    # 2y schools and 1 for the effect in 4y schools. 
    
    amount_type = 2 
    past = past + amount_exp
    param_type = param_utility[past:past+amount_type]
    param_type = build_param_type(param_type,fields,size)


    utility_parameters = [param_g_x1,param_g_work,param_g_last,param_educ,param_period,param_period_work,param_first,param_exp,param_type]
    
    return utility_parameters
    

def notlogs_likelihood(param_g,choices_all,vjt_all,x1_new,choices_array_all,x_change,x_educ, x_first2, x_first4, x_firstgrad, x_exp, em_type):
    
    """This function computes the likelihood associated with the data"""
    utility_parameters = build_param_g(param_g)

    # Get the data of each period 
    
    likelihood = np.ones((np.shape(choices_all[0])[0],T-1))
    jacobian = np.zeros((np.shape(param_g)[0],T-1))
    
    for period in prange(1,T,1):

        # get the period of interest. 
        choices = choices_all[period-1]
        vjt_notusable = vjt_all[period-1]
        choices_index = choices_array_all[period-1]
        x_change_p = x_change[f"period{period}"]
        x1_new_p = x1_new[f"period{period}"]
        x_educ_p = x_educ[f"period{period}"]
        x_first2_p = x_first2[f"period{period}"]
        x_first4_p = x_first4[f"period{period}"]
        x_firstgrad_p = x_firstgrad[f"period{period}"]
        x_exp_p = x_exp[f"period{period}"]
        
        # Notice that since the base category is the same for everybody, all the
        # g() functions of the future will cancel out. 
        
        # Sum the corresponding g() function to each vjt
        
        g = get_all_g(utility_parameters,x1_new_p,x_change_p,x_educ_p,x_first2_p,x_first4_p,x_firstgrad_p,x_exp_p,period,em_type)
        
        
        #vjt = vjt + g
        vjt = g   
        # set a base category: Remember, always a possible chioce! 
            
        base = -1  # home production
            
        vjt = vjt - np.repeat(vjt[:,base][...,np.newaxis],np.shape(vjt)[1]).reshape(np.shape(vjt))
        
        # now get the exponent
            
        vjt = np.exp(vjt)
            
        # Get the sum
        #vjt= vjt / np.nansum(vjt,axis=1)[...,np.newaxis]
        vjt = vjt / np.repeat(np.sum(vjt,axis=1)[...,None],np.shape(vjt)[1]).reshape(np.shape(vjt))  # I coded nans as -inf, so this should do the job
        
        # Get the choice
        vjt_i = get_column_choice(vjt,choices_index)
        
        
        # Now sum across all individuals
        
        likelihood[:,period-1] = vjt_i[:,0]
        
    return  np.prod(likelihood,axis=1)

def get_likelihood_prod(x0,choices_all,vjt_all,x1_new,choices_array_all,x_change,x_educ, x_first2, x_first4, x_firstgrad, x_exp, measures):
    
    '''
    This function generates the product of the different likelihoods. Those are: 
        1- Product of t multinomial choices
        2- Ordered logit : arrive late at class
        3- Logit Model: Take summer classes 
    '''
    
    global total_n_multi
    global total_n_late
    global total_n_summer
    
    type_numbers = 2
    
    likes = np.zeros((np.shape(choices_all[0])[0],type_numbers))
    
    
    
    #  First get the parameters that correspond to each function -------------#
    
    param_g = x0[:total_n_multi]
    param_late = x0[total_n_multi:total_n+total_n_late]
    param_summer = x0[total_n+total_n_late:total_n+total_n_late+total_n_summer]
    #-------------0
    
    #------------------------------------------------------------#
    
    # First, product of multinomial choices ----------------------------------#

    
    likes[:,0]  = notlogs_likelihood(param_g,choices_all,vjt_all,x1_new,choices_array_all,x_change,x_educ, x_first2, x_first4, x_firstgrad,x_exp,SCHOOL_REPRESENTATIVE_TYPE_IDS[0])
    likes[:,1]  = notlogs_likelihood(param_g,choices_all,vjt_all,x1_new,choices_array_all,x_change,x_educ, x_first2, x_first4, x_firstgrad,x_exp,SCHOOL_REPRESENTATIVE_TYPE_IDS[1])
    
    #-------------------------------------------------------------------------#
    
    # Now  logit for late to school ------------------------------------------#

    likes[:,0] = likes[:,0]*logit_like(param_late,x1_new["period1"],np.array(measures["late_school"]),1)
    likes[:,1] = likes[:,1]*logit_like(param_late,x1_new["period1"],np.array(measures["late_school"]),2)
    
    #-------------------------------------------------------------------------#
    
    # Now  logit for summer classes ------------------------------------------#

    likes[:,0] = likes[:,0]*logit_like(param_summer,x1_new["period1"],np.array(measures["summer_class"]),1)
    likes[:,1] = likes[:,1]*logit_like(param_summer,x1_new["period1"],np.array(measures["summer_class"]),2)
    
    #-------------------------------------------------------------------------#
    
    return likes
    

def get_q(pi,x0,choices_all,vjt_all,x1_new,choices_array_all,x_change,x_educ, x_first2, x_first4, x_firstgrad,x_exp, measures):
    
    
    '''
    This function returns the updated probabilities of beloning to each type. 
    '''
    
    
    likes = get_likelihood_prod(x0,choices_all,vjt_all,x1_new,choices_array_all,x_change,x_educ, x_first2, x_first4, x_firstgrad, x_exp,measures)
    
    temp  = pi*likes
    
    sumtemp = np.sum(temp,axis=1)
    
    q = np.vstack((temp[:,0]/sumtemp,temp[:,1]/sumtemp)).T
    
    return q
        
def get_pi_new(q):
    
    '''
    This function updates the pis based on the posterior
    '''
    
    pinew = np.sum(q,axis=0)
    
    # make sure they sum up to 1
    
    pinew  = pinew / np.sum(pinew)
    
    return pinew
    
#@njit(parallel=True)
def likelihood(param_g,choices_all,vjt_all_types,x1_new,choices_array_all,x_change,x_educ, x_first2, x_first4,x_firstgrad,x_exp,q):
    
    """Compute the posterior-weighted structural likelihood for all joint types.

    ``vjt_all_types[type_index][period_index]`` contains the continuation values
    for one joint type. Posterior columns use the same shared type ordering. The
    posterior slice is passed to the numerical gradient as a one-dimensional
    array, so the hot path never relies on ``em_type - 1`` indexing.
    """
    utility_parameters = build_param_g(param_g)

    if len(choices_all) == 0:
        raise ValueError("choices_all must contain at least one period.")
    q = validate_q(q, n_individuals=np.shape(choices_all[0])[0])
    if len(vjt_all_types) != N_STRUCTURAL_TYPES:
        raise ValueError(
            f"Expected continuation values for {N_STRUCTURAL_TYPES} types; "
            f"received {len(vjt_all_types)}."
        )

    # Get the data of each period 
    
    finallike = np.zeros((N_STRUCTURAL_TYPES,1))
    finaljac = np.zeros((np.shape(param_g)[0],N_STRUCTURAL_TYPES))
    
    for type_index, em_type in enumerate(TYPE_IDS):
        vjt_all = vjt_all_types[type_index]
        type_weights = q[:,type_index]
        school_type = int(TYPE_SCHOOL[type_index])
        likelihood = np.zeros((T-1,1))
        jacobian = np.zeros((np.shape(param_g)[0],T-1))
        for period in prange(1,T,1):
    
            # get the period of interest. 
            choices = choices_all[period-1]
            vjt = vjt_all[period-1]
            choices_index = choices_array_all[period-1]
            x_change_p = x_change[f"period{period}"]
            x1_new_p = x1_new[f"period{period}"]
            x_educ_p = x_educ[f"period{period}"]
            x_first2_p = x_first2[f"period{period}"]
            x_first4_p = x_first4[f"period{period}"]
            x_firstgrad_p = x_firstgrad[f"period{period}"]
            x_exp_p = x_exp[f"period{period}"]
            
            # Notice that since the base category is the same for everybody, all the
            # g() functions of the future will cancel out. 
            
            # Sum the corresponding g() function to each vjt
            
            g = get_all_g(utility_parameters,x1_new_p,x_change_p,x_educ_p,x_first2_p,x_first4_p,x_firstgrad_p,x_exp_p,period,em_type)
            
            
            vjt = vjt + g
   
            # set a base category: Remember, always a possible chioce! 
                
            base = -1  # home production
                
            vjt = vjt - np.repeat(vjt[:,base][...,np.newaxis],np.shape(vjt)[1]).reshape(np.shape(vjt))
            
            # now get the exponent
                
            vjt = np.exp(vjt)
                
            # Get the sum
            #vjt= vjt / np.nansum(vjt,axis=1)[...,np.newaxis]
            vjt = vjt / np.repeat(np.sum(vjt,axis=1)[...,None],np.shape(vjt)[1]).reshape(np.shape(vjt))  # I coded nans as -inf, so this should do the job
            
            # Get the choice
            vjt_i = get_column_choice(vjt,choices_index)
            
            
            # Compute the contribution to the gradiant. 
            jacobian[:,period-1] = jacobian_likelihood_numba(choices_index,vjt,x1_new_p,x_change_p,x_educ_p,x_first2_p,x_first4_p,x_firstgrad_p,x_exp_p,period,type_weights,school_type)
    
            # Now obtain the optimal based on the choice
            
            #vjt_i = vjt[range(np.shape(vjt)[0]),choices]
            
            
            #now get it in logs. 
            
            log_vjt = np.log(vjt_i)
            
            
            # Now sum across all individuals
            
            likelihood[period-1] = np.nansum(log_vjt*type_weights[...,None])
        
        finallike[type_index] = np.nansum(likelihood)
        finaljac[:,type_index] = np.sum(jacobian,axis=1)
        
    # now sum across all periods
    print("likelihood is", np.nansum(finallike))
    return (-1*np.nansum(finallike),-1*np.sum(finaljac,axis=1))
    #return -1*np.nansum(finallike)


def get_vjt_static(utility_parameters,x1_new_p,x_change_p,x_educ_p,x_first2_p,x_first4_p,x_firstgrad_p,x_exp_p,period,em_type):
    
    # Sum the corresponding g() function to each vjt

    g = get_all_g(utility_parameters,x1_new_p,x_change_p,x_educ_p,x_first2_p,x_first4_p,x_firstgrad_p,x_exp_p,period,em_type)
    
    vjt = g 
    
    return vjt
    
def likelihood_simple(param_g,choices_all,vjt_all,x1_new,choices_array_all,x_change,x_educ, x_first2, x_first4,x_firstgrad,x_exp,q,number_types=2):
    
    """This function computes the likelihood associated with the data"""
    utility_parameters = build_param_g(param_g)

    # Get the data of each period 
    
    finallike = np.zeros((2,1))
    finaljac = np.zeros((np.shape(param_g)[0],2))
    
    for em_type in range(1,number_types+1):
        type_index = em_type - 1
        type_weights = q[:,type_index]
        school_type = type_index
        joint_type_id = SCHOOL_REPRESENTATIVE_TYPE_IDS[school_type]
        likelihood = np.zeros((T-1,1))
        jacobian = np.zeros((np.shape(param_g)[0],T-1))
        for period in prange(1,T,1):
    
            # get the period of interest. 
            choices = choices_all[period-1]
            vjt_notusable = vjt_all[period-1]
            choices_index = choices_array_all[period-1]
            x_change_p = x_change[f"period{period}"]
            x1_new_p = x1_new[f"period{period}"]
            x_educ_p = x_educ[f"period{period}"]
            x_first2_p = x_first2[f"period{period}"]
            x_first4_p = x_first4[f"period{period}"]
            x_firstgrad_p = x_firstgrad[f"period{period}"]
            x_exp_p = x_exp[f"period{period}"]
            
            # Notice that since the base category is the same for everybody, all the
            # Get vjt for the static model
            vjt = get_vjt_static(utility_parameters,x1_new_p,x_change_p,x_educ_p,x_first2_p,x_first4_p,x_firstgrad_p,x_exp_p,period,joint_type_id)
            #vjt = vjt + g
            
            # set a base category: Remember, always a possible chioce! 
                
            base = -1  # home production
                
            vjt = vjt - np.repeat(vjt[:,base][...,np.newaxis],np.shape(vjt)[1]).reshape(np.shape(vjt))
            
            # now get the exponent
                
            vjt = np.exp(vjt)
                
            # Get the sum
            #vjt= vjt / np.nansum(vjt,axis=1)[...,np.newaxis]
            vjt = vjt / np.repeat(np.sum(vjt,axis=1)[...,None],np.shape(vjt)[1]).reshape(np.shape(vjt))  # I coded nans as -inf, so this should do the job
            
            # Get the choice
            vjt_i = get_column_choice(vjt,choices_index)
            
            
            jacobian[:,period-1] = jacobian_likelihood_numba(choices_index,vjt,x1_new_p,x_change_p,x_educ_p,x_first2_p,x_first4_p,x_firstgrad_p,x_exp_p,period,type_weights,school_type)

            # Now obtain the optimal based on the choice
            
            #vjt_i = vjt[range(np.shape(vjt)[0]),choices]
            
            
            #now get it in logs. 
            
            log_vjt = np.log(vjt_i)
            
            
            # Now sum across all individuals
            
            likelihood[period-1] = np.nansum(log_vjt*type_weights[...,None])
        
        finallike[em_type-1] = np.nansum(likelihood)
        finaljac[:,em_type-1] = np.sum(jacobian,axis=1)
        
    # now sum across all periods
    print("likelihood is", np.nansum(finallike))
    return (-1*np.nansum(finallike),-1*np.sum(finaljac,axis=1))
    #return -1*np.nansum(finallike)

def temp_jacobian(param_g,choices_all,vjt_type1,vjt_type2,x1_new,choices_array_all,x_change,x_educ, x_first2, x_first4,x_firstgrad,x_exp,q,number_types=2):
    
    """This function computes the likelihood associated with the data"""
    utility_parameters = build_param_g(param_g)

    # Get the data of each period 
    
    finallike = np.zeros((2,1))
    finaljac = np.zeros((np.shape(param_g)[0],2))
    
    for em_type in range(1,number_types+1):
        type_index = em_type - 1
        type_weights = q[:,type_index]
        school_type = type_index
        joint_type_id = SCHOOL_REPRESENTATIVE_TYPE_IDS[school_type]
        if em_type == 1:
            vjt_all = vjt_type1
        else: 
            vjt_all = vjt_type2
        likelihood = np.zeros((T-1,1))
        jacobian = np.zeros((np.shape(param_g)[0],T-1))
        for period in prange(1,T,1):
    
            # get the period of interest. 
            choices = choices_all[period-1]
            vjt = vjt_all[period-1]
            choices_index = choices_array_all[period-1]
            x_change_p = x_change[f"period{period}"]
            x1_new_p = x1_new[f"period{period}"]
            x_educ_p = x_educ[f"period{period}"]
            x_first2_p = x_first2[f"period{period}"]
            x_first4_p = x_first4[f"period{period}"]
            x_firstgrad_p = x_firstgrad[f"period{period}"]
            x_exp_p = x_exp[f"period{period}"]
            
            # Notice that since the base category is the same for everybody, all the
            # g() functions of the future will cancel out. 
            
            # Sum the corresponding g() function to each vjt
            
            g = get_all_g(utility_parameters,x1_new_p,x_change_p,x_educ_p,x_first2_p,x_first4_p,x_firstgrad_p,x_exp_p,period,joint_type_id)
            
            
            vjt = vjt + g

            # set a base category: Remember, always a possible chioce! 
                
            base = -1  # home production
                
            vjt = vjt - np.repeat(vjt[:,base][...,np.newaxis],np.shape(vjt)[1]).reshape(np.shape(vjt))
            
            # now get the exponent
                
            vjt = np.exp(vjt)
                
            # Get the sum
            #vjt= vjt / np.nansum(vjt,axis=1)[...,np.newaxis]
            vjt = vjt / np.repeat(np.sum(vjt,axis=1)[...,None],np.shape(vjt)[1]).reshape(np.shape(vjt))  # I coded nans as -inf, so this should do the job
            
            # Get the choice
            vjt_i = get_column_choice(vjt,choices_index)
            # Compute the contribution to the gradiant. 
            jacobian[:,period-1] = jacobian_likelihood_numba(choices_index,vjt,x1_new_p,x_change_p,x_educ_p,x_first2_p,x_first4_p,x_firstgrad_p,x_exp_p,period,type_weights,school_type)
            
            
            #now get it in logs. 
            
            log_vjt = np.log(vjt_i)
            
            
            # Now sum across all individuals

        
        finallike[em_type-1] = np.nansum(likelihood)
        finaljac[:,em_type-1] = np.sum(jacobian,axis=1)
        
    # now sum across all periods
    print("likelihood is", np.nansum(finallike))
    #return (-1*np.nansum(finallike),-1*np.sum(finaljac,axis=1))
    return -1*np.nansum(finaljac,axis=1)


@njit(parallel=True)
def sum_numba_axis(array):
    
    results = np.zeros(np.shape(array)[1])
    
    for i in prange(np.shape(array)[1]):
        
        results[i] = array[:,i].sum()
        
    return results


@njit()
def sum_numba_axis_special(array):
    
    ones = np.ones(np.shape(array)[0])

    results = array.T@ones      

    return results

def build_previous_affects(x_change_p):
    
    global fields
    global occupations
    
    previous = (1+fields+1+occupations)*9 + 2+2*fields+occupations+2
    
    previousaffects = np.zeros((np.shape(x_change_p)[0],1+fields+occupations))
    
    for i in range(1+fields+occupations):
    
        
        previousaffects[:,i] = get_previous_affects(i,x_change_p)
        
        
    return previousaffects


        

def jacobian_likelihood_numba(choices_index,vjt,x1_new_p,x_change_p,x_educ_p,x_first2_p,x_first4_p,x_firstgrad_p,x_exp_p,period,type_weights,school_type):
    
    tic = time.time()
    # Something for the Jacobian
    affected = map_param_to_choice(fields,occupations)
    
    #Previous Effects for the Jacobian as well:
    previousaffects = build_previous_affects(x_change_p)
    toc = time.time()
    #print("Prepare this things",toc-tic)
    # Compute the contribution to the gradiant. 
    jacobian_t = jacobian_likelihood(choices_index,vjt,x1_new_p,x_change_p,x_educ_p,x_first2_p,x_first4_p,x_firstgrad_p,x_exp_p,period,type_weights,school_type,affected,previousaffects)

    return jacobian_t


@njit()
def jacobian_education_effects(previous,ccps,choices,affected,x_educ_p,type_weights):
    
    """
    This function computes the jacobian wrt the education complementarity effects
    given that it is quite complex to have. I will do the derivative for each
    occuaption, the difficulty is that there is a different amount of parameters
    that affects each occupation. 
    """
    
    jactemp = np.zeros(get_amount_educ())
    
    # For Business: (Easy, since there are 9 parameters)
    
        
    temp = affected[previous,:]
    
    check = ccps@temp
    
    idx = temp[choices]
    
    derivative  = (idx-check)[...,None]*x_educ_p*type_weights[...,None]
    
    jactemp[:9] = sum_numba_axis(derivative)
    
    # For STEM
    
    temp = affected[previous+9,:]
    
    check = ccps@temp
    
    idx = temp[choices]
    
    x_educ_stem = np.concatenate((x_educ_p[:,0][...,None],x_educ_p[:,2][...,None],
                                  x_educ_p[:,7][...,None],x_educ_p[:,8][...,None]),axis=1)
    
    derivative  = (idx-check)[...,None]*x_educ_stem*type_weights[...,None]
    
    jactemp[9:13] = sum_numba_axis(derivative)
    
    # For Social Sciences
    
    temp = affected[previous+13,:]
    
    check = ccps@temp
    
    idx = temp[choices]
    
    x_educ_social = np.concatenate((x_educ_p[:,4][...,None],x_educ_p[:,5][...,None],
                                    x_educ_p[:,7][...,None],x_educ_p[:,8][...,None]),axis=1)
    
    derivative  = (idx-check)[...,None]*x_educ_social*type_weights[...,None]
    
    jactemp[13:17] = sum_numba_axis(derivative)
    
    # For Education
    
    temp = affected[previous+17,:]
    
    check = ccps@temp
    
    idx = temp[choices]
    
    x_educ_educ = np.concatenate((x_educ_p[:,3:7],x_educ_p[:,8][...,None]),axis=1)
    
    derivative  = (idx-check)[...,None]*x_educ_educ*type_weights[...,None]
    
    jactemp[17:22] = sum_numba_axis(derivative)
    
    # For Humanities
    
    temp = affected[previous+22,:]
    
    check = ccps@temp
    
    idx = temp[choices]
    
    x_educ_human = np.concatenate((x_educ_p[:,0][...,None],x_educ_p[:,5][...,None],
                                   x_educ_p[:,8][...,None]),axis=1)
    
    derivative  = (idx-check)[...,None]*x_educ_human*type_weights[...,None]
    
    jactemp[22:25] = sum_numba_axis(derivative)
    
    
    # For Health
    
    temp = affected[previous+25,:]
    
    check = ccps@temp
    
    idx = temp[choices]
    
    x_educ_health = np.concatenate((x_educ_p[:,0][...,None],x_educ_p[:,2][...,None],
                                    x_educ_p[:,6][...,None],x_educ_p[:,8][...,None]),axis=1)
    
    derivative  = (idx-check)[...,None]*x_educ_health*type_weights[...,None]
    
    jactemp[25:29] = sum_numba_axis(derivative)
    
    # For Services
    
    temp = affected[previous+29,:]
    
    check = ccps@temp
    
    idx = temp[choices]
    
    derivative  = (idx-check)[...,None]*x_educ_p*type_weights[...,None]
    
    jactemp[29:38] = sum_numba_axis(derivative)
    
    # For Production
    
    temp = affected[previous+38,:]
    
    check = ccps@temp
    
    idx = temp[choices]
    
    derivative  = (idx-check)[...,None]*x_educ_p*type_weights[...,None]
    
    jactemp[38:47] = sum_numba_axis(derivative)
    
    return jactemp

@njit(parallel=True)
def numba_broadcast(a,x,q):
    
    size = np.shape(x)[1]
    
    results = np.zeros(np.shape(x))
    
    for i in range(size):
        
        results[:,i] = a*x[:,i]*q
    
    return results

@njit()
def numba_multiply(a,x,q):
    
    size = np.shape(x)
    
    a = np.repeat(a,size[1]).reshape(size)
    q = np.repeat(q,size[1]).reshape(size)
    
    return a*x*q
    

@njit(parallel=True) 
def jacobian_likelihood(choices,ccps,x1_new_p,x_change_p,x_educ_p,x_first2_p,x_first4_p,x_firstgrad_p,x_exp_p,period,type_weights,school_type,affected,previousaffects, fields=8,occupations=8):
    
    """This function computes the jacobian of the likelihood """
    
    
    jac = np.zeros((np.shape(affected)[0],))
    
    # First the derivative of x1. I am computing for each choice category that shares
    # parameters (For example: associate degree) the derivative of the 9 parameters
    # simultaneously. 
    
    
    for i in prange((1+fields+1+occupations)):
        
        temp = affected[i*9,:]
        
        idx = temp[choices]
        
        check = ccps@temp
                
        derivative = numba_multiply(idx-check,x1_new_p,type_weights)
        
        jac[i*9:(1+i)*9] = sum_numba_axis_special(derivative)
        
    #-------------------------------------------------------------------------#
    # now for the work parameters
    
    previous = (1+fields+1+occupations)*9
    
    
    for i in prange(2+2*fields+occupations+2):
        
        temp = affected[previous+i,:]
        
        idx = temp[choices]
        
        check = ccps@temp
                
        derivative = (idx-check)*type_weights
        
        jac[previous+i] = np.sum(derivative)
        
    #-------------------------------------------------------------------------#
    # now for the last choice parameters
    
    
    previous = (1+fields+1+occupations)*9 + 2+2*fields+occupations+2
    
    for i in prange(1+fields+occupations):
        
        temp = affected[previous+i,:]
        
        idx = temp[choices]
        
        check = ccps@temp
        
        derivative = (idx-check)*previousaffects[:,i]*type_weights
        
        
        jac[previous+i] = np.sum(derivative)
        
    #-------------------------------------------------------------------------#   
    # Now for education effects
    
    previous = (1+fields+1+occupations)*9 + 2+2*fields+occupations+2 + 1+fields+occupations 
    
    # here is like with x1, I can compute the different js (which in this case are occupations)
    # simultaneously

    jac[previous:previous+get_amount_educ()] = jacobian_education_effects(previous,ccps,choices,affected,x_educ_p,type_weights)
    
    #-------------------------------------------------------------------------#
    # Now for period effects
    
    previous = previous + get_amount_educ()
    
    # Notice the derivative only affects the current period!
    if period > 1:
        for i in prange((1+fields+occupations)):
            
            temp = affected[previous+i*(T-2),:]
            
            idx = temp[choices]
            
            check = ccps@temp
                        
            derivative = (idx-check)*type_weights
            
            jac[previous+i*(T-2)+(period-2)] = np.sum(derivative)
            
        #grad school:
        if period > 5:
            
            temp = affected[previous+(1+fields+occupations)*(T-2),:]  # check if this is ok!!
            
            idx = temp[choices]
            
            check = ccps@temp
                        
            derivative = (idx-check)*type_weights
            
            jac[previous+(1+fields+occupations)*(T-2)+period-5-1] = np.sum(derivative)
            
    #-------------------------------------------------------------------------#
            
    # Now for period work effects
    
    previous = previous + np.maximum((1+fields+occupations)*(T-1-1),0) + np.maximum((T-5-1),0)
    
    # Notice the derivative only affects the current period!
    if period > 1:
        for i in prange((1+1+1)):  # educ part-time, educ full-time, work full-time
            
            temp = affected[previous+i*(T-2),:]
            
            idx = temp[choices]
            
            check = ccps@temp
                        
            derivative = (idx-check)*type_weights
            
            jac[previous+i*(T-2)+(period-2)] = np.sum(derivative)
            
        #grad school:
        if period > 5:
            
            for i in prange(2): # part time, full time
            
                temp = affected[previous+3*(T-2)+(T-5)*(i),:]  # check if this is ok!!
                
                idx = temp[choices]
                
                check = ccps@temp
                                
                derivative = (idx-check)*type_weights
                
                jac[previous+3*(T-2)+period-5-1+(T-5-1)*i] = np.sum(derivative)
                
    #-------------------------------------------------------------------------#
        
    # now for first time enrolled in associate degree
    
    previous = previous + np.maximum(3*(T-1-1),0) + np.maximum((T-5-1)*2,0)
    
    temp = affected[previous,:]
    
    idx = temp[choices]
    
    check = ccps@temp
        
    derivative = (idx-check)*x_first2_p[:,0]*type_weights
    
    jac[previous] = np.sum(derivative)
    
    # Now for first time enrolled in bachelor
    
    previous = previous + 1
    
    # notice I can do all parameters at the same time. 
    
    temp = affected[previous,:]
    
    idx = temp[choices]
    
    check = ccps@temp
        
    derivative = numba_broadcast((idx-check),x_first4_p,type_weights)
    
    jac[previous:previous+4] = sum_numba_axis(derivative)
    
    # Now for first time enrolled in graduate school
    
    previous = previous + 4
    
    if period > 5:
        
        temp = affected[previous,:]
        
        idx = temp[choices]
        
        check = ccps@temp
                
        derivative = (idx-check)*x_firstgrad_p[:,0]*type_weights
        
        jac[previous] = np.sum(derivative)
        
    #-------------------------------------------------------------------------#
    
    # Now for experience ability effects
    
    previous = previous + 1
    
    for i in prange(fields-1):
        
        temp = affected[previous+i*6*4]
        
        idx = temp[choices]
        
        check = ccps@temp
                
        derivative = numba_multiply(idx-check,x_exp_p,type_weights)
        
        jac[previous+6*4*i:previous+6*4*(i+1)] = sum_numba_axis_special(derivative)
    
    
    #-------------------------------------------------------------------------#
    
    # Now for type effects
    # type 1:
        
    if school_type == 1:
    
        previous = previous + (fields-1)*6*4
            
        temp = affected[previous,:]
        
        idx = temp[choices]
        
        check = ccps@temp
                        
        derivative = (idx-check)*type_weights
            
        jac[previous] = np.sum(derivative)
            
        # type 2:    
        temp = affected[previous+1,:]
            
        check = ccps@temp
            
        idx = temp[choices]
            
        derivative = (idx-check)*type_weights
            
        jac[previous+1] = np.sum(derivative)    
    
    return jac
        


def get_previous_affects(i,x_change_p):
    
    """ This function check whether the last choice affects the current alternative
    i.
    i 0 : associate
    i: 1-..: fields
    i: ... : occupations"""
    
    affects = np.zeros(np.shape(x_change_p)[0])
    
    # associate degree, affected only by fields or home production
    if i == 0: 
        # fields
        affects[np.any(x_change_p[:,1:fields+1]==1,axis=1)] = 1
        # home production
        affects[x_change_p[:,-1]==1] = 1
     
    # field chocie, affected by fields or home production or associate
    elif (i > 0) & (i <= fields):
        # fields or associate
        affects[np.any(x_change_p[:,:fields+1]==1,axis=1)] = 1
        # home production
        affects[x_change_p[:,-1]==1] = 1
        # if it was the same field, set to 0
        affects[x_change_p[:,i]==1] = 0
    
    # occupation, affected by other occupations or home production
    elif i > fields:
        #  occupations or home production
        affects[np.any(x_change_p[:,fields+1:]==1,axis=1)] = 1
        # if it was the same occupation, set to 0
        affects[x_change_p[:,i]==1] = 0
        
    return affects
    
    

def map_param_x1(fields,occupations,totchoices):
    
    """ This function maps the parameters of the likelihood to which choices
    are affected. Only for those parameters related to x1"""
    
    nx1 = 9*(1+fields+occupations+1) # associate, fields,occupations,grad school
    
    x1param = np.zeros((nx1,totchoices))
    
    
    # First for fields and associate
    
    for i in range(1+fields):
        
        x1param[i*9:(1+i)*9,i*3:(1+i)*3] = 1
        
        
    # Now for occupations
    
    for i in range(occupations):
        
        x1param[9*(1+fields)+i*9:9*(1+fields)+(1+i)*9,(1+fields)*3+i*2:(1+fields)*3+(i+1)*2] = 1
        
    
    # Finally for grad school
    
    x1param[9*(1+fields+occupations):,(1+fields)*3+occupations*2:-1] = 1
    
    
    return x1param


def map_param_period(fields,occupations,totchoices):
    
    """ This function maps the parameters of the likelihood to which choices
    are affected. Only for those parameters related to the period effects"""
    
    nx1 = np.maximum((1+fields+occupations)*(T-1-1),0) + np.maximum((T-5-1),0)  # associate, fields,occupations,grad school
    
    periodparam = np.zeros((nx1,totchoices))
    
    
    # First for field
    
    for i in range(1+fields):
        
        periodparam[i*(T-2):(i+1)*(T-2),i*3:(i+1)*3] = 1
        
    # Now for occupations
    start = (T-2)*(1+fields)
    for i in range(occupations):
        
        periodparam[start+i*(T-2):start+(i+1)*(T-2),i*2+(1+fields)*3:(i+1)*2+(1+fields)*3] = 1
    
    if T > 5:
        # Graduate school
        start  = start + (T-2)*occupations
        periodparam[start:,(1+fields)*3+occupations*2:-1] = 1
    
    
    return periodparam

def map_param_period_work(fields,occupations,totchoices):
    
    """ This function maps the parameters of the likelihood to which choices
    are affected. Only for those parameters related to the work period effects"""
    
    nx1 = np.maximum(3*(T-1-1),0) + np.maximum((T-5-1)*2,0)  # associate, fields,occupations,grad school
    
    periodparam = np.zeros((nx1,totchoices))
    
    
    # First for associate degree and fields
    
    for i in range(1+fields):
        
        periodparam[:(T-2),i*3+1] = 1
        periodparam[(T-2):(T-2)*2,i*3+2] = 1
        
    # Now for occupations

    for i in range(occupations):
        
        periodparam[(T-2)*2:(T-2)*3,(1+fields)*3+2*(i)+1] = 1
    
    if T > 5:
        # Graduate school
        periodparam[(T-2)*3:(T-2)*3+(T-5-1),(1+fields)*3+(occupations)*2+1] = 1
        periodparam[(T-2)*3+(T-5-1):(T-2)*3+(T-5-1)*2,(1+fields)*3+(occupations)*2+2] = 1
    
    
    return periodparam


def map_param_affected(fields,occupations,totchoices):
    
    """This function returns the indicator for the choices that are affected by
    the different parameters of the part vs full time dummies"""
    
    nwork = (2+2*fields+occupations+2)
    
    xwork = np.zeros((nwork,totchoices))
    
    # First affect the different education categories
    
    for i in range(1+fields):
        
        xwork[i*2,i+1+i*2] = 1
        xwork[i*2+1,i+2+i*2] = 1
        
    # Now the occupations
    
    for i in range(occupations):
        
        xwork[2+2*fields+i,(1+fields)*3+(1+i*2)] = 1
        
        
    # Now for graduate school
    
    xwork[2+2*fields+occupations,(1+fields)*3+occupations*2+1] = 1
    xwork[2+2*fields+occupations+1,(1+fields)*3+occupations*2+2] = 1
    
    
    return xwork


def map_param_last(fields,occupations,totchoices):
    
    """This function returns the indicator for the choices that are affected by
    the different parameters of the last choice"""
    
    nlast = 1+ fields + occupations
    
    xlast = np.zeros((nlast,totchoices))
    
    
    # Basically each parameter affects one category
    
    # First education categories
    
    for i in range(1+fields):
        
        xlast[i,i+i*2:i+3+i*2] = 1
        
    # Now for occupation
    
    for i in range(occupations):
        
        xlast[1+fields+i,(1+fields)*3+i*2:(1+fields)*3+i*2+2] = 1
        
    
    return xlast

def map_param_educ(fields,occupations,totchoices):
    
    """This function returns the indicator for the choices that are affected by
    the different parameters of the educational parameters"""
    
    neduc = get_amount_educ()
    
    xeduc = np.zeros((neduc,totchoices))
    
    # The idea is that each possible education category affects each occupation
    
    # First 9 affect business
    xeduc[:9,(1+fields)*3:(1+fields)*3+(1)*2] = 1 
    # Next 4 affect stem
    xeduc[9:13,(1+fields)*3+1*2:(1+fields)*3+(1+1)*2] = 1
    #Next 4 affect Social Sciences
    xeduc[13:17,(1+fields)*3+2*2:(1+fields)*3+(2+1)*2] = 1
    #Next 5 affect education
    xeduc[17:22,(1+fields)*3+3*2:(1+fields)*3+(3+1)*2] = 1
    #Next 3 affect humanities
    xeduc[22:25,(1+fields)*3+4*2:(1+fields)*3+(4+1)*2] = 1 
    #Next 4 affect health
    xeduc[25:29,(1+fields)*3+5*2:(1+fields)*3+(5+1)*2] = 1 
    #Next 9 affect services
    xeduc[29:38,(1+fields)*3+6*2:(1+fields)*3+(6+1)*2] = 1 
    #Next 9 affect production
    xeduc[38:47,(1+fields)*3+7*2:(1+fields)*3+(7+1)*2] = 1 
        
    return xeduc


def map_param_first(fields,occupations,totchoices,coltype):
    
    """ this function maps parameters to whcih choices are affected"""
    
    if coltype == 1:
        nfirst =  1
        affected_first = np.zeros((nfirst,totchoices))
        affected_first[0,:3] = 1
    elif coltype == 2:
        nfirst = 4
        affected_first = np.zeros((nfirst,totchoices))
        affected_first[:,3:3+fields*3] = 1
        
    elif coltype == 3:
        nfirst =  1 
        affected_first = np.zeros((nfirst,totchoices))
        affected_first[0,(3+fields*3+occupations*2):(3+fields*3+occupations*2)+3] = 1
        
    
    return affected_first

def map_param_types(fields,totchoices):
    
    """
    This function maps parameters of thetype effects to the choices that
    are affected by those parameters
    """
    
    ntypes = 2
    affected_types = np.zeros((ntypes,totchoices))
    
    # First parameter affects 2y schools
    affected_types[0,:3] = 1
    # Second parameter affects 4y schools
    affected_types[1,3:(fields+1)*3] = 1
    
    return affected_types


def map_param_exp(fields,totchoices):
    
    """
    This function maps parameters of the experience effects to the choices that
    are affected by those parameters
    """
    
    ntypes = (fields-1)*6*4
    affected_exp = np.zeros((ntypes,totchoices))
    
    for field in range(fields):
        
        if field < 2: 
            affected_exp[6*4*field:6*4*(field+1),3+field*3:3+(1+field)*3] = 1
            
        if field > 2: 
            affected_exp[6*4*(field-1):6*4*(field),3+field*3:3+(1+field)*3] = 1
    
    return affected_exp
    

def map_param_to_choice(fields,occupations):
    
    """This function maps for each parameters which choices are affected """
    
    totchoices = np.shape(get_total_choices())[0]
    
    
    # First for x1 param:
        
    affected_x1 = map_param_x1(fields,occupations,totchoices)
    
    # Now for work
    
    affected_work = map_param_affected(fields,occupations,totchoices)
    
    
    # For last choice
    
    affected_last = map_param_last(fields,occupations,totchoices)
    
    
    # For education complementaritites
    
    affected_educ = map_param_educ(fields,occupations,totchoices)
    
    # For period effects
    
    affected_period = map_param_period(fields,occupations,totchoices)
    
    # For period work effects
    
    affected_period_work = map_param_period_work(fields,occupations,totchoices)
    
    # For first time enrollen in 2y school
    
    affected_first2 = map_param_first(fields,occupations,totchoices,1)
    
    # For first time enrolled in 4y school
    
    affected_first4 = map_param_first(fields,occupations,totchoices,2)
    
    # For first time enrolled in grad school
    
    affected_first_grad = map_param_first(fields,occupations,totchoices,3)
    
    # For experience ability effects
    
    affexted_experience = map_param_exp(fields,totchoices)
    
    # Affected for the type parameters
    
    affected_type = map_param_types(fields,totchoices)
    
    # All together
    
    affected = np.concatenate((affected_x1,affected_work,affected_last,
                               affected_educ,affected_period,affected_period_work, 
                               affected_first2,affected_first4,
                               affected_first_grad,affexted_experience,affected_type),axis=0)
    
    
    return affected

def load_param_g(real=1):
    
    if real == 1:
        param_utility = np.load(f"{path_estimates}/param_g.npy")
    else:
        fields=8
        occupations = 8
        n_param_g_x1 = 9*(1+fields+occupations+1)
        n_param_g_work = 2 + 2*fields+occupations+2
        n_param_g_change = 1+ fields  + occupations
        n_param_g_educ = get_amount_educ()
        n_param_g_period = np.maximum((1+fields+occupations)*(T-1-1),0) + np.maximum((T-5-1),0)
        n_param_g_period_work = np.maximum((T-1-1)*3,0)  + np.maximum((T-5-1)*2,0)
        n_param_g_first = 1 + 4 +1 
        total_n = n_param_g_x1 + n_param_g_change + n_param_g_work + n_param_g_educ + n_param_g_period + n_param_g_period_work + n_param_g_first
        param_utility = np.zeros((total_n,))
    
        
    return param_utility
 
@njit()   
def get_amount_educ():
    
    """
    This function returns the amount of total parameters that there are in the
    education occupation complementarities. Notice that this depend on which 
    occupations can be chosen by each field. As a reminder: 
        1- Business: All
        2- STEM: STEM,Other,Associate
        3- Social: Social, Humanities, Other
        6- Education: Education,Social,Humanities,Health
        7- Humanities: Humanities, Associate
        8- Health: Health, STEM, Associate
        9- Service: All
        10- Production: All
    I need to sum grad school to all of them
    """
    
    amount_business = 8 + 1
    amount_STEM = 3 + 1
    amount_social = 3 + 1
    amount_educ = 4 +1 
    amount_human = 2 + 1 
    amount_health = 3 + 1 
    amount_service = 8 + 1 
    amount_prod = 8 + 1 
    
    amount_educ = amount_business+  amount_STEM + amount_social + amount_educ + amount_human  + amount_health + amount_service + amount_prod
    
    return amount_educ

def store(x):

    np.save(f"{path_estimates}/temp_evaluation.npy",x)
    
    
def ordered_logit_loglike(param,x,choices,alternatives,q):
    
    """
    This functioncion computes the log likelihood of an ordered
    logit model.  First params are for x + type, the others are for thresholds.
    """
    
    betas = param[:np.shape(x)[1]]
    
    type_param = param[np.shape(x)[1]]

    thresholds = param[np.shape(x)[1]+1:]
    
    loglike = np.zeros(2)
    
    # Now I need to get the probability of each alternative
    
    for em_type in range(1,2):
        
        xbeta = x@betas[...,None] + type_param*(em_type-1)   # only sum em_type when necesary
        
        # Get the real thresholds
        
        real_thresholds = np.zeros(np.shape(thresholds)[0]+2)
        
        real_thresholds[0]  = -np.inf
        real_thresholds[-1] = np.inf 
        real_thresholds[1:-1] = thresholds
        
        # Now compute the probabilities: 
            
        temp  = 1/(1+np.exp(xbeta-real_thresholds))
        
        # Now the probabilities are: 
            
            
        probs = temp[:,1:] - temp[:,:-1]
        
        # And the chosen probability is
        
        probs_hat = np.take_along_axis(probs,choices[...,None].astype('int'),axis=1)
        
        logpprob  = np.log(probs_hat)
        
        logpprob[logpprob==-np.inf] = -100000
    
        loglike[em_type-1] = np.sum(logpprob)
        
    return -1*np.sum(loglike)
    
def ordered_logit_like(param,x,choices,alternatives,em_type):
    
    """
    This functioncion computes the log likelihood of an ordered
    logit model.  First params are for x + type, the others are for thresholds.
    """
    
    betas = param[:np.shape(x)[1]]
    
    type_param = param[np.shape(x)[1]]

    thresholds = param[np.shape(x)[1]+1:]
    
    # Now I need to get the probability of each alternative
    
    xbeta = x@betas[...,None] + type_param*(em_type-1)   # only sum em_type when necesary
    
    # Get the real thresholds
    
    real_thresholds = np.zeros(np.shape(thresholds)[0]+2)
    
    real_thresholds[0]  = -np.inf
    real_thresholds[-1] = np.inf 
    real_thresholds[1:-1] = thresholds
    
    # Now compute the probabilities: 
        
    temp  = 1/(1+np.exp(xbeta-real_thresholds))
    
    # Now the probabilities are: 
        
        
    probs = temp[:,1:] - temp[:,:-1]
    
    # And the chosen probability is
    
    probs_hat = np.take_along_axis(probs,choices[...,None].astype('int'),axis=1)
    
    return probs_hat[:,0]

def logit_loglike(param,x,choices,q):
    
    """
    This functioncion computes the log likelihood of a
    logit model.  First params are for x + type, the others are for thresholds.
    """
    
    betas = param[:np.shape(x)[1]]
    
    type_param = param[np.shape(x)[1]]
    
    loglike = np.zeros(2)
    
    for em_type in range(1,3):
    
        # Now I need to get the probability of each alternative
        
        xbeta = -1*(x@betas[...,None] + type_param*(em_type-1))   # only sum em_type when necesary
        
        # Now compute the probabilities: 
            
        temp  = 1/(1+np.exp(xbeta))
        
        # Now the probabilities are: 
            
            
        probs = np.concatenate((1-temp,temp),axis=1)
        
        # And the chosen probability is
        
        probs_hat = np.take_along_axis(probs,choices[...,None].astype('int'),axis=1)
        
        loglike[em_type-1] = np.sum(np.log(probs_hat)*q[:,em_type-1][...,None])
    
    return -1*np.sum(loglike)


def logit_like(param,x,choices,em_type):
    
    """
    This functioncion computes the log likelihood of a
    logit model.  First params are for x + type, the others are for thresholds.
    """
    
    betas = param[:np.shape(x)[1]]
    
    type_param = param[np.shape(x)[1]]
    
    # Now I need to get the probability of each alternative
    
    xbeta = -1*(x@betas[...,None] + type_param*(em_type-1))   # only sum em_type when necesary
    
    # Now compute the probabilities: 
        
    temp  = 1/(1+np.exp(xbeta))
    
    # Now the probabilities are: 
        
        
    probs = np.concatenate((1-temp,temp),axis=1)
    
    # And the chosen probability is
    
    probs_hat = np.take_along_axis(probs,choices[...,None].astype('int'),axis=1)
    
    return probs_hat[:,0]


def logit_margineffect(param,x):
    
    """
    This function computes the marginal effect of the logit model wrt the 
    type effect
    """
    
    betas = param[:np.shape(x)[1]]
    
    type_param = param[np.shape(x)[1]]
    
    # Now I need to get the probability of each alternative
    
    xbeta = -1*(x@betas[...,None] + type_param)   
    xbeta_notype = -1*(x@betas[...,None] )  
    # Now compute the probabilities: 
        
    temp_type  = 1/(1+np.exp(xbeta))
    temp_notype =  1/(1+np.exp(xbeta_notype))
    # Now the marginal effect is: 
        
        
    margeffect = np.mean(temp_type - temp_notype)
    
    return margeffect


#-----------------------------------------------------------------------------#
# Sixteen-type auxiliary EM: schooling x grant x parental transfer x loan.
# The shared layout is authoritative for both auxiliary and structural code.
#-----------------------------------------------------------------------------#

MONEY_SCALE = 1000.0

AUXILIARY_TYPE_COMPONENTS = TYPE_COMPONENTS
AUXILIARY_TYPE_SCHOOL = TYPE_SCHOOL
AUXILIARY_TYPE_GRANT = TYPE_GRANT
AUXILIARY_TYPE_TRANSFER = TYPE_TRANSFER
AUXILIARY_TYPE_LOAN = TYPE_LOAN
AUXILIARY_TYPE_NAMES = TYPE_NAMES
N_AUXILIARY_TYPES = N_STRUCTURAL_TYPES


def _financial_alt_features(choices):
    """Alternative-level controls: four-year, graduate, part-time, full-time."""
    choices = np.asarray(choices)
    return np.column_stack(
        (
            choices[:, 1] == 2,
            choices[:, 1] == 3,
            choices[:, 2] == 1,
            choices[:, 2] == 2,
        )
    ).astype(float)


def _state_wage_design(state):
    """Vectorized version of the full-model ``get_x2_new`` wage state."""
    state = np.asarray(state)
    design = np.zeros((state.shape[0], 10), dtype=float)
    design[:, 0] = state[:, 0]
    major = state[:, 8].astype(int)

    regular = (major != 0) & (major != 12)
    regular_rows = np.flatnonzero(regular)
    regular_major = major[regular]
    regular_columns = np.where(regular_major < 3, regular_major - 1, regular_major - 2)
    valid = (regular_columns >= 0) & (regular_columns < 8)
    design[regular_rows[valid], 1 + regular_columns[valid]] = 1.0
    design[major == 12, 8] = 1.0
    design[:, 9] = state[:, 6]
    return design


def _occupation_wage_design(state_design, occupation):
    """Apply the exact occupation-specific wage mapping used by the full model."""
    column_map = {
        1: tuple(range(10)),
        2: (0, 1, 6, 7, 8),
        3: (0, 3, 4, 6, 7),
        6: (0, 2, 3, 4, 5, 8),
        7: (0, 4, 7, 8),
        8: (0, 1, 5, 7, 8),
        9: tuple(range(10)),
        10: tuple(range(10)),
    }
    return state_design[:, column_map[int(occupation)]]


def load_fixed_wage_parameters():
    """Load, but never estimate, the wage equations used by the auxiliary EM."""
    occupations = (1, 2, 3, 6, 7, 8, 9, 10)
    parameters = {
        occupation: np.load(f"{pathfunctions}/wage_{occupation}.npy")
        for occupation in occupations
    }
    return {
        "school": np.load(f"{pathfunctions}/wage_0.npy"),
        "occupations": parameters,
        "sigmas": np.asarray(np.load(f"{pathfunctions}/sigmas.npy"), dtype=float),
        "occupation_order": occupations,
    }


def predict_expected_wages(x1_design, state, feasible_choices, wage_parameters):
    """Expected annual wage for every observation and feasible alternative.

    Wage equations are in log hourly wages. The lognormal correction is included,
    and labor supply follows the full-model convention: 0, 20, or 40 hours/week.
    """
    choices = np.asarray(feasible_choices)
    nobs, nchoices = x1_design.shape[0], choices.shape[0]
    expected = np.zeros((nobs, nchoices), dtype=float)
    state_design = _state_wage_design(state)
    sigmas = wage_parameters["sigmas"]

    school_mu = np.column_stack((x1_design, state_design[:, :-1])) @ wage_parameters["school"]
    school_hourly = np.exp(np.clip(school_mu + 0.5 * sigmas[0] ** 2, -20.0, 20.0))

    occupation_hourly = {}
    for sigma_index, occupation in enumerate(wage_parameters["occupation_order"], start=1):
        x2_wage = _occupation_wage_design(state_design, occupation)
        wage_x = np.column_stack((x1_design, x2_wage))
        mu = wage_x @ wage_parameters["occupations"][occupation]
        occupation_hourly[occupation] = np.exp(
            np.clip(mu + 0.5 * sigmas[sigma_index] ** 2, -20.0, 20.0)
        )

    annual_hours = choices[:, 2].astype(float) * 0.5 * 40.0 * 52.0
    for j, (occupation, education, work) in enumerate(choices.astype(int)):
        if work == 0:
            continue
        if education > 0:
            expected[:, j] = school_hourly * annual_hours[j]
        elif occupation in occupation_hourly:
            expected[:, j] = occupation_hourly[occupation] * annual_hours[j]
    return expected


def _feasible_choice_mask(states, total_choices):
    """Cache legal alternatives; this individual loop is never run inside EM."""
    mask = np.zeros((states.shape[0], total_choices.shape[0]), dtype=bool)
    for i, state in enumerate(states.astype(int)):
        possible = get_possible_choices(state)
        indices = np.where((total_choices == possible[:, None]).all(-1))[1]
        mask[i, indices] = True
    return mask


def _observed_financial_design(x1_design, choices):
    return np.column_stack((x1_design, _financial_alt_features(choices)))


def _observed_grant_design(x1_design, choices):
    """Grant controls available in every structural alternative."""
    choices = np.asarray(choices)
    return np.column_stack(
        (
            x1_design,
            choices[:, 2] == 1,
            choices[:, 2] == 2,
        )
    ).astype(float)


def _observed_loan_design(x1_design, grant, transfer):
    """Loan controls: demographics plus realized grants and transfers ($1,000s)."""
    return np.column_stack(
        (
            np.asarray(x1_design, dtype=float),
            np.asarray(grant, dtype=float) / MONEY_SCALE,
            np.asarray(transfer, dtype=float) / MONEY_SCALE,
        )
    )


def build_auxiliary_em_data(
    choices_all,
    choices_array_all,
    x1_new,
    x_change,
    x_educ,
    x_first2,
    x_first4,
    x_firstgrad,
    x_exp,
):
    """Load and cache every object that is fixed across auxiliary EM iterations."""
    total_choices = get_total_choices().astype(int)
    wage_parameters = load_fixed_wage_parameters()
    debt_grid = get_debt_range()
    periods = []
    grant_rows = {education: [] for education in (1, 2, 3)}
    transfer_rows = []
    loan_rows = {education: [] for education in (1, 2, 3)}
    n_individuals = len(choices_all[0])

    for period in range(1, T):
        x1_raw, state, debt_index, debtchoice, choices, income = load_data_superfeasible(
            period, return_income=True
        )
        loan_flow = np.asarray(
            np.load(f"{path}/loanflow_superfeasible_t{period}.npy"), dtype=float
        )
        x1_period = np.asarray(x1_new[f"period{period}"], dtype=float)
        if len(choices) != n_individuals:
            raise ValueError("The auxiliary EM requires the same balanced individuals each period.")

        feasible = _feasible_choice_mask(state, total_choices)
        chosen_index = np.asarray(choices_array_all[period - 1], dtype=int)
        if not np.all(feasible[np.arange(n_individuals), chosen_index]):
            raise ValueError(f"Observed infeasible choice found in auxiliary period {period}.")

        expected_wage = predict_expected_wages(
            x1_period, state, total_choices, wage_parameters
        )
        debt_index = np.asarray(debt_index, dtype=int)
        debt_dollars = debt_grid[np.clip(debt_index, 0, len(debt_grid) - 1)]

        periods.append(
            {
                "period": period,
                "choices": np.asarray(choices),
                "chosen_index": chosen_index,
                "state": np.asarray(state),
                "x1": x1_period,
                "feasible": feasible,
                "expected_wage": expected_wage,
                "debt_dollars": debt_dollars,
                "x_change": np.asarray(x_change[f"period{period}"]),
                "x_educ": np.asarray(x_educ[f"period{period}"]),
                "x_first2": np.asarray(x_first2[f"period{period}"]),
                "x_first4": np.asarray(x_first4[f"period{period}"]),
                "x_firstgrad": np.asarray(x_firstgrad[f"period{period}"]),
                "x_exp": np.asarray(x_exp[f"period{period}"]),
            }
        )

        education = choices[:, 1].astype(int)
        individual_index = np.arange(n_individuals, dtype=int)
        financial_x = _observed_financial_design(x1_period, choices)
        grant_x = _observed_grant_design(x1_period, choices)

        grant = np.asarray(income[:, 1], dtype=float)
        transfer = np.asarray(income[:, 2], dtype=float) + np.asarray(income[:, 3], dtype=float)
        for education_level in (1, 2, 3):
            grant_observed = (education == education_level) & np.isfinite(grant)
            grant_rows[education_level].append(
                {
                    "x": grant_x[grant_observed],
                    "amount": grant[grant_observed],
                    "individual": individual_index[grant_observed],
                }
            )

        transfer_observed = (
            np.isin(education, (1, 2))
            & np.isfinite(income[:, 2])
            & np.isfinite(income[:, 3])
        )
        transfer_rows.append(
            {
                "x": financial_x[transfer_observed],
                "amount": transfer[transfer_observed],
                "individual": individual_index[transfer_observed],
            }
        )

        enrolled = np.isin(education, (1, 2, 3))
        if np.any(~np.isfinite(loan_flow[enrolled])):
            raise ValueError(
                f"Missing annual loan flows among enrolled observations in period {period}. "
                "Rerun clean_loans_final.do and data_model_fullfields.do."
            )
        if np.any(loan_flow[enrolled] < 0.0):
            raise ValueError(f"Negative annual loan flow found in period {period}.")
        loan_x = _observed_loan_design(x1_period, grant, transfer)
        for education_level in (1, 2, 3):
            loan_observed = education == education_level
            loan_rows[education_level].append(
                {
                    "x": loan_x[loan_observed],
                    "amount": loan_flow[loan_observed],
                    "individual": individual_index[loan_observed],
                }
            )

    def stack_financial(rows):
        return {
            "x": np.concatenate([row["x"] for row in rows], axis=0),
            "amount": np.concatenate([row["amount"] for row in rows]),
            "individual": np.concatenate([row["individual"] for row in rows]),
        }

    alt_features = _financial_alt_features(total_choices)
    tuition = np.select(
        [total_choices[:, 1] == 1, total_choices[:, 1] == 2, total_choices[:, 1] == 3],
        [4000.0, 8000.0, 14000.0],
        default=0.0,
    )
    return {
        "n_individuals": n_individuals,
        "total_choices": total_choices,
        "periods": periods,
        "grant": {
            education: stack_financial(grant_rows[education])
            for education in (1, 2, 3)
        },
        "transfer": stack_financial(transfer_rows),
        "loan": {
            education: stack_financial(loan_rows[education])
            for education in (1, 2, 3)
        },
        "alt_financial_features": alt_features,
        "grant_eligible": total_choices[:, 1] > 0,
        "transfer_eligible": np.isin(total_choices[:, 1], (1, 2)),
        "tuition": tuition,
        "nonhome": np.any(total_choices != 0, axis=1).astype(float),
        "x1_measure": np.asarray(x1_new["period1"], dtype=float),
    }


def _collapse_type_weights(q, type_dimension):
    q = np.asarray(q, dtype=float)
    return np.column_stack(
        (
            q[:, np.asarray(type_dimension) == 0].sum(axis=1),
            q[:, np.asarray(type_dimension) == 1].sum(axis=1),
        )
    )


def collapse_school_weights(q):
    return _collapse_type_weights(q, AUXILIARY_TYPE_SCHOOL)


def collapse_grant_weights(q):
    return _collapse_type_weights(q, AUXILIARY_TYPE_GRANT)


def collapse_transfer_weights(q):
    return _collapse_type_weights(q, AUXILIARY_TYPE_TRANSFER)


def collapse_loan_weights(q):
    return _collapse_type_weights(q, AUXILIARY_TYPE_LOAN)


def collapse_sgt_weights(q):
    """Collapse loan type for explicit eight-cell reporting/compatibility."""
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[1] != N_AUXILIARY_TYPES:
        raise ValueError(
            f"Expected an N x {N_AUXILIARY_TYPES} auxiliary posterior; received {q.shape}."
        )
    return q.reshape(len(q), 2, 2, 2, 2).sum(axis=4).reshape(len(q), 8)


def _weighted_type_logit_objective(parameters, x, outcome, weights):
    beta, shift = parameters[:-1], parameters[-1]
    eta_low = x @ beta
    eta_high = eta_low + shift
    probability_low = expit(eta_low)
    probability_high = expit(eta_high)

    loglike_low = outcome * eta_low - np.logaddexp(0.0, eta_low)
    loglike_high = outcome * eta_high - np.logaddexp(0.0, eta_high)
    objective = -np.sum(weights[:, 0] * loglike_low + weights[:, 1] * loglike_high)

    residual_low = weights[:, 0] * (probability_low - outcome)
    residual_high = weights[:, 1] * (probability_high - outcome)
    gradient_beta = x.T @ (residual_low + residual_high)
    gradient_shift = np.sum(residual_high)
    return objective, np.append(gradient_beta, gradient_shift)


def _weighted_type_normal_fit(x, log_amount, weights):
    nobs, nbase = x.shape
    design_low = np.column_stack((x, np.zeros(nobs)))
    design_high = np.column_stack((x, np.ones(nobs)))
    design = np.vstack((design_low, design_high))
    outcome = np.tile(log_amount, 2)
    weight = np.concatenate((weights[:, 0], weights[:, 1]))
    root_weight = np.sqrt(np.clip(weight, 0.0, None))
    weighted_design = design * root_weight[:, None]
    weighted_outcome = outcome * root_weight
    coefficients = np.linalg.lstsq(weighted_design, weighted_outcome, rcond=None)[0]
    residual = outcome - design @ coefficients
    variance = np.sum(weight * residual ** 2) / max(np.sum(weight), 1.0)
    return coefficients, float(np.sqrt(max(variance, 1.0e-8)))


def estimate_financial_source(dataset, q_financial, previous=None):
    """Weighted receipt logit plus weighted log-normal positive-amount model."""
    x = dataset["x"]
    amount = dataset["amount"]
    row_weights = q_financial[dataset["individual"]]
    receipt = (amount > 0).astype(float)
    initial_receipt = (
        previous["receipt"]
        if previous is not None
        else np.zeros(x.shape[1] + 1, dtype=float)
    )
    result = minimize(
        _weighted_type_logit_objective,
        initial_receipt,
        args=(x, receipt, row_weights),
        jac=True,
        method="L-BFGS-B",
        options={"disp": False, "maxiter": 300, "gtol": 1.0e-6, "ftol": 1.0e-12},
    )

    positive = (amount > 0) & np.isfinite(amount)
    if not np.any(positive):
        raise ValueError("No positive observed financial amounts are available.")
    amount_coefficients, amount_sigma = _weighted_type_normal_fit(
        x[positive], np.log(amount[positive]), row_weights[positive]
    )
    return {
        "receipt": np.asarray(result.x),
        "amount": amount_coefficients,
        "sigma": amount_sigma,
        "receipt_success": bool(result.success),
        "receipt_message": str(result.message),
    }


def initialize_financial_source(dataset, financial_typeeffect=0.25):
    n_individuals = int(np.max(dataset["individual"])) + 1
    equal_weights = np.full((n_individuals, 2), 0.5)
    parameters = estimate_financial_source(dataset, equal_weights)
    parameters["receipt"][-1] = financial_typeeffect
    parameters["amount"][-1] = 0.25 * financial_typeeffect
    return parameters


def estimate_grant_processes(datasets, q_financial, previous=None):
    """Estimate separate two-year, four-year, and graduate grant equations."""
    estimates = {}
    for education in (1, 2, 3):
        dataset = datasets[education]
        positive_count = int(np.sum(np.asarray(dataset["amount"]) > 0))
        minimum_positive = dataset["x"].shape[1] + 1
        if positive_count < minimum_positive:
            raise ValueError(
                f"Only {positive_count} positive grants are available for education="
                f"{education}; at least {minimum_positive} are required for its "
                "positive-amount equation. Rerun data_model_fullfields.do so graduate "
                "grants are exported in totgrants."
            )
        prior = None if previous is None else previous[education]
        estimates[education] = estimate_financial_source(
            dataset, q_financial, previous=prior
        )
    return estimates


def initialize_grant_processes(datasets, financial_typeeffect=0.25):
    n_individuals = max(
        int(np.max(dataset["individual"]))
        for dataset in datasets.values()
        if len(dataset["individual"])
    ) + 1
    equal_weights = np.full((n_individuals, 2), 0.5)
    estimates = estimate_grant_processes(datasets, equal_weights)
    for parameters in estimates.values():
        parameters["receipt"][-1] = financial_typeeffect
        parameters["amount"][-1] = 0.25 * financial_typeeffect
    return estimates


def estimate_loan_processes(datasets, q_loan, previous=None):
    """Estimate education-specific loan receipt and positive-flow equations."""
    estimates = {}
    for education in (1, 2, 3):
        dataset = datasets[education]
        positive_count = int(np.sum(np.asarray(dataset["amount"]) > 0))
        minimum_positive = dataset["x"].shape[1] + 1
        if positive_count < minimum_positive:
            raise ValueError(
                f"Only {positive_count} positive annual loan flows are available for "
                f"education={education}; at least {minimum_positive} are required for "
                "its positive-amount equation."
            )
        prior = None if previous is None else previous[education]
        estimates[education] = estimate_financial_source(
            dataset, q_loan, previous=prior
        )
    return estimates


def initialize_loan_processes(datasets, loan_typeeffect=0.25):
    n_individuals = max(
        int(np.max(dataset["individual"]))
        for dataset in datasets.values()
        if len(dataset["individual"])
    ) + 1
    equal_weights = np.full((n_individuals, 2), 0.5)
    estimates = estimate_loan_processes(datasets, equal_weights)
    for parameters in estimates.values():
        parameters["receipt"][-1] = loan_typeeffect
        parameters["amount"][-1] = 0.25 * loan_typeeffect
    return estimates


def _source_log_likelihood_by_financial_type(parameters, dataset, n_individuals):
    x, amount = dataset["x"], dataset["amount"]
    individual = dataset["individual"]
    receipt = (amount > 0).astype(float)
    beta_receipt, shift_receipt = parameters["receipt"][:-1], parameters["receipt"][-1]
    beta_amount, shift_amount = parameters["amount"][:-1], parameters["amount"][-1]
    sigma_amount = max(float(parameters["sigma"]), 1.0e-8)

    eta_low = x @ beta_receipt
    eta_high = eta_low + shift_receipt
    row_loglike = np.column_stack(
        (
            receipt * eta_low - np.logaddexp(0.0, eta_low),
            receipt * eta_high - np.logaddexp(0.0, eta_high),
        )
    )

    positive = (amount > 0) & np.isfinite(amount)
    if np.any(positive):
        log_amount = np.log(amount[positive])
        mean_low = x[positive] @ beta_amount
        mean_high = mean_low + shift_amount
        constant = np.log(2.0 * np.pi * sigma_amount ** 2)
        row_loglike[positive, 0] += -0.5 * (
            constant + ((log_amount - mean_low) / sigma_amount) ** 2
        )
        row_loglike[positive, 1] += -0.5 * (
            constant + ((log_amount - mean_high) / sigma_amount) ** 2
        )

    individual_loglike = np.zeros((n_individuals, 2), dtype=float)
    np.add.at(individual_loglike, individual, row_loglike)
    return individual_loglike


def financial_log_likelihoods(
    grant_parameters, transfer_parameters, loan_parameters, auxiliary_data
):
    n = auxiliary_data["n_individuals"]
    grant = np.zeros((n, 2), dtype=float)
    for education in (1, 2, 3):
        grant += _source_log_likelihood_by_financial_type(
            grant_parameters[education], auxiliary_data["grant"][education], n
        )
    transfer = _source_log_likelihood_by_financial_type(
        transfer_parameters, auxiliary_data["transfer"], n
    )
    loan = np.zeros((n, 2), dtype=float)
    for education in (1, 2, 3):
        loan += _source_log_likelihood_by_financial_type(
            loan_parameters[education], auxiliary_data["loan"][education], n
        )
    return (
        grant[:, AUXILIARY_TYPE_GRANT]
        + transfer[:, AUXILIARY_TYPE_TRANSFER]
        + loan[:, AUXILIARY_TYPE_LOAN]
    )


def _expected_source_by_period(parameters, period_data, auxiliary_data, eligibility):
    n = auxiliary_data["n_individuals"]
    x1 = period_data["x1"]
    alt = auxiliary_data["alt_financial_features"]
    n_x1 = x1.shape[1]

    receipt_base = x1 @ parameters["receipt"][:n_x1]
    receipt_alt = alt @ parameters["receipt"][n_x1:-1]
    amount_base = x1 @ parameters["amount"][:n_x1]
    amount_alt = alt @ parameters["amount"][n_x1:-1]

    eta_receipt_low = receipt_base[:, None] + receipt_alt[None, :]
    eta_amount_low = amount_base[:, None] + amount_alt[None, :]
    expected = np.empty((2, n, alt.shape[0]), dtype=float)
    for financial_type in (0, 1):
        receipt_shift = financial_type * parameters["receipt"][-1]
        amount_shift = financial_type * parameters["amount"][-1]
        probability = expit(eta_receipt_low + receipt_shift)
        conditional_level = np.exp(
            np.clip(
                eta_amount_low + amount_shift + 0.5 * parameters["sigma"] ** 2,
                -20.0,
                20.0,
            )
        )
        expected[financial_type] = probability * conditional_level
        expected[financial_type, :, ~eligibility] = 0.0
    return expected


def _expected_grants_by_period(parameters, period_data, auxiliary_data):
    """Expected grants from three education-specific hurdle models."""
    n = auxiliary_data["n_individuals"]
    choices = auxiliary_data["total_choices"]
    x1 = period_data["x1"]
    part_time = (choices[:, 2] == 1).astype(float)
    full_time = (choices[:, 2] == 2).astype(float)
    expected = np.zeros((2, n, len(choices)), dtype=float)

    for education in (1, 2, 3):
        selected = choices[:, 1] == education
        if not np.any(selected):
            continue
        source = parameters[education]
        receipt_base = x1 @ source["receipt"][:9]
        amount_base = x1 @ source["amount"][:9]
        receipt_work = (
            part_time[selected] * source["receipt"][9]
            + full_time[selected] * source["receipt"][10]
        )
        amount_work = (
            part_time[selected] * source["amount"][9]
            + full_time[selected] * source["amount"][10]
        )
        for financial_type in (0, 1):
            probability = expit(
                receipt_base[:, None]
                + receipt_work[None, :]
                + financial_type * source["receipt"][-1]
            )
            conditional_level = np.exp(
                np.clip(
                    amount_base[:, None]
                    + amount_work[None, :]
                    + financial_type * source["amount"][-1]
                    + 0.5 * source["sigma"] ** 2,
                    -20.0,
                    20.0,
                )
            )
            expected[financial_type][:, selected] = probability * conditional_level
    return expected


def build_expected_consumption(
    wage_parameters,
    grant_parameters,
    transfer_parameters,
    model_data,
    feasible_choices,
):
    """Expected consumption for all grant/parental-transfer type pairs.

    Fixed expected wages are cached in ``model_data``. Grants are available for
    all schooling alternatives; parental transfers follow the full-model rule
    and are available for two- and four-year enrollment, but not graduate school.
    No quadrature, debt choice, or continuation value enters this calculation.
    """
    del wage_parameters, feasible_choices  # predictions/mappings are already cached
    by_resource_type = [[] for _ in range(4)]
    for period_data in model_data["periods"]:
        expected_grant = _expected_grants_by_period(
            grant_parameters, period_data, model_data
        )
        expected_transfer = _expected_source_by_period(
            transfer_parameters,
            period_data,
            model_data,
            model_data["transfer_eligible"],
        )
        for grant_type in (0, 1):
            for transfer_type in (0, 1):
                resource_index = 2 * grant_type + transfer_type
                resources = (
                    period_data["expected_wage"]
                    + expected_grant[grant_type]
                    + expected_transfer[transfer_type]
                    - model_data["tuition"][None, :]
                )
                resources[:, -1] = 0.0  # preserve home-production normalization
                by_resource_type[resource_index].append(resources)
    return by_resource_type


def _choice_components(choice_parameters, auxiliary_data, expected_consumption):
    legacy_count = total_n_choice_legacy
    legacy_parameters = choice_parameters[:legacy_count]
    consumption_coefficient = choice_parameters[legacy_count]
    debt_coefficient = choice_parameters[legacy_count + 1]
    utility_parameters = build_param_g(legacy_parameters)
    return (
        utility_parameters,
        consumption_coefficient,
        debt_coefficient,
        expected_consumption,
    )


def auxiliary_choice_log_likelihoods(choice_parameters, auxiliary_data, expected_consumption):
    """Individual choice-sequence log likelihood for all sixteen joint types."""
    utility_parameters, alpha_c, alpha_b, consumption_by_resource_type = (
        _choice_components(choice_parameters, auxiliary_data, expected_consumption)
    )
    n = auxiliary_data["n_individuals"]
    loglike = np.zeros((n, N_AUXILIARY_TYPES), dtype=float)

    for period_index, period_data in enumerate(auxiliary_data["periods"]):
        g_school = []
        for school_type in (0, 1):
            g_school.append(
                get_all_g(
                    utility_parameters,
                    period_data["x1"],
                    period_data["x_change"],
                    period_data["x_educ"],
                    period_data["x_first2"],
                    period_data["x_first4"],
                    period_data["x_firstgrad"],
                    period_data["x_exp"],
                    period_data["period"],
                    SCHOOL_REPRESENTATIVE_TYPE_IDS[school_type],
                )
            )
        debt_regressor = (
            period_data["debt_dollars"][:, None]
            * auxiliary_data["nonhome"][None, :]
            / MONEY_SCALE
        )
        for latent_type in range(N_AUXILIARY_TYPES):
            resource_index = (
                2 * AUXILIARY_TYPE_GRANT[latent_type]
                + AUXILIARY_TYPE_TRANSFER[latent_type]
            )
            consumption = consumption_by_resource_type[resource_index][period_index]
            utility = (
                g_school[AUXILIARY_TYPE_SCHOOL[latent_type]]
                + alpha_c * consumption / MONEY_SCALE
                + alpha_b * debt_regressor
            )
            utility = np.where(period_data["feasible"], utility, -np.inf)
            log_probability = utility - logsumexp(utility, axis=1, keepdims=True)
            loglike[:, latent_type] += log_probability[
                np.arange(n), period_data["chosen_index"]
            ]
    return loglike


def auxiliary_choice_objective_legacy(choice_parameters, auxiliary_data, expected_consumption, q):
    """Reference sixteen-type objective using the original analytical score path."""
    utility_parameters, alpha_c, alpha_b, consumption_by_resource_type = (
        _choice_components(choice_parameters, auxiliary_data, expected_consumption)
    )
    legacy_gradient = np.zeros(total_n_choice_legacy, dtype=float)
    resource_gradient = np.zeros(2, dtype=float)
    loglike = 0.0
    affected = map_param_to_choice(fields, occupations)

    for period_index, period_data in enumerate(auxiliary_data["periods"]):
        g_school = []
        for school_type in (0, 1):
            g_school.append(
                get_all_g(
                    utility_parameters,
                    period_data["x1"],
                    period_data["x_change"],
                    period_data["x_educ"],
                    period_data["x_first2"],
                    period_data["x_first4"],
                    period_data["x_firstgrad"],
                    period_data["x_exp"],
                    period_data["period"],
                    SCHOOL_REPRESENTATIVE_TYPE_IDS[school_type],
                )
            )
        previous_affects = build_previous_affects(period_data["x_change"])
        debt_regressor = (
            period_data["debt_dollars"][:, None]
            * auxiliary_data["nonhome"][None, :]
            / MONEY_SCALE
        )
        chosen = period_data["chosen_index"]
        rows = np.arange(len(chosen))

        for latent_type in range(N_AUXILIARY_TYPES):
            school_type = AUXILIARY_TYPE_SCHOOL[latent_type]
            resource_index = (
                2 * AUXILIARY_TYPE_GRANT[latent_type]
                + AUXILIARY_TYPE_TRANSFER[latent_type]
            )
            consumption = consumption_by_resource_type[resource_index][period_index]
            consumption_scaled = consumption / MONEY_SCALE
            utility = (
                g_school[school_type]
                + alpha_c * consumption_scaled
                + alpha_b * debt_regressor
            )
            utility = np.where(period_data["feasible"], utility, -np.inf)
            log_denominator = logsumexp(utility, axis=1, keepdims=True)
            probability = np.exp(utility - log_denominator)
            chosen_log_probability = utility[rows, chosen] - log_denominator[:, 0]
            loglike += np.sum(q[:, latent_type] * chosen_log_probability)

            legacy_gradient += jacobian_likelihood(
                chosen,
                probability,
                period_data["x1"],
                period_data["x_change"],
                period_data["x_educ"],
                period_data["x_first2"],
                period_data["x_first4"],
                period_data["x_firstgrad"],
                period_data["x_exp"],
                period_data["period"],
                q[:, latent_type],
                school_type,
                affected,
                previous_affects,
            )
            resource_gradient[0] += np.sum(
                q[:, latent_type]
                * (
                    consumption_scaled[rows, chosen]
                    - np.sum(probability * consumption_scaled, axis=1)
                )
            )
            resource_gradient[1] += np.sum(
                q[:, latent_type]
                * (
                    debt_regressor[rows, chosen]
                    - np.sum(probability * debt_regressor, axis=1)
                )
            )

    gradient = np.concatenate((legacy_gradient, resource_gradient))
    return -float(loglike), -gradient


@njit(parallel=True)
def _parallel_type_probabilities(
    g_school,
    consumption_by_resource,
    debt_regressor,
    feasible,
    chosen,
    alpha_c,
    alpha_b,
    type_school,
    type_grant,
    type_transfer,
):
    """Stable softmax for every individual/type row in parallel."""
    ntypes = type_school.shape[0]
    nobs = g_school.shape[1]
    nchoices = g_school.shape[2]
    probability = np.zeros((ntypes, nobs, nchoices), dtype=np.float64)
    chosen_log_probability = np.empty((ntypes, nobs), dtype=np.float64)

    for flat_index in prange(ntypes * nobs):
        latent_type = flat_index // nobs
        individual = flat_index - latent_type * nobs
        school_type = type_school[latent_type]
        resource_type = 2 * type_grant[latent_type] + type_transfer[latent_type]

        maximum = -np.inf
        chosen_utility = -np.inf
        for alternative in range(nchoices):
            if feasible[individual, alternative]:
                utility = (
                    g_school[school_type, individual, alternative]
                    + alpha_c
                    * consumption_by_resource[resource_type, individual, alternative]
                    + alpha_b * debt_regressor[individual, alternative]
                )
                if utility > maximum:
                    maximum = utility
                if alternative == chosen[individual]:
                    chosen_utility = utility

        denominator = 0.0
        for alternative in range(nchoices):
            if feasible[individual, alternative]:
                utility = (
                    g_school[school_type, individual, alternative]
                    + alpha_c
                    * consumption_by_resource[resource_type, individual, alternative]
                    + alpha_b * debt_regressor[individual, alternative]
                )
                denominator += np.exp(utility - maximum)

        log_denominator = maximum + np.log(denominator)
        chosen_log_probability[latent_type, individual] = (
            chosen_utility - log_denominator
        )
        for alternative in range(nchoices):
            if feasible[individual, alternative]:
                utility = (
                    g_school[school_type, individual, alternative]
                    + alpha_c
                    * consumption_by_resource[resource_type, individual, alternative]
                    + alpha_b * debt_regressor[individual, alternative]
                )
                probability[latent_type, individual, alternative] = np.exp(
                    utility - log_denominator
                )

    return probability, chosen_log_probability


@njit(parallel=True)
def _weighted_probability_by_school(probability, q, type_school):
    """Collapse type-specific probabilities using current posterior weights."""
    ntypes, nobs, nchoices = probability.shape
    weighted_all = np.zeros((nobs, nchoices), dtype=np.float64)
    weighted_high_school = np.zeros((nobs, nchoices), dtype=np.float64)
    q_all = np.zeros(nobs, dtype=np.float64)
    q_high_school = np.zeros(nobs, dtype=np.float64)

    for individual in prange(nobs):
        for latent_type in range(ntypes):
            weight = q[individual, latent_type]
            q_all[individual] += weight
            if type_school[latent_type] == 1:
                q_high_school[individual] += weight
            for alternative in range(nchoices):
                contribution = weight * probability[latent_type, individual, alternative]
                weighted_all[individual, alternative] += contribution
                if type_school[latent_type] == 1:
                    weighted_high_school[individual, alternative] += contribution
    return weighted_all, weighted_high_school, q_all, q_high_school


@njit(parallel=True)
def _parallel_column_score(score_residual, individual_regressor):
    """Reduce independent analytical score columns across individuals."""
    nobs, nparameters = score_residual.shape
    result = np.zeros(nparameters, dtype=np.float64)
    for parameter in prange(nparameters):
        total = 0.0
        for individual in range(nobs):
            total += (
                score_residual[individual, parameter]
                * individual_regressor[individual, parameter]
            )
        result[parameter] = total
    return result


def _legacy_individual_regressors(period_data):
    """Individual-side regressors in the exact 602-parameter legacy ordering."""
    nobs = period_data["x1"].shape[0]
    regressors = np.zeros((nobs, total_n_choice_legacy), dtype=float)
    cursor = 0

    # Alternative groups x the nine invariant characteristics.
    for group in range(1 + fields + 1 + occupations):
        regressors[:, cursor + 9 * group:cursor + 9 * (group + 1)] = period_data["x1"]
    cursor += 9 * (1 + fields + 1 + occupations)

    # Work-choice parameters.
    work_count = 2 + 2 * fields + occupations + 2
    regressors[:, cursor:cursor + work_count] = 1.0
    cursor += work_count

    # Previous-choice parameters.
    previous_affects = build_previous_affects(period_data["x_change"])
    previous_count = 1 + fields + occupations
    regressors[:, cursor:cursor + previous_count] = previous_affects
    cursor += previous_count

    # Education/occupation complementarities.
    x_educ = period_data["x_educ"]
    education_columns = (
        tuple(range(9))
        + (0, 2, 7, 8)
        + (4, 5, 7, 8)
        + (3, 4, 5, 6, 8)
        + (0, 5, 8)
        + (0, 2, 6, 8)
        + tuple(range(9))
        + tuple(range(9))
    )
    regressors[:, cursor:cursor + len(education_columns)] = x_educ[:, education_columns]
    cursor += len(education_columns)

    period = int(period_data["period"])
    period_groups = 1 + fields + occupations
    period_slots = T - 2
    if period > 1:
        for group in range(period_groups):
            regressors[:, cursor + group * period_slots + period - 2] = 1.0
    cursor += period_groups * period_slots
    graduate_period_slots = T - 5 - 1
    if period > 5:
        regressors[:, cursor + period - 6] = 1.0
    cursor += graduate_period_slots

    # Period-by-work effects.
    if period > 1:
        for group in range(3):
            regressors[:, cursor + group * period_slots + period - 2] = 1.0
    cursor += 3 * period_slots
    if period > 5:
        for group in range(2):
            regressors[:, cursor + group * graduate_period_slots + period - 6] = 1.0
    cursor += 2 * graduate_period_slots

    regressors[:, cursor] = period_data["x_first2"][:, 0]
    cursor += 1
    regressors[:, cursor:cursor + 4] = period_data["x_first4"]
    cursor += 4
    if period > 5:
        regressors[:, cursor] = period_data["x_firstgrad"][:, 0]
    cursor += 1

    # Seven field groups, each with the same 24 experience/ability regressors.
    experience_width = period_data["x_exp"].shape[1]
    for group in range(fields - 1):
        regressors[
            :, cursor + group * experience_width:cursor + (group + 1) * experience_width
        ] = period_data["x_exp"]
    cursor += (fields - 1) * experience_width

    # The last two parameters are the high-schooling-type utility shifts.
    regressors[:, cursor:cursor + 2] = 1.0
    cursor += 2
    if cursor != total_n_choice_legacy:
        raise RuntimeError(
            f"Legacy regressor map produced {cursor} parameters; expected "
            f"{total_n_choice_legacy}."
        )
    return regressors


def auxiliary_choice_objective_parallel(
    choice_parameters, auxiliary_data, expected_consumption, q
):
    """Weighted choice likelihood with batched, parallel closed-form score."""
    utility_parameters, alpha_c, alpha_b, consumption_by_resource_type = (
        _choice_components(choice_parameters, auxiliary_data, expected_consumption)
    )
    affected = np.asarray(map_param_to_choice(fields, occupations), dtype=float)
    common_count = total_n_choice_legacy - 2
    legacy_gradient = np.zeros(total_n_choice_legacy, dtype=float)
    resource_gradient = np.zeros(2, dtype=float)
    loglike = 0.0

    for period_index, period_data in enumerate(auxiliary_data["periods"]):
        g_school = np.stack(
            [
                get_all_g(
                    utility_parameters,
                    period_data["x1"],
                    period_data["x_change"],
                    period_data["x_educ"],
                    period_data["x_first2"],
                    period_data["x_first4"],
                    period_data["x_firstgrad"],
                    period_data["x_exp"],
                    period_data["period"],
                    SCHOOL_REPRESENTATIVE_TYPE_IDS[school_type],
                )
                for school_type in (0, 1)
            ],
            axis=0,
        )
        debt_regressor = (
            period_data["debt_dollars"][:, None]
            * auxiliary_data["nonhome"][None, :]
            / MONEY_SCALE
        )
        consumption_by_resource = np.stack(
            [
                consumption_by_resource_type[resource][period_index] / MONEY_SCALE
                for resource in range(4)
            ],
            axis=0,
        )
        probability, chosen_log_probability = _parallel_type_probabilities(
            np.ascontiguousarray(g_school),
            np.ascontiguousarray(consumption_by_resource),
            np.ascontiguousarray(debt_regressor),
            np.ascontiguousarray(period_data["feasible"]),
            np.ascontiguousarray(period_data["chosen_index"], dtype=np.int64),
            alpha_c,
            alpha_b,
            AUXILIARY_TYPE_SCHOOL,
            AUXILIARY_TYPE_GRANT,
            AUXILIARY_TYPE_TRANSFER,
        )
        loglike += float(np.sum(q.T * chosen_log_probability))

        weighted_probability, weighted_high, q_all, q_high = (
            _weighted_probability_by_school(
                probability, q, AUXILIARY_TYPE_SCHOOL
            )
        )
        chosen = np.asarray(period_data["chosen_index"], dtype=int)
        regressors = _legacy_individual_regressors(period_data)

        expected_common = weighted_probability @ affected[:common_count].T
        chosen_common = affected[:common_count, chosen].T * q_all[:, None]
        legacy_gradient[:common_count] += _parallel_column_score(
            chosen_common - expected_common,
            regressors[:, :common_count],
        )

        expected_type = weighted_high @ affected[common_count:].T
        chosen_type = affected[common_count:, chosen].T * q_high[:, None]
        legacy_gradient[common_count:] += _parallel_column_score(
            chosen_type - expected_type,
            regressors[:, common_count:],
        )

        rows = np.arange(len(chosen))
        resource_gradient[1] += np.sum(
            q_all * debt_regressor[rows, chosen]
            - np.sum(weighted_probability * debt_regressor, axis=1)
        )
        for latent_type in range(N_AUXILIARY_TYPES):
            resource_index = (
                2 * AUXILIARY_TYPE_GRANT[latent_type]
                + AUXILIARY_TYPE_TRANSFER[latent_type]
            )
            consumption = consumption_by_resource[resource_index]
            resource_gradient[0] += np.sum(
                q[:, latent_type]
                * (
                    consumption[rows, chosen]
                    - np.sum(probability[latent_type] * consumption, axis=1)
                )
            )

    gradient = np.concatenate((legacy_gradient, resource_gradient))
    return -float(loglike), -gradient


def auxiliary_choice_objective(choice_parameters, auxiliary_data, expected_consumption, q):
    """Active analytical objective; set AUXILIARY_LEGACY_JACOBIAN=1 to fall back."""
    if os.environ.get("AUXILIARY_LEGACY_JACOBIAN", "0") == "1":
        return auxiliary_choice_objective_legacy(
            choice_parameters, auxiliary_data, expected_consumption, q
        )
    return auxiliary_choice_objective_parallel(
        choice_parameters, auxiliary_data, expected_consumption, q
    )


def estimate_school_measure(initial, x, outcome, q_school):
    observed = np.isfinite(outcome)
    result = minimize(
        _weighted_type_logit_objective,
        initial,
        args=(x[observed], outcome[observed], q_school[observed]),
        jac=True,
        method="L-BFGS-B",
        options={"disp": False, "maxiter": 300, "gtol": 1.0e-6, "ftol": 1.0e-12},
    )
    return np.asarray(result.x), float(result.fun), result


def school_measure_log_likelihoods(parameters, x, outcome):
    observed = np.isfinite(outcome)
    result = np.zeros((len(outcome), N_AUXILIARY_TYPES), dtype=float)
    beta, shift = parameters[:-1], parameters[-1]
    eta_low = x @ beta
    eta_high = eta_low + shift
    by_school = np.column_stack(
        (
            outcome * eta_low - np.logaddexp(0.0, eta_low),
            outcome * eta_high - np.logaddexp(0.0, eta_high),
        )
    )
    by_school[~observed] = 0.0
    result[:] = by_school[:, AUXILIARY_TYPE_SCHOOL]
    return result


def update_type_posteriors(pi, conditional_log_likelihood):
    safe_pi = np.clip(np.asarray(pi, dtype=float), 1.0e-12, 1.0)
    joint = conditional_log_likelihood + np.log(safe_pi)[None, :]
    log_mixture = logsumexp(joint, axis=1)
    q = np.exp(joint - log_mixture[:, None])
    return q, float(np.sum(log_mixture))


def all_auxiliary_log_likelihoods(
    pi,
    choice_parameters,
    measure_late,
    measure_summer,
    grant_parameters,
    transfer_parameters,
    loan_parameters,
    auxiliary_data,
    measures,
    expected_consumption,
):
    conditional = auxiliary_choice_log_likelihoods(
        choice_parameters, auxiliary_data, expected_consumption
    )
    conditional += school_measure_log_likelihoods(
        measure_late,
        auxiliary_data["x1_measure"],
        np.asarray(measures["late_school"], dtype=float),
    )
    conditional += school_measure_log_likelihoods(
        measure_summer,
        auxiliary_data["x1_measure"],
        np.asarray(measures["summer_class"], dtype=float),
    )
    conditional += financial_log_likelihoods(
        grant_parameters, transfer_parameters, loan_parameters, auxiliary_data
    )
    return update_type_posteriors(pi, conditional)

    
def perform_em(
    typeffect,
    max_iterations=250,
    tolerance=3.0e-4,
    return_details=False,
    verbose=True,
    resume=False,
    checkpoint_file=None,
):
    """Estimate the sixteen-type schooling x grant x transfer x loan model.

    The orchestration deliberately mirrors the economic algorithm: current
    posteriors -> financial M-step -> expected consumption -> choice M-step ->
    schooling-measure M-step -> new likelihoods and posteriors. Wage equations
    are fixed throughout and all invariant arrays are cached before iteration.
    Set ``resume=True`` to restore the complete saved EM state; otherwise the
    estimator retains its original fresh-start initialization.
    """
    global total_n_multi
    global total_n_late
    global total_n_summer

    estimation_start = time.perf_counter()

    def progress(message):
        if verbose:
            elapsed = time.perf_counter() - estimation_start
            print(f"[Auxiliary EM +{elapsed:9.2f}s] {message}", flush=True)

    def finish_step(label, step_start, extra=""):
        elapsed = time.perf_counter() - step_start
        suffix = f" | {extra}" if extra else ""
        progress(f"Finished {label} in {elapsed:.2f}s{suffix}")

    progress(
        "Starting sixteen-type auxiliary estimation "
        f"(max_iterations={max_iterations}, tolerance={tolerance:g}, "
        f"resume={resume})"
    )
    progress(
        "Analytical choice objective: parallel probability/score path | "
        f"Numba available={NUMBA_AVAILABLE}, threads={get_num_threads()}"
    )

    # Load choice arrays and all fixed data once.
    step_start = time.perf_counter()
    progress("Setup 1/8: loading cached choice and state arrays...")
    (
        choices_all,
        _vjt_unused,
        x1_new,
        choices_array_all,
        x_change,
        x_educ,
        x_first2,
        x_first4,
        x_firstgrad,
        x_exp,
    ) = load_all_arrays_feasible(auxiliar=1)
    finish_step("loading cached arrays", step_start)

    step_start = time.perf_counter()
    progress("Setup 2/8: loading schooling measures...")
    measures = pd.read_csv(f"{path}/feasible_measures.csv")
    finish_step("loading schooling measures", step_start)

    step_start = time.perf_counter()
    progress("Setup 3/8: building fixed auxiliary data and expected wages...")
    auxiliary_data = build_auxiliary_em_data(
        choices_all,
        choices_array_all,
        x1_new,
        x_change,
        x_educ,
        x_first2,
        x_first4,
        x_firstgrad,
        x_exp,
    )
    finish_step(
        "building fixed auxiliary data",
        step_start,
        extra=(
            f"N={auxiliary_data['n_individuals']}, "
            f"periods={len(auxiliary_data['periods'])}, "
            f"alternatives={len(auxiliary_data['total_choices'])}"
        ),
    )
    n = auxiliary_data["n_individuals"]
    if len(measures) != n:
        raise ValueError("Schooling measures and auxiliary panel have different sample sizes.")

    start_iteration = 0
    checkpoint_file = (
        f"{path_estimates}/auxiliary_em_checkpoint.npz"
        if checkpoint_file is None
        else os.fspath(checkpoint_file)
    )

    if resume:
        step_start = time.perf_counter()
        progress(f"Setup 4-6/8: loading EM checkpoint {checkpoint_file}...")
        if not os.path.isfile(checkpoint_file):
            raise FileNotFoundError(
                f"Cannot resume auxiliary EM: checkpoint not found: {checkpoint_file}"
            )
        required_keys = {
            "type_names", "type_school", "type_grant", "type_transfer", "type_loan",
            "pi", "q",
            "choice_parameters", "measure_late", "measure_summer",
            "grant_education_levels", "grant_receipt", "grant_amount", "grant_sigma",
            "transfer_receipt", "transfer_amount", "transfer_sigma",
            "loan_education_levels", "loan_receipt", "loan_amount", "loan_sigma",
            "observed_loglike",
        }
        with np.load(checkpoint_file, allow_pickle=False) as checkpoint:
            missing = sorted(required_keys.difference(checkpoint.files))
            if missing:
                raise ValueError(
                    "Cannot resume auxiliary EM: checkpoint is missing "
                    + ", ".join(missing)
                )
            for key, expected in (
                ("type_names", AUXILIARY_TYPE_NAMES),
                ("type_school", AUXILIARY_TYPE_SCHOOL),
                ("type_grant", AUXILIARY_TYPE_GRANT),
                ("type_transfer", AUXILIARY_TYPE_TRANSFER),
                ("type_loan", AUXILIARY_TYPE_LOAN),
            ):
                if not np.array_equal(checkpoint[key], expected):
                    raise ValueError(
                        f"Cannot resume auxiliary EM: checkpoint {key} does not "
                        "match the current sixteen-type auxiliary ordering."
                    )

            choice_parameters = np.asarray(checkpoint["choice_parameters"], dtype=float)
            measure_late = np.asarray(checkpoint["measure_late"], dtype=float)
            measure_summer = np.asarray(checkpoint["measure_summer"], dtype=float)
            pinew = np.asarray(checkpoint["pi"], dtype=float)
            q = np.asarray(checkpoint["q"], dtype=float)
            loglike_history = list(
                np.asarray(checkpoint["observed_loglike"], dtype=float).reshape(-1)
            )
            education_levels = np.asarray(
                checkpoint["grant_education_levels"], dtype=int
            )
            grant_receipt = np.asarray(checkpoint["grant_receipt"], dtype=float)
            grant_amount = np.asarray(checkpoint["grant_amount"], dtype=float)
            grant_sigma = np.asarray(checkpoint["grant_sigma"], dtype=float)
            transfer_receipt = np.asarray(checkpoint["transfer_receipt"], dtype=float)
            transfer_amount = np.asarray(checkpoint["transfer_amount"], dtype=float)
            transfer_sigma = float(np.asarray(checkpoint["transfer_sigma"]).reshape(()))
            loan_education_levels = np.asarray(
                checkpoint["loan_education_levels"], dtype=int
            )
            loan_receipt = np.asarray(checkpoint["loan_receipt"], dtype=float)
            loan_amount = np.asarray(checkpoint["loan_amount"], dtype=float)
            loan_sigma = np.asarray(checkpoint["loan_sigma"], dtype=float)

        expected_shapes = {
            "choice_parameters": (total_n_multi,),
            "measure_late": (total_n_late,),
            "measure_summer": (total_n_summer,),
            "pi": (N_AUXILIARY_TYPES,),
            "q": (n, N_AUXILIARY_TYPES),
        }
        restored_arrays = {
            "choice_parameters": choice_parameters,
            "measure_late": measure_late,
            "measure_summer": measure_summer,
            "pi": pinew,
            "q": q,
        }
        for name, expected_shape in expected_shapes.items():
            if restored_arrays[name].shape != expected_shape:
                raise ValueError(
                    f"Cannot resume auxiliary EM: checkpoint {name} has shape "
                    f"{restored_arrays[name].shape}, expected {expected_shape}."
                )
            if not np.all(np.isfinite(restored_arrays[name])):
                raise ValueError(
                    f"Cannot resume auxiliary EM: checkpoint {name} contains "
                    "non-finite values."
                )
        if not np.array_equal(education_levels, np.array([1, 2, 3])):
            raise ValueError(
                "Cannot resume auxiliary EM: checkpoint grant education levels "
                "must be [1, 2, 3]."
            )
        if not np.array_equal(loan_education_levels, np.array([1, 2, 3])):
            raise ValueError(
                "Cannot resume auxiliary EM: checkpoint loan education levels "
                "must be [1, 2, 3]."
            )
        if not loglike_history or not np.all(np.isfinite(loglike_history)):
            raise ValueError(
                "Cannot resume auxiliary EM: observed likelihood history is invalid."
            )
        if np.any(pinew <= 0.0) or not np.isclose(pinew.sum(), 1.0, atol=1.0e-8):
            raise ValueError(
                "Cannot resume auxiliary EM: prior probabilities must be positive "
                "and sum to one."
            )
        if np.any(q < 0.0) or not np.allclose(q.sum(axis=1), 1.0, atol=1.0e-8):
            raise ValueError(
                "Cannot resume auxiliary EM: posterior rows must be nonnegative "
                "and sum to one."
            )

        grant_parameters = {}
        for row, education in enumerate((1, 2, 3)):
            expected_financial_size = auxiliary_data["grant"][education]["x"].shape[1] + 1
            if (
                grant_receipt[row].shape != (expected_financial_size,)
                or grant_amount[row].shape != (expected_financial_size,)
            ):
                raise ValueError(
                    f"Cannot resume auxiliary EM: grant parameters for education "
                    f"{education} do not match the current data design."
                )
            grant_parameters[education] = {
                "receipt": grant_receipt[row].copy(),
                "amount": grant_amount[row].copy(),
                "sigma": float(grant_sigma[row]),
                "receipt_success": True,
                "receipt_message": "Loaded from EM checkpoint",
            }
        expected_transfer_size = auxiliary_data["transfer"]["x"].shape[1] + 1
        if (
            transfer_receipt.shape != (expected_transfer_size,)
            or transfer_amount.shape != (expected_transfer_size,)
        ):
            raise ValueError(
                "Cannot resume auxiliary EM: transfer parameters do not match "
                "the current data design."
            )
        transfer_parameters = {
            "receipt": transfer_receipt.copy(),
            "amount": transfer_amount.copy(),
            "sigma": transfer_sigma,
            "receipt_success": True,
            "receipt_message": "Loaded from EM checkpoint",
        }
        loan_parameters = {}
        for row, education in enumerate((1, 2, 3)):
            expected_loan_size = auxiliary_data["loan"][education]["x"].shape[1] + 1
            if (
                loan_receipt[row].shape != (expected_loan_size,)
                or loan_amount[row].shape != (expected_loan_size,)
            ):
                raise ValueError(
                    f"Cannot resume auxiliary EM: loan parameters for education "
                    f"{education} do not match the current data design."
                )
            loan_parameters[education] = {
                "receipt": loan_receipt[row].copy(),
                "amount": loan_amount[row].copy(),
                "sigma": float(loan_sigma[row]),
                "receipt_success": True,
                "receipt_message": "Loaded from EM checkpoint",
            }
        if (
            np.any(grant_sigma <= 0.0)
            or not np.all(np.isfinite(grant_sigma))
            or not np.isfinite(transfer_sigma)
            or transfer_sigma <= 0.0
            or np.any(loan_sigma <= 0.0)
            or not np.all(np.isfinite(loan_sigma))
        ):
            raise ValueError(
                "Cannot resume auxiliary EM: financial amount standard deviations "
                "must be finite and positive."
            )
        start_iteration = len(loglike_history) - 1
        finish_step(
            "loading and validating the EM checkpoint",
            step_start,
            extra=f"resuming after iteration {start_iteration}",
        )
    else:
        # Choice parameters retain the old g(.) block and append one common
        # expected-consumption coefficient and one debt-vs-home coefficient.
        choice_parameters = np.zeros(total_n_multi, dtype=float)
        choice_parameters[total_n_choice_legacy - 2:total_n_choice_legacy] = typeffect
        choice_parameters[total_n_choice_legacy] = 0.05
        choice_parameters[total_n_choice_legacy + 1] = -0.02

        measure_late = np.zeros(total_n_late, dtype=float)
        measure_summer = np.zeros(total_n_summer, dtype=float)
        measure_late[-1] = typeffect
        measure_summer[-1] = typeffect

        # Positive seeds separately orient grant and parental-transfer labels.
        resource_type_seed = max(0.05, 0.125 * abs(float(typeffect)))

        step_start = time.perf_counter()
        progress("Setup 4/8: initializing grant equations...")
        grant_parameters = initialize_grant_processes(
            auxiliary_data["grant"], financial_typeeffect=resource_type_seed
        )
        finish_step("initializing grant equations", step_start)

        step_start = time.perf_counter()
        progress("Setup 5/8: initializing transfer equations...")
        transfer_parameters = initialize_financial_source(
            auxiliary_data["transfer"], financial_typeeffect=resource_type_seed
        )
        finish_step("initializing transfer equations", step_start)

        step_start = time.perf_counter()
        progress("Setup 6/8: initializing loan equations...")
        loan_parameters = initialize_loan_processes(
            auxiliary_data["loan"], loan_typeeffect=resource_type_seed
        )
        finish_step("initializing loan equations", step_start)

    step_start = time.perf_counter()
    progress("Setup 7/8: computing initial expected consumption...")
    expected_consumption = build_expected_consumption(
        None,
        grant_parameters,
        transfer_parameters,
        auxiliary_data,
        auxiliary_data["total_choices"],
    )
    finish_step("initial expected consumption", step_start)

    step_start = time.perf_counter()
    if resume:
        progress("Setup 8/8: using checkpoint likelihood and posterior weights...")
        finish_step(
            "restoring likelihood and posterior weights",
            step_start,
            extra=f"last observed log likelihood={loglike_history[-1]:.6f}",
        )
    else:
        progress("Setup 8/8: computing initial likelihoods and posteriors...")
        pinew = np.full(N_AUXILIARY_TYPES, 1.0 / N_AUXILIARY_TYPES, dtype=float)
        q, initial_loglike = all_auxiliary_log_likelihoods(
            pinew,
            choice_parameters,
            measure_late,
            measure_summer,
            grant_parameters,
            transfer_parameters,
            loan_parameters,
            auxiliary_data,
            measures,
            expected_consumption,
        )
        finish_step(
            "initial likelihood and posterior calculation",
            step_start,
            extra=f"initial observed log likelihood={initial_loglike:.6f}",
        )
        loglike_history = [initial_loglike]
    xnew = np.concatenate((choice_parameters, measure_late, measure_summer))
    err = np.inf

    def financial_vector(parameters):
        if "receipt" in parameters:
            return np.concatenate(
                (parameters["receipt"], parameters["amount"], [parameters["sigma"]])
            )
        return np.concatenate(
            [financial_vector(parameters[education]) for education in (1, 2, 3)]
        )

    for run_iteration in range(max_iterations):
        iteration = start_iteration + run_iteration
        iteration_start = time.perf_counter()
        progress(
            f"Iteration {iteration + 1} started "
            f"({run_iteration + 1}/{max_iterations} in this run)"
        )
        pi = pinew.copy()
        q_current = q.copy()
        x0 = xnew.copy()
        grant0 = financial_vector(grant_parameters)
        transfer0 = financial_vector(transfer_parameters)
        loan0 = financial_vector(loan_parameters)

        # 1. Each financial process uses only its own posterior collapse.
        q_grant = collapse_grant_weights(q_current)
        step_start = time.perf_counter()
        progress(f"Iteration {iteration + 1}: estimating grant equations...")
        grant_parameters = estimate_grant_processes(
            auxiliary_data["grant"], q_grant, previous=grant_parameters
        )
        finish_step(
            f"iteration {iteration + 1} grant equations",
            step_start,
            extra=(
                "receipt optimizer success [2y, 4y, grad]="
                f"{[grant_parameters[e]['receipt_success'] for e in (1, 2, 3)]}"
            ),
        )

        step_start = time.perf_counter()
        progress(f"Iteration {iteration + 1}: estimating transfer equations...")
        q_transfer = collapse_transfer_weights(q_current)
        transfer_parameters = estimate_financial_source(
            auxiliary_data["transfer"], q_transfer, previous=transfer_parameters
        )
        finish_step(
            f"iteration {iteration + 1} transfer equations",
            step_start,
            extra=f"receipt optimizer success={transfer_parameters['receipt_success']}",
        )

        step_start = time.perf_counter()
        progress(f"Iteration {iteration + 1}: estimating loan equations...")
        q_loan = collapse_loan_weights(q_current)
        loan_parameters = estimate_loan_processes(
            auxiliary_data["loan"], q_loan, previous=loan_parameters
        )
        finish_step(
            f"iteration {iteration + 1} loan equations",
            step_start,
            extra=(
                "receipt optimizer success [2y, 4y, grad]="
                f"{[loan_parameters[e]['receipt_success'] for e in (1, 2, 3)]}"
            ),
        )

        # 2. Rebuild consumption for all four grant/transfer combinations.
        step_start = time.perf_counter()
        progress(f"Iteration {iteration + 1}: rebuilding expected consumption...")
        expected_consumption = build_expected_consumption(
            None,
            grant_parameters,
            transfer_parameters,
            auxiliary_data,
            auxiliary_data["total_choices"],
        )
        finish_step(f"iteration {iteration + 1} expected consumption", step_start)

        # 3. Estimate the choice block with all sixteen posterior weights.
        step_start = time.perf_counter()
        progress(
            f"Iteration {iteration + 1}: optimizing the education-choice block "
            f"({len(choice_parameters)} parameters)..."
        )
        choice_result = minimize(
            auxiliary_choice_objective,
            choice_parameters,
            args=(auxiliary_data, expected_consumption, q_current),
            jac=True,
            method="BFGS",
            options={"disp": False, "maxiter": 200, "gtol": 1.0e-5},
        )
        choice_parameters = np.asarray(choice_result.x)
        finish_step(
            f"iteration {iteration + 1} education-choice optimization",
            step_start,
            extra=(
                f"success={choice_result.success}, nit={getattr(choice_result, 'nit', 'NA')}, "
                f"nfev={getattr(choice_result, 'nfev', 'NA')}, "
                f"njev={getattr(choice_result, 'njev', 'NA')}"
            ),
        )

        # 4. Estimate measures using schooling posterior collapses.
        q_school = collapse_school_weights(q_current)
        step_start = time.perf_counter()
        progress(f"Iteration {iteration + 1}: estimating the late-school measure...")
        measure_late, late_fun, late_result = estimate_school_measure(
            measure_late,
            auxiliary_data["x1_measure"],
            np.asarray(measures["late_school"], dtype=float),
            q_school,
        )
        finish_step(
            f"iteration {iteration + 1} late-school measure",
            step_start,
            extra=f"success={late_result.success}, nit={getattr(late_result, 'nit', 'NA')}",
        )

        step_start = time.perf_counter()
        progress(f"Iteration {iteration + 1}: estimating the summer-class measure...")
        measure_summer, summer_fun, summer_result = estimate_school_measure(
            measure_summer,
            auxiliary_data["x1_measure"],
            np.asarray(measures["summer_class"], dtype=float),
            q_school,
        )
        finish_step(
            f"iteration {iteration + 1} summer-class measure",
            step_start,
            extra=f"success={summer_result.success}, nit={getattr(summer_result, 'nit', 'NA')}",
        )
        xnew = np.concatenate((choice_parameters, measure_late, measure_summer))

        # 5. Recompute all sixteen conditional likelihoods, then update pi and q.
        step_start = time.perf_counter()
        progress(f"Iteration {iteration + 1}: recomputing likelihoods and posteriors...")
        q, observed_loglike = all_auxiliary_log_likelihoods(
            pi,
            choice_parameters,
            measure_late,
            measure_summer,
            grant_parameters,
            transfer_parameters,
            loan_parameters,
            auxiliary_data,
            measures,
            expected_consumption,
        )
        pinew = np.clip(q.mean(axis=0), 1.0e-10, None)
        pinew /= pinew.sum()
        q, observed_loglike = all_auxiliary_log_likelihoods(
            pinew,
            choice_parameters,
            measure_late,
            measure_summer,
            grant_parameters,
            transfer_parameters,
            loan_parameters,
            auxiliary_data,
            measures,
            expected_consumption,
        )
        finish_step(
            f"iteration {iteration + 1} likelihood and posterior update",
            step_start,
            extra=f"observed log likelihood={observed_loglike:.6f}",
        )
        loglike_history.append(observed_loglike)

        parameter_change = max(
            float(np.max(np.abs(xnew - x0))),
            float(np.max(np.abs(financial_vector(grant_parameters) - grant0))),
            float(np.max(np.abs(financial_vector(transfer_parameters) - transfer0))),
            float(np.max(np.abs(financial_vector(loan_parameters) - loan0))),
            float(np.max(np.abs(q - q_current))),
        )
        err = parameter_change

        progress(
            f"Iteration {iteration + 1} estimates: "
            f"pi {AUXILIARY_TYPE_NAMES.tolist()}={np.array2string(pinew, precision=6)}, "
            f"maximum update={err:.6g}"
        )

        # Lightweight backward-compatible checkpoints.
        step_start = time.perf_counter()
        progress(f"Iteration {iteration + 1}: saving checkpoints...")
        np.save(f"{path_estimates}/param_em_{iteration}_typeff{typeffect}.npy", xnew)
        np.save(f"{path_estimates}/param_em_latest.npy", xnew)
        # This legacy filename is consumed by structural estimation/simulation,
        # which currently know only the two schooling types. Keep its two-column
        # contract while the full joint posterior lives in the EM checkpoint.
        np.save(
            f"{path_estimates}/em_q_typeff{typeffect}.npy",
            collapse_school_weights(q),
        )
        np.save(
            f"{path_estimates}/likelihood_external_em_{typeffect}.npy",
            np.asarray(loglike_history),
        )
        np.savez_compressed(
            f"{path_estimates}/auxiliary_em_checkpoint.npz",
            type_names=AUXILIARY_TYPE_NAMES,
            type_school=AUXILIARY_TYPE_SCHOOL,
            type_grant=AUXILIARY_TYPE_GRANT,
            type_transfer=AUXILIARY_TYPE_TRANSFER,
            type_loan=AUXILIARY_TYPE_LOAN,
            pi=pinew,
            q=q,
            choice_parameters=choice_parameters,
            measure_late=measure_late,
            measure_summer=measure_summer,
            grant_education_levels=np.array([1, 2, 3]),
            grant_receipt=np.stack(
                [grant_parameters[e]["receipt"] for e in (1, 2, 3)]
            ),
            grant_amount=np.stack(
                [grant_parameters[e]["amount"] for e in (1, 2, 3)]
            ),
            grant_sigma=np.array(
                [grant_parameters[e]["sigma"] for e in (1, 2, 3)]
            ),
            transfer_receipt=transfer_parameters["receipt"],
            transfer_amount=transfer_parameters["amount"],
            transfer_sigma=transfer_parameters["sigma"],
            loan_education_levels=np.array([1, 2, 3]),
            loan_receipt=np.stack(
                [loan_parameters[e]["receipt"] for e in (1, 2, 3)]
            ),
            loan_amount=np.stack(
                [loan_parameters[e]["amount"] for e in (1, 2, 3)]
            ),
            loan_sigma=np.array(
                [loan_parameters[e]["sigma"] for e in (1, 2, 3)]
            ),
            observed_loglike=np.asarray(loglike_history),
        )
        finish_step(f"iteration {iteration + 1} checkpoint saving", step_start)

        step_start = time.perf_counter()
        progress(f"Iteration {iteration + 1}: writing reporting tables...")
        table_paths = write_auxiliary_em_tables(
            iteration=iteration + 1,
            pi=pinew,
            choice_parameters=choice_parameters,
            measure_late=measure_late,
            measure_summer=measure_summer,
            grant_parameters=grant_parameters,
            transfer_parameters=transfer_parameters,
            loan_parameters=loan_parameters,
            auxiliary_data=auxiliary_data,
        )
        finish_step(
            f"iteration {iteration + 1} reporting tables",
            step_start,
            extra=f"saved to {table_paths['iteration_dir']}",
        )

        progress(
            f"Iteration {iteration + 1} completed in "
            f"{time.perf_counter() - iteration_start:.2f}s"
        )

        if err <= tolerance:
            progress(
                f"Converged after {iteration + 1} iterations: "
                f"maximum update {err:.6g} <= tolerance {tolerance:g}"
            )
            break

    step_start = time.perf_counter()
    progress("Saving final auxiliary estimates and expected-consumption arrays...")
    expected_consumption_arrays = {}
    for grant_type in (0, 1):
        for transfer_type in (0, 1):
            resource_index = 2 * grant_type + transfer_type
            key = f"g{grant_type}_t{transfer_type}"
            values = np.stack(expected_consumption[resource_index], axis=0)
            expected_consumption_arrays[key] = values
            np.save(
                f"{path_estimates}/expected_consumption_{key}.npy",
                values,
            )
    np.save(f"{path_estimates}/auxiliary_type_probabilities.npy", pinew)
    np.save(f"{path_estimates}/auxiliary_q_sixteen_types.npy", q)
    # Preserve an explicit S x G x T collapse for reporting and compatibility.
    # The full 16-column posterior is the structural model input.
    np.save(f"{path_estimates}/auxiliary_q_eight_types.npy", collapse_sgt_weights(q))
    np.save(
        f"{path_estimates}/auxiliary_q_schooling_types.npy",
        collapse_school_weights(q),
    )
    np.save(
        f"{path_estimates}/auxiliary_q_grant_types.npy",
        collapse_grant_weights(q),
    )
    np.save(
        f"{path_estimates}/auxiliary_q_transfer_types.npy",
        collapse_transfer_weights(q),
    )
    np.save(
        f"{path_estimates}/auxiliary_q_loan_types.npy",
        collapse_loan_weights(q),
    )
    np.save(
        f"{path_estimates}/auxiliary_observed_loglike_history.npy",
        np.asarray(loglike_history),
    )
    np.savez_compressed(
        f"{path_estimates}/auxiliary_em_results.npz",
        type_names=AUXILIARY_TYPE_NAMES,
        type_school=AUXILIARY_TYPE_SCHOOL,
        type_grant=AUXILIARY_TYPE_GRANT,
        type_transfer=AUXILIARY_TYPE_TRANSFER,
        type_loan=AUXILIARY_TYPE_LOAN,
        pi=pinew,
        q=q,
        choice_parameters=choice_parameters,
        measure_late=measure_late,
        measure_summer=measure_summer,
        grant_education_levels=np.array([1, 2, 3]),
        grant_receipt=np.stack(
            [grant_parameters[e]["receipt"] for e in (1, 2, 3)]
        ),
        grant_amount=np.stack(
            [grant_parameters[e]["amount"] for e in (1, 2, 3)]
        ),
        grant_sigma=np.array(
            [grant_parameters[e]["sigma"] for e in (1, 2, 3)]
        ),
        transfer_receipt=transfer_parameters["receipt"],
        transfer_amount=transfer_parameters["amount"],
        transfer_sigma=transfer_parameters["sigma"],
        loan_education_levels=np.array([1, 2, 3]),
        loan_receipt=np.stack(
            [loan_parameters[e]["receipt"] for e in (1, 2, 3)]
        ),
        loan_amount=np.stack(
            [loan_parameters[e]["amount"] for e in (1, 2, 3)]
        ),
        loan_sigma=np.array(
            [loan_parameters[e]["sigma"] for e in (1, 2, 3)]
        ),
        observed_loglike=np.asarray(loglike_history),
    )
    finish_step("saving final auxiliary estimates", step_start)

    details = {
        "pi": pinew,
        "q": q,
        "type_names": AUXILIARY_TYPE_NAMES.copy(),
        "choice_parameters": choice_parameters,
        "measure_late": measure_late,
        "measure_summer": measure_summer,
        "grant_parameters": grant_parameters,
        "transfer_parameters": transfer_parameters,
        "loan_parameters": loan_parameters,
        "expected_consumption": expected_consumption_arrays,
        "observed_loglike_history": np.asarray(loglike_history),
        "converged": bool(err <= tolerance),
        "iterations": len(loglike_history) - 1,
    }
    if return_details:
        progress(f"Auxiliary estimation finished in {time.perf_counter() - estimation_start:.2f}s")
        return details
    progress(f"Auxiliary estimation finished in {time.perf_counter() - estimation_start:.2f}s")
    return pinew, xnew, q

def get_marginal_effects():
    
    global total_n_multi
    global total_n_late
    global total_n_summer
    
    choices_all, _vjt_all_types, x1_new, choices_array_all, x_change, x_educ, x_first2, x_first4, x_firstgrad, x_exp = load_all_arrays_feasible(auxiliar=1)

    
    x0 = np.load(f"{path_estimates}/param_em_{64}_typeff{2}.npy")
    
    margin_late = logit_margineffect(x0[total_n_multi:total_n_multi+total_n_late], x1_new["period1"])
    margin_summer = logit_margineffect(x0[total_n_multi+total_n_late:total_n_multi+total_n_late+total_n_summer], x1_new["period1"])
    
    return margin_late, margin_summer



# Data preparation is intentionally not run on import. Call get_feasible(),
# get_data_superfeasible(), or get_x_g_superfeasible() explicitly when rebuilding
# the auxiliary sample and its cached design arrays.
#get_feasible_pubid()
#get_data_superfeasible()
#get_x_g_superfeasible()
#for period in range(1,T):
    #prepare_vjt_feasible(period)
#prepare_vjt_feasible(10)
#choices_all, vjt_all_type1,vjt_all_type2, x1_new, choices_array_all, x_change, x_educ, x_first2, x_first4, x_firstgrad, x_exp = load_all_arrays_feasible() 
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
total_n_choice_legacy = total_n
n_param_auxiliary_consumption = 1
n_param_auxiliary_debt = 1
total_n_multi = (
    total_n_choice_legacy
    + n_param_auxiliary_consumption
    + n_param_auxiliary_debt
)
total_n_late    = 9 + 1 # 9 coefficients + 1 type em
total_n_summer  = 9 + 1 # 9 coefficients + 1 type em


#get_marginal_effects()
#param_g = np.zeros(total_n)
#param_g = np.linspace(1,total_n,total_n)
#param_utility = np.linspace(1,total_n,total_n)
#a = build_param_g(param_utility)
#-----------------------------------------------------------------------------#
#                       CODE TO IMPLEMENT THE ALGORITHM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the auxiliary EM estimator."
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help=(
            "Rebuild feasible and superfeasible caches from the raw Model input "
            "panels before estimation. Use this after rerunning the Stata export."
        ),
    )
    parser.add_argument(
        "--numba-threads",
        type=int,
        default=None,
        help=(
            "Set the Numba worker count for the parallel probability and closed-form "
            "score kernels. By default Numba uses all CPUs visible to the process."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from Model/Estimates/auxiliary_em_checkpoint.npz, restoring "
            "all parameters, sixteen-type prior probabilities, posterior weights, "
            "and likelihood history. Without this flag estimation starts fresh."
        ),
    )
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help=(
            "Optional checkpoint .npz path to use with --resume. The default is "
            "the current Model/Estimates auxiliary_em_checkpoint.npz."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=250,
        help=(
            "Maximum EM iterations for this run (default: 250). With --resume, "
            "these are additional iterations after the saved checkpoint."
        ),
    )
    arguments = parser.parse_args()
    if arguments.max_iterations < 1:
        parser.error("--max-iterations must be positive.")
    if arguments.checkpoint_file is not None and not arguments.resume:
        parser.error("--checkpoint-file requires --resume.")
    if arguments.numba_threads is not None:
        if arguments.numba_threads < 1:
            parser.error("--numba-threads must be positive.")
        set_num_threads(arguments.numba_threads)
    if arguments.rebuild_data:
        print("Rebuilding feasible auxiliary-data caches from raw panels...", flush=True)
        get_feasible()
        get_feasible_pubid()
        get_superfeasible()
        get_x_g_superfeasible()
        print("Finished rebuilding auxiliary-data caches.", flush=True)
    pi, xnew, q = perform_em(
        2,
        max_iterations=arguments.max_iterations,
        resume=arguments.resume,
        checkpoint_file=arguments.checkpoint_file,
    )

#.............................................................................#
#                       CODE TO CHECK THE JACOBIAN ETC...

#param_g = load_param_g(real=1)

#pi = np.array([0.5,0.5])


#q = get_q(pi,param_g,choices_all,vjt_all_type1,x1_new,choices_array_all,x_change,x_educ, x_first2, x_first4, x_firstgrad)
#q = np.concatenate((np.ones(4982)[...,None]*0.5,np.ones(4982)[...,None]*0.5),axis=1)
#a = likelihood(param_g,choices_all,vjt_all_type1,vjt_all_type2,x1_new,choices_array_all,x_change,x_educ,x_first2,x_first4,x_firstgrad,q)
#real_grad = temp_jacobian(param_g,choices_all,vjt_all_type1,vjt_all_type2,x1_new,choices_array_all,x_change,x_educ,x_first2,x_first4,x_firstgrad,x_exp,q)

#epsilons = np.ones(total_n)* np.sqrt(np.finfo(float).eps)
#simulated_grad  = approx_fprime(param_g,likelihood,epsilons,choices_all,vjt_all_type1,vjt_all_type2,x1_new,choices_array_all,x_change,x_educ,x_first2,x_first4,x_firstgrad,x_exp,q)
#check = real_grad/simulated_grad

#check_grad(likelihood,temp_jacobian,param_g,choices_all,vjt_all,x1_new,choices_array_all,x_change,x_educ)
#x0 = np.zeros(total_n) 
#res = minimize(likelihood,x0,args=(choices_all, vjt_all,x1_new, choices_array_all,x_change,x_educ,x_first2,x_first4),
#               jac=True,options = {'disp':True})
#print(res.x)
# Save param g to generate the model

#np.save("estimates/param_g.npy",res.x)

#a = build_param_g(res.x)
# standard errors:
#se = np.diag(np.sqrt(res.hess_inv))

#old = np.load("estimates/param_g.npy")
#old = build_param_g(old)
#------------------------------------ RUN ------------------------------------#
#%%
# Those two functions avoid that I need to run this at each iteraction of
# the likelihood! 

#if __name__ == '__main__':
    #pool_obj = multiprocessing.Pool(T-1)
    #args = [period for period in range(1,T,1)]
    #results = pool_obj.map(prepare_vjt_feasible, args)
    #pool_obj.close()

#prepare_vjt()
#get_x1_new()

# Load all arrays to avoid loading at each iteration:
#choices_all, vjt_all, x1_new, choices_array_all = load_all_arrays()


#x0 = np.ones((20*47,))*10
#res = minimize(likelihood,x0,args=(choices_all, vjt_all,x1_new, choices_array_all),
#               jac=True,options={'disp':True})

#print(res.x)
#print(errrorrssss)
#tic  = time.time()
#likelihood(x0,choices_all, vjt_all,x1_new,choices_array_all)
#toc = time.time()
#print("Time elapsed",toc-tic)
#x0 = np.zeros((20*47,))
#tic  = time.time()
#likelihood(x0,choices_all, vjt_all,x1_new,choices_array_all)
#toc = time.time()
#print("Time elapsed",toc-tic)

#print(errorsss)

#%%
# Notice that for param_g there are 20 parameters and 48 chioces. 
# Also notice that there is one whcih is the base category. Therefore,
# I dont need to include it in the initial guess

#mkl.set_num_threads(44)
#if __name__ == '__main__':
    
    
    #x0 = np.zeros((20*47,))
    #choices_all, vjt_all, x1_new = load_all_arrays()
    #res = minimize_parallel(likelihood,x0,args=(choices_all, vjt_all,x1_new, choices_array_all))

    #print(res)
    #print(res.x)
