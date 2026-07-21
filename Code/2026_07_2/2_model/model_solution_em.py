# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 09:28:16 2023

@author: Sergi
"""

import numpy as np
import scipy
import os
import time
#import numba
from numba import njit,jit,prange
import joblib
import multiprocessing 
import multiprocessing as mp
import pandas as pd
from sklearn.linear_model import LogisticRegression
from os import environ
#print(environ['MKL_NUM_THREADS'])

#import mkl
#mkl.set_num_threads(1)

#os.chdir(r"C:\Users\Sergi\Dropbox\PhD\Projects\Papers\1_financial_constraints\Model\Temp")
#os.chdir(r"C:/Users/Sergi/Project/Real")
mu = 0
gamma = 0.57721
beta = 0.98
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
from config import DIR, OUT, INP, FUN, RDATA, CONT, EST, STATES
import budget_shock as bs
from debt_limits import (
    CONSUMPTION_FLOOR,
    INTEREST_RATE,
    get_annual_cap_by_stage,
    get_debt_region_bounds,
    get_lifetime_cap_by_stage,
    lower_bound_index,
    upper_bound_index,
)
from financial_process import (
    expected_financial_help_numba,
    load_auxiliary_financial_process,
    prepare_type_financial_parameters,
)
from latent_types import N_TYPES, TYPE_NAMES, type_components
r = INTEREST_RATE
pathfunctions = DIR["MODEL_FUNCOEF"]
path = DIR["MODEL_REALDATA"]
pathcont = DIR["MODEL_CONTINUATION"]
pathcontfinal = DIR["MODEL_CONTINUATION_FINAL"]
path_estimates       = DIR["MODEL_ESTIMATES"]
pathout       = DIR["MODEL_OUTPUT"]
#-----------------------------------------------------------------------------#

# Each worker loads the common EM financial estimates at most once and caches
# the numeric coefficient arrays selected for every joint type it encounters.
_auxiliary_financial_process = None
_type_financial_parameters = {}


def get_type_financial_parameters(type_id):
    """Return Numba-ready grant/transfer arrays for joint ``type_id``.

    Validation, disk loading, and type selection happen before the Bellman
    loops. Inner functions therefore receive only numeric arrays and never map
    types or inspect parameter dictionaries.
    """
    global _auxiliary_financial_process

    _, grant_type, transfer_type, _ = type_components(type_id)
    if type_id not in _type_financial_parameters:
        if _auxiliary_financial_process is None:
            _auxiliary_financial_process = load_auxiliary_financial_process(
                EST("auxiliary_em_results.npz")
            )
        _type_financial_parameters[type_id] = prepare_type_financial_parameters(
            _auxiliary_financial_process, grant_type, transfer_type
        )
    return _type_financial_parameters[type_id]

#-----------------------------------------------------------------------------#
# --> Define the functions
def save_npz_here(rel_path: str, names, arrays, compressed=True):
    """
    Save {name: array} -> Model/Output/... directly.
    rel_path should be relative under Output/ (we’ll join with OUT()).
    """
    full_path = OUT(*rel_path.split("/"))
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    payload = {n: a for n, a in zip(names, arrays)}
    if compressed:
        np.savez_compressed(full_path, **payload)
    else:
        np.savez(full_path, **payload)

def load_all_parameters():
    wage_0  = np.load(f"{pathfunctions}/wage_0.npy")[..., None].T
    wage_1  = np.load(f"{pathfunctions}/wage_1.npy")[..., None].T
    wage_2  = np.load(f"{pathfunctions}/wage_2.npy")[..., None].T
    wage_3  = np.load(f"{pathfunctions}/wage_3.npy")[..., None].T
    wage_6  = np.load(f"{pathfunctions}/wage_6.npy")[..., None].T
    wage_7  = np.load(f"{pathfunctions}/wage_7.npy")[..., None].T
    wage_8  = np.load(f"{pathfunctions}/wage_8.npy")[..., None].T
    wage_9  = np.load(f"{pathfunctions}/wage_9.npy")[..., None].T
    wage_10 = np.load(f"{pathfunctions}/wage_10.npy")[..., None].T

    param_wage = [wage_1, wage_2, wage_3, wage_6, wage_7, wage_8, wage_9, wage_10]

    sigmas = np.load(f"{pathfunctions}/sigmas.npy")

    grad_2    = np.load(f"{pathfunctions}/prob_grad_twoyear.npy")[..., None].T
    grad_4    = np.load(f"{pathfunctions}/prob_grad_four.npy")[..., None].T
    grad_grad = np.load(f"{pathfunctions}/prob_grad_grad.npy")[..., None].T
    param_prob_grad = [grad_2, grad_4, grad_grad]

    return (
        wage_0,
        param_wage,
        sigmas,
        param_prob_grad,
    )

def reload_budgetshock_params(raise_if_missing=True):
    global sigma_u_parinc, budget_params, debt_pen_vec
    sigma_u_parinc, budget_params, debt_pen_vec = load_params_frombudget()
    if raise_if_missing and (budget_params is None):
        raise FileNotFoundError(f"Missing {EST('budgetshock_params.npy')}")
    return sigma_u_parinc, budget_params, debt_pen_vec

def build_debt_pen_vec(budget_params):
    """
    Build a length-9 vector so that (x1_new @ debt_pen_vec) returns the right
    debt disutility for the individual's parental-income group.

    x1_new layout in your code (get_x1_new):
      [1,
       parinc_2, parinc_3, parinc_4,
       ability_2, ability_3, ability_4,
       sex, eth]

    So only entries 1:4 matter for parinc-specific penalties.

    Current estimator output uses ``baseline_plus_deviations``:
      [dp0, dp2, dp3, dp4], giving penalties
      [dp0, dp0+dp2, dp0+dp3, dp0+dp4].

    Older files without an explicit parameterization are interpreted as four
    group levels, preserving compatibility with prior saved estimates.
    """
    return bs.debt_penalty_design_vector(budget_params)


def load_params_frombudget():
    budget_params = bs.load(raise_if_missing=False)
    if budget_params is None:
        return None, None, None
    sigma_u_parinc = budget_params.get("risk_aversion")
    debt_pen_vec = build_debt_pen_vec(budget_params)
    return sigma_u_parinc, budget_params, debt_pen_vec
    



def get_params_wage(j):
    
    """ This function returns the parameters of the wage equations
    depending on the choice"""
    if j[0] < 4:
        param_wage = params_wage[j[0]-1]
    else:
        param_wage = params_wage[j[0]-3]
        
    return param_wage

@njit()
def wage0(x1_new,x2_new):
    
    """ Different wage fucntion for numba"""
    
    param_wage = wage_0
    
    # This part puts together x1 and x2. x1 is time invariant, x2 is time variant. 
    x = np.hstack((x1_new[...,None].T,x2_new[:-1][...,None].T)) # do not include grad school coefficient
    
    w = x@param_wage.T
    return w

@njit()
def get_x2_wage(x2_new,j):
    
    """
    This function 
    Remember:
        0-Business
        1-Stem
        2-Education
        3-Social
        4-Huamnities
        5-Health
        6-Other
        7-Associate
        8-GradSchool
    """
    
    if j[0] == 1: #Business
        x2_wage = x2_new
    elif j[0] == 2: #STEM
        x2_wage = np.zeros(4)
        x2_wage[0] = x2_new[1] #stem
        x2_wage[1] = x2_new[6] #other
        x2_wage[2] = x2_new[7] #associate
        x2_wage[3] = x2_new[8] #grad degree
        x2_wage = np.append(np.array(x2_new[0]),x2_wage) #Experience
    elif j[0] == 3: #Social Sciences
        x2_wage = np.zeros(4)
        x2_wage[0] = x2_new[3] #Social Sciences
        x2_wage[1] = x2_new[4] #Humanities
        x2_wage[2] = x2_new[6] #Other
        x2_wage[3] = x2_new[7] #Assicuate
        x2_wage = np.append(np.array(x2_new[0]),x2_wage) #Experience
    elif j[0] == 6: # Education
        x2_wage = np.zeros(5)
        x2_wage[0] = x2_new[2] # Education
        x2_wage[1] = x2_new[3] # Social Sciences
        x2_wage[2] = x2_new[4] # Humanities
        x2_wage[3] = x2_new[5] # Health
        x2_wage[4] = x2_new[8] # GradDegree
        x2_wage = np.append(np.array(x2_new[0]),x2_wage) #Experience
    elif j[0] == 7: #Humanities
        x2_wage = np.zeros(3)
        x2_wage[0] = x2_new[4] #Humanities
        x2_wage[1] = x2_new[7] #AssociateDegree
        x2_wage[2] = x2_new[8] #AssociateDegree
        x2_wage = np.append(np.array(x2_new[0]),x2_wage) #Experience
    elif j[0] ==8: #Health
        x2_wage = np.zeros(4)
        x2_wage[0] = x2_new[1] #Stem
        x2_wage[1] = x2_new[5] #health
        x2_wage[2] = x2_new[7] #AssociateDegree
        x2_wage[3] = x2_new[8] #GradSchool
        x2_wage = np.append(np.array(x2_new[0]),x2_wage) #Experience
        
    elif j[0] == 9: #Services
        x2_wage = x2_new
    elif j[0] == 10: #Production
        x2_wage = x2_new
    
    return x2_wage
    

@njit()
def wage(x1_new,x2_new,j,param_wage):
    
    "This function returs the wage based on the current state"

        
    x2_wage = get_x2_wage(x2_new,j)
        
    # This part puts together x1 and x2. x1 is time invariant, x2 is time variant. 
    x = np.hstack((x1_new[...,None].T,x2_wage[...,None].T))
    
    w = x@param_wage.T
    return w

@njit(cache=True)
def fin_help(x1_new, j, financial_parameters):
    """Expected grant plus transfer for one schooling alternative.

    The coefficient tuple is preselected for the permanent joint type before
    entering the Bellman loops. This function therefore performs no type
    mapping, file access, allocation of design vectors, or validation.
    """
    (
        grant_receipt,
        grant_amount,
        grant_sigma,
        transfer_receipt,
        transfer_amount,
        transfer_sigma,
    ) = financial_parameters
    return expected_financial_help_numba(
        x1_new,
        int(j[1]),
        int(j[2]),
        grant_receipt,
        grant_amount,
        grant_sigma,
        transfer_receipt,
        transfer_amount,
        transfer_sigma,
    )

@njit()
def tuition(j):
    
    if j[1] == 1:
        return 4000
    elif j[1] == 2:
        return 8000
    elif j[1] == 3:
        return 14000
    
@njit()  
def get_param_g(j,param_g):
    
    """This function returns the parameters of the g(x1) function given the current choice"""

    if ((j[1] == 0) & (j[2] == 1)) | ((j[1] == 0) & (j[2] == 2)):  #not study, work partime/fulltime
        index = 0
        pg = param_g[(j[0]-1)+(8*index)]
    elif ((j[1] == 1) & (j[2] == 1)) | ((j[1] == 1) & (j[2] == 0)): #study 2y, work partime/notwork
        index = 1
        pg = param_g[(j[0]-1)+(8*index)]
    elif ((j[1] == 2) & (j[2] == 1)) | ((j[1] == 2) & (j[2] == 0)): #study 4y, work partime/notwork
        index = 2
        pg = param_g[(j[0]-1)+(8*index)]
    elif ((j[1] == 3) & (j[2] == 1)) | ((j[1] == 3) & (j[2] == 0)): #grad school, work partime/nowork
        pg = param_g[-1]
        
    return pg
@njit
def get_x1_poli(x1):
    
    """This function creates the polinomial of x1"""
    
    x1_poli = np.ones(9)
        
    x1_poli[1] = x1[0,0]
    x1_poli[2] = x1[0,0]**2
    x1_poli[3] = x1[0,0]**3
    x1_poli[4] = x1[0,1]
    x1_poli[5] = x1[0,1]**2
    x1_poli[6] = x1[0,1]**3
    x1_poli[7] = x1[0,2]
    x1_poli[8] = x1[0,3]
    
    return x1_poli

@njit
def get_power_utility(sigma_u,c):
    
    c = np.maximum(c,CONSUMPTION_FLOOR)
    
    return 0.1*((0.00001*c)**(1-sigma_u)/(1-sigma_u))

@njit()
def numba_tile_new(x,reps):
    
    """This function performs the same as tile but in a 
    numba usable way"""
    shape_x = np.shape(x)[0]
    final  = np.zeros(shape_x*reps)
    for i in range(reps):
        
        final[i*shape_x:(i+1)*shape_x] = x
        
    return final

@njit()
def check_consumption(c):
    
    rows,columns = np.shape(c)
    
    for col in range(columns):
        for row in range(rows):
            if c[row,col] < CONSUMPTION_FLOOR:
                c[row,col] = CONSUMPTION_FLOOR
    return c

@njit()
def check_consumption_new(c):
    
    shape = np.shape(c)
    
    c = c.flatten()

    c[c<CONSUMPTION_FLOOR] = CONSUMPTION_FLOOR

    c.reshape(shape)
    return c


@njit()
def get_param_g_last(param_g_last,j):
    
    # 8 2y
    # 8 4y
    # 8 occupations
    # 1 grad school
    
    if ((j[1] == 1) & (j[2] == 1)) | ((j[1] == 1) & (j[2] == 0)): # 2y
        index = 0
        pg = param_g_last[(j[0]-1)+(8*index)] 
    elif ((j[1] == 2) & (j[2] == 1)) | ((j[1] == 2) & (j[2] == 0)): #4y
        index = 1
        pg = param_g_last[(j[0]-1)+(8*index)]      
    elif ((j[1] == 0) & (j[2] == 1)) | ((j[1] == 0) & (j[2] == 2)): #occupation
        index = 2
        pg = param_g_last[(j[0]-1)+(8*index)] 
    elif ((j[1] == 3) & (j[2] == 1)) | ((j[1] == 3) & (j[2] == 0)): # grad school
        pg = param_g_last[-1]
        
    return pg
    

@njit()
def get_x_last(x2,j):
    
    # For now x is either you switched or not and wether part or full time
    
    x_last = np.zeros(1)   
    
    
    if j[0] != x2[9]: # this means you have switechd
    
        x_last[0] = 1
        
        
    return x_last
    
    
@njit()
def get_g(x1_new,x2,j,param_g,param_g_last):

    # First load the parameters
        
    pg_x1 = get_param_g(j,param_g).astype("float64")
            
    # Now prepare the x. Basically, it is x1 + switching cost + part_time
        
    g_x1 = x1_new@pg_x1
        
    # Identify last choice and part time
        
    x_last = get_x_last(x2,j)
        
    # Get parameters
        
    pg_last = get_param_g_last(param_g_last,j).astype("float64")
            
    g_last = x_last@pg_last
        
    g = g_last + g_x1
    
    return g



#@njit()
def get_utility(
    sigma_u, x1, x1_new, x2, b, b1, e, j, period,
    financial_parameters, z=0.0
):
    """
    Studying consumption for each (shock point, b) pair.
    e: array length Q
    z: either scalar or array length Q (budget shock)
    """

    nb = np.shape(b)[0]
    Q  = np.shape(e)[0]

    h0 = fin_help(x1_new, j, financial_parameters)  # type-specific scalar
    h_vis = np.repeat(h0, nb*Q)         # length Q*nb

    # Add the budget shock. It may be one value per quadrature point or one
    # value per (quadrature point, current-debt state) when its conditional
    # mean depends on pre-choice resources.
    if np.ndim(z) == 0:
        z_vis = 0.0
    elif np.asarray(z).size == Q * nb:
        z_vis = np.asarray(z, dtype=np.float64).reshape(-1)
    else:
        z_vis = np.repeat(z, nb)        # length Q*nb
    h_vis = h_vis + z_vis

    w = wage0(x1_new,x2)
    real_wage = np.exp(w+e)*(j[2]/2)*52*40
    real_wage = np.repeat(real_wage, nb)     # length Q*nb

    b_vis = numba_tile_new(b, Q)             # length Q*nb

    c = (h_vis - (1+r)*b_vis - tuition(j) + real_wage)
    c = c[...,np.newaxis] + b1
    return c



def fieldummies(j):
    
    fieldummies = np.zeros(7)
    if j[0] > 1:
        fieldummies[j[0]-2] = 1 # set field
    return fieldummies

def get_x_grad(x1_new,x2,j):
    """
    This function gets the x's for the graduation probability. 
    x1_new + i.twoyear_exp + i.fouryear_exp in the normal case.
    Coefficients are:
        -> 2y graduation: 1-5 dummies 2y exp, 0-4 dummies 4y exp
        -> 4y graduation: 1-4 dummies 2y exp + dummy twoyear grad, 3-6 dummies 4y exp, + field dummies
        -> grad graduation: 0-2 dummies for grad experience.
    """
    if j[0]< 9 : # 4y choice
        twodummies = np.zeros(5)
        if x2[4] == 1: #Two year grad
            twodummies[-1] = 1 
        else:
            if x2[1] > 0 : # More than 0 experience. In principle you can't arrive here with 0 but just in case!
                twodummies[int(np.minimum(x2[1],4))-1] = 1
        fourdummies = np.zeros(4)  # You can only have 3to6 experience
        if x2[2] > 2:
            fourdummies[int(np.minimum(x2[2],6))-3] = 1
        x_dummies = np.append(twodummies,fourdummies)
        # Now field dummies
        fieldummies = np.zeros(6)
        if (j[0]>2):
            fieldummies[int(j[0])-3] = 1 
        elif j[0]==2:
            fieldummies[0] = 1
        x_dummies = np.append(x_dummies,fieldummies)
        x_grad = np.append(x1_new,x_dummies)
        
    elif j[0] == 12: # associate choice
        twodummies = np.zeros(5)
        if x2[1] > 0 :
            twodummies[int(np.minimum(x2[1],5))-1] = 1
        fourdummies = np.zeros(5)
        fourdummies[int(np.minimum(x2[2],4))] = 1
        x_dummies = np.append(twodummies,fourdummies)
        x_grad = np.append(x1_new,x_dummies)
    
    elif j[0] == 13 : # grad school choice
        graddummies = np.zeros(3)
        graddummies[int(np.minimum(x2[3],2))] = 1
        x_grad = np.append(x1_new,graddummies)
    return x_grad

#@njit()
# Not sure if this is faster njited! 
def probability_graduation(x1_new,x2,j):
    """This function returns the graduation probability of each x2 depending
    on if it is 2y or 4y school. Right now I am taking the parameters as given,
    but at some point I need to create a function that estimate them"""
    
    # At the moment I will just use x1 info, but in the future I might use x2. 
    # Concatenate information 
    #x = np.concatenate((x1_new,x2_new),axis=1)

    # Check if it is a two year or a four year school
    
    if j[1] == 1: #two-year school
        x_grad = get_x_grad(x1_new,x2,j)
        param_grad_two  = param_prob_grad[0]
        temp = np.exp(param_grad_two@x_grad.T)
        p = temp / (1+ temp)    
    elif j[1] == 2: # four-year school
        x_grad = get_x_grad(x1_new,x2,j)
        param_grad_four = param_prob_grad[1]
        temp = np.exp(param_grad_four@x_grad.T)
        p = temp / (1+ temp)
    elif j[1] == 3: # graduate school
    
        x_grad = get_x_grad(x1_new,x2,j)
        param_grad_grad = param_prob_grad[2]
        temp = np.exp(param_grad_grad@x_grad.T)
        p = temp / (1+ temp)

    return p



def VT(x1,x1_new,x2,x2_new,b,period,j,evt,repayment):
    """ This function returns the continuation value for each individual
    in non terminal periods.  """

    # First check if the choice and state include a possible graduation state. 
    if ((x2[1] >=1) & (j[1] == 1) & (x2[4] == 0)) | ((x2[2]>=3) & (j[1] == 2) & (x2[5] == 0))  | (j[1] == 3) :
        # The choice could induce a graduation state. For this reason, take the expectation.
        grad_x2  =  move_state_grad(x2,j,period,grad=1)
        notgrad_x2 = move_state_grad(x2,j,period)
        #evt = np.load(f"evt/evt_t{period+1}_{x1}.npz")
        evt_grad =  evt[f"evt_t{period+1}_{x1}_{grad_x2}"]
        evt_nograd = evt[f"evt_t{period+1}_{x1}_{notgrad_x2}"]
        p_grad = probability_graduation(x1_new,x2,j)
        vt = p_grad*evt_grad + (1-p_grad)*evt_nograd
            
    else: 
        # load the data
        x2 = move_state_grad(x2,j,period)
        evt_new = evt[f"evt_t{period+1}_{x1}_{x2}"]
        # Now for each individual take the one corresponding with its tmr level of debt
        # Notice that if on the future the repayment scheme changes for different states
        # I should correct for that here.
        if j[1] == 0:  # if the individual is not studying, debt has evolved.
            vt = evt_new[repayment]
        else:  # if the individual is studying, a debt decision has to be made
            vt = evt_new
            
    return vt

def move_state(x,j,period):
    
    "this function moves the state space given your current state space and choice"
    
    z = np.copy(x)
        
    # Right now the field of study does not matter, unless it is your graduation
    # year. Therefore, I will just focus on occupation vs education choices

    if j[1] == 1:  # two-year choice
    
        z[1] = z[1] + 1
        
        z[7] = period # last period enrolled	
        
        z[9] = j[0]  # track previous choice
    
    elif j[1] == 2: # four year choice
    
        z[2] = z[2] + 1
        
        z[7] = period # last period enrolled
        
        z[9] = j[0]  # track previous choice
        
    elif j[1] == 3 : # grad school choice
    
        z[3] = z[3] + 1
        
        z[7] = period # last period enrolled
        
        z[9] = j[0]  # track previous choice
        
    
    elif (j[1] == 0) & (j[2] >= 1) & (z[4]!=1) & (z[5]!=1): # work full or partime before graduation
    
        z[0] = z[0] + 1
        
        z[9] = j[0]  # track previous choice
        
    elif (j[1] == 0) & (j[2] >= 1) & ((z[4] == 1) | (z[5] == 1)): # work full or partime after graduation
    
        z[0] = z[0] + 1
        
        z[9] = j[0]  # track previous choice
        
    elif (j[1] == 0) & (j[2]==0):  #home production
    
        z[9] = 0  # track previous choice
    
    return z

def move_state_grad(x,j,period,grad=0):
    
    """this function moves the state space given your current state space and choice
    after the graduation uncertainty has been realised."""
    
    z = np.copy(x)
    
    if grad == 0:  # If there is no graduation at this period, just leave it as it was. 
            
        # Right now the field of study does not matter, unless it is your graduation
        # year. Therefore, I will just focus on occupation vs education choices

        if j[1] == 1:  # two-year choice
        
            z[1] = z[1] + 1
            
            z[7] = period # last period enrolled
            
            z[9] = j[0]  # track previous choice
        
        elif j[1] == 2: # four year choice
        
            z[2] = z[2] + 1
            
            z[7] = period # last period enrolled
            
            z[9] = j[0]  # track previous choice
            
        elif j[1] == 3: # grad chioce
        
            z[3] = z[3] + 1
            
            z[7] = period
    
            z[9] = j[0]
            
        
        elif (j[1] == 0) & (j[2] >= 1) & (z[4]!=1) & (z[5]!=1): # work full or partime before graduation
        
            z[0] = z[0] + 1
            
            z[9] = j[0]  # track previous choice
            
        elif (j[1] == 0) & (j[2] >= 1) & ((z[4] == 1) | (z[5] == 1)): # work full or partime after graduation
        
            z[0] = z[0] + 1
            
            z[9] = j[0]  # track previous choice
            
        elif (j[1] == 0) & (j[2]==0):  #home production
        
            z[9] = 0  # track previous choice
        
        
    else:  # If there is graduation at this period, it is either we are at a two-year, four year or grad choice
    
        if j[1] == 1:  # two-year choice
            
            z[1] = 99
            
            z[4] = 1
            
            z[9] = j[0]  # track previous choice
            
            z[7] = period  # set period of graduation
            
            z[8] = j[0]   # set the major the individual is getting
        
        if j[1] == 2: # four-year choice
        
            z[2] = z[2] + 1
            
            z[5] = 1
            
            z[9] = j[0]  # track previous choice
            
            z[7] = period  # set period of graduation
            
            z[8] = j[0]  # set the major the individual is getting
            
            z[1] = 99 
            
            z[2] = 99
            
        if j[1] == 3:  # grad school choice
        
            z[3] = z[3] + 1
            
            z[6] = 1
            
            z[7] = period
            
            z[9] = j[0]  # track previous choice
            
            z[3] = 99
            
    
    return z



                    

@njit
def get_maximum_loop_modified_c(sigma_u,b,c,continuation,j,x2):
    """This function loops over states. The only reason it is an external function is
     to perform the loop using numba. To save computation time this function uses some
     features of the objective function to just search over reasonable maximum candidates"""
    
    payoff = np.zeros(np.shape(c)[0])
                
    # Loop over all states today
        
    continuation = continuation[:,0]    
    # Loop over the amount of different "e" shocks. The amount of gaussian quadrature points
    # that I am using. 
    
    quadrature = int(np.shape(c)[0]/np.shape(continuation)[0])
    
    for e in range(quadrature):
        
        # now loop over all possible bt today at each e state. 
        
        it = 0
        check = 0
        alert = 0
        while check < 1:
            
            c2 = c[it+np.shape(continuation)[0]*e]
            
            if it == 0 :  # check that this is the first iteration of the current quadrature
            
                # Get the first not negative consumption value
                
                c2new = c2[c2>0]
                
                # Get the bound
                firstbound = np.shape(b)[0]-len(c2new)
            
                u = get_power_utility(sigma_u,np.maximum(c2new,CONSUMPTION_FLOOR))
                
                final = u + continuation[firstbound:]
                
                amax = np.argmax(final)
                amax_new = amax  + firstbound  # just neecssay to save the payoff later
            # Analyze the case where it is not equal to 0 
            
            else:
                
                # The new boundaries are defined -10/+10 the previous argmax
                
                bound_left = np.maximum(amax_new-10,it)
                bound_right = amax_new + 10
                # check that the boundaries are well defined.
                # The boundari on the left should be the maximum of the
                # current iteration or bound_left -10 
                
                if bound_right > np.shape(continuation)[0]:
                    
                    bound_right = np.shape(continuation)[0]
                    
                # notice the individual should at least have positive consumption
                if c2[bound_left] < 0:
                    c2new = c2[c2>0]
                    # Notice that if there is no posible positive value of consumption
                    # I will set debt to the maximum possible level. 
                    if len(c2new) == 0: 
                        
                        # If this is the case debt is the maximum for all possible
                        # values of debt today, because no matter your debt consumption
                        # wont be positive.
                        vjt = get_power_utility(sigma_u,np.maximum(c[it+np.shape(continuation)[0]*e:(e+1)*np.shape(continuation)[0],-1],CONSUMPTION_FLOOR)) + continuation[-1]
                        
                        payoff[it+np.shape(continuation)[0]*e:(e+1)*np.shape(continuation)[0]] = vjt
                        alert = 1
                        check = 1
                    else: 
                        # if this is the case there are still things to do. 
                    
                        bound_left = np.shape(b)[0]-len(c2new)
                        bound_right = 100 

                    
                # Get the set of maximum candidates
                
                c_new  =  np.maximum(c2[bound_left:bound_right],CONSUMPTION_FLOOR) # Not sure if this will work with numba
                
                # compute the payoff
                u = get_power_utility(sigma_u,c_new)
                
                # sum the continuation value
                
                u = u + continuation[bound_left:bound_right]
                         
                
                final = u
                # Compute the maximum and the argmax

                amax = np.argmax(final)
                

                # Check if the optimal is the right boundary
                
                if amax== (np.shape(final)[0]-1):
                    
                    # Now everything is 99? Yes

                    vjt = get_power_utility(sigma_u,np.maximum(c[it+np.shape(continuation)[0]*e:(e+1)*np.shape(continuation)[0],-1],CONSUMPTION_FLOOR)) + continuation[-1]
                    
                    payoff[it+np.shape(continuation)[0]*e:(e+1)*np.shape(continuation)[0]] =vjt
                    check = 1
                    alert = 1
                # Get the right amax:
                    
                if (amax!= (np.shape(final)[0]-1)) :
                    
                    amax_old = amax_new
                    
                    amax_new = amax + bound_left  # I just need to account for how much elements there are on the left
                    #print(amax_old,amax)
                    # security if condition, rarely used. Sometimes the objective function is wierd
                    
                    if (amax_new-amax_old) > 9:  # in this case I probably have the wrong optimal for previous values
                        #print(amax_new,amax_old,"here")
                        # The jamp is too big, let's compute the whole array within the positives.
                        c2new = c2[c2>0]
                        # Get the bound
                        firstbound = np.shape(b)[0]-len(c2new)
                        clast = np.maximum(c2new,CONSUMPTION_FLOOR)
                        u = get_power_utility(sigma_u,clast)
                        final = u + continuation[firstbound:]
                        amax = np.argmax(final)
                        amax_new= amax + firstbound
                            
            
            # Store the result (only if it has not been done already)
            if alert == 0 :
                payoff[it+np.shape(continuation)[0]*e] = final[amax]
            
            # Sum one iteration
            
            it += 1
            
            # Estabish terminal condition
            
            if it == (np.shape(continuation)[0]):
                
                check = 1
                
                
    return payoff

@njit
def get_maximum_loop_modified_c_maxdebt(sigma_u, b, c, continuation, j, x2):
    """
    Two-region version:

    Region 1: accrued debt < lifetime cap
        -> search over [lo_idx[it], hi_idx[it]] among choices with c >= 2000
        -> if none meets the consumption floor, force hi_idx[it]

    Region 2: accrued debt >= lifetime cap
        -> no search, debt evolves mechanically
    """
    payoff = np.zeros(np.shape(c)[0])
    consumption_floor = CONSUMPTION_FLOOR

    continuation = continuation[:, 0]
    ncont = np.shape(continuation)[0]
    quadrature = int(np.shape(c)[0] / ncont)

    lo_idx, hi_idx, cap_start = get_debt_region_bounds(b, x2, j)

    for e in range(quadrature):
        it = 0
        check = 0
        amax_new = 0

        while check < 1:
            row_idx = it + ncont * e
            c2 = c[row_idx]
            alert = 0

            # --------------------------------------------------------
            # Region 2: no borrowing choice left
            # --------------------------------------------------------
            if it >= cap_start:
                idx_use = lo_idx[it]   # here lo_idx[it] == hi_idx[it]
                payoff[row_idx] = (
                    get_power_utility(
                        sigma_u, np.maximum(c2[idx_use], consumption_floor)
                    )
                    + continuation[idx_use]
                )
                it += 1
                if it == ncont:
                    check = 1
                continue

            # --------------------------------------------------------
            # Region 1: active borrowing region
            # --------------------------------------------------------
            lo = lo_idx[it]
            hi = hi_idx[it]

            if it == 0:
                c2temp = c2[lo:hi+1]
                c2new = c2temp[c2temp >= consumption_floor]

                if len(c2new) == 0:
                    idx_use = hi
                    payoff[row_idx] = (
                        get_power_utility(
                            sigma_u, np.maximum(c2[idx_use], consumption_floor)
                        )
                        + continuation[idx_use]
                    )
                    amax_new = idx_use
                else:
                    firstbound = hi + 1 - len(c2new)
                    u = get_power_utility(sigma_u, c2new)
                    final = u + continuation[firstbound:hi+1]
                    amax = np.argmax(final)
                    amax_new = amax + firstbound
                    payoff[row_idx] = final[amax]

            else:
                # local window around previous argmax, clipped to feasible set
                bound_left = np.maximum(amax_new - 10, lo)
                bound_left = np.maximum(bound_left, it)
                bound_right = np.minimum(bound_left + 20, hi + 1)

                if bound_right <= bound_left:
                    bound_left = lo
                    if bound_left < it:
                        bound_left = it
                    bound_right = hi + 1

                if c2[bound_left] < consumption_floor:
                    c2temp = c2[lo:hi+1]
                    c2new = c2temp[c2temp >= consumption_floor]

                    if len(c2new) == 0:
                        idx_use = hi
                        payoff[row_idx] = (
                            get_power_utility(
                                sigma_u, np.maximum(c2[idx_use], consumption_floor)
                            )
                            + continuation[idx_use]
                        )
                        amax_new = idx_use
                        it += 1
                        if it == ncont:
                            check = 1
                        continue
                    else:
                        bound_left = hi + 1 - len(c2new)
                        if bound_left < lo:
                            bound_left = lo
                        if bound_left < it:
                            bound_left = it
                        bound_right = hi + 1

                c_new = c2[bound_left:bound_right]
                u = get_power_utility(sigma_u, c_new)
                final = u + continuation[bound_left:bound_right]
                amax = np.argmax(final)

                if amax != (np.shape(final)[0] - 1):
                    amax_old = amax_new
                    amax_new = amax + bound_left

                    if (amax_new - amax_old) > 9:
                        c2temp = c2[lo:hi+1]
                        c2new = c2temp[c2temp >= consumption_floor]

                        if len(c2new) == 0:
                            idx_use = hi
                            payoff[row_idx] = (
                                get_power_utility(
                                    sigma_u,
                                    np.maximum(c2[idx_use], consumption_floor),
                                )
                                + continuation[idx_use]
                            )
                            amax_new = idx_use
                            it += 1
                            if it == ncont:
                                check = 1
                            continue
                        else:
                            firstbound = hi + 1 - len(c2new)
                            if firstbound < lo:
                                firstbound = lo
                            if firstbound < it:
                                firstbound = it

                            clast = c2[firstbound:hi+1]
                            u = get_power_utility(sigma_u, clast)
                            final = u + continuation[firstbound:hi+1]
                            amax = np.argmax(final)
                            amax_new = amax + firstbound

                payoff[row_idx] = final[amax]

            it += 1
            if it == ncont:
                check = 1

    return payoff

def get_maximum_loop_modified_c_maxdebt_debug(sigma_u, b, c, continuation, j, x2):
    """
    Only prints REAL argmax threats.
    Retained diagnostic for the legacy fixed-index cap, using the shared
    consumption floor.
    """

    maxdebt = 76
    payoff = np.zeros(np.shape(c)[0])

    continuation = continuation[:, 0]
    ncont = np.shape(continuation)[0]
    quadrature = int(np.shape(c)[0] / ncont)

    for e in range(quadrature):
        it = 0
        check = 0

        while check < 1:
            alert = 0
            row_idx = it + ncont * e
            c2 = c[row_idx]

            if it == 0:
                c2temp = c2[:maxdebt]
                c2new = c2temp[c2temp >= CONSUMPTION_FLOOR]

                firstbound = maxdebt - len(c2new)
                u = get_power_utility(sigma_u, np.maximum(c2new, CONSUMPTION_FLOOR))
                final = u + continuation[firstbound:maxdebt]

                if len(final) == 0:
                    print("\n[REAL THREAT: FIRST ARGMAX EMPTY]")
                    print("j =", j)
                    print("x2 =", x2)
                    print("e =", e, "it =", it, "row_idx =", row_idx)
                    print("len(c2new) =", len(c2new))
                    print("firstbound =", firstbound)
                    print("len(final) =", len(final))
                    print("any feasible capped =", np.any(c2[:maxdebt] >= CONSUMPTION_FLOOR))
                    print("any feasible full =", np.any(c2 >= CONSUMPTION_FLOOR))
                    print("min capped =", np.min(c2[:maxdebt]))
                    print("max capped =", np.max(c2[:maxdebt]))
                    print("min full =", np.min(c2))
                    print("max full =", np.max(c2))
                    print("ABOUT TO ARGMAX [FIRST]")

                amax = np.argmax(final)
                amax_new = amax + firstbound

            elif (it > 0) & (it < maxdebt):
                bound_left = np.maximum(amax_new - 10, it)
                bound_right = np.minimum(bound_left + 20, maxdebt)

                if bound_right <= bound_left:
                    print("\n[REAL THREAT: INNER SLICE EMPTY BEFORE ARGMAX]")
                    print("j =", j)
                    print("x2 =", x2)
                    print("e =", e, "it =", it, "row_idx =", row_idx)
                    print("amax_new =", amax_new)
                    print("bound_left =", bound_left, "bound_right =", bound_right)

                if c2[bound_left] < CONSUMPTION_FLOOR:
                    c2temp = c2[:maxdebt]
                    c2new = c2temp[c2temp >= CONSUMPTION_FLOOR]

                    if len(c2new) == 0:
                        idx_fallback = np.maximum(it, maxdebt)
                        vjt = (
                            get_power_utility(
                                sigma_u,
                                np.maximum(c2[idx_fallback], CONSUMPTION_FLOOR)
                            )
                            + continuation[idx_fallback]
                        )
                        payoff[row_idx] = vjt
                        alert = 1
                    else:
                        bound_left = maxdebt - len(c2new)
                        bound_right = maxdebt

                c_new = np.maximum(c2[bound_left:bound_right], CONSUMPTION_FLOOR)
                u = get_power_utility(sigma_u, c_new)
                final = u + continuation[bound_left:bound_right]

                if alert == 0 and len(final) == 0:
                    print("\n[REAL THREAT: INNER ARGMAX EMPTY]")
                    print("j =", j)
                    print("x2 =", x2)
                    print("e =", e, "it =", it, "row_idx =", row_idx)
                    print("amax_new =", amax_new)
                    print("bound_left =", bound_left, "bound_right =", bound_right)
                    print("len(final) =", len(final))
                    print("ABOUT TO ARGMAX [INNER]")

                if alert == 0:
                    amax = np.argmax(final)

                    if amax != (np.shape(final)[0] - 1):
                        amax_old = amax_new
                        amax_new = amax + bound_left

                        if (amax_new - amax_old) > 9:
                            c2temp = c2[:maxdebt]
                            c2new = c2temp[c2temp >= CONSUMPTION_FLOOR]

                            firstbound = maxdebt - len(c2new)
                            clast = np.maximum(c2new, CONSUMPTION_FLOOR)
                            u = get_power_utility(sigma_u, clast)
                            final = u + continuation[firstbound:maxdebt]

                            if len(final) == 0:
                                print("\n[REAL THREAT: BIGJUMP ARGMAX EMPTY]")
                                print("j =", j)
                                print("x2 =", x2)
                                print("e =", e, "it =", it, "row_idx =", row_idx)
                                print("amax_old =", amax_old)
                                print("amax_new(pre-reset) =", amax_new)
                                print("len(c2new) =", len(c2new))
                                print("firstbound =", firstbound)
                                print("len(final) =", len(final))
                                print("any positive capped =", np.any(c2[:maxdebt] > 0))
                                print("any positive full =", np.any(c2 > 0))
                                print("ABOUT TO ARGMAX [BIGJUMP]")

                            amax = np.argmax(final)
                            amax_new = amax + firstbound

            elif it >= maxdebt:
                idx_fallback = np.maximum(it, maxdebt)
                payoff[row_idx] = (
                    get_power_utility(
                        sigma_u,
                        np.maximum(c2[idx_fallback], CONSUMPTION_FLOOR)
                    )
                    + continuation[idx_fallback]
                )
                alert = 1

            if alert == 0:
                payoff[row_idx] = final[amax]

            it += 1

            if it == ncont:
                check = 1

    return payoff

#@njit()
def get_maximum(sigma_u,c,continuation,x1,b,j,x2,maxdebt):
    
    """This function computes the maximum of the conditional value function, minimizing
    the amount of times that c**(1-sigma)/1-sigma needs to be computed, since it is
    very time consuming to compute.
    
    There are always three candidates for maximum:
        1. the point most to the left.
        2. the point most to the right.
        3. the maximum in the overlap notflat regions or wathever point in the
            overlap flat regions"""
    
    if maxdebt == False:
            
        payoff = get_maximum_loop_modified_c(sigma_u,b,c,continuation,j,x2)
    else:
        payoff = get_maximum_loop_modified_c_maxdebt(sigma_u,b,c,continuation,j,x2)
        
    return payoff
    

def get_debt_tomorrow(x1_new,x2,j,b,e):
    
    """
    This function moves debt based on your income shock!
    """
    global debt_range
    global params_wage
    
    params_wage_j = get_params_wage(j)
    
    w = wage(x1_new,x2,j,params_wage_j)
    
    real_wage = np.exp(w+e)*(j[2]/2)*52*40
    
    real_wage = np.repeat(real_wage,np.shape(b)[0])
    
    b_vis = numba_tile_new(b,np.shape(e)[0])
   
    b1 = b_vis - real_wage*0.1  # 10% repayment
    
    b1[b1<0] = 0
    
    # Now map debt to the closer value on the debt range
    
    diff = (debt_range - b1[...,None])**2
    
    debt_position= np.argmin(diff,axis=1)
    
    return debt_position

@njit
def get_debt_income_home(x1_new,x2_new,x2,period,j,debt,e,conter):
    
    '''
    This function computes the next period value of debt based on todays debt
    and income shock. The value will be different depending on which couterfactual
    scenario is being solved. 
    
    '''
    
    periods = period - x2[7] # Get last schooling period. Notice that if you have never atteded school this does not matter, since you will always have 0 debt.
    
    w = 0

    real_wage = np.exp(w+e)*(j[2]/2)*52*40
    
    real_wage = np.repeat(real_wage,np.shape(debt)[0])
    
    debt_vis = numba_tile_new(debt,np.shape(e)[0])
    
    if conter == 1:
        
        discretionary_income = real_wage-2.25*15000
        
        paid = np.maximum(0,np.minimum(real_wage-CONSUMPTION_FLOOR,np.minimum(0.05*discretionary_income,(1/periods)*debt_vis*(1+r))))
        
        income = real_wage-paid
        
        debtnew = (1+r)*debt_vis - paid
        
        # map debt to the index position
        
        #diff = (debt - debtnew[...,None])**2
        
        #debt_position= np.argmin(diff,axis=1)
        
        #dif = np.sqrt(np.min(diff,axis=1))
        
        #income  = income + dif
        
    else:
        
        
        paid = np.maximum(0,np.minimum(real_wage-CONSUMPTION_FLOOR,(1/periods)*debt_vis*(1+r)))
        
        debtnew = (1+r)*debt_vis - paid
        
        # map debt to the index position
        
        #diff = (debt - debtnew[...,None])**2
        
        #debt_position= np.argmin(diff,axis=1)
        
        # Get disposable income
        
        income = real_wage - paid
        
        #dif = np.sqrt(np.min(diff,axis=1))
        
        #income  = income + dif
    
    return debtnew, income

@njit
def get_debt_income(x1_new,x2_new,x2,period,j,debt,e,conter,param_wage_j):
    
    '''
    This function computes the next period value of debt based on todays debt
    and income shock. The value will be different depending on which couterfactual
    scenario is being solved. 
    
    '''
    
    periods = period - x2[7] # Get last schooling period. Notice that if you have never atteded school this does not matter, since you will always have 0 debt.
    
    w = wage(x1_new,x2_new,j,param_wage_j)

    real_wage = np.exp(w+e)*(j[2]/2)*52*40
    
    real_wage = np.repeat(real_wage,np.shape(debt)[0])
    
    debt_vis = numba_tile_new(debt,np.shape(e)[0])
    
    if conter == 1:
        
        discretionary_income = real_wage-2.25*15000
        
        paid = np.maximum(0,np.minimum(real_wage-CONSUMPTION_FLOOR,np.minimum(0.05*discretionary_income,(1/periods)*debt_vis*(1+r))))
        
        income = real_wage-paid
        
        debtnew = (1+r)*debt_vis - paid
        
        # map debt to the index position
        
        #diff = (debt - debtnew[...,None])**2
        
        #debt_position= np.argmin(diff,axis=1)
        
        #dif = np.sqrt(np.min(diff,axis=1))
        
        #income  = income + dif
        
    else:
        
        
        paid = np.maximum(0,np.minimum(real_wage-CONSUMPTION_FLOOR,(1/periods)*debt_vis*(1+r)))
        
        debtnew = (1+r)*debt_vis - paid
        
        # map debt to the index position
        
        #diff = (debt - debtnew[...,None])**2
        
        #debt_position= np.argmin(diff,axis=1)
        
        # Get disposable income
        
        income = real_wage - paid
        
        #dif = np.sqrt(np.min(diff,axis=1))
        
        #income  = income + dif
    
    return debtnew, income

def map_debt_position(debt,debtnew):
    
    diff = (debt - debtnew[...,None])**2
    
    debt_position= np.argmin(diff,axis=1)
    
    return debt_position

def evolve_continuation(x1,x1_new,x2,x2_new,b,period,e,j,evt,debt_tomorrow):
    
    continuation = np.zeros((np.shape(debt_tomorrow)[0]))
    for i in range(np.shape(e)[0]):
        continuation[i*100:(i+1)*100] =  beta*VT(x1,x1_new,x2,x2_new,b,period,j,evt,debt_tomorrow[i*100:(i+1)*100])[:,0]
    
    
    return continuation

def get_conditional(
    sigma_u, x1, x1_new, x2, x2_new, b, b1, e, j, period, evt,
    conterfactual, maxdebt, financial_parameters, z=0.0
):

    global debt_pen_vec

    # scalar penalty for this individual (depends only on x1_new)
    debt_pen = float(x1_new @ debt_pen_vec)   # typically negative

    nb = b.shape[0]
    Q  = e.shape[0]

    # mask for "today debt > 0" repeated over shocks
    mask_nb = (b > 0).astype(np.float64)      # length nb
    mask    = np.tile(mask_nb, Q)             # length Q*nb

    if j[1] == 0:
        # ---- NOT STUDYING ----
        if j[2] != 0:
            param_wage_j = get_params_wage(j)
            debtnew, income = get_debt_income(x1_new,x2_new,x2,period,j,b,e,conterfactual,param_wage_j)
        else:
            debtnew, income = get_debt_income_home(x1_new,x2_new,x2,period,j,b,e,conterfactual)

        debt_position = map_debt_position(b,debtnew)

        u = get_power_utility(sigma_u,income)
        continuation = evolve_continuation(x1,x1_new,x2,x2_new,b,period,e,j,evt,debt_position)

        vjt = u[...,None] + continuation[...,None]    # (Q*nb,1)

        # apply penalty when b>0
        vjt[:,0] += debt_pen * mask
        return vjt[:,0]

    else:
        # ---- STUDYING ----
        c = get_utility(
            sigma_u, x1, x1_new, x2_new, b, b1, e, j, period,
            financial_parameters, z=z
        )
        continuation = beta*VT(x1,x1_new,x2,x2_new,b1,period,j,evt,0)  # shape (nb,1) or (nb,?)

        # The flow penalty is attached to the candidate next-period debt.
        # This charges the first penalty in the period in which borrowing is
        # chosen. Future explicit periods charge the same flow parameter
        # recursively; the period-T terminal value adds no further penalty.
        candidate_debt_mask = (b1 > 0).astype(np.float64)
        continuation[:,0] += debt_pen * candidate_debt_mask

        max_vjt = get_maximum(sigma_u,c,continuation,x1,b,j,x2,maxdebt)
        return max_vjt



def get_sigma(j,sigmas):
    
    """This function introduces the variance of the distribution
    depending on the choice"""
    
    if j[1] != 0 : # the individual is choosing education
    
        sigma = sigmas[0]
        
    else:  # the individual works. What matters is the field now. 
        
        if j[0] <4:
            sigma = sigmas[j[0]]
        else:
            sigma = sigmas[j[0]-2]
        
    return sigma


def get_quadrature_budget(deg, x1, x2, j, period):
    return bs.quadrature(
        budget_params, x1, period, degree=deg,
        education=int(j[1]), state=x2,
    )


def budget_mu_sigma_from_params(x1, x2, j, period):
    keywords = {"education": int(j[1]), "state": x2}
    mean = float(np.asarray(
        bs.conditional_mean(budget_params, x1, period, **keywords)
    ).reshape(-1)[0])
    return mean, bs.conditional_sigma(budget_params, period, **keywords)


def get_quadrature(deg,mu,sigma):
    
    [x,w] = scipy.special.roots_hermite(deg, mu=False)
    w = w*1/np.sqrt(np.pi)
    y = np.sqrt(2)*sigma*x+mu
    
    return y,w

def get_quadrature_wage(deg,mu,j):
    
    ''' This function computes the range and weight of the Gauss-Hermite 
    quadrature, for a normal distribution. 
    
    deg -> number of points in the range (degree).
    mu -> mean of the variable to integrate.
    sigma -> st. deviation of the variable to integrate. 
    
    '''
    
    if j[2] == 0 : # no labor supply decision
    
        y = np.array([0.0], dtype=np.float64)
        w = np.array([1.0], dtype=np.float64)
        
    else:
    
        sigma = get_sigma(j,sigmas)  # get the variance of this choice
        
        y,w = get_quadrature(deg,mu,sigma)
    
    return y,w


def get_expected_conditional(
    sigma_u, x1, x1_new, x2, x2_new, b, b1, j, period, deg, evt, conterfactual, maxdebt,
    financial_parameters, deg_budget=5
):
    nb = np.shape(b)[0]

    e_nodes, we = get_quadrature_wage(deg, mu, j)

    if j[1] == 0:
        w_vis = np.repeat(we, nb)
        v = get_conditional(
            sigma_u, x1, x1_new, x2, x2_new, b, b1, e_nodes, j, period,
            evt, conterfactual, maxdebt, financial_parameters, z=0.0
        ) * w_vis
        v = v.reshape((len(e_nodes), nb)).T
        return np.sum(v, axis=1)

    standard_nodes, wz = np.polynomial.hermite.hermgauss(deg_budget)
    standard_nodes = np.sqrt(2.0) * standard_nodes
    wz = wz / np.sqrt(np.pi)

    e_joint = np.tile(e_nodes, len(standard_nodes))
    z_standard_joint = np.repeat(standard_nodes, len(e_nodes))
    w_joint = np.kron(wz, we)

    # The fitted residual-shock mean may depend on resources available before
    # choosing next-period debt. Construct those resources for every wage
    # quadrature point and current-debt state, exactly where the Bellman budget
    # is formed.
    h0 = float(fin_help(x1_new, j, financial_parameters))
    wage_index = float(np.asarray(wage0(x1_new, x2)).reshape(-1)[0])
    real_wage = (
        np.exp(wage_index + e_joint) * (j[2] / 2.0) * 52.0 * 40.0
    )
    pre_choice_resources = (
        h0 + real_wage[:, None] - tuition(j)
        - (1.0 + r) * b[None, :]
    )
    z_joint = bs.realization(
        budget_params,
        x1,
        period,
        z_standard_joint[:, None],
        education=int(j[1]),
        state=x2,
        pre_choice_resources=pre_choice_resources,
    ).reshape(-1)

    w_vis = np.repeat(w_joint, nb)

    v = get_conditional(
        sigma_u, x1, x1_new, x2, x2_new, b, b1,
        e_joint, j, period, evt, conterfactual, maxdebt,
        financial_parameters, z=z_joint
    ) * w_vis

    v = v.reshape((len(w_joint), nb)).T
    return np.sum(v, axis=1)


# Not sure if this is actually saving time!
@njit()
def get_expected_numba(all_vjt):
    
    all_vjt = np.exp(all_vjt).sum(axis=1)
        
    all_vjt  = np.log(all_vjt) + gamma
    
    return all_vjt

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



def get_x_change(x2,period,fields=8,occupations=8):
    
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
            
            
            x_change[x2[9]] = 1
            
        # Check if last period was an occupation period
        
        elif (x2[7]!= (period-1)) & (x2[9]!=0):
            
            if x2[9] < 4:
            
                x_change[fields+x2[9]] = 1
                
            else:
                
                x_change[fields+x2[9]-2] = 1
        # Check if last period was a graduate school period
        
            # Nothing NOW here
        
        # Check if last period was home production
        
        elif (x2[7]!=period-1) & (x2[9]==0):
            
            x_change[-1] = 1        
    
    return x_change

def get_x_educ(x2,period,fields=8):
    
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
                x_afqt[int(x1[0,1]-1)] = 1
    
    if coltype == 3:
        
        if period > 5:
            if (x2[7] ==(period-1)) & (x2[9]==13):
                x_afqt = 1
            else:
                x_afqt = 0
        else:
            x_afqt = 0
            
       
    return x_afqt
    
    
def get_all_g(utility_parameters,x1,x1_new,x2,Jx,period):

    """This function computes all g() values for a given individual.
    It is important to notice that it """
    # First get the different parameters
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
    param_g_type  = utility_parameters[8]
    
    # first get gx1: ----------------------------------------------------------
    
    total_choices = get_total_choices()
    
    idx = np.where( (total_choices==Jx[:,None]).all(-1) )[1]  # This tells which index in tota_choices corresponds to each choice in Jx
    
    idx = np.concatenate((idx[...,np.newaxis],np.arange(0,np.shape(idx)[0]).reshape(np.shape(idx)[0],1)),axis=1)  # This just includes the index of Jx as a column in the array. 
    
    # Get only the parameters of the choices that are feasible on this state. 
    
    param_g_state = param_g[idx[:,0],:]
    
    g_x1 = x1_new@param_g_state.T
    #--------------------------------------------------------------------------
    
    # Now g work --------------------------------------------------------------
        
    g_work  = param_g_work[0,idx[:,0]]    
    
    #--------------------------------------------------------------------------
    
    # Now get get g_change: ---------------------------------------------------
    x_change = get_x_change(x2,period)
    param_g_last_state = param_g_last[idx[:,0],:]
    g_change = x_change@param_g_last_state.T
    
    #--------------------------------------------------------------------------
    
    # Now g educ preferences --------------------------------------------------
    
    x_educ = get_x_educ(x2,period)
    param_g_educ_state = param_g_educ[idx[:,0],:]
    g_educ = x_educ@param_g_educ_state.T
    
    #--------------------------------------------------------------------------
    
    # Now g for period ----- --------------------------------------------------
    
    g_period = param_g_period[period-1,idx[:,0]]
    
    #--------------------------------------------------------------------------
    
    # Now g for period work ---------------------------------------------------
    
    g_period_work = param_g_period_work[period-1,idx[:,0]]
    
    #--------------------------------------------------------------------------
    
    # Now g for first enrollment time in associate ----------------------------
    
    x_afqt = get_x_afqt_first(x1,x2,period,1)
    param_g_first_state2 = param_g_first_2[idx[:,0],0]
    g_first2 = x_afqt[0]*param_g_first_state2.T
    
    #--------------------------------------------------------------------------
    
    # Now g for first enrollment time in bachelor  ----------------------------
    
    x_afqt = get_x_afqt_first(x1,x2,period,2)
    param_g_first_state4 = param_g_first_4[idx[:,0],:]
    g_first4 = x_afqt@param_g_first_state4.T
    
    #--------------------------------------------------------------------------
    
    # Now g for first enrollment time in grad school --------------------------
    
    x_afqt = get_x_afqt_first(x1,x2,period,3)
    param_g_first_state_grad = param_g_first_grad[idx[:,0],:]
    g_firstgrad = x_afqt*param_g_first_state_grad.T
    
    #--------------------------------------------------------------------------
    
    # Now g for the experience effects  ---------------------------------------
    
    x_exp = get_x_exp(x1,x2)
    param_g_exp_state = param_g_exp[idx[:,0],:]
    g_exp = x_exp@param_g_exp_state.T
    
    #--------------------------------------------------------------------------
    
    # Now g for unobserved type effect  ---------------------------------------
    
    g_type = param_g_type[idx[:,0],0]
    
    #--------------------------------------------------------------------------
    
    g =g_x1  + g_work +g_change +g_educ  +g_period+ g_period_work + g_first2 + g_first4 + g_firstgrad + g_exp + g_type
    
    return g
    
@njit()
def get_x_exp(x1,x2):
    
    """
    This funciton returs the x_exp for the experience aftq effects.
    """
    
    x_exp = np.zeros(6*4)
    
    ability = x1[0,1] - 1
    
    exp = np.minimum(x2[2],5)
    
    x_exp[int(exp + ability*6)] = 1
    
    return x_exp

def get_all_choices(
    x1, x1_new, x2, x2_new, b, b1, period, evt, ccp_real, sigma_u,
    utility_parameters, models, solution_mode, conterfactual, maxdebt,
    financial_parameters
):
    
    # For each state x1 and x2 loop over all possible choices.
    
    Jx  = get_possible_choices(x2)
    # now loop over all the entrances in Jx
    # Create a matrix that will store the values
    
    all_vjt = np.zeros((np.shape(b)[0],np.shape(Jx)[0]))
    
    count = 0
    for j in Jx:
        
        # get the choice
        #tic = time.time()
        vjt = get_expected_conditional(
            sigma_u, x1, x1_new, x2, x2_new, b, b1, j, period, 5,
            evt, conterfactual, maxdebt, financial_parameters
        )
        #toc = time.time()
        
        #print("J Elapsed time: ",toc-tic)
        # now save it
        
        all_vjt[:,count] = vjt
        count+=1
        

    #np.savez_compressed(f"vjt/vjt_t{period}_{x1}_{x2}.npz",a=all_vjt)
    #np.save(f"vjt/vjt_t{period}_{x1}_{x2}.npy",all_vjt)   
    # once all are created, perform the expectation. 
    
    base  = -1
    
    if solution_mode == 0:
    
        if ccp_real == 0:
            
            #ccp = get_estimated_ccps(x1,x2,b,period,models)
            
            ccp = models[f"ccp_t{period}_{x1}_{x2}"]    
            
        else:
            g = get_all_g(utility_parameters,x1,x1_new,x2,Jx,period)
            
            all_vjt_temp = all_vjt + g
            
            log_ccp = (
                all_vjt_temp[:,base]
                - scipy.special.logsumexp(all_vjt_temp,axis=1)
            )
            ccp = np.exp(log_ccp)
            
        vjt_ccp = all_vjt[:,base]
        
        if ccp_real == 1:
            evt = vjt_ccp - log_ccp + gamma
        else:
            evt = vjt_ccp -np.log(ccp) + gamma
        
    elif solution_mode == 1:
        
        g = get_all_g(utility_parameters,x1,x1_new,x2,Jx,period)
        
        all_vjt = all_vjt + g
        
        evt = np.log(np.exp(all_vjt).sum(axis=1)) + gamma

        ccp = None
    
    #all_vjt = get_expected_numba(all_vjt)
    
    #np.savez_compressed(f"evt/evt_t{period}_{x1}_{x2}.npz",a=exp_all_vjt[...,np.newaxis])
    return all_vjt,evt[...,np.newaxis],ccp
        
    
        


def get_expected_continuation(x1,x1_new,x2,x2_new,b1,period):
    
    # This function used to do more things that now are done by others.
    # Maybe I could kil it .
    

    vt = VT(x1,x1_new,x2[...,None].T,x2_new,b1,period,1,evt=0)  # I am setting j = 1 since it does not matter
    #np.savez_compressed(f"evt/evt_t{period}_{x1}_{x2}.npz",a=vt)
    return vt


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

def get_x1_new(x1a):
    """This function expands the x1 variables into dummies"""
    types = 4
    parinc  = np.array([np.concatenate((np.zeros(x1a[0]-1),np.array([1]),np.zeros(types-x1a[0])),axis=0)],)
    ability = np.array([np.concatenate((np.zeros(x1a[1]-1),np.array([1]),np.zeros(types-x1a[1])),axis=0)],)
    x1_new = np.array([np.concatenate((parinc[0,1:],ability[0,1:],x1a[2:]))],)
    return np.append(1,x1_new) # add a constant


def get_x2_new(x2,fields=8):
    x2 = np.array(x2,dtype="int")
    """Generates the x2 variables of the wage equation"""
    # exp,fields (first 4y then twoyear), grad school
    majordummies = np.zeros((fields-1)+1)

    if (x2[8] !=12) & (x2[8]!=0):
        if x2[8] < 3:
            majordummies[x2[8]-1] = 1 # set major 
        else:
            majordummies[x2[8]-2] = 1 # set major 
        
    elif x2[8] == 12:  #  major is associate degree
        majordummies[-1] = 1
        
    x2_new = np.append(np.array(x2[0]),majordummies)

    return np.append(x2_new,np.array(x2[6])) # include whether the individual has grad school

def get_educ_level(x2):
    
    if (x2[4]==0) & (x2[5]==0): 
        
        if (x2[1]==0) & (x2[2]==0):
            
            educ = 0  #only highschool
            
        else: 
            
            educ = 1 # some college, never grad
    
    if (x2[4] == 1) & (x2[5] == 0):
        
        educ = 2 # associate degree
        
    if (x2[5] == 1) & (x2[6]==0):
        
        educ = 3 # bachelor degree
        
    if (x2[5]==1) & (x2[6]==1):
        
        educ = 4  #graduate degree
        
    return educ   

def terminal_from_interp(x1_row, x2, sigma_u, debt_grid):
    interp_dict = load_interp_dict_local()  # joblib.load once per process

    sex = int(x1_row[0,2])
    eth = int(x1_row[0,3])
    educ = get_educ_level(x2)
    major = int(x2[8])
    lastschool = int(x2[7])

    model = interp_dict[(sex, eth, lastschool, educ, major)]
    return model((float(sigma_u), debt_grid))   # shape (100,)

# ---------------------------------------------------------------------
_INTERP_DICT = None

def load_interp_dict_local(cache_path=None):
    global _INTERP_DICT
    if _INTERP_DICT is not None:
        return _INTERP_DICT

    if cache_path is None:
        cache_path = OUT("cache", "interp_dict.joblib")  # same place as your SMM code

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Missing interp cache: {cache_path}")

    _INTERP_DICT = joblib.load(cache_path)
    return _INTERP_DICT

def get_terminal_pandas(evt,x1,x2,sigma_u,conterfactual):
    
    """
    This function loads the terminal continuation value for different scenarios
    """    

    # get relevant data
    sex = x1[0,2]
    eth = x1[0,3]
    educ = get_educ_level(x2)
    major = x2[8]
    lastschool = x2[7]
    
    evt = evt[f"con_last{lastschool}_educ{educ}_major{major}"]
    
    return evt

def loop_rows(
    i, x1, x2, b, b1, period, evt, ccp_real, sigma_u,
    utility_parameters, models, solution_mode, conterfactual, maxdebt,
    financial_parameters
):
    inv = x1[i,:]
    inv = inv[..., None].T
    # Generate x1
    x1_new = get_x1_new(inv[0])
    # Generate dummies for majors in x2
    x2_new = get_x2_new(x2)
    x2 = np.array(x2,dtype="int")
    if period == T:
        if conterfactual == 1:
            vt = get_terminal_pandas(evt, inv, x2, sigma_u, conterfactual)[..., None]
        else:
            vt = terminal_from_interp(inv, x2, sigma_u, b)[..., None]  # b is debt_range (len=100)
        return vt
    elif period != 1:
        all_vjt, exp_vjt, ccp = get_all_choices(
            inv, x1_new, x2, x2_new, b, b1, period, evt, ccp_real,
            sigma_u, utility_parameters, models, solution_mode,
            conterfactual, maxdebt, financial_parameters
        )
        #get_expected_continuation(inv,x1_new,x2,x2_new,b1,period)
        return all_vjt, exp_vjt, ccp
    else:
        all_vjt, exp_vjt, ccp = get_all_choices(
            inv, x1_new, x2, x2_new, b, b1, period, evt, ccp_real,
            sigma_u, utility_parameters, models, solution_mode,
            conterfactual, maxdebt, financial_parameters
        )
        return all_vjt, exp_vjt, ccp



       
def get_x2(period):
    
    """This function returns the set of all possible states at a given period"""
    
    return np.load(STATES(f"states_t{period}.npy"))

def persist_outputs_for_period(
    period: int,
    x1i: np.ndarray,
    em_type: int,
    solution_mode: int,
    conterfactual: int,
    maxdebt: bool,
    save_evt: int,
    names_vjt: list,
    result_vjt: list,
    names_exp: list,
    result_exp: list,
    save_fn
):
    """
    Persist VJT/EVT artifacts for a single (period, x1i) result bundle.

    Parameters
    ----------
    save_fn : callable
        Function with signature save_fn(path:str, names:list[str], arrays:list[np.ndarray], compressed:bool)
        e.g. your existing `save_npz_here`.
    """
    # choose base paths depending on flags
    if period < T:
        if solution_mode == 0:
            # vjt_nog / evt_nog
            save_fn(f"vjt_nog/{period}/vjt_t{period}_{x1i}_em{em_type}.npz", names_vjt, result_vjt, compressed=True)
            if save_evt == 1:
                save_fn(f"evt_nog/{period}/evt_t{period}_{x1i}_em{em_type}.npz", names_exp, result_exp, compressed=True)

        elif solution_mode == 1:
            if conterfactual == 0:
                # vjt / evt
                save_fn(f"vjt/{period}/vjt_t{period}_{x1i}_em{em_type}.npz",  names_vjt, result_vjt, compressed=True)
                if save_evt == 1:
                    save_fn(f"evt/{period}/evt_t{period}_{x1i}_em{em_type}.npz", names_exp, result_exp, compressed=True)
            else:
                # vjt_conter / evt_conter
                suffix = f"_em{em_type}_maxdebt{maxdebt}.npz"
                save_fn(f"vjt_conter/{period}/vjt_t{period}_{x1i}{suffix}",    names_vjt, result_vjt, compressed=True)
                if save_evt == 1:
                    save_fn(f"evt_conter/{period}/evt_t{period}_{x1i}{suffix}", names_exp, result_exp, compressed=True)
    else:
        # Terminal period: only EVT bundles are written
        if save_evt == 1:
            if solution_mode == 0:
                save_fn(f"evt_nog/{period}/evt_t{period}_{x1i}_em{em_type}.npz", names_exp, result_exp, compressed=True)
            elif solution_mode == 1:
                if conterfactual == 0:
                    save_fn(f"evt/{period}/evt_t{period}_{x1i}_em{em_type}.npz", names_exp, result_exp, compressed=True)
                else:
                    suffix = f"_em{em_type}_maxdebt{maxdebt}.npz"
                    save_fn(f"evt_conter/{period}/evt_t{period}_{x1i}{suffix}",  names_exp, result_exp, compressed=True)

def loop_over_states(i, x1, x2_set, b, b1, period, ccp_real, sigma_u, 
                     utility_parameters,models,solution_mode,conterfactual,
                     evtnew,em_type,maxdebt,financial_parameters,save_evt = 1):
    
    """This function just facilitates debugging since a full x2 iteration
    can be done by just calling this function"""
    names_vjt=[]
    names_exp = []
    names_ccp = []
    results_ccp = []
    result_vjt = []
    result_exp = []
    x1i = x1[i,:]
    x1i = x1i[..., None].T
    # load the corresponding continuation values.With this I will only need to
    # load them once. 
    if period < T:
        evt = evtnew
    elif period == T:
        if conterfactual == 1:
            evt = np.load(f"{pathcontfinal}/continuation_conter_s{x1i[0,2]}_eth{x1i[0,3]}_sigma{sigma_u}.npz")
        else:
            # baseline: use interpolation; evt becomes a local cache dict
            evt = {}
        
    for x2 in x2_set:
        #print(i,x2)
        tic = time.time()
        if period < T:
            all_vjt, exp_vjt, ccp = loop_rows(
                i, x1, x2, b, b1, period, evt, ccp_real, sigma_u,
                utility_parameters, models, solution_mode, conterfactual,
                maxdebt, financial_parameters
            )
            names_exp.append(f"evt_t{period}_{x1i}_{x2.astype(int)}")
            names_vjt.append(f"vjt_t{period}_{x1i}_{x2.astype(int)}")
            names_ccp.append(f"ccp_t{period}_{x1i}_{x2.astype(int)}")
            results_ccp.append(ccp)
            result_vjt.append(all_vjt)
            result_exp.append(exp_vjt)
        else:
            models = 0
            exp_vjt = loop_rows(
                i, x1, x2, b, b1, period, evt, ccp_real, sigma_u,
                utility_parameters, models, solution_mode, conterfactual,
                maxdebt, financial_parameters
            )
            names_exp.append(f"evt_t{period}_{x1i}_{x2.astype(int)}")
            result_exp.append(exp_vjt)
            
        toc = time.time()
        #if i == 0:
        #print("Period",period,"Time elapsed : ", toc-tic)
    
    # Store output
    persist_outputs_for_period(
        period=period,
        x1i=x1i,
        em_type=em_type,
        solution_mode=solution_mode,
        conterfactual=conterfactual,
        maxdebt=maxdebt,
        save_evt=save_evt,
        names_vjt=names_vjt,
        result_vjt=result_vjt,
        names_exp=names_exp,
        result_exp=result_exp,
        save_fn=save_npz_here,  # your existing writer
    )
    # Format evt for next period
    evtnext = dict(zip(names_exp,result_exp))
    
    # Store CCPs if needed
    
    if ccp_real == 1: 
        save_npz_here(
            f"ccp/{period}/ccp_t{period}_{x1i}_em{em_type}.npz",
            names_ccp,
            results_ccp,
            compressed=True,
        )
    
    return evtnext
        

def get_all_evt(i,x1,b,b1,ccp_real,utility_parameters,models,
                solution_mode,conterfactual,em_type,maxdebt):

    """Solve and save one invariant state's Bellman problem for one joint type.

    ``em_type`` is the permanent joint type ID in ``1, ..., 16``. Its schooling
    component is already encoded in ``utility_parameters`` by ``build_param_g``.
    Grant and transfer components are mapped once here and passed downward as
    a preselected tuple of numeric arrays.
    """
    time.sleep(0)

    financial_parameters = get_type_financial_parameters(em_type)
        
    evtnext = 0 
    
    models = 0
    
    sigma_u = float(bs.risk_aversion(budget_params, x1[i, :]))
    task_started = time.perf_counter()
    total_states = len(x1)
    total_tasks = N_TYPES * total_states
    task_number = (em_type - 1) * total_states + i + 1
    if ccp_real == 1:
        ccp_mode = "updated"
    elif ccp_real == 0:
        ccp_mode = "initial"
    else:
        ccp_mode = "supplied"
    for period in range(T,0,-1):
        period_started = time.perf_counter()
        if period < T:
            models = np.load(f"{pathout}/ccp/{period}/ccp_t{period}_[{x1[i,:]}]_em{em_type}.npz")
        
        # Get set of states
        
        x2_set = get_x2(period)
        
        # get new evt
        
        evt = evtnext
        
        # loop over all states of x2
        
        evtnext = loop_over_states(i, x1, x2_set, b, b1, period, ccp_real,
                                   sigma_u,utility_parameters,models,
                                   solution_mode,conterfactual,evt,
                                   em_type,maxdebt,financial_parameters)
        print(
            f"[Bellman | pid={os.getpid()} | task={task_number}/{total_tasks} "
            f"| type={em_type}/{N_TYPES}:{TYPE_NAMES[em_type - 1]} "
            f"| state={i + 1}/{total_states} | CCP={ccp_mode}] "
            f"completed period {period}/{T} | period={time.perf_counter() - period_started:.2f}s "
            f"| task={time.perf_counter() - task_started:.2f}s",
            flush=True,
        )


def _trace_expected_conditional_debug(
    sigma_u, x1, x1_new, x2, x2_new, b, b1, j, period, evt,
    conterfactual, maxdebt, financial_parameters, deg=5, deg_budget=5
):
    """Replay one choice without aggregation and retain quadrature identities."""
    nb = np.shape(b)[0]
    e_nodes, we = get_quadrature_wage(deg, mu, j)

    if j[1] == 0:
        if j[2] != 0:
            param_wage_j = get_params_wage(j)
            debtnew, income = get_debt_income(
                x1_new, x2_new, x2, period, j, b, e_nodes,
                conterfactual, param_wage_j,
            )
        else:
            debtnew, income = get_debt_income_home(
                x1_new, x2_new, x2, period, j, b, e_nodes, conterfactual
            )
        debt_position = map_debt_position(b, debtnew)
        flow_utility = get_power_utility(sigma_u, income)
        continuation = evolve_continuation(
            x1, x1_new, x2, x2_new, b, period, e_nodes, j, evt, debt_position
        )
        debt_penalty = float(x1_new @ debt_pen_vec) * np.tile(
            (b > 0).astype(np.float64), len(e_nodes)
        )
        raw_flat = flow_utility + continuation + debt_penalty
        raw = raw_flat.reshape((len(e_nodes), nb)).T
        weighted = raw * we[None, :]
        return {
            "income_by_node_debt": income.reshape((len(e_nodes), nb)),
            "next_debt_by_node_debt": debtnew.reshape((len(e_nodes), nb)),
            "next_debt_index_by_node_debt": debt_position.reshape((len(e_nodes), nb)),
            "flow_utility_by_node_debt": flow_utility.reshape((len(e_nodes), nb)),
            "continuation_by_node_debt": continuation.reshape((len(e_nodes), nb)),
            "debt_penalty_by_node_debt": debt_penalty.reshape((len(e_nodes), nb)),
            "raw_vjt_by_debt_node": raw,
            "weighted_vjt_by_debt_node": weighted,
            "integrated_vjt": np.sum(weighted, axis=1),
            "wage_nodes": e_nodes,
            "wage_weights": we,
            "budget_standard_nodes": np.array([0.0]),
            "budget_weights": np.array([1.0]),
            "actual_budget_shock_by_node_debt": np.zeros((len(e_nodes), nb)),
            "pre_choice_resources_by_node_debt": np.zeros((len(e_nodes), nb)),
            "joint_wage_index": np.arange(len(e_nodes), dtype=int),
            "joint_budget_index": np.zeros(len(e_nodes), dtype=int),
        }

    standard_nodes, wz = np.polynomial.hermite.hermgauss(deg_budget)
    standard_nodes = np.sqrt(2.0) * standard_nodes
    wz = wz / np.sqrt(np.pi)
    e_joint = np.tile(e_nodes, len(standard_nodes))
    z_standard_joint = np.repeat(standard_nodes, len(e_nodes))
    w_joint = np.kron(wz, we)

    h0 = float(fin_help(x1_new, j, financial_parameters))
    wage_index = float(np.asarray(wage0(x1_new, x2)).reshape(-1)[0])
    real_wage = np.exp(wage_index + e_joint) * (j[2] / 2.0) * 52.0 * 40.0
    pre_choice_resources = (
        h0 + real_wage[:, None] - tuition(j) - (1.0 + r) * b[None, :]
    )
    z_joint = bs.realization(
        budget_params, x1, period, z_standard_joint[:, None],
        education=int(j[1]), state=x2,
        pre_choice_resources=pre_choice_resources,
    ).reshape(-1)
    conditional_utility = get_utility(
        sigma_u, x1, x1_new, x2_new, b, b1, e_joint, j, period,
        financial_parameters, z=z_joint,
    )
    continuation = beta * VT(x1, x1_new, x2, x2_new, b1, period, j, evt, 0)
    continuation = np.asarray(continuation).copy()
    candidate_debt_penalty = float(x1_new @ debt_pen_vec) * (
        b1 > 0
    ).astype(np.float64)
    continuation[:, 0] += candidate_debt_penalty
    raw_flat = get_maximum(
        sigma_u, conditional_utility, continuation, x1, b, j, x2, maxdebt
    )
    raw = raw_flat.reshape((len(w_joint), nb)).T
    weighted = raw * w_joint[None, :]
    return {
        "raw_vjt_by_debt_node": raw,
        "weighted_vjt_by_debt_node": weighted,
        "integrated_vjt": np.sum(weighted, axis=1),
        "wage_nodes": e_nodes,
        "wage_weights": we,
        "budget_standard_nodes": standard_nodes,
        "budget_weights": wz,
        "actual_budget_shock_by_node_debt": z_joint.reshape(len(w_joint), nb),
        "pre_choice_resources_by_node_debt": pre_choice_resources,
        "conditional_utility": conditional_utility,
        "continuation_by_candidate_debt": continuation,
        "candidate_debt_penalty": candidate_debt_penalty,
        "joint_wage_index": np.tile(
            np.arange(len(e_nodes), dtype=int), len(standard_nodes)
        ),
        "joint_budget_index": np.repeat(
            np.arange(len(standard_nodes), dtype=int), len(e_nodes)
        ),
    }


def get_all_choices_debug(
    x1, x1_new, x2, x2_new, b, b1, period, evt, ccp_real, sigma_u,
    utility_parameters, models, solution_mode, conterfactual, maxdebt,
    financial_parameters, recorder, state_metadata
):
    """Checked Bellman choice calculation; production entry point is untouched."""
    from model_debug_checks import ccp_problems, finite_problems, vjt_problems

    Jx = get_possible_choices(x2)
    all_vjt = np.zeros((np.shape(b)[0], np.shape(Jx)[0]))
    for choice_index, j in enumerate(Jx):
        metadata = {
            **state_metadata,
            "choice_index": int(choice_index),
            "choice": np.asarray(j, dtype=int),
        }
        vjt = get_expected_conditional(
            sigma_u, x1, x1_new, x2, x2_new, b, b1, j, period, 5,
            evt, conterfactual, maxdebt, financial_parameters
        )
        problems = vjt_problems(vjt)
        if problems:
            trace = _trace_expected_conditional_debug(
                sigma_u, x1, x1_new, x2, x2_new, b, b1, j, period,
                evt, conterfactual, maxdebt, financial_parameters
            )
            bad = np.argwhere(
                np.isnan(trace["raw_vjt_by_debt_node"])
                | np.isposinf(trace["raw_vjt_by_debt_node"])
            )
            if bad.size:
                debt_index, joint_index = (int(value) for value in bad[0])
                metadata.update(
                    debt_index=debt_index,
                    debt=float(np.asarray(b)[debt_index]),
                    joint_node_index=joint_index,
                    wage_node_index=int(trace["joint_wage_index"][joint_index]),
                    budget_node_index=int(trace["joint_budget_index"][joint_index]),
                    wage_shock=float(
                        trace["wage_nodes"][trace["joint_wage_index"][joint_index]]
                    ),
                    budget_standard_shock=float(
                        trace["budget_standard_nodes"][
                            trace["joint_budget_index"][joint_index]
                        ]
                    ),
                    budget_actual_shock=float(
                        trace["actual_budget_shock_by_node_debt"][
                            joint_index, debt_index
                        ]
                    ),
                    pre_choice_resources=float(
                        trace["pre_choice_resources_by_node_debt"][
                            joint_index, debt_index
                        ]
                    ),
                )
            replay_matches = np.allclose(
                trace["integrated_vjt"], vjt, rtol=1e-12, atol=1e-12,
                equal_nan=True,
            )
            metadata["quadrature_replay_matches"] = bool(replay_matches)
            for intermediate_name in (
                "income_by_node_debt",
                "next_debt_by_node_debt",
                "flow_utility_by_node_debt",
                "conditional_utility",
                "continuation_by_node_debt",
                "continuation_by_candidate_debt",
                "raw_vjt_by_debt_node",
                "weighted_vjt_by_debt_node",
                "integrated_vjt",
            ):
                if intermediate_name not in trace:
                    continue
                intermediate = np.asarray(trace[intermediate_name])
                invalid = np.isnan(intermediate) | np.isposinf(intermediate)
                coordinates = np.argwhere(invalid)
                if coordinates.size:
                    metadata["first_nonfinite_stage"] = intermediate_name
                    metadata["first_nonfinite_stage_index"] = tuple(
                        int(value) for value in coordinates[0]
                    )
                    break
            recorder.check(
                "choice_vjt",
                vjt,
                problems,
                metadata,
                arrays={
                    "x1": x1,
                    "x1_new": x1_new,
                    "x2": x2,
                    "choice": j,
                    "debt_grid": b,
                    "next_debt_grid": b1,
                    "production_integrated_vjt": vjt,
                    **trace,
                },
            )
        all_vjt[:, choice_index] = vjt

    recorder.check(
        "all_choice_vjt", all_vjt, vjt_problems(all_vjt), state_metadata,
        arrays={"debt_grid": b, "choices": Jx, "all_choice_vjt": all_vjt},
    )
    home = np.flatnonzero(np.all(Jx == 0, axis=1))
    if home.size != 1:
        recorder.record(
            "home_choice_count",
            {**state_metadata, "home_choice_count": int(home.size)},
            arrays={"choices": Jx},
        )
    base = -1
    if home.size == 1 and int(home[0]) != base % len(Jx):
        recorder.record(
            "home_choice_not_last",
            {**state_metadata, "home_choice_index": int(home[0])},
            arrays={"choices": Jx},
        )

    if solution_mode == 0:
        if ccp_real == 0:
            key = f"ccp_t{period}_{x1}_{x2}"
            if key not in models:
                recorder.record("initial_ccp_key_missing", {**state_metadata, "key": key})
                ccp = np.full(np.shape(b), np.nan)
            else:
                ccp = np.asarray(models[key])
        else:
            g = get_all_g(utility_parameters, x1, x1_new, x2, Jx, period)
            recorder.check("g", g, finite_problems(g, "g"), state_metadata)
            all_vjt_temp = all_vjt + g
            log_ccp = (
                all_vjt_temp[:, base]
                - scipy.special.logsumexp(all_vjt_temp, axis=1)
            )
            recorder.check(
                "home_log_ccp",
                log_ccp,
                finite_problems(log_ccp, "home_log_ccp"),
                state_metadata,
                arrays={
                    "debt_grid": b,
                    "choices": Jx,
                    "choice_vjt": all_vjt,
                    "g": g,
                    "total_choice_utility": all_vjt_temp,
                    "home_log_ccp": log_ccp,
                },
            )
            recorder.observe("updated_home_log_ccp", log_ccp)
            ccp = np.exp(log_ccp)
        recorder.observe(
            "initial_home_ccp_consumed" if ccp_real == 0 else "updated_home_ccp",
            ccp,
        )
        recorder.check(
            "home_ccp", ccp, ccp_problems(ccp), state_metadata,
            arrays={
                "debt_grid": b,
                "choices": Jx,
                "choice_vjt": all_vjt,
                "home_log_ccp": (
                    np.array([]) if ccp_real == 0 else log_ccp
                ),
                "ccp": ccp,
            },
        )
        vjt_ccp = all_vjt[:, base]
        recorder.check(
            "home_vjt", vjt_ccp, finite_problems(vjt_ccp, "home_vjt"),
            {**state_metadata, "choice_index": int(base % len(Jx)), "choice": Jx[base]},
        )
        if ccp_real == 1:
            expected_vjt = vjt_ccp - log_ccp + gamma
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                expected_vjt = vjt_ccp - np.log(ccp) + gamma
        recorder.observe("outgoing_evt", expected_vjt)
    elif solution_mode == 1:
        g = get_all_g(utility_parameters, x1, x1_new, x2, Jx, period)
        recorder.check("g", g, finite_problems(g, "g"), state_metadata)
        all_vjt = all_vjt + g
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            expected_vjt = np.log(np.exp(all_vjt).sum(axis=1)) + gamma
        ccp = None
    else:
        raise ValueError(f"Unknown solution_mode={solution_mode}")

    recorder.check(
        "outgoing_evt",
        expected_vjt,
        finite_problems(expected_vjt, "outgoing_evt"),
        state_metadata,
        arrays={
            "debt_grid": b,
            "choices": Jx,
            "choice_vjt": all_vjt,
            "home_ccp": np.array([]) if ccp is None else ccp,
            "outgoing_evt": expected_vjt,
        },
    )
    return all_vjt, expected_vjt[..., None], ccp


def loop_over_states_debug(
    i, x1, x2_set, b, b1, period, ccp_real, sigma_u,
    utility_parameters, models, solution_mode, conterfactual, evtnew,
    em_type, maxdebt, financial_parameters, recorder, save_evt=1
):
    """Debug-only counterpart of :func:`loop_over_states`."""
    from model_debug_checks import finite_problems

    names_vjt, names_exp, names_ccp = [], [], []
    result_vjt, result_exp, results_ccp = [], [], []
    x1i = x1[i, :][None, :]
    if period < T:
        evt = evtnew
    elif conterfactual == 1:
        evt = np.load(
            f"{pathcontfinal}/continuation_conter_s{x1i[0,2]}_eth{x1i[0,3]}_sigma{sigma_u}.npz"
        )
    else:
        evt = {}

    for x2_index, x2_value in enumerate(x2_set):
        x2 = np.asarray(x2_value, dtype=int)
        metadata = {
            "period": int(period),
            "x1": x1i[0],
            "x2_index": int(x2_index),
            "x2": x2,
        }
        name_exp = f"evt_t{period}_{x1i}_{x2}"
        if period < T:
            x1_new = get_x1_new(x1i[0])
            x2_new = get_x2_new(x2)
            all_vjt, exp_vjt, ccp = get_all_choices_debug(
                x1i, x1_new, x2, x2_new, b, b1, period, evt, ccp_real,
                sigma_u, utility_parameters, models, solution_mode,
                conterfactual, maxdebt, financial_parameters, recorder, metadata
            )
            names_vjt.append(f"vjt_t{period}_{x1i}_{x2}")
            names_ccp.append(f"ccp_t{period}_{x1i}_{x2}")
            result_vjt.append(all_vjt)
            results_ccp.append(ccp)
        else:
            if conterfactual == 1:
                exp_vjt = get_terminal_pandas(evt, x1i, x2, sigma_u, conterfactual)[..., None]
            else:
                exp_vjt = terminal_from_interp(x1i, x2, sigma_u, b)[..., None]
            recorder.check(
                "terminal_evt", exp_vjt,
                finite_problems(exp_vjt, "terminal_evt"), metadata,
                arrays={"debt_grid": b, "terminal_evt": exp_vjt},
            )
        names_exp.append(name_exp)
        result_exp.append(exp_vjt)

    persist_outputs_for_period(
        period, x1i, em_type, solution_mode, conterfactual, maxdebt,
        save_evt, names_vjt, result_vjt, names_exp, result_exp, save_npz_here
    )
    if ccp_real == 1 and period < T:
        relative_ccp_path = f"ccp/{period}/ccp_t{period}_{x1i}_em{em_type}.npz"
        save_npz_here(
            relative_ccp_path,
            names_ccp, results_ccp, compressed=True,
        )
        if recorder.config.verify_saved:
            saved_ccp_path = os.path.join(pathout, *relative_ccp_path.split("/"))
            with np.load(saved_ccp_path, allow_pickle=False) as saved_ccps:
                expected_keys = set(names_ccp)
                saved_keys = set(saved_ccps.files)
                if saved_keys != expected_keys:
                    recorder.record(
                        "saved_updated_ccp_keys_mismatch",
                        {
                            "period": int(period),
                            "path": saved_ccp_path,
                            "missing_keys": sorted(expected_keys - saved_keys),
                            "extra_keys": sorted(saved_keys - expected_keys),
                        },
                    )
                for key, expected in zip(names_ccp, results_ccp):
                    if key in saved_ccps and not np.array_equal(
                        saved_ccps[key], expected
                    ):
                        recorder.record(
                            "saved_updated_ccp_value_mismatch",
                            {"period": int(period), "path": saved_ccp_path, "key": key},
                            arrays={"expected": expected, "saved": saved_ccps[key]},
                        )
    return dict(zip(names_exp, result_exp))


def get_all_evt_debug(
    i, x1, b, b1, ccp_real, utility_parameters, models,
    solution_mode, conterfactual, em_type, maxdebt, debug_config
):
    """Solve one full Bellman task with exact-state and draw diagnostics."""
    from model_debug_checks import DebugRecorder, finite_problems

    recorder = DebugRecorder(debug_config, "bellman", em_type, i)
    financial_parameters = get_type_financial_parameters(em_type)
    evtnext = 0
    sigma_u = float(bs.risk_aversion(budget_params, x1[i, :]))
    recorder.check(
        "risk_aversion", np.asarray(sigma_u),
        finite_problems(np.asarray(sigma_u), "risk_aversion"),
        {"x1": np.asarray(x1[i, :], dtype=int)},
    )
    total_states = len(x1)
    task_started = time.perf_counter()
    for period in range(T, 0, -1):
        period_started = time.perf_counter()
        archive = None
        if period < T:
            archive_path = f"{pathout}/ccp/{period}/ccp_t{period}_[{x1[i,:]}]_em{em_type}.npz"
            try:
                archive = np.load(archive_path, allow_pickle=False)
                models_period = archive
            except Exception as error:
                recorder.record(
                    "initial_ccp_bundle_load_failed",
                    {"period": int(period), "path": archive_path, "exception": repr(error)},
                )
                models_period = {}
        else:
            models_period = 0
        try:
            evtnext = loop_over_states_debug(
                i, x1, get_x2(period), b, b1, period, ccp_real, sigma_u,
                utility_parameters, models_period, solution_mode, conterfactual,
                evtnext, em_type, maxdebt, financial_parameters, recorder,
            )
        finally:
            if archive is not None:
                archive.close()
        print(
            f"[Bellman DEBUG | pid={os.getpid()} | type={em_type}/{N_TYPES}:"
            f"{TYPE_NAMES[em_type - 1]} | state={i + 1}/{total_states}] "
            f"period {period}/{T} checked | period={time.perf_counter() - period_started:.2f}s "
            f"| task={time.perf_counter() - task_started:.2f}s",
            flush=True,
        )
    return recorder.finalize()
        
        
        
def simulate_all_states(periods):
    """This function simulates the set of all possible states
    at a specific period by simulating all choices over all states at t-1"""
    
    for t in range(1,periods+1,1):
        print("current period is", t)
        
        variant_states = []
        
        # load previous period possible states
        
        if t > 1: 
        
            previous_states = np.load(STATES(f"states_t{t-1}.npy"))               
        
        else:
            
            # generate the initial variant state as 0s. 
            
            previous_states = np.zeros((1,10))
            
            # save it
            
            np.save(STATES(f"states_t{t}.npy"), previous_states)
            
            continue
        
        # iterate over all possible states here
        
        for i in range(np.shape(previous_states)[0]):
            
            x2 = previous_states[i,:]
            
            # now get the set of choices that can be done at this state
            
            possible_choices = get_possible_choices(x2)
            
            # Iterate over those choices and move the state
            
            for j in possible_choices:
                    
                x2_new = move_state(x2, j, t-1)
                
                variant_states.append(x2_new)
                
                # check if the choice can induce a graduation stage:
                    
                if (j[1] == 1) & (x2_new[1] >= 2 ) & (x2_new[4] == 0):  # Notice you gan graduate with 1 year (in the data...)
                # this if checks if the chioce can induce graduation at two year schools
                        
                    # then include graduation
                        
                    x2_new_graduation = np.copy(x2_new)
                        
                    # change graduation status
                    x2_new_graduation[4] = 1
                    
                    # change experience
                    x2_new_graduation[1] = 99
                        
                    #include major at graduation
                    x2_new_graduation[8] = j[0]
                        
                    variant_states.append(x2_new_graduation)
                        
                elif (j[1] ==2) & (x2_new[2] >= 4 ) & (x2_new[5] == 0):
                # this if should check if the chioce can induce graduation in four-year schools
                        
                    # then include graduation
                        
                    x2_new_graduation = np.copy(x2_new)
                        
                    # change graduation status
                    x2_new_graduation[5] = 1
                        
                    #include major at graduation
                    x2_new_graduation[8] = j[0]
                    
                    # modify experience
                    x2_new_graduation[1] = 99
                    x2_new_graduation[2] = 99
                        
                    variant_states.append(x2_new_graduation)
                    
                elif (j[1] == 3):  # graduation of grad school
                    
                    # then include graduation
                        
                    x2_new_graduation = np.copy(x2_new)
                        
                    # change graduation status
                    x2_new_graduation[6] = 1
                    
                    # modify experience
                    x2_new_graduation[3] = 99
                        
                    variant_states.append(x2_new_graduation)
                    
                    
                      
        # now store it to the computer
        np.save(STATES(f"states_t{t}.npy"), np.unique(np.array(variant_states), axis=0))
    

    

def load_param_g(em_type,real=1):
    """Load flow-utility parameters for one permanent joint type ID."""
    if real == 1:
        param_utility = np.load(f"{path_estimates}/param_g.npy")
    else:
        fields= 8
        occupations = 8
        n_param_g_x1 = 9*(1+fields+occupations+1)
        n_param_g_work = 2 + 2*fields+occupations+2
        n_param_g_change = 1+ fields  + occupations
        n_param_g_educ = get_amount_educ()
        n_param_g_period = np.maximum((1+fields+occupations)*(T-1-1),0) + np.maximum((T-5-1),0)
        n_param_g_period_work = np.maximum((T-1-1)*3,0) + np.maximum((T-5-1)*2,0)
        n_param_g_first = 1 + 4 + 1
        n_param_exp_ability = (fields-1)*6*4 #foreach field, 6 experience cells and 4 ability levels 
        n_param_g_type = 2
        total_n = n_param_g_x1 + n_param_g_change + n_param_g_work + n_param_g_educ + n_param_g_period + n_param_g_period_work+ n_param_g_first + n_param_exp_ability + n_param_g_type
        param_utility = np.zeros((total_n,))
        param_utility = np.linspace(1,total_n,total_n)
        
    utility_parameters = build_param_g(em_type,param_utility)
    
    return utility_parameters


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

def build_param_g(em_type,param_utility):
    
    """Construct flow-utility arrays for one permanent joint type.

    The joint type's school component determines whether the two estimated
    schooling-preference shifts are active. Grant and transfer components do
    not enter flow utility; they enter the Bellman budget through ``fin_help``.
    This function is setup code and runs once per parameter vector and type.

    The returned matrices are:
    
    param_g_x1      ---   associated with preferences of individulas
    param_g_work    ---   full vs part time preference dislike of work
    param_g_last    ---   last choice effect on current choice
    param_g_educ    ---   effect of education level on occupations
    param_g_period  ---   period effect on choices
    param_g_first   ---   cost of first time enrollment 
    
    """

    school_type, _, _, _ = type_components(em_type)
    
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
    past = amount_x1+amount_work+amount_last+amount_educ+amount_period+amount_period_work
    param_first = param_utility[past:past+amount_first]
    param_first = build_param_first(param_first,fields,occupation,size)
    
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
    if school_type == 0:  # Low-schooling component is the utility base type.
        param_type[:,:] = 0

    utility_parameters = [param_g_x1,param_g_work,param_g_last,param_educ,param_period,param_period_work,param_first,param_exp,param_type]
    
    return utility_parameters
    
    


def get_data(period,real_data=1):
    
    """This function loads the data of each corresponding period"""
    
    global debt_range
    
    if real_data == 0 :
    
        pass
    
    else:
    
        state = np.load(f"{path}/real_data/state_t{period}.npy")
        
        choices = np.load(f"{path}/real_data/choice_t{period}.npy")
            
        x1 = np.load(f"{path}/real_data/invariant_state_t{period}.npy")[:,1:]  # do not take PUBID

        debt = np.load(f"{path}/real_data/financial_t{period}.npy")[:,0]  # only loans
        
        # map debt to the closer point on the grid
        
        value = np.zeros(np.shape(state)[0])
    
        for i in range(np.shape(state)[0]):
            
            diff = (debt_range - debt[i])**2
            
            value[i] = np.argmin(diff)
            
        debt = value
        
    
    return x1,state,debt,choices

def get_feasible():
    
    """This is a temporary function to solve data cleaning problems. 
    I will only get individuals that belong to a feasible possible 
    state. """
    
    for period in range(1,T):
        
        x1,state,debt,choices = get_data(period)
        
        feasible = np.load(STATES(period) / f"states_t{period}.npy")
                
        idx = np.sort(np.where( (state==feasible[:,None]).all(-1) )[1]) # index of feasible elements
        
        x1_feasible  = x1[idx,:].astype("int")
        state_feasible = state[idx,:].astype("int")
        debt_feasible = debt[idx].astype("int")
        choices_feasible = choices[idx,:].astype("int")
        
        np.save(f"{path}/state_feasible_t{period}.npy", state_feasible)
        np.save(f"{path}/invariant_state_feasible_t{period}.npy", x1_feasible)
        np.save(f"{path}/debt_feasible_t{period}.npy", debt_feasible)
        np.save(f"{path}/choice_feasible_t{period}.npy", choices_feasible)      
        print("Percentage of destroied data", np.shape(x1_feasible)[0]/np.shape(x1)[0])


def get_data_feasible(period):
    
    state = np.load(f"{path}/state_feasible_t{period}.npy")
    choices = np.load(f"{path}/choice_feasible_t{period}.npy")
    x1 = np.load(f"{path}/invariant_state_feasible_t{period}.npy")
    debt = np.load(f"{path}/debt_feasible_t{period}.npy")
    return x1,state,debt,choices

def estimate_logit_ccps():
    
    # This function estimates all the logit models that will be used
    # to predict the ccps.
    
    models = []
    
    total_choices = get_total_choices()
    
    for period in range(1,T):
    
        print(f"Estimateing model {period} out of {T}")
        x1,state,debt,choices  = get_data_feasible(period)
        
        if period == 1:
            
            # Notice that at period  1 all agents have same x2. Therefore, the
            # ccps will only depend on x1. 
            
            names = ["parinc","afqt","gender","ethnicity"]  # Not sure if this is the right order!
            
            x = pd.DataFrame(np.copy(x1),columns= names)
            
        else:
            
            xfull = np.concatenate((x1,state,debt[...,None]),axis=1)
            
            names = ["parinc","afqt","gender","ethnicity","exp","two","four","grad","two_grad","four_grad","grad_grad","last_educ","field","last_choice","debt"]
            
            x = pd.DataFrame(np.copy(xfull),columns = names)
            
            if period > 2 :
            
                # Generate dummies for different fields
                
                x = x.reset_index().merge(pd.get_dummies(x["field"], prefix="field", prefix_sep='_').reset_index(),
                                            how="left",
                                            on ='index',
                                            ).set_index('index')
                
                x.drop(["field"],axis=1,inplace=True)
                
                # Generate dummies for different previous chioces
                
                x = x.reset_index().merge(pd.get_dummies(x["last_choice"], prefix="last_choice", prefix_sep='_').reset_index(),
                                            how="left",
                                            on ='index',
                                            ).set_index('index')
                
                x.drop(["last_choice"],axis=1,inplace=True)
                
                # Generate dummies for different last educ states
                
                x = x.reset_index().merge(pd.get_dummies(x["last_educ"], prefix="last_educ", prefix_sep='_').reset_index(),
                                            how="left",
                                            on ='index',
                                            ).set_index('index')
                
                x.drop(["last_educ"],axis=1,inplace=True)
            
            
        # Now get dummies for different quartiles:
          
        x = x.reset_index().merge(pd.get_dummies(x["afqt"], prefix="afqt", prefix_sep='_').reset_index(),
                                    how="left",
                                    on ='index',
                                    ).set_index('index')
        x = x.reset_index().merge(pd.get_dummies(x["parinc"], prefix="parinc", prefix_sep='_').reset_index(),
                                    how="left",
                                    on ='index',
                                    ).set_index('index')
        
        x.drop(["afqt","parinc"],axis=1,inplace=True)  # drop the previous ones
    
        # map choices to numbers
        choice_idx = np.where( (total_choices==choices[:,None]).all(-1) )[1]  # This tells which index in tota_choices corresponds to each choice in Jx
        
        # Estimate the model
        
        model = LogisticRegression(solver='liblinear', random_state=0).fit(x, choice_idx)
        
        models.append(model)
    
    return models


def get_estimated_ccps(x1,x2,b,period,models):
    
    """"This function estimates a logistic regression always with the same data (so I can try to 
    skip the re-estimation at some point) and predicts the ccps for a given x1,x2,b"""
    
    model = models[period-1]
    x1 = x1[0]

    if period == 1:
        
        # Notice that at period  1 all agents have same x2. Therefore, the
        # ccps will only depend on x1. 
        
        names = ["parinc","afqt","gender","ethnicity"]  # Not sure if this is the right order!
        
        x_new = pd.DataFrame(np.repeat(x1[...,None].T,np.shape(b)[0],axis=0),columns=names)
        # Notice that here I am creating 1000 times the same and then compute
        # the ccp on the matrix of equal values. If things become too computatioanlly
        # expensive, I can just compute the ccp of one example and then repeat it 1000 times. 
        
        
    else:
        
        names = ["parinc","afqt","gender","ethnicity","exp","two","four","grad","two_grad","four_grad","grad_grad","last_educ","field","last_choice","debt"]
        
        x_new = np.concatenate((x1[...,None].T,x2[...,None].T),axis=1)
        x_new = np.concatenate((np.repeat(x_new,np.shape(b)[0],axis=0),b[...,None]),axis=1)
        x_new = pd.DataFrame(x_new,columns=names)
        
        # Notice that if period == 2 there is one only possible field, which
        # is 0. For this reason I will not include it. 
        
        if period > 2:
        
            for i in range(0,9,1):
                x_new[f"field_{i}"] = False
                x_new.loc[x_new["field"] == i, f"field_{i}"] = True
                
            for i in range(0,9,1):
                x_new[f"last_choice_{i}"] = False
                x_new.loc[x_new["last_choice"] == i, f"last_choice_{i}"] = True
                
            for i in range(0,period,1):
                x_new[f"last_educ_{i}"] = False
                x_new.loc[x_new["last_educ"] == i, f"last_educ_{i}"] = True
                
            x_new.drop(["field","last_choice","last_educ"],axis=1,inplace=True)  # drop the previous ones
            
            
        
    
    # Prepare the data for the model

    for i in range(1,5,1):
        x_new[f"afqt_{i}"] = False
        x_new.loc[x_new["afqt"] == i, f"afqt_{i}"] = True
        
    for i in range(1,5,1):
        x_new[f"parinc_{i}"] = False
        x_new.loc[x_new["parinc"] == i, f"parinc_{i}"] = True
        
    x_new.drop(["afqt","parinc"],axis=1,inplace=True)  # drop the previous ones
    
    # Predic the ccps

    ccps = model.predict_proba(x_new)

        
    return ccps[:,-1]  # only for home production



def get_debt_range():
    
    debtrange1 = np.array([0,300,500,620,770,950])
    debtrange2 = np.linspace(1166,3500,16)
    debtrange3 = np.linspace(3720,8800,25)
    debtrange4 = np.linspace(9200,20000,25)
    debtrange5 = np.linspace(22700,100000,28)

    debt_range = np.concatenate((debtrange1,debtrange2,debtrange3,debtrange4,debtrange5))
    
    return debt_range

#-----------------------------------------------------------------------------#
#                                 EXECUTION                                   #
#-----------------------------------------------------------------------------#


(wage_0, params_wage, sigmas, param_prob_grad) = load_all_parameters()
sigma_u_parinc, budget_params, debt_pen_vec = load_params_frombudget()

#simulate_all_states(11)
#a = np.load("states/states_t8.npy")
invariant_states = np.array(np.meshgrid([1,2,3,4], [1,2,3,4], [0,1],[0,1])).T.reshape(-1,4)
debt_range =  get_debt_range()
if __name__ == "__main__":
    ccp_real = 1
    em_type = 2
    utility_parameters = load_param_g(em_type, real=0)
    models = []
    solution_mode = 1
    conterfactual = 0
    maxdebt = True

    # choose one invariant-state row to debug
    i = 32

    print("Starting direct debug run...")
    get_all_evt(
        i,
        invariant_states,
        debt_range,
        debt_range,
        ccp_real,
        utility_parameters,
        models,
        solution_mode,
        conterfactual,
        em_type,
        maxdebt
    )
# Get repayment trajectories
#get_debt_trajectory(debt_range)
#get_debt_nextperiod(debt_range)
# get other things
#a = load_param_g(0,real=0)
#utility_parameters = load_param_g(em_type,real=0)
#ccp_real = 1
#solution_mode = 0
#get_feasible()
#models = estimate_logit_ccps()
#models = []
#sigma_u = 1.4
#conterfactual = 0 
#em_type = 2
#maxdebt = True
#utility_parameters = load_param_g(em_type,real=0)
#get_all_evt(32,invariant_states,debt_range,debt_range,ccp_real,utility_parameters,models,solution_mode,conterfactual,em_type,maxdebt)
# Generate all possible variant state space
# Experience , Years Education 4y, Years Education 2y, Experience
# after, 2ydegree, 4ydegree,     
# I will not inlcude major in my state space by now. 
#%%
#if __name__ == '__main__':
    #simulate_all_states(11)
    #ccp_real=1
    #print("Started!")
    #em_type = 2
    #utility_parameters = load_param_g(em_type,real=0)
    #models = estimate_logit_ccps()
    #models = []
    #solution_mode = 0
    #sigma_u = 1.4
    #conterfactual = 0
    #print("Parameters loaded!")
    #args = [(i,invariant_states,debt_range,debt_range,ccp_real,sigma_u,utility_parameters,models,solution_mode,conterfactual,em_type) for i in range(np.shape(invariant_states)[0])]
    #pool_obj = multiprocessing.Pool(60)
    #results = pool_obj.starmap(get_all_evt, args)
    #pool_obj.close()

# The alternative is to simulate all the possible states at each time "t" by
# simulating choices over all previous states. This means that I first define choices
# and a transition rule, things might be easier. 

# Generate choices: 
    
# Choices are:
#   -> occupation or field of study (8 categories)
#   -> education (2y vs 4y)
#   -> labor supply decision (full, part or zero(only when education))



# There are 56 possible choices. 

# I need to think how to deal with graduation and with individuals
# being observed enrolled more than 4years in school. 

# I HAVE A PROBLEM WITH THE DIMENSIONS OF THE VJT MATRIX AT SOME POINT. 
# I NEED TO CHECK THAT THE CODE IS DOING WHAT I THINK DOES.


