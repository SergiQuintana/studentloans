# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:46:43 2024

@author: Sergi
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
from scipy import optimize
from os import environ
#from optimparallel import minimize_parallel
#print(environ['MKL_NUM_THREADS'])
from scipy.optimize import check_grad
from scipy.optimize import approx_fprime
import mkl
import warnings
import sys
from tqdm import tqdm 

from numba.core.errors import NumbaPendingDeprecationWarning,NumbaDeprecationWarning

warnings.simplefilter('ignore',category=NumbaDeprecationWarning)
warnings.simplefilter('ignore',category=NumbaPendingDeprecationWarning)

#mkl.set_num_threads(1)

#os.chdir(r"C:\Users\Sergi\Dropbox\PhD\Projects\Papers\1_financial_constraints\Model\Temp")
os.chdir(r"C:/Users/Sergi/Project/Real")
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




path = "C:/Users/Sergi/Dropbox/PhD/Projects/Papers/1_financial_constraints/Model"
pathfunctions = "C:/Users/Sergi/Dropbox/PhD/Projects/Papers/1_financial_constraints/Model/"

#-----------------------------------------------------------------------------#

def load_all_parameters():
    
    """This function loads all the parameters for the different functions
    estimated from the regresions """

    # Wages    

    wage_0 = np.load(f"{pathfunctions}/function_coefficients/wage_0.npy")[...,None].T
    wage_1 = np.load(f"{pathfunctions}/function_coefficients/wage_1.npy")[...,None].T 
    wage_2 = np.load(f"{pathfunctions}/function_coefficients/wage_2.npy")[...,None].T 
    wage_3 = np.load(f"{pathfunctions}/function_coefficients/wage_3.npy")[...,None].T 
    wage_6 = np.load(f"{pathfunctions}/function_coefficients/wage_6.npy")[...,None].T 
    wage_7 = np.load(f"{pathfunctions}/function_coefficients/wage_7.npy")[...,None].T
    wage_8 = np.load(f"{pathfunctions}/function_coefficients/wage_8.npy")[...,None].T
    wage_9 = np.load(f"{pathfunctions}/function_coefficients/wage_9.npy")[...,None].T
    wage_10 = np.load(f"{pathfunctions}/function_coefficients/wage_10.npy")[...,None].T
    
    param_wage = [wage_1,wage_2,wage_3,wage_6,
                  wage_7,wage_8,wage_9,wage_10]
    
    # Standard Deviation Wages
    
    sigmas = np.load(f"{pathfunctions}/function_coefficients/sigmas.npy")
    
    # Grants
    
    param_grants = np.load(f"{pathfunctions}/function_coefficients/grants.npy")
    grants_grad = np.load(f"{pathfunctions}/function_coefficients/paramgrants_grad.npy")[...,None].T
    # Grants Probability
    
    pgrants_2 = np.load(f"{pathfunctions}/function_coefficients/prob_grants_twoyear.npy")[...,None].T
    pgrants_4 = np.load(f"{pathfunctions}/function_coefficients/prob_grants_fouryear.npy")[...,None].T
    pgrants_grad = np.load(f"{pathfunctions}/function_coefficients/dummy_grants_grad.npy")[...,None].T
    
    param_prob_grants = np.concatenate((pgrants_2,pgrants_4),axis=0)
    
    # Parental Transfers
    
    param_fam = np.load(f"{pathfunctions}/function_coefficients/parental_transfers.npy")
    prob_trans = np.load(f"{pathfunctions}/function_coefficients/prob_parental_transfers.npy")	
    
    # Graduation Probability
    
    grad_2 = np.load(f"{pathfunctions}/function_coefficients/prob_grad_twoyear.npy")[...,None].T
    grad_4 = np.load(f"{pathfunctions}/function_coefficients/prob_grad_four.npy")[...,None].T
    grad_grad = np.load(f"{pathfunctions}/function_coefficients/prob_grad_grad.npy")[...,None].T
    
    param_prob_grad = [grad_2,grad_4,grad_grad]
    
    return wage_0, param_wage,sigmas,param_grants,param_prob_grants,param_fam,param_prob_grad, prob_trans, pgrants_grad, grants_grad


def load_data_superfeasible(period):
    
    state = np.load(f"{path}/real_data/state_superfeasible_t{period}.npy").astype('int')
    
    choices = np.load(f"{path}/real_data/choice_superfeasible_t{period}.npy")
        
    x1 = np.load(f"{path}/real_data/invariant_state_superfeasible_t{period}.npy")[:,1:]

    debt = np.load(f"{path}/real_data/debt_superfeasible_t{period}.npy")
    
    debtchoice = np.load(f"{path}/real_data/debtchoice_superfeasible_t{period}.npy")
    
    return x1,state,debt,choices, debtchoice

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
    
    
@njit()
def map_vjt_columns(vjt_i,column_map,idx,total_choices,i):
    
    
    vjt_column = np.zeros((1,np.shape(total_choices)[0]))
    
    for choice in range(np.shape(total_choices)[0]):
        if choice in list(idx[:,0]):
            vjt_column[0,choice] = vjt_i[int(column_map[i,choice])]  # Get the corresponding chioce with the corresponding debt level.
        else:
            vjt_column[0,choice] =-np.inf
                
    return vjt_column

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

def load_all_evt(x1,x2,j,period,em_type,debtchoice):
    
    evt = np.zeros((np.shape(x2)[0],100))
            
    # Now loop over all individuals
            
    for i in range(np.shape(x2)[0]):
            
        # Get individual i
            
        x2i = x2[i,:].astype("int")
        x1i = x1[i,:].astype("int")
        ji = j[i,:].astype("int")
        
        evt_temp = np.load(f"evt/{period+1}/evt_t{period+1}_[{x1i}]_em{em_type}.npz")
        
        # First check if the choice and state include a possible graduation state. 
        if ((x2i[1] >=1) & (ji[1] == 1) & (x2i[4] == 0)) | ((x2i[2]>=3) & (ji[1] == 2) & (x2i[5] == 0))  | (ji[1] == 3) :
            
            grad_x2  =  move_state_grad(x2i,ji,period,grad=1)
            notgrad_x2 = move_state_grad(x2i,ji,period)
            #evt = np.load(f"evt/evt_t{period+1}_{x1}.npz")
            evt_grad =  evt_temp[f"evt_t{period+1}_[{x1i}]_{grad_x2}"][:,0]
            evt_nograd = evt_temp[f"evt_t{period+1}_[{x1i}]_{notgrad_x2}"][:,0]
            x1_new = get_x1_new(x1i)
            p_grad = probability_graduation(x1_new,x2i,ji)
            evt_new = p_grad*evt_grad + (1-p_grad)*evt_nograd
            
        else: 
            # load the data
            x2inew = move_state_grad(x2i,ji,period)
            evt_new = evt_temp[f"evt_t{period+1}_[{x1i}]_{x2inew}"][:,0]
            
            
    
        # I should use the column map strategy:

        evt[i,:] = evt_new
        
    return evt

def prepare_evt():
    
    """"This function loads the vjts to avoid doing that at each iteration of the 
    likelihood function. I should parallelize this function at some point,
    it takes a lot of time..."""
    
    q = np.load("estimates/em_q_typeff2.npy")
        
    for period in range(1,T):
        
        print(f"Loading period {period}")
        
        x1,state,debt,choices,debtchoice = load_data_superfeasible(period)
        
        
        
        # Get only individuals making educational choices
        
        x1 = x1[choices[:,1]!=0]
        state = state[choices[:,1]!=0]
        debt = debt[choices[:,1]!=0]
        debtchoice = debtchoice[choices[:,1]!=0]
        qperiod = q[choices[:,1]!=0]
        choices = choices[choices[:,1]!=0]   
        
                
        evt_em1 = load_all_evt(x1,state,choices,period,1,debtchoice)
        evt_em2 = load_all_evt(x1,state,choices,period,2,debtchoice)  
        
        periods = np.ones(np.shape(x1)[0])*period
        
        if period == 1:
            
            x1all = x1
            stateall = state
            debtall = debt
            choicesall = choices
            debtchoiceall = debtchoice
            periodsall = periods
            evt1all = evt_em1
            evt2all = evt_em2
            qall = qperiod
        else:
            x1all = np.concatenate((x1all,x1))
            stateall = np.concatenate((stateall,state))
            debtall = np.concatenate((debtall,debt))
            choicesall = np.concatenate((choicesall,choices))
            debtchoiceall = np.concatenate((debtchoiceall,debtchoice))
            periodsall = np.concatenate((periodsall,periods))
            evt1all = np.concatenate((evt1all,evt_em1))
            evt2all = np.concatenate((evt2all,evt_em2))
            qall = np.concatenate((qall,qperiod))
            
    return x1all, stateall, debtall, choicesall, debtchoiceall, periodsall , evt1all, evt2all, qall

def get_debt_range():
    
    debtrange1 = np.array([0,300,500,620,770,950])
    debtrange2 = np.linspace(1166,3500,16)
    debtrange3 = np.linspace(3720,8800,25)
    debtrange4 = np.linspace(9200,20000,25)
    debtrange5 = np.linspace(22700,100000,28)

    debt_range = np.concatenate((debtrange1,debtrange2,debtrange3,debtrange4,debtrange5))
    
    return debt_range


#@njit()     
def get_utility(sigma_u,x1,x2,b,j,period):
    
    """This function computes the utility of individuals studying"""
    
    global debt_range
    
    global epsilon_range
    
    x1_new = get_x1_new(x1.astype("int"))
    
    h = fin_help(x1_new,x2,j,period)
    
    x2new = get_x2_new(x2)
    
    w = wage0(x1_new,x2new)
    real_wage = np.exp(w)*(j[2]/2)*52*40 
        
    c = (h-(1+r)*debt_range[int(b)]-tuition(j)+real_wage + epsilon_range)
    c = c.T + debt_range

    return c

@njit()
def tuition(j):
    
    if j[1] == 1:
        return 4000
    elif j[1] == 2:
        return 8000
    elif j[1] == 3:
        return 14000


 
@njit()
def get_x1_new(x1a):
    """This function expands the x1 variables into dummies"""
    types = 4
    parinc  = np.concatenate((np.zeros(x1a[0]-1),np.array([1]),np.zeros(types-x1a[0])))[...,None].T
    ability = np.concatenate((np.zeros(x1a[1]-1),np.array([1]),np.zeros(types-x1a[1])))[...,None].T
    x1_new = np.concatenate((parinc[0,1:],ability[0,1:],x1a[2:]))
    return np.append(1,x1_new) # add a constant


#@njit()
def fin_help(x1_new,x2,j,period):
    
    "this function returns the amount of financial help individuals will recieve"
    
    global param_fam
    global param_grants
    global prob_trans
    global param_prob_grants
    global pgrants_grad
    global grants_grad 

    # help = b0 + x1_new +  4ydummy
    if j[1] == 1:
        x = np.append(x1_new,0)  # include 4ydummy
        prob_grants = param_prob_grants[0,:]
        temp = np.exp(x@prob_trans)
        trans_prob =temp / (1+ temp)
        p_trans = x@param_fam
        temp = np.exp(x1_new@prob_grants)
        grants_prob = temp/(1+temp)
        grants =  x@param_grants
        h = np.exp(p_trans)*trans_prob + np.exp(grants)*grants_prob
    elif j[1] == 2:
        x = np.append(x1_new,1) # include 4ydummy
        prob_grants = param_prob_grants[1,:]
        temp = np.exp(x@prob_trans)
        trans_prob =temp / (1+ temp)
        p_trans = x@param_fam
        temp = np.exp(x1_new@prob_grants)
        grants_prob = temp/(1+temp)
        grants =  x@param_grants
        h = np.exp(p_trans)*trans_prob + np.exp(grants)*grants_prob
    elif j[1] == 3: # For now assume is the same for simplicity
        x = x1_new
        prob_grants = pgrants_grad
        temp = np.exp(x1_new@prob_grants.T)
        grants_prob = temp/(1+temp)
        grants =  x@grants_grad.T
        h = np.exp(grants)*grants_prob  
        
    return h

@njit()
def get_x2_new(x2,fields=8):
    """Generates the x2 variables of the wage equation"""
    # exp,fields (first 4y then twoyear), grad school
    majordummies = np.zeros((fields-1)+1)

    if (int(x2[8]) !=12) & (int(x2[8])!=0):
        if x2[8] < 3:
            majordummies[int(x2[8])-1] = 1 # set major 
        else:
            majordummies[int(x2[8])-2] = 1 # set major 
        
    elif int(x2[8]) == 12:  #  major is associate degree
        majordummies[-1] = 1
        
    x2_new = np.append(np.array(x2[0]),majordummies)
    x2_new = np.append(x2_new,np.array(x2[6])) # include whether the individual has grad school

    return x2_new

@njit()
def wage0(x1_new,x2_new):
    
    """ Different wage fucntion for numba"""
    
    param_wage = wage_0
    
    # This part puts together x1 and x2. x1 is time invariant, x2 is time variant. 
    x = np.hstack((x1_new[...,None].T,x2_new[:-1][...,None].T)) # do not include grad school coefficient
    
    w = x@param_wage.T
    return w

#@njit(parallel=True) 
def get_epsilons(x1all,stateall,debtall,debtchoiceall,choicesall,periodsall,evt1all,evt2all):
    
    
    epsilons1 = np.zeros(np.shape(x1all)[0])
    epsilons2 = np.zeros(np.shape(x1all)[0])
    
    budget1 = np.zeros(np.shape(x1all)[0])
    budget2 = np.zeros(np.shape(x1all)[0])
    
    
    for i in prange(np.shape(x1all)[0]):
        epsilons1[i], budget1[i] = get_epsilon_i(x1all[i,:],stateall[i,:],debtall[i],debtchoiceall[i],choicesall[i,:],periodsall[i],evt1all[i])   
        epsilons2[i], budget2[i] = get_epsilon_i(x1all[i,:],stateall[i,:],debtall[i],debtchoiceall[i],choicesall[i,:],periodsall[i],evt2all[i]) 
    
    
    return epsilons1, epsilons2


def perform_pararell(x1all,stateall,debtall,debtchoiceall,choicesall,periodsall,evt1all,evt2all):
    
    
    pool_obj = multiprocessing.Pool(60)
    args = [(i,x1all,stateall,debtall,debtchoiceall,choicesall,periodsall,evt1all,evt2all) for i in range(np.shape(x1all)[0])]
    results = pool_obj.starmap(get_epsilons_parallel, args)
    pool_obj.close()
    
    np.save("fitdebt/results_sigma0.4.npy",np.array(results))
    
    pass

def get_epsilons_parallel(i,x1all,stateall,debtall,debtchoiceall,choicesall,periodsall,evt1all,evt2all):
    
    print(i)

    
    epsilons1, budget1 = get_epsilon_i(x1all[i,:],stateall[i,:],debtall[i],debtchoiceall[i],choicesall[i,:],periodsall[i],evt1all[i])   
    epsilons2, budget2 = get_epsilon_i(x1all[i,:],stateall[i,:],debtall[i],debtchoiceall[i],choicesall[i,:],periodsall[i],evt2all[i]) 
    
    return epsilons1, epsilons2, budget1, budget2

#@njit()
def get_epsilon_i(x1,state,debt,debtchoice,choices,period,evt):
    
    sigma_u = 1.4
    
    c = get_utility(sigma_u,x1,state,debt,choices,period)
    
    e, cval = find_epsilon(c,debtchoice,evt)
           
    
    return e, cval

@njit()
def check_consumption(vjt,c):
    
    shape = np.shape(vjt)
    
    c = c.flatten()
    vjt = vjt.flatten()

    vjt[c<2000] = -100000000

    vjt = vjt.reshape(shape)
    return vjt

@njit()
def get_debt(debt_range,debt):
    
    debtvalues = np.zeros(np.shape(debt)[0])
    
    for i in range(np.shape(debt)[0]):
        
        debtvalues[i] = debt_range[int(debt[i])]
        
    return debtvalues
        
        

#@njit() 
def find_epsilon(c,debtchoice,evt1):
    
    global debt_range
    
    sigma_u = 1.4
    
    u = get_power_utility(sigma_u,c)

    vjt = u + beta*evt1
       
    vjt = check_consumption(vjt,c)
    
    debttheory = np.argmax(vjt,axis=1)
    
    debttheory[vjt[:,-1]== -100000000] = 99
    
    debtvalue = get_debt(debt_range,debttheory)
    
    diff = (debtvalue - debt_range[int(debtchoice)])**2
    
    if int(debtchoice) != 99:
        
        debtidx = int(np.argmin(diff))
        if debtidx == 0: 
            first99 = np.shape(debttheory[debttheory==99])[0]
            e = epsilon_range[first99]
            cvalue = c[first99,int(debtchoice)]
        else:
            e  = epsilon_range[debtidx]
            cvalue = c[debtidx,int(debtchoice)]
        
    else:  # here the epsilon value is the first debt value that generates 99 debt
    
        first99 = np.shape(debttheory[debttheory==99])[0]
        e = epsilon_range[first99]
        cvalue = c[first99,int(debtchoice)]
    

    return e, cvalue
    

def save_to_stata():
    
    x1all, stateall, debtall, choicesall, debtchoiceall, periodsall , evt1all, evt2all, qall = load_all_temporary()
    
    epsilons = np.load("fitdebt/results_sigma0.4.npy")
    
    debtall = debt_range[debtall.astype("int")]
    debtchoiceall = debt_range[debtchoiceall.astype("int")]
        
    states = np.concatenate((x1all,stateall,debtall[...,None],choicesall,periodsall[...,None],debtchoiceall[...,None],epsilons),axis=1)
    
    names = ["parinc","ability","sex","race","exp","twoyear_exp","fouryear_exp",
             "grad_exp","twograd","fourgrad","gradgrad","last_school","majorgrad",
             "last_choice","debt","field","educ","work","period","debt_choice",
             "epsilon1","epsilon2","budget1","budget2"]
    
    toexport = pd.DataFrame(states,columns=names)
    
    toexport.to_stata(f"{path}/dataepsilons.dta",write_index=False)
    
def fitdata_pararell():
    
    x1all, stateall, debtall, choicesall, debtchoiceall, periodsall , evt1all, evt2all, qall = load_all_temporary()
    
    
    debtall = debt_range[debtall.astype("int")]
    debtchoiceall = debt_range[debtchoiceall.astype("int")]
    
    
    states = np.concatenate((x1all,stateall,debtall[...,None],choicesall,periodsall[...,None],debtchoiceall[...,None],qall),axis=1)
    
    # Loop over all states
    
    args = [(2,ability,parinc,exp,states,x1all,stateall,choicesall,evt1all,evt2all) for ability in range(1,5) for parinc in range(1,5) for exp in range(0,6) ]
    pool_obj = multiprocessing.Pool(60)
    results4y = pool_obj.starmap(loop_over_all_states_pararell, args)
    pool_obj.close()
    
    args = [(1,ability,parinc,exp,states,x1all,stateall,choicesall,evt1all,evt2all) for ability in range(1,3) for parinc in range(1,5) for exp in range(0,5) ]
    pool_obj = multiprocessing.Pool(60)
    results2y = pool_obj.starmap(loop_over_all_states_pararell, args)
    pool_obj.close()
    
    args = [(3,ability,parinc,exp,states,x1all,stateall,choicesall,evt1all,evt2all) for ability in range(1,5) for parinc in range(1,5) for exp in range(0,2) ]
    pool_obj = multiprocessing.Pool(60)
    resultsgrad = pool_obj.starmap(loop_over_all_states_pararell, args)
    pool_obj.close()
    
    np.save("fitdebt/grad_distribution.npy",resultsgrad)
    np.save("fitdebt/fouryear_distribution.npy",results4y)
    np.save("fitdebt/twoyear_distribution.npy",results2y)
    
    
def loop_over_all_states_pararell(eductype,ability,parinc,exp,states,x1all,stateall,choicesall,evt1all,evt2all):
    
    print(eductype,ability,parinc,exp)
    
    if eductype == 2:  # 4y school
            
        if exp < 5:
                
            states1 = states[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                             & (stateall[:,2]==exp) & (choicesall[:,1]==2)]
                    
            evt1_1 = evt1all[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                             & (stateall[:,2]==exp) & (choicesall[:,1]==2)]
                    
            evt2_1 = evt2all[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                             & (stateall[:,2]==exp) & (choicesall[:,1]==2)]
                    
                
                    
            results = fit_the_mean(states1,evt1_1, evt2_1)
            results_joint = [ability,parinc,exp,results[0][0],results[0][1],results[1]]
                    
        elif exp == 5:
                    
            states1 = states[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                             & (stateall[:,2]>4)& (choicesall[:,1]==2)]
                        
            evt1_1 = evt1all[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                             & (stateall[:,2]>4) & (choicesall[:,1]==2)]
                        
            evt2_1 = evt2all[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                             & (stateall[:,2]>4) & (choicesall[:,1]==2)]
                        
                    
                        
            results = fit_the_mean(states1,evt1_1, evt2_1)
            results_joint = [ability,parinc,exp,results[0][0],results[0][1],results[1]]
            
    
    # For 2y choices there are only 2 ability groups!
    elif eductype == 1: 
 
        if exp < 4:
            
            if ability == 1:
                
                states1 = states[(x1all[:,0]==parinc) & (x1all[:,1]<3) 
                                 & (stateall[:,2]==exp) & (choicesall[:,1]==1)]
                        
                evt1_1 = evt1all[(x1all[:,0]==parinc) & (x1all[:,1]<3) 
                                 & (stateall[:,2]==exp) & (choicesall[:,1]==1)]
                        
                evt2_1 = evt2all[(x1all[:,0]==parinc) & (x1all[:,1]<3) 
                                 & (stateall[:,2]==exp) & (choicesall[:,1]==1)]
                        
                    
                results = fit_the_mean(states1,evt1_1, evt2_1)
                results_joint = [1,parinc,exp,results[0][0],results[0][1],results[1]]
            
            elif ability ==2:
                
                states1 = states[(x1all[:,0]==parinc) & (x1all[:,1]>2) 
                                 & (stateall[:,2]==exp) & (choicesall[:,1]==1)]
                    
                evt1_1 = evt1all[(x1all[:,0]==parinc) & (x1all[:,1]>2) 
                                 & (stateall[:,2]==exp) & (choicesall[:,1]==1)]
                    
                evt2_1 = evt2all[(x1all[:,0]==parinc) & (x1all[:,1]>2) 
                                 & (stateall[:,2]==exp) & (choicesall[:,1]==1)]
                    
                
                results = fit_the_mean(states1,evt1_1, evt2_1)
                results_joint = [3,parinc,exp,results[0][0],results[0][1],results[1]]


        elif exp == 4:
            
            if ability == 1:
                
                states1 = states[(x1all[:,0]==parinc) & (x1all[:,1]<3) 
                                 & (stateall[:,2]>3) & (choicesall[:,1]==1)]
                    
                evt1_1 = evt1all[(x1all[:,0]==parinc) & (x1all[:,1]<3) 
                                 & (stateall[:,2]>3) & (choicesall[:,1]==1)]
                    
                evt2_1 = evt2all[(x1all[:,0]==parinc) & (x1all[:,1]<3) 
                                 & (stateall[:,2]>3) & (choicesall[:,1]==1)]
                    
                
                results = fit_the_mean(states1,evt1_1, evt2_1)
                results_joint = [1,parinc,exp,results[0][0],results[0][1],results[1]]

            elif ability == 2:
                
                states1 = states[(x1all[:,0]==parinc) & (x1all[:,1]>2) 
                                 & (stateall[:,2]>3) & (choicesall[:,1]==1)]
                    
                evt1_1 = evt1all[(x1all[:,0]==parinc) & (x1all[:,1]>2) 
                                 & (stateall[:,2]>3) & (choicesall[:,1]==1)]
                    
                evt2_1 = evt2all[(x1all[:,0]==parinc) & (x1all[:,1]>2) 
                                 & (stateall[:,2]>3) & (choicesall[:,1]==1)]                
                
                results = fit_the_mean(states1,evt1_1, evt2_1)
                results_joint = [3,parinc,exp,results[0][0],results[0][1],results[1]]


    
    
    elif eductype == 3:
            
        if exp == 0:
                
            states1 = states[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                             & (stateall[:,3]==exp) & (choicesall[:,1]==3)]
                    
            evt1_1 = evt1all[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                             & (stateall[:,3]==exp) & (choicesall[:,1]==3)]
                    
            evt2_1 = evt2all[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                             & (stateall[:,3]==exp) & (choicesall[:,1]==3)]
                    
                    
            results = fit_the_mean(states1,evt1_1, evt2_1)
            results_joint = [ability,parinc,exp,results[0][0],results[0][1],results[1]]
                    
        elif exp == 1:
                
            states1 = states[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                             & (stateall[:,3]>0) & (choicesall[:,1]==3)]
                    
            evt1_1 = evt1all[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                             & (stateall[:,3]>0) & (choicesall[:,1]==3)]
                    
            evt2_1 = evt2all[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                             & (stateall[:,3]>0) & (choicesall[:,1]==3)]
                    
                    
            results = fit_the_mean(states1,evt1_1, evt2_1)
            results_joint = [ability,parinc,exp,results[0][0],results[0][1],results[1]]
                    
    return results_joint
    
def fitdata():
    
    
    x1all, stateall, debtall, choicesall, debtchoiceall, periodsall , evt1all, evt2all, qall = load_all_temporary()
        
    debtall = debt_range[debtall.astype("int")]
    debtchoiceall = debt_range[debtchoiceall.astype("int")]
    
    
    
    states = np.concatenate((x1all,stateall,debtall[...,None],choicesall,periodsall[...,None],debtchoiceall[...,None],qall),axis=1)
    
    # Loop over all states
    
    loop_over_all_states(states,x1all,stateall,choicesall,evt1all,evt2all)
    
    # Re-check states with sigma = 1 ! Its random!?
    
    


def loop_over_all_states(states,x1all,stateall,choicesall,evt1all,evt2all):
    
    # For 4y choices
    #states = np.array(np.meshgrid([1,2,3,4], [1,2,3,4], [1,2,3,4,5,6])).T.reshape(-1,3)
    
    results_all = []
    
    for ability in range(1,5): 
    
        for parinc in range(1,5):
            
            for four_exp in range(0,6):
                
                print(ability,parinc,four_exp)
            
                if four_exp < 5:
                
                    states1 = states[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                                     & (stateall[:,2]==four_exp) & (choicesall[:,1]==2)]
                    
                    evt1_1 = evt1all[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                                     & (stateall[:,2]==four_exp) & (choicesall[:,1]==2)]
                    
                    evt2_1 = evt2all[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                                     & (stateall[:,2]==four_exp) & (choicesall[:,1]==2)]
                    
                
                    
                    results = fit_the_mean(states1,evt1_1, evt2_1)
                    results_joint = [ability,parinc,four_exp,results[0][0],results[0][1],results[1]]
                    results_all.append(results_joint)
                    
                elif four_exp == 5:
                    
                    states1 = states[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                                     & (stateall[:,2]>4)& (choicesall[:,1]==2)]
                        
                    evt1_1 = evt1all[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                                     & (stateall[:,2]>4) & (choicesall[:,1]==2)]
                        
                    evt2_1 = evt2all[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                                     & (stateall[:,2]>4) & (choicesall[:,1]==2)]
                        
                    
                        
                    results = fit_the_mean(states1,evt1_1, evt2_1)
                    results_joint = [ability,parinc,four_exp,results[0][0],results[0][1],results[1]]
                    results_all.append(results_joint)
                    
    results4y = np.array(results_all)
            
    results_all = []
    
    # For 2y choices there are only 2 ability groups!
            
    for parinc in range(1,5):
        
        for twoyear_exp in range(0,5):
            
            if twoyear_exp < 4:
                
                print(parinc,twoyear_exp)
                
                states1 = states[(x1all[:,0]==parinc) & (x1all[:,1]<3) 
                                 & (stateall[:,2]==twoyear_exp) & (choicesall[:,1]==1)]
                    
                evt1_1 = evt1all[(x1all[:,0]==parinc) & (x1all[:,1]<3) 
                                 & (stateall[:,2]==twoyear_exp) & (choicesall[:,1]==1)]
                    
                evt2_1 = evt2all[(x1all[:,0]==parinc) & (x1all[:,1]<3) 
                                 & (stateall[:,2]==twoyear_exp) & (choicesall[:,1]==1)]
                    
                
                results = fit_the_mean(states1,evt1_1, evt2_1)
                results_joint = [1,parinc,twoyear_exp,results[0][0],results[0][1],results[1]]
                results_all.append(results_joint) 
                
                states1 = states[(x1all[:,0]==parinc) & (x1all[:,1]>2) 
                                 & (stateall[:,2]==twoyear_exp) & (choicesall[:,1]==1)]
                    
                evt1_1 = evt1all[(x1all[:,0]==parinc) & (x1all[:,1]>2) 
                                 & (stateall[:,2]==twoyear_exp) & (choicesall[:,1]==1)]
                    
                evt2_1 = evt2all[(x1all[:,0]==parinc) & (x1all[:,1]>2) 
                                 & (stateall[:,2]==twoyear_exp) & (choicesall[:,1]==1)]
                    
                
                results = fit_the_mean(states1,evt1_1, evt2_1)
                results_joint = [3,parinc,twoyear_exp,results[0][0],results[0][1],results[1]]
                results_all.append(results_joint)
            elif twoyear_exp == 4:
                
                states1 = states[(x1all[:,0]==parinc) & (x1all[:,1]<3) 
                                 & (stateall[:,2]>3) & (choicesall[:,1]==1)]
                    
                evt1_1 = evt1all[(x1all[:,0]==parinc) & (x1all[:,1]<3) 
                                 & (stateall[:,2]>3) & (choicesall[:,1]==1)]
                    
                evt2_1 = evt2all[(x1all[:,0]==parinc) & (x1all[:,1]<3) 
                                 & (stateall[:,2]>3) & (choicesall[:,1]==1)]
                    
                
                results = fit_the_mean(states1,evt1_1, evt2_1)
                results_joint = [1,parinc,twoyear_exp,results[0][0],results[0][1],results[1]]
                results_all.append(results_joint) 
                
                states1 = states[(x1all[:,0]==parinc) & (x1all[:,1]>2) 
                                 & (stateall[:,2]>3) & (choicesall[:,1]==1)]
                    
                evt1_1 = evt1all[(x1all[:,0]==parinc) & (x1all[:,1]>2) 
                                 & (stateall[:,2]>3) & (choicesall[:,1]==1)]
                    
                evt2_1 = evt2all[(x1all[:,0]==parinc) & (x1all[:,1]>2) 
                                 & (stateall[:,2]>3) & (choicesall[:,1]==1)]                
                
                results = fit_the_mean(states1,evt1_1, evt2_1)
                results_joint = [3,parinc,twoyear_exp,results[0][0],results[0][1],results[1]]
                results_all.append(results_joint)
                
    results2y = np.array(results_all)
    
    
    # now for graduate school
    
    results_all = []
    
    for ability in range(1,5): 
    
        for parinc in range(1,5):
            
            for grad_exp in range(0,2):
                
                print(ability,parinc,grad_exp)
            
                if grad_exp == 0:
                
                    states1 = states[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                                     & (stateall[:,3]==grad_exp) & (choicesall[:,1]==3)]
                    
                    evt1_1 = evt1all[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                                     & (stateall[:,3]==grad_exp) & (choicesall[:,1]==3)]
                    
                    evt2_1 = evt2all[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                                     & (stateall[:,3]==grad_exp) & (choicesall[:,1]==3)]
                    
                    
                    results = fit_the_mean(states1,evt1_1, evt2_1)
                    results_joint = [ability,parinc,grad_exp,results[0][0],results[0][1],results[1]]
                    results_all.append(results_joint)
                    
                elif grad_exp == 1:
                
                    states1 = states[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                                     & (stateall[:,3]>0) & (choicesall[:,1]==3)]
                    
                    evt1_1 = evt1all[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                                     & (stateall[:,3]>0) & (choicesall[:,1]==3)]
                    
                    evt2_1 = evt2all[(x1all[:,0]==parinc) & (x1all[:,1]==ability) 
                                     & (stateall[:,3]>0) & (choicesall[:,1]==3)]
                    
                    
                    results = fit_the_mean(states1,evt1_1, evt2_1)
                    results_joint = [ability,parinc,grad_exp,results[0][0],results[0][1],results[1]]
                    results_all.append(results_joint)
                    
    resultsgrad = np.array(results_all)
    
    # store the results
    
    np.save("fitdebt/grad_distribution.npy",resultsgrad)
    np.save("fitdebt/fouryear_distribution.npy",results4y)
    np.save("fitdebt/twoyear_distribution.npy",results2y)
    
    pass
        
        
def view_distributions():


    resultsgrad = np.load("fitdebt/grad_distribution.npy")
    results4y   = np.load("fitdebt/fouryear_distribution.npy")
    results2y   = np.load("fitdebt/twoyear_distribution.npy")
    
    return results2y, results4y, resultsgrad
    
    
               


def get_sample(data,evt1,evt2):
    
    
    s = 100     # number of samples to generate 
    n = 1000    # sample size 
    
    random_indices = np.random.choice(np.shape(data)[0],  
                                  size=n,  
                                  replace=True) 
  
    sample = data[random_indices,:]
    
    evt1sample = evt1[random_indices,:]
    evt2sample = evt2[random_indices,:]
    
    # draw education type
    eductype  = np.random.binomial(1, sample[:,25], np.shape(sample)[0])
    
    evt = get_evt(evt1sample,evt2sample,eductype)
    budget = sample[:,22]
    
    e = np.random.normal(loc=0, scale=1, size=n)
    
    return budget, evt, e


def get_budget(data, random_indices):
    
    
    budgetdata = np.zeros(np.shape(data)[0])
    
    for i in range(np.shape(data)[0]):
        
        x1 = data[i,:4]
        x2 = data[i,4:14]
        debt = data[i,14]
        j = data[i,15:18]
        period = data[i,18]
    
    
        x1_new = get_x1_new(x1.astype("int"))
    
        h = fin_help(x1_new,x2,j,period)
    
        x2new = get_x2_new(x2)
    
        w = wage0(x1_new,x2new)
        real_wage = np.exp(w)*(j[2]/2)*52*40 
        
        budgetdata[i] = (h-(1+r)*debt-tuition(j)+real_wage)
        
    budget = budgetdata
    return budget
    
    

def get_sample_new(data,evt1,evt2):
    
    
    s = 10     # number of samples to generate 
    n = 1000    # sample size 
    #samples = []
    
    random_indices = np.random.choice(np.shape(data)[0],  
                                  size=n,  
                                  replace=True) 
  
    #sample = data[random_indices,:]
    
    #evt1sample = evt1[random_indices,:]
    #evt2sample = evt2[random_indices,:]
    
    # draw education type
    eductype  = np.random.binomial(1, data[:,21], np.shape(data)[0])
    
    evt = get_evt(evt1,evt2,eductype)

    debtprevious = data[:,14]
    
    budget = get_budget(data, random_indices)
    
    
    return budget, evt, debtprevious 
    
def fit_the_mean(data,evt1,evt2):
    
    
    # Get the target moment
    
    debt = np.copy(data[:,19])

    debt2 = np.copy(debt)
    
    debt2[debt2>0] = 1
    
    anydebt = np.mean(debt2)
    
    if anydebt == 0:
        
        vardebt = 0.01
        anydebt = 0.01
        meandebt = 0.01
        p80 = 0.01
        
    else: 
        
        meandebt = np.mean(debt[debt>0])
        vardebt = np.var(debt[debt>0])
        p80 =  np.percentile(debt[debt>0], 80)
    
    # guess a mean and a sigma
    
    #params = [1000,1000]  
    params = [-12000,12000]
    
    b1 = (-np.inf,np.inf)
    b2 = (1,np.inf)
    #b3 = (0.2,2)
    bounds = (b1,b2)
    
    
    tolr = 0.01
    newerror = 100000000000000000000000000000000000000
    it = 0
    while newerror > tolr:
        print(it) 
        # change random state
        np.random.seed(it)
        # Simulate the samples        
        budget, evt, previousdebt = get_sample_new(data,evt1,evt2)
        # Minimize
        res = minimize(minimize_distance,params,args=(meandebt,anydebt,vardebt,p80,budget,evt,previousdebt),
                       bounds=bounds,method='Nelder-Mead',options={'disp':True, })
        if res.fun < newerror:
            
            paramshat = res.x
            funhat = res.fun
            
        newerror = res.fun
        
        it = it+1
        
        if it == 10:
            
            tolr = 0.03
            
        elif it == 15:
            
            tolr = 0.04
        
        if it == 20:
            newerror = 0

    print(paramshat,funhat)
    return paramshat, funhat
    
    
def minimize_distance(params,meandebt,anydebt,vardebt,p80,budget,evt,previousdebt):
    
    """
    This function fits the mean debt of the data with a normal distribution
    that takes parameters: 
        
        - constant, variance
        
    It simulates 10k individuals from the state distribution. 
    
    """
    
    mu = params[0]
    sigma = params[1]
    
    
    moments = np.zeros(4)
    
    s = 200 
    
    meandebtsimu = np.zeros(s)
    anydebtsimu = np.zeros(s)
    vardebtsimu = np.zeros(s)
    p80simu = np.zeros(s)
    
    for i in range(s):

        enew = np.random.normal(loc=mu, scale=sigma, size=np.shape(budget)[0]) 
        #enew = e = np.random.normal(loc=mu, scale=sigma, size=1000) 
        #enew = enew*0 
        meandebtsimu[i], anydebtsimu[i], vardebtsimu[i], p80simu[i] = get_optimal_debt_new(budget,evt,enew,previousdebt)
    
    meandebtsimu = np.mean(meandebtsimu)
    anydebtsimu = np.mean(anydebtsimu)
    vardebtsimu = np.mean(vardebtsimu)
    p80simu = np.mean(p80simu)
    #print(p80simu,p80) 
    moments[0] = (meandebtsimu-meandebt)/meandebt 
    moments[1] = (anydebtsimu-anydebt)/anydebt 
    moments[2] = (vardebtsimu-vardebt)/vardebt
    moments[2] = 0
    moments[3] = (p80simu-p80)/p80 
    moments[3] = 0 
    #print("mean:",meandebt,meandebtsimu)
    #print("anydebt:",anydebt,anydebtsimu)
    #print("p80:",p80simu,p80) 
    return np.sum(moments**2)


@njit()
def get_evt(evt1,evt2,eductype):
    
    evt = np.zeros(np.shape(evt1))
    
    for i in range(np.shape(eductype)[0]):
        
        if eductype[i] == 0:
        
            evt[i,:] = evt1[i,:]
        else:
            evt[i,:] = evt2[i,:]
            
    return evt


@njit()
def get_range_number(b,maxdebt):
    
    value = np.zeros(np.shape(b)[0])
    value2 = np.zeros(np.shape(b)[0])
    
    for i in range(np.shape(b)[0]):
        
        diff = (debt_range - b[i])**2
            
        value[i] = np.argmin(diff)
        
        diff2 = (debt_range - maxdebt[i])**2
            
        value2[i] = np.argmin(diff2)
        
    return value, value2


@njit()
def minimum_debt_maxdebt(u,b,maxdebt):
    
    
    b, maxdebt  = get_range_number(b,maxdebt)
    
    for i in range(np.shape(b)[0]):
        
        u[i,:int(b[i])] = -100000000
        #u[i,int(maxdebt[i])+1:] = -100000000
        #if u[i,-1] == -100000000:  # at least you need to be able to get the maximum amount...
            #u[i,-1] = 0 
        
    return u
        
        

def get_debt_rules(c,vjt,previousdebt):
    
    global debt_range
    
    
    vjt[c<2000] = -100000000
    
    maxdebt = np.minimum(debt_range[-1],previousdebt + 22300)
        
    
    #vjt = minimum_debt_maxdebt(vjt,previousdebt,maxdebt)

    return vjt


def get_optimal_debt_new(budget,evt,e,previousdebt):
    
    global debt_range
     
    
    c = (budget[...,None]+e[...,None]+debt_range)
    
    sigma_u = 1.4 
    
    u = get_power_utility(sigma_u,c)

    vjt = u + beta*evt
       
    vjt = get_debt_rules(c,vjt,previousdebt)
        
    debttheory = np.argmax(vjt,axis=1)
    
    debttheory[vjt[:,-1]== -100000000] = 99
    
    debtvalue = get_debt(debt_range,debttheory)
    
    debt2 = np.copy(debtvalue)
    debt2[debt2>0] = 1
    
    anydebt = np.mean(debt2)
    
    if anydebt == 0 :
        meandebt = 0.01
        anydebt = 0.01
        vardebt = 0.01
        p80 = 0.01
    else:
        meandebt = np.mean(debtvalue[debtvalue>0])
        vardebt = np.var(debtvalue[debtvalue>0])
        p80 = np.percentile(debtvalue[debtvalue>0], 80)

    
    return meandebt,  anydebt, vardebt,  p80


    
    
def get_optimal_debt(budget,evt,e):
    
    global debt_range
     
    
    c = (budget[...,None]+e[...,None]+debt_range)
    
    sigma_u = 1.4 
    
    u = get_power_utility(sigma_u,c)

    vjt = u + beta*evt
       
    vjt = check_consumption(vjt,c)
    
    debttheory = np.argmax(vjt,axis=1)
    
    debttheory[vjt[:,-1]== -100000000] = 99
    
    debtvalue = get_debt(debt_range,debttheory)
    
    debt2 = np.copy(debtvalue)
    debt2[debt2>0] = 1
    
    return np.mean(debtvalue[debtvalue>0]),  np.mean(debt2), np.var(debtvalue[debtvalue>0])
    
    

def save_all_temporary(x1all, stateall, debtall, choicesall, debtchoiceall, periodsall , evt1all, evt2all, qall):
    
    np.save("fitdebt/x1all.npy",x1all)
    np.save("fitdebt/stateall.npy",stateall)
    np.save("fitdebt/debtall.npy",debtall)
    np.save("fitdebt/debtchoiceall.npy",debtchoiceall)
    np.save("fitdebt/choicesall.npy",choicesall)
    np.save("fitdebt/periodsall.npy",periodsall)
    np.save("fitdebt/evt1all.npy",evt1all)
    np.save("fitdebt/evt2all.npy",evt2all)
    np.save("fitdebt/qall.npy",qall)
    
    pass

@njit
def get_power_utility(sigma_u,c):
    
    c = np.maximum(c,2000)
    
    return 0.1*((0.00001*c)**(1-sigma_u)/(1-sigma_u))

def load_all_temporary_temp():
    
    x1all = np.load(f"{pathtemp}/fitdebt/x1all.npy")
    stateall =np.load(f"{pathtemp}/fitdebt/stateall.npy")
    debtall =np.load(f"{pathtemp}/fitdebt/debtall.npy")
    debtchoiceall =np.load(f"{pathtemp}/fitdebt/debtchoiceall.npy")
    choicesall =np.load(f"{pathtemp}/fitdebt/choicesall.npy")
    periodsall =np.load(f"{pathtemp}/fitdebt/periodsall.npy")
    evt1all =np.load(f"{pathtemp}/fitdebt/evt1all.npy")
    evt2all =np.load(f"{pathtemp}/fitdebt/evt2all.npy")
    qall =np.load(f"{pathtemp}/fitdebt/qall.npy")
    
    return x1all, stateall, debtall, choicesall, debtchoiceall, periodsall , evt1all, evt2all, qall

def load_all_temporary():
    
    x1all = np.load(f"fitdebt/x1all.npy")
    stateall =np.load("fitdebt/stateall.npy")
    debtall =np.load("fitdebt/debtall.npy")
    debtchoiceall =np.load("fitdebt/debtchoiceall.npy")
    choicesall =np.load("fitdebt/choicesall.npy")
    periodsall =np.load("fitdebt/periodsall.npy")
    evt1all =np.load("fitdebt/evt1all.npy")
    evt2all =np.load("fitdebt/evt2all.npy")
    qall =np.load("fitdebt/qall.npy")
    
    return x1all, stateall, debtall, choicesall, debtchoiceall, periodsall , evt1all, evt2all, qall

#-----------------------------------------------------------------------------#

# Test:
    
wage_0, params_wage,sigmas,param_grants,param_prob_grants,param_fam,param_prob_grad, prob_trans, pgrants_grad, grants_grad = load_all_parameters()

debt_range = get_debt_range()

epsilon_range = np.linspace(-200000,200000,10000)

x1all, stateall, debtall, choicesall, debtchoiceall, periodsall , evt1all, evt2all, qall = prepare_evt()

save_all_temporary(x1all, stateall, debtall, choicesall, debtchoiceall, periodsall , evt1all, evt2all, qall)
#x1all, stateall, debtall, choicesall, debtchoiceall, periodsall , evt1all, evt2all, qall = load_all_temporary()
#x1all, stateall, debtall, choicesall, debtchoiceall, periodsall , evt1all, evt2all, qall = load_all_temporary_temp()
results2y, results4y, resultsgrad = view_distributions()

#fitdata() 
#tic =time.time()
#epsilon1, epsion2 = get_epsilons(x1all,stateall,debtall,debtchoiceall,choicesall,periodsall,evt1all,evt2all)
#toc = time.time()
#print("Time is!",toc-tic) 
#get_epsilons_parallel(9425,x1all,stateall,debtall,debtchoiceall,choicesall,periodsall,evt1all,evt2all)

#%%
# Pararelll Ppooool Version
if __name__ == '__main__':
    print("Loading Data!")
    tic = time.time()
    x1all, stateall, debtall, choicesall, debtchoiceall, periodsall , evt1all, evt2all, qall = load_all_temporary()
    fitdata_pararell()
    toc = time.time()
    print("Time elapsed is", toc-tic)
    save_to_stata()
