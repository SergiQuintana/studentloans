# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 18:00:02 2023

@author: Sergi
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import scipy
import os
import time
from numba import njit

#os.chdir(r"C:\Users\Sergi\Dropbox\PhD\Projects\Papers\1_financial_constraints\Model\Temp")
mu = 0
gamma = 0.57721
beta = 0.98
T = 10 
sigma_u = 1.4
from debt_limits import (
    CONSUMPTION_FLOOR,
    INTEREST_RATE,
    get_annual_cap_by_stage,
    get_simulation_bounds_indices,
    get_lifetime_cap_by_stage,
    lower_bound_index,
    upper_bound_index,
)
r = INTEREST_RATE

# Forward-simulation debt-penalty convention used in ``get_utility_agents``.
# "legacy_multiplier" (default) keeps the current behavior: the SMM-shortcut
# discounted multiplier applied in flow utility. "single_flow" charges the
# per-period flow penalty exactly once, matching the Bellman convention in
# ``model_solution_em.get_conditional``. Read the comment block inside
# ``get_utility_agents`` before switching; "single_flow" awaits researcher
# confirmation and must not become the default without sign-off.
SIM_DEBT_PENALTY_CONVENTION = "legacy_multiplier"

# Parameters for the wage equations:
from pathlib import Path
from config import DIR, OUT, INP, FUN, RDATA, CONT, EST, LIK
import budget_shock as bs
from financial_process import (
    draw_grants_vectorized,
    draw_transfers_vectorized,
    expected_grants_vectorized,
    expected_transfers_vectorized,
    load_auxiliary_financial_process,
)
from latent_types import (
    TYPE_GRANT,
    TYPE_LOAN,
    TYPE_TRANSFER,
    draw_type_ids,
    validate_q,
)
pathfunctions   = DIR["MODEL_FUNCOEF"]
path_realdata   = DIR["MODEL_REALDATA"]
path_estimates  = DIR["MODEL_ESTIMATES"]
#---------------------------------------------------#
# Now simulate the data

# All agents start at the same period, but they are heterogeneous in 
# the invariant state space. I will start with 100 agents and then increase. 

def load_all_parameters():
    
    """This function loads all the parameters for the different functions
    estimated from the regresions """

    # Wages    

    wage_0  = np.load(f"{pathfunctions}/wage_0.npy")[...,None].T
    wage_1  = np.load(f"{pathfunctions}/wage_1.npy")[...,None].T
    wage_2  = np.load(f"{pathfunctions}/wage_2.npy")[...,None].T
    wage_3  = np.load(f"{pathfunctions}/wage_3.npy")[...,None].T
    wage_6  = np.load(f"{pathfunctions}/wage_6.npy")[...,None].T
    wage_7  = np.load(f"{pathfunctions}/wage_7.npy")[...,None].T
    wage_8  = np.load(f"{pathfunctions}/wage_8.npy")[...,None].T
    wage_9  = np.load(f"{pathfunctions}/wage_9.npy")[...,None].T
    wage_10 = np.load(f"{pathfunctions}/wage_10.npy")[...,None].T
    
    param_wage = [wage_1,wage_2,wage_3,wage_6,
                  wage_7,wage_8,wage_9,wage_10]
    
    # Standard Deviation Wages
    
    sigmas = np.load(f"{pathfunctions}/sigmas.npy")
    
    # Graduation Probability
    
    grad_2 = np.load(f"{pathfunctions}/prob_grad_twoyear.npy")[...,None].T
    grad_4 = np.load(f"{pathfunctions}/prob_grad_four.npy")[...,None].T
    grad_grad = np.load(f"{pathfunctions}/prob_grad_grad.npy")[...,None].T
    
    param_prob_grad = [grad_2,grad_4,grad_grad]
    
    return wage_0, param_wage, sigmas, param_prob_grad

def initial_state():

        # The state space is composed by: 
        # Sex,Black,Hispanic, AFQT,Parental Income
        
        #data1 = np.random.binomial(3, 0.5, (n,2)).reshape(n,2) +1  # AFQT, Parental Income
        #data2 = np.random.binomial(1, 0.5, (n,2)).reshape(n,2)  # sex, african american
        
        #results = np.hstack((data1,data2))
        
        # now initialize states usin the real data at period 1
        
        x1 = np.load(f"{path_realdata}/invariant_state_superfeasible_t1.npy")
        n = np.shape(x1)[0]
        
        return x1,n
    
def get_types(x1,q,conterfactual,cohort):
    
    """
    This function draws types for each individuals based on their observed
    probability of belonigng to each type.
    """
    
    if conterfactual == 0:
        q = validate_q(q, n_individuals=np.shape(x1)[0])
        types = draw_type_ids(q, np.random.random(np.shape(q)[0]))
        
        np.save(f"{OUT('types')}/types_{cohort}.npy", types)
    else:
        types = np.load(f"{OUT('types')}/types_{cohort}.npy").astype(np.int64)
        if len(types) != np.shape(x1)[0]:
            raise ValueError(
                f"Saved cohort {cohort} types contain {len(types)} rows; "
                f"expected {np.shape(x1)[0]}."
            )
    return types
    
    
    
def initialize_states(q,conterfactual,cohort):
    
    x1_initial,n = initial_state()
    types = get_types(x1_initial,q,conterfactual,cohort)
    x2_initial = np.array(([0,0,0,0,0,0,0,0,0,0],))
    x2_initial = np.repeat(x2_initial,np.shape(x1_initial)[0],axis=0)
    b2_initial = np.zeros((n,1))
    x = np.concatenate((x1_initial,x2_initial,b2_initial),axis=1).astype(np.int32)
    
    return x, types




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
    
def load_vjti(x1,x2,period,conter,types,maxdebt):
    
    """
    This function loads the corresponding vjt depnding on which conterfactual is running. 
    """
    
    if conter == 0:
    
        vjti = np.load(f"{OUT('vjt', str(period))}/vjt_t{period}_[{x1}]_em{types}.npz")[f"vjt_t{period}_[{x1}]_{x2}"]
        
    if conter == 1:
        
        vjti = np.load(f"{OUT('vjt_conter', str(period))}/vjt_t{period}_[{x1}]_em{types}_maxdebt{maxdebt}.npz")[f"vjt_t{period}_[{x1}]_{x2}"]
        
    if conter == 2:
        
        vjti = np.load(f"{OUT('vjt_conter_not', str(period))}/vjt_t{period}_[{x1}]_em{types}.npz")[f"vjt_t{period}_[{x1}]_{x2}"]
        
    return vjti

def get_expected_conditional_x(x,period,conter,types,maxdebt,uparams):
    
    # Loop over all individuals
    total_choices = get_total_choices()  # This should load the array with all possible choices
    payoff = np.zeros((np.shape(x)[0],np.shape(total_choices)[0])) # Total amount of possible choices
    # Generate column map
    column_map = np.repeat(np.nan,np.shape(x)[0]*np.shape(total_choices)[0]).reshape(np.shape(x)[0],np.shape(total_choices)[0])  # This array will hold how to map from columns in payoff, to the corresponding choice. As safety I will create it with nans
    
    gvalue = np.zeros((np.shape(x)[0],np.shape(total_choices)[0]))
    
    x1new = get_x1_new(x[:,1:5])
    
    for i in range(np.shape(x)[0]):
        xi = x[i,1:]   # get the first individual
    
        # First check which choices are feasible 
        
        x2 = xi[4:14]
        x1 = xi[:4]
        debt = xi[14]
        
        Jx  = get_possible_choices(x2)

        # Perform the map
        
        idx = np.where( (total_choices==Jx[:,None]).all(-1) )[1]  # This tells which index in tota_choices corresponds to each choice in Jx
        
        idx = np.concatenate((idx[...,np.newaxis],np.arange(0,np.shape(idx)[0]).reshape(np.shape(idx)[0],1)),axis=1)  # This just includes the index of Jx as a column in the array. 

        column_map[i,idx[:,0]] = idx[:,1]  # This performs the match to this individual of the corresponding mapping.
        
        # load the vjts        
        vjt_jx = load_vjti(x1,x2,period,conter,types[i],maxdebt)
        
        gfunctions = get_all_g(uparams[types[i]-1],x1,x1new[i,:],x2,Jx,period)
        
        column = 0  # This will track which choice is in the column. 
        for choice in range(np.shape(total_choices)[0]):
            if choice in list(idx[:,0]):
                payoff[i,column] = vjt_jx[debt,int(column_map[i,column])]  # Get the corresponding chioce with the corresponding debt level.
                gvalue[i,column] = gfunctions[0,int(column_map[i,column])]
            else:
                payoff[i,column] = np.nan
                gvalue[i,column] = np.nan
            
            column +=1 # Identify next column!
            
    return payoff, gvalue


def save_choice(choices,period,cohort,conter,maxdebt):
    
    """
    This function saves the simulatd choices depending on which conterfactual is
    """

    if conter == 0:
        np.save(f"{OUT('choice')}/choice_t{period}_s{cohort}.npy", choices)
    elif conter == 1:
        np.save(f"{OUT('choice')}/choice_conter_t{period}_s{cohort}_maxdebt{maxdebt}.npy", choices)
    elif conter == 2:
        np.save(f"{OUT('choice')}/choice_conter_not_t{period}_s{cohort}.npy", choices)
        
def save_welfare(w,wepsi,period,cohort,conter,maxdebt):

    if conter == 0:
        np.save(f"{OUT('welfare')}/w_t{period}_s{cohort}.npy", w)
        np.save(f"{OUT('welfare')}/wepsi_t{period}_s{cohort}.npy", wepsi)
    elif conter == 1:
        np.save(f"{OUT('welfare')}/w_conter_t{period}_s{cohort}_maxdebt{maxdebt}.npy", w)
        np.save(f"{OUT('welfare')}/wepsi_conter_t{period}_s{cohort}_maxdebt{maxdebt}.npy", wepsi)
    elif conter == 2:
        np.save(f"{OUT('welfare')}/w_conter_not_t{period}_s{cohort}.npy", w)
        np.save(f"{OUT('welfare')}/wepsi_conter_not_t{period}_s{cohort}.npy", wepsi)
        
def load_epsilons(x,cohort,period):
    
    colnames = ["id","e"]
    
    epsilons = pd.DataFrame(np.load(f"{OUT('epsilon')}/e_cohort{cohort}_period{period}.npy")).rename(columns={0:"id"})
        
    xdf = pd.DataFrame(x).rename(columns={0:"id"})
    
    xdf = pd.merge(xdf,epsilons,on="id")
    
    epsilons_ordered = np.array(xdf)[:,16:]
    
    return epsilons_ordered
    
    
    
    
def save_epsilons(x,e,period,cohort):
    
    """"This function will store epsilons considering the 
    PUBID to later match it to the right individual"""
    
    epsilons = np.concatenate((x[:,0][...,None],e),axis=1)
    
    np.save(f"{OUT('epsilon')}/e_cohort{cohort}_period{period}.npy", epsilons)    

def compare_alternatives(x,period,cohort,conterfactual,types,maxdebt,uparams,get_welfare=1):
    
    # Compute the payoff
    
    payoff, gvalue = get_expected_conditional_x(x,period,conterfactual,types,maxdebt,uparams)
    
    total_choices_size = np.shape(get_total_choices())[0]
    
    # Draw the epsilons
    
    if conterfactual == 0:
        epsilons = np.random.gumbel(loc=0.0, scale=1.0, size=(np.shape(x)[0],total_choices_size))
        save_epsilons(x,epsilons,period,cohort)
    else: 
        epsilons = load_epsilons(x,cohort,period)
        #epsilons = np.random.gumbel(loc=0.0, scale=1.0, size=(np.shape(x)[0],total_choices_size))
    # Sum both
    
    evjt = payoff + epsilons
        
    # Get the choice with the maximum utility value
    
    choices = np.nanargmax(evjt,axis=1)
    
    # Save it   
    save_choice(choices,period,cohort,conterfactual,maxdebt)
    
    # Get Welfare if necessary
    
    if get_welfare == 1:
        
        welfare = np.take_along_axis(gvalue, choices[...,None], axis=1)
        epsichoice = np.take_along_axis(epsilons, choices[...,None], axis=1)
        welfareepsi = welfare + epsichoice
        
        save_welfare(welfare,welfareepsi,period,cohort,conterfactual,maxdebt)
    
    return choices
        

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
    
    ability = x1[1] - 1
    
    exp = np.minimum(x2[2],5)
    
    x_exp[int(exp + ability*6)] = 1
    
    return x_exp

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
                x_afqt[int(x1[1]-1)] = 1
    
    if coltype == 3:
        
        if period > 5:
            if (x2[7] ==(period-1)) & (x2[9]==13):
                x_afqt = 1
            else:
                x_afqt = 0
        else:
            x_afqt = 0
            
       
    return x_afqt

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


def which_educ():
    
    total_choices = get_total_choices()
    
    total_choices = np.concatenate((total_choices,np.arange(0,np.shape(total_choices)[0]).reshape(np.shape(total_choices)[0],1)),axis=1) 
    
    educ_choices = total_choices[:,3][total_choices[:,1]!=0]
    
    not_educ_choices = total_choices[:,3][total_choices[:,1]==0]
    
    return educ_choices[...,None], not_educ_choices[...,None]


def map_debt_position(debt,debtnew):
    
    diff = (debt - debtnew[...,None])**2
    
    debt_position= np.argmin(diff,axis=1)
    
    return debt_position


def move_debt_repayment(x1,x2,period,debt,j,conter):
    
    global debt_range
    
    x1_new = get_x1_new(x1)
    
    total_choices = get_total_choices()
    
    # Get the wage and psi for each individual
    
    real_wage = np.zeros(np.shape(x1)[0])
    
    periods = np.zeros(np.shape(x1)[0])
    
    for i in range(np.shape(x1)[0]):
        
        choice = total_choices[j[i],:]
        
        paramwage = get_params_wage(choice)
        
        if choice[0] > 0 :
    
            w = wage(x1_new[i,:],x2[i,:],choice,paramwage)
            
        else: 
            
            w = 0
        
        if choice[0] <4:
        
            sigma = sigmas[choice[0]]
        else:
            sigma = sigmas[choice[0]-2]
            
        psi = np.random.normal(loc=mu, scale=sigma, size=1)
        
        real_wage[i] = np.exp((w + psi))*choice[2]*1/2*(40*52)
    
        periods[i] = 10 + x2[i,7] + 1 - period
        
    if conter == 0 :
        
        paid = np.maximum(0,np.minimum(real_wage-CONSUMPTION_FLOOR,(1/periods)*debt_range[debt]*(1+r)))
    
        debtnew = (1+r)*debt_range[debt] - paid
        
    elif conter == 1:
        
        discretionary_income = real_wage-2.25*15000
        
        paid = np.maximum(0,np.minimum(real_wage-CONSUMPTION_FLOOR,np.minimum(0.05*discretionary_income,(1/periods)*debt_range[debt]*(1+r))))
                
        debtnew = (1+r)*debt_range[debt] - paid
        
    debtnew = map_debt_position(debt_range, debtnew)
        
    return debtnew
        

        

def move_states_and_debt(sigma_u,x,choices,period,conterfactual,types,maxdebt,cohort):
    
    # First identify individuals that are making educational choices
    educ_choices,not_educ_choices = which_educ()
    
    # Break the set of individuals into those who make educational choices and
    # those who do not.  ``choices`` contains indices into ``get_total_choices``.
    # Keep one common row map for states, choices, and permanent types.
    education_rows = np.flatnonzero(np.isin(choices, educ_choices.reshape(-1)))
    noneducation_rows = np.flatnonzero(
        np.isin(choices, not_educ_choices.reshape(-1))
    )
    x_educ = x[education_rows]
    x_noteduc = x[noneducation_rows]
    choices_educ = choices[education_rows]
    choices_noteduc = choices[noneducation_rows]
    types_educ = types[education_rows].astype(np.int64)
    types_noteduc = types[noneducation_rows].astype(np.int64)
    total_choices = get_total_choices()
    choices_educ_original = total_choices[choices_educ]

    # Draw distinct shocks in their correct units. The wage shock is in log
    # wages; the fitted budget shock is additive in dollar consumption.
    if budget_params is None:
        reload_budget_shock_params(raise_if_missing=True)
    if auxiliary_financial_process is None:
        reload_auxiliary_financial_process()
    n_educ = np.shape(x_educ)[0]
    wage_psi_educ = np.random.normal(loc=0.0, scale=sigmas[0], size=n_educ)
    x1_educ = x_educ[:, 1:5]
    x2_educ = x_educ[:, 5:15]
    x1_new_educ = get_x1_new(x1_educ)
    grant_types = TYPE_GRANT[types_educ - 1]
    transfer_types = TYPE_TRANSFER[types_educ - 1]
    loan_types = TYPE_LOAN[types_educ - 1]

    # Realize the grant and parental-transfer processes estimated inside the
    # auxiliary EM.  These are the same typed parameter blocks used by the
    # Bellman solver; supplied draws keep them under the cohort's simulation
    # seed instead of creating independent random generators internally.
    grants = draw_grants_vectorized(
        x1_new_educ,
        choices_educ_original[:, 1],
        choices_educ_original[:, 2],
        auxiliary_financial_process["grant"],
        grant_type=grant_types,
        receipt_uniform=np.random.random(n_educ),
        amount_standard_normal=np.random.standard_normal(n_educ),
    )
    transfers = draw_transfers_vectorized(
        x1_new_educ,
        choices_educ_original[:, 1],
        choices_educ_original[:, 2],
        auxiliary_financial_process["transfer"],
        transfer_type=transfer_types,
        receipt_uniform=np.random.random(n_educ),
        amount_standard_normal=np.random.standard_normal(n_educ),
    )
    realized_help = grants + transfers
    wage_index = wage0(x1_new_educ, x2_educ)
    realized_wage = (
        np.exp(wage_index + wage_psi_educ)
        * choices_educ_original[:, 2] * 0.5 * (40 * 52)
    )
    pre_choice_resources = (
        realized_help + realized_wage
        - tuition_agents(conterfactual, choices_educ_original)
        - (1.0 + r) * debt_range[x_educ[:, 15].astype(np.int64)]
    )
    budget_psi_educ = np.empty(n_educ, dtype=np.float64)
    budget_standard_draw = np.random.standard_normal(n_educ)
    for education in (1, 2, 3):
        education_rows = np.flatnonzero(
            choices_educ_original[:, 1].astype(np.int64) == education
        )
        if not education_rows.size:
            continue
        cell_codes = bs.budget_education_cell_from_state(
            x2_educ[education_rows], education
        )
        for cell_code in np.unique(cell_codes):
            rows = education_rows[cell_codes == cell_code]
            program_year = int(cell_code - 100 * education)
            budget_psi_educ[rows] = bs.realization(
                budget_params,
                x1_educ[rows],
                period,
                budget_standard_draw[rows],
                loan_type=loan_types[rows],
                education=education,
                program_year=program_year,
                pre_choice_resources=pre_choice_resources[rows],
            )
    sigma_educ = bs.risk_aversion(
        budget_params, x_educ[:, 1:5], loan_type=loan_types
    )
    
    # Move debt for not education choices.
    # The individuals that are not making educational choices will have tomorrow debt based on the repayment rule.
    # For now it is independent on income but I could make it dependent. 
    
    debt_noteduc = move_debt_repayment(x_noteduc[:,1:5],x_noteduc[:,5:15],period,x_noteduc[:,15],choices_noteduc,conterfactual)

    
    # The individuals that are making educational choices will endogenize the choice of debt given a shock:
    # Notice here hta the choices are only for the subset of individuals that make education chioces
    
    payoff = get_conditional_agents(sigma_educ,x_educ[:,1:5],x_educ[:,5:15],x_educ[:,15],
                                    realized_help,budget_psi_educ,wage_psi_educ,
                                    choices_educ,period,conterfactual,
                                    types_educ,maxdebt)
    debt_educ = np.nanargmax(payoff,axis=1)

    
    # Put the individuals back together:
        
    
    debt_new = np.concatenate((debt_noteduc,debt_educ))
    x_together = np.concatenate((x_noteduc,x_educ))
    choices_together = np.concatenate((choices_noteduc,choices_educ))
    
    # Notice all the mess with choices together etc is just to make sure that he index is preserved
    # across individuals,and that the first choice corresponds to the firts individual and so on...!
    
    # Compute the new state
    
    # Before that, map choices back to its original form.
    
    choices_original = total_choices[choices_together,:]
    
            
    x_new = np.concatenate((x_together[:,:5],move_state_agents(x_together[:,:5],x_together[:,5:15],choices_original,period,conterfactual,cohort),debt_new[...,None]),axis=1).astype(np.int32)
    
    # Also move types since the individuals have now changed
    
    types = np.concatenate((types_noteduc,types_educ))
    
    return x_new, types
    
def move_types(types,choices,educ_choices,not_educ_choices):
    types_educ = types[np.isin(choices, educ_choices.reshape(-1))]
    types_noteduc = types[np.isin(choices, not_educ_choices.reshape(-1))]
    typesnew = np.concatenate((types_noteduc,types_educ))
    return typesnew


def simulate_choices(cohort,sigma_u,conterfactual,q,maxdebt,uparams):
    
    """This function simulates choices for all the periods for n agents. The function
    is meant to be paralallelized and simulates different cohorts. The input 
    corresponds to the current cohort(sample) being simulated. 
    
    Conterfactuals are: 
        0 -- Real Model
        1 -- Income Driven Plan
        2 -- No Tuition
        
    Q: EM weights that I will use to draw types    
    
    """
    np.random.seed(seed=cohort)
    reload_budget_shock_params(raise_if_missing=True)
    reload_auxiliary_financial_process()
    if len(uparams) != validate_q(q).shape[1]:
        raise ValueError(
            "uparams must contain one utility-parameter block per joint type."
        )
    # initialize states
    
    x, types = initialize_states(q,conterfactual,cohort)

    
    # loop over periods
    
    for period in range(1,T,1):
        print("Current period is", period)
        save_state(x,period,cohort,conterfactual,maxdebt)
        choices = compare_alternatives(x,period,cohort,conterfactual,types,maxdebt,uparams)
        x, types = move_states_and_debt(sigma_u,x,choices,period,conterfactual,types,maxdebt,cohort)
        if period == T-1:
            save_state(x,period+1,cohort,conterfactual,maxdebt)

def save_state(x,period,cohort,conterfactual,maxdebt):
    
    if conterfactual == 0 :
        np.save(f"{OUT('state')}/state_t{period}_s{cohort}.npy", x)
    elif conterfactual == 1 :
        np.save(f"{OUT('state')}/state_conter_t{period}_s{cohort}_maxdebt{maxdebt}.npy", x)
    elif conterfactual == 2 :
        np.save(f"{OUT('state')}/state_conter_not_t{period}_s{cohort}.npy", x)
        
        
def get_debt_range():
    
    debtrange1 = np.array([0,300,500,620,770,950])
    debtrange2 = np.linspace(1166,3500,16)
    debtrange3 = np.linspace(3720,8800,25)
    debtrange4 = np.linspace(9200,20000,25)
    debtrange5 = np.linspace(22700,100000,28)

    debt_range = np.concatenate((debtrange1,debtrange2,debtrange3,debtrange4,debtrange5))
    
    return debt_range
    
def get_conditional_agents(sigma_u,x1,x2,b,financial_help,budget_psi,wage_psi,j,period,conterfactual,types,maxdebt):
    
    "this function returns the conditional value function associated with each alternative"
    
    
    b1 = get_debt_range()
    
    # Map choice back to tuple
    
    total_choices = get_total_choices()
    
    j = total_choices[j,:]
            
    # Forward the permanent latent loan type so the utility can charge the
    # loan-type-specific new-borrowing event cost (kappa).
    loan_types = TYPE_LOAN[np.asarray(types, dtype=np.int64) - 1]

    u = get_utility_agents(
        sigma_u, x1, x2, b, b1, financial_help, budget_psi,
        wage_psi, j, period, conterfactual, maxdebt,
        loan_types=loan_types,
    )

    continuation = beta*VT_agents(x1,x2,b1,period,j,conterfactual,types,maxdebt)
        
    vjt = u + continuation

    return vjt


@njit()
def wage0(x1_new,x2_new):
    
    """ Different wage fucntion for numba"""
    
    param_wage = wage_0

    # This part puts together x1 and x2. x1 is time invariant, x2 is time variant. 
    x = np.hstack((x1_new,x2_new[:,:-1])) # do not include grad school coefficient
    
    w = x@param_wage.T
    return w

#@njit()
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




#@njit()
def move_state_agents(x1,x2,j,period,conter,cohort):
    
    "this function moves the state space given your current state space and choice"
    
    z = np.copy(x2)
    x1_new = get_x1_new(x1[:,1:])
    
    if conter == 0:  # I need to store graduation probabilities
    
        gradall = np.zeros((np.shape(x1)[0],3)) # PUBID, field,  grad
        
    else: 
        
        gradall = np.load(f"{OUT('grad_prob')}/gradall_c{cohort}_t{period}.npy")
    
    # Loop over all agents. For now I will do it brute force.
    
    for i in range(np.shape(x2)[0]):
        
        # First, check if the chioce could induce graduation! 
        
        if ((z[i,1] >=1) & (j[i,1] == 1) & (z[i,4] == 0)) | ((z[i,2]>=3) & (j[i,1] == 2) & (z[i,5] == 0)) | (j[i,1]==3)  :
            
            # It is the same as the other case, but it could get graduated with some probability
            
            if j[i,1] == 1:  # two-year choice
            
                z[i,1] = z[i,1] + 1  # plus one experiemce
                
                # Compute graduation probability
                
                if conter == 0 :
                
                    grad_prob = probability_graduation(x1_new[i,:],x2[i,:],j[i,:])
                    
                    # Draw gradaution with some probability
                    grad = np.random.binomial(1,grad_prob,1)[0]
                    
                    # Save the information
                    
                    gradall[i,:] = np.array([x1[i,0],j[i,0],grad])
                    
                else: 
                    
                    grad_temp = gradall[gradall[:,0]== x1[i,0]]
                    
                    #if (grad_temp[0,2]!=99) & (grad_temp[0,1] == j[i,0]):
                    if grad_temp[0,2]!=99:
                        
                        grad = grad_temp[0,2]
                        
                    else:
                        
                        grad_prob = probability_graduation(x1_new[i,:],x2[i,:],j[i,:])
                        
                        # Draw gradaution with some probability
                        grad = np.random.binomial(1,grad_prob,1)[0]
                                 

                # Now if the graduation is true, change field at graduation
                
                z[i,7] = period # last period enrolled
                
                z[i,9] = j[i,0]  # track previous choice
                
                if grad == 1:
                    
                    # Input graduation
                    
                    z[i,4] = grad
                    
                    # Set mejor at graduation:
                        
                    z[i,8] = j[i,0]
                                       
                    # Chnge experience
                    
                    z[i,1] = 99
            
            elif j[i,1] == 2: # four year choice
            
                z[i,2] = z[i,2] + 1
                
                # Compute graduation probability
                
                if conter == 0:
                
                    grad_prob = probability_graduation(x1_new[i,:],x2[i,:],j[i,:])
                    
                    # Draw gradaution with some probability
                    
                    grad = np.random.binomial(1,grad_prob,1)[0]
                    
                    # Save the information
                    
                    gradall[i,:] = np.array([x1[i,0],j[i,0],grad])
                    
                else: 
                    
                    grad_temp = gradall[gradall[:,0]== x1[i,0]]
                    
                    if grad_temp[0,2]!=99:
                        
                        grad = grad_temp[0,2]
                        
                    else:
                        
                        grad_prob = probability_graduation(x1_new[i,:],x2[i,:],j[i,:])
                        
                        # Draw gradaution with some probability
                        grad = np.random.binomial(1,grad_prob,1)[0]
                    
                    
                
                # Now if the graduation is true, change field at graduation
                
                z[i,7] = period # last period enrolled
                
                z[i,9] = j[i,0]  # track previous choice
                
                if grad == 1:
                    
                    # Input graduation
                    
                    z[i,5] = grad
                    
                    # Set mejor at graduation:
                        
                    z[i,8] = j[i,0]
                    
                    #Change experience
                    
                    z[i,2] = 99
                    z[i,1] = 99
                    
            elif j[i,1] == 3:  # grad choice
            
                z[i,3] = z[i,3] + 1
                
                # Drw graduaiton probability
                
                if conter == 0:
                
                    grad_prob = probability_graduation(x1_new[i,:],x2[i,:],j[i,:])
                    
                    grad = np.random.binomial(1,grad_prob,1)[0]
                    
                    # Save the information
                    
                    gradall[i,:] = np.array([x1[i,0],j[i,0],grad])
                    
                else: 
                    
                    grad_temp = gradall[gradall[:,0]== x1[i,0]]
                    
                    if grad_temp[0,2]!=99:
                        
                        grad = grad_temp[0,2]
                        
                    else:
                        
                        grad_prob = probability_graduation(x1_new[i,:],x2[i,:],j[i,:])
                        
                        # Draw gradaution with some probability
                        grad = np.random.binomial(1,grad_prob,1)[0]
                    
                
                z[i,7] = period
                
                z[i,9] = j[i,0]
                
                if grad == 1:
                    
                    z[i,6] = 1
                    
                    z[i,3] = 99
                    
            elif (j[i,1] == 0) & (j[i,2] >= 1): # work full or partime
            
                z[i,0] = z[i,0] + 1
                
                z[i,9] = j[i,0]  # track previous choice
                

            
        else:
            
            # Save the information
            
            if conter == 0:
            
                gradall[i,:] = np.array([x1[i,0],99,99])
        
            if j[i,1] == 1:  # two-year choice
            
                z[i,1] = z[i,1] + 1
                
                z[i,7] = period # last period enrolled
                
                z[i,9] = j[i,0]  # track previous choice
            
            elif j[i,1] == 2: # four year choice
            
                z[i,2] = z[i,2] + 1
                
                z[i,7] = period # last period enrolled
                
                z[i,9] = j[i,0]  # track previous choice
            
            elif (j[i,1] == 0) & (j[i,2] >= 1): # work full or partime 
            
                z[i,0] = z[i,0] + 1
                
                z[i,9] = j[i,0]  # track previous choice
                
            elif (j[i,1] == 0) & (j[i,2] == 0): # home production
            
                z[i,9] = 0  # track previous choice
     
            
    np.save(f"{OUT('grad_prob')}/gradall_c{cohort}_t{period}.npy", gradall)
    
    return z

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


def VT_agents(x1,x2,b1,period,choices,conter,types,maxdebt):

    " This function returns the final continuation value for each individual"
    x1 = np.array(x1,dtype="int")
    x2 = np.array(x2,dtype="int")
    x1_new = get_x1_new(x1)
    vt = np.zeros((np.shape(x2)[0],np.shape(b1)[0])) 
    for space in range(np.shape(x2)[0]):
        x2i = x2[space,:]
        x1i = x1[space,:]
        ji = choices[space,:]

        if conter == 0:
            evt = np.load(f"{OUT('evt', str(period+1))}/evt_t{period+1}_[{x1i}]_em{types[space]}.npz")
        elif conter == 1: 
            evt = np.load(f"{OUT('evt_conter', str(period+1))}/evt_t{period+1}_[{x1i}]_em{types[space]}_maxdebt{maxdebt}.npz")
        # First check if the choice and state include a possible graduation state. 
        if ((x2i[1] >=1) & (ji[1] == 1) & (x2i[4] == 0)) | ((x2i[2]>=3) & (ji[1] == 2) & (x2i[5] == 0)) | (ji[1]==3):
            # The choice could induce a graduation state. For this reason, take the expectation.
            grad_x2  =  move_state_grad(x2i,ji,period,grad=1)
            notgrad_x2 = move_state_grad(x2i,ji,period)
            #evt = np.load(f"evt/evt_t{period+1}_{x1}.npz")
            evt_grad =  evt[f"evt_t{period+1}_[{x1i}]_{grad_x2}"]
            evt_nograd = evt[f"evt_t{period+1}_[{x1i}]_{notgrad_x2}"]
            p_grad = probability_graduation(x1_new[space,:],x2[space,:],ji)
            evt_new = p_grad*evt_grad + (1-p_grad)*evt_nograd
            vt[space,:] = evt_new[:,0]
        else: 
            # load the data
            x2_next = move_state_grad(x2i,ji,period)
            evt_new = evt[f"evt_t{period+1}_[{x1i}]_{x2_next}"]
            # Now for each individual take the one corresponding with its tmr level of debt
            # Notice that if on the future the repayment scheme changes for different states
            # I should correct for that here.
            vt[space,:] = evt_new[:,0]

    return vt
    
@njit()
def check_consumption(c):
    
    rows,columns = np.shape(c)
    
    for col in range(columns):
        for row in range(rows):
            if c[row,col] < CONSUMPTION_FLOOR:
                c[row,col] = CONSUMPTION_FLOOR
    return c


def get_x2_new_simulation(x2,fields=2):
    """This function expands the x2 variables into dummies"""
    # exp,fields (first 4y then twoyear) + grad
    
    x2_new = np.zeros((np.shape(x2)[0],1+fields+1+1))  
    for i in range(np.shape(x2)[0]):
        x2i = x2[i,:]   
        x2_new[i,:] = get_x2_new(x2i)
        
    return x2_new




def get_params_wage(j):
    
    """ This function returns the parameters of the wage equations
    depending on the choice"""
    if j[0] < 4:
        param_wage = params_wage[j[0]-1]
    else:
        param_wage = params_wage[j[0]-3]
        
    return param_wage

#@njit()
def wage(x1_new,x2_new,j,param_wage):
    
    "This function returs the wage based on the current state"

        
    x2_wage = get_x2_wage(x2_new,j)
        
    # This part puts together x1 and x2. x1 is time invariant, x2 is time variant. 
    x = np.hstack((x1_new[...,None].T,x2_wage[...,None].T))
    
    w = x@param_wage.T
    return w


#@njit()
def get_utility_agents(sigma_u,x1,x2,b,b1,financial_help,budget_psi,wage_psi,j,period,conterfactual,maxdebt,loan_types=None):
    
    " this function returns the utility associated with a particular alternative"
    
    global debt_range
    
    r  = 0.05
    # Here I am making sure all the dimensions are correct, so that the final product is
    # (x2 X b) x b . This will allow to vectorize and obtain results without loops. 
    # Get expanded version of x1
    
    x1_new = get_x1_new(x1)
    h = np.asarray(financial_help, dtype=np.float64)
    
    full_part_time = j[:,2].copy()
    
    w = wage0(x1_new,x2)
    
    wage_shock = np.exp((w + wage_psi[...,None]))*full_part_time[...,None]*1/2*(40*52)
    
    c = (h + wage_shock[:,0] + budget_psi
         -(1+r)*debt_range[np.array(b,dtype="int")]-tuition_agents(conterfactual,j))
    
    c =c[...,None]
    c = np.repeat(c,np.shape(b1)[0],axis=1)
    b2 = b1[...,None]
    b2 = np.repeat(b2.T,np.shape(c)[0],axis=0)
    c  = c + b2
    #c = check_consumption(c)

    sigma_rows = np.asarray(sigma_u, dtype=np.float64)
    if sigma_rows.ndim == 0:
        sigma_rows = np.full(c.shape[0], float(sigma_rows))
    sigma_matrix = sigma_rows[:, None]
    # Choices below the consumption floor are removed by ``get_debt_rules``.
    # Evaluate them at the floor here so the forced maximum-debt fallback has
    # a finite payoff even if raw consumption remains non-positive.
    scaled_c = 0.00001 * np.maximum(c, CONSUMPTION_FLOOR)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(
            np.abs(sigma_matrix - 1.0) < 1e-8,
            0.1 * np.log(scaled_c),
            0.1 * scaled_c ** (1.0 - sigma_matrix) / (1.0 - sigma_matrix),
        )

    # ------------------------------------------------------------------
    # Debt-penalty convention (module switch SIM_DEBT_PENALTY_CONVENTION).
    #
    # "legacy_multiplier" (default, current behavior): match the SMM
    # shortcut, in which debt persists along the home-reference path, so
    # the full discounted stream of future flow penalties through explicit
    # period T-1 is charged here in flow utility via
    # ``explicit_debt_penalty_multiplier``. In the SMM shortcut that
    # multiplier is paired with a penalty-free continuation, so the
    # accounting is internally consistent THERE. THIS simulator, however,
    # adds ``beta * VT_agents(...)``, which loads the solved Bellman EVT —
    # and the Bellman recursion already charges one flow penalty in every
    # future explicit period (see ``model_solution_em.get_conditional``:
    # candidate ``b1 > 0`` on school paths, current debt stock on
    # non-school paths). Combining the multiplier with the Bellman
    # continuation therefore double-counts every future period's penalty
    # in the forward debt choice.
    #
    # "single_flow": charge the per-period flow penalty exactly once
    # (multiplier = 1), attached to the candidate next-period debt
    # ``b1 > 0`` — the same object ``model_solution_em.get_conditional``
    # penalizes on this education/debt-choice path. This matches the
    # Bellman convention and removes the double count. It awaits
    # researcher confirmation and is NOT the default; the default remains
    # numerically identical to the pre-switch code.
    # ------------------------------------------------------------------
    if SIM_DEBT_PENALTY_CONVENTION == "legacy_multiplier":
        penalty_multiplier = bs.explicit_debt_penalty_multiplier(
            period, beta=beta, terminal_period=T
        )
    elif SIM_DEBT_PENALTY_CONVENTION == "single_flow":
        penalty_multiplier = 1.0
    else:
        raise ValueError(
            "SIM_DEBT_PENALTY_CONVENTION must be 'legacy_multiplier' or "
            f"'single_flow'; received {SIM_DEBT_PENALTY_CONVENTION!r}"
        )
    # Loan-type debt-penalty shift (Spec B, heterogeneous debt aversion):
    # added per agent to the parental-income penalty for the debt-averse
    # latent loan type (loan type 0, the low-borrowing type; see
    # budget_shock.DEBT_PENALTY_SHIFT_LOAN_TYPE). Guarded so that with a
    # zero (or absent) shift, or without loan types, the penalty is exactly
    # the current ``bs.debt_penalty`` output.
    if (
        loan_types is not None
        and float(budget_params.get("debt_penalty_loan_type_shift", 0.0))
        != 0.0
    ):
        base_penalty = bs.debt_penalty_by_loan_type(
            budget_params, x1, loan_types
        )
    else:
        base_penalty = bs.debt_penalty(budget_params, x1)
    penalty = (base_penalty * penalty_multiplier)[:, None]
    u = u + penalty * (b1[None, :] > 0.0)

    # One-shot new-borrowing event cost (kappa). Charged only in the period
    # in which the chosen next-period debt exceeds the accrued current debt,
    # i.e. the individual takes out a new loan: entry cost kappa0[loan_type]
    # when current debt is zero, continuation cost kappa1 when it is
    # positive. No discounting multiplier — one event, one charge (see
    # budget_shock.NEW_BORROWING_COST_TIMING). Skipped entirely when both
    # kappas are zero so default behavior is numerically unchanged.
    kappa_entry, kappa_continuation = bs.new_borrowing_cost_parameters(
        budget_params
    )
    if np.any(kappa_entry != 0.0) or kappa_continuation != 0.0:
        current_debt = debt_range[np.array(b, dtype="int")]
        new_borrowing = b1[None, :] > (1 + r) * current_debt[:, None]
        event_cost = bs.new_borrowing_cost(
            budget_params, current_debt > 0.0, loan_type=loan_types
        )
        u = u + np.asarray(event_cost, dtype=np.float64)[:, None] * new_borrowing

    # Incorporate debt  rules! 
    
    u = get_debt_rules(c, u, b, x2, j, maxdebt)

    return u

@njit()
def minimum_debt(u,b):
    
    for i in range(np.shape(b)[0]):
        
        u[i,:b[i]] = -100000
        
    return u


@njit()
def minimum_debt_maxdebt(u,b):
    
    for i in range(np.shape(b)[0]):
        
        u[i,:b[i]] = -100000
        u[i,76:] = -100000
        if b[i] >= 76:  # at least you need to be able to get the current debt level
            u[i,b[i]] = 0 
        
    return u



def get_debt_rules(c, u, b, x2, j, maxdebt):
    """
    Impose debt-choice feasibility directly on the already-computed utility
    matrix for students in the simulation.

    Inputs
    ------
    c : (n, B) float array
        Consumption associated with each individual and each debt-grid choice.
    u : (n, B) float array
        Utility associated with c before debt-choice restrictions are imposed.
    b : (n,) int array
        Current debt positions on the debt grid.
    x2 : (n, K) int array
        Current time-varying state variables.
    j : (n, 3) int array
        Current choices in original tuple form.
    maxdebt : bool
        Whether to apply the borrowing-cap rules.

    Returns
    -------
    u : (n, B) float array
        Utility matrix after debt feasibility rules are imposed.

    Rules imposed
    -------------
    1. Choices outside the admissible debt interval are assigned a very negative
       value so they cannot be selected by np.argmax().

    2. In the active borrowing region:
         - admissible debt choices are those between lo_idx and hi_idx
         - among those, choices with c < 2000 are assigned a very negative value
         - if none of the admissible choices satisfies c >= 2000, then only the
           maximum admissible debt choice remains available

    3. In the capped region:
         - there is no new borrowing choice
         - only the mechanically evolved debt level remains available

    This function modifies the utility matrix so that later maximization
    selects the correct debt choice automatically.
    """
    BIGNEG = -100000.0

    # Without borrowing caps, the only restrictions are:
    #   - consumption must satisfy the floor
    #   - tomorrow debt cannot be below today's debt index
    if maxdebt == False:
        u[c < CONSUMPTION_FLOOR] = BIGNEG
        u = minimum_debt(u, b)
        return u

    lo_idx, hi_idx, cap_region = get_simulation_bounds_indices(
        b.astype(np.int64),
        x2.astype(np.int64),
        j.astype(np.int64),
        debt_range
    )

    n, B = u.shape

    for i in range(n):
        lo = int(lo_idx[i])
        hi = int(hi_idx[i])

        # Keep the already-computed utility at the fallback points.
        u_lo = u[i, lo]
        u_hi = u[i, hi]

        # Eliminate all debt choices outside the admissible interval.
        u[i, :lo] = BIGNEG
        u[i, hi+1:] = BIGNEG

        # Capped region:
        # only the mechanically evolved debt level is admissible.
        if cap_region[i] == 1:
            u[i, :] = BIGNEG
            u[i, lo] = u_lo
            continue

        # Active borrowing region:
        # keep admissible choices that satisfy the consumption floor.
        found_feasible = False

        for k in range(lo, hi + 1):
            if c[i, k] >= CONSUMPTION_FLOOR:
                found_feasible = True
            else:
                u[i, k] = BIGNEG

        # If no admissible choice satisfies the consumption floor,
        # force the maximum admissible debt choice for the current period.
        if not found_feasible:
            u[i, :] = BIGNEG
            u[i, hi] = u_hi

    return u



#@njit()
def get_x1_new(x1):
    
    x1_new = np.ones(shape=(np.shape(x1)[0],9))
    for i in range(np.shape(x1)[0]):
        x1a = x1[i,:]
        parinc  = np.array([np.concatenate((np.zeros(int(x1a[0]-1)),np.array([1]),np.zeros(int(4-x1a[0]))),axis=0)],)
        ability = np.array([np.concatenate((np.zeros(int(x1a[1]-1)),np.array([1]),np.zeros(int(4-x1a[1]))),axis=0)],)
    
        x1_new[i,1:] = np.array([np.concatenate((parinc[0,1:],ability[0,1:],x1[i,2:]))],)
        
    return x1_new
    

#@njit()
def fin_help_agents(x1_new,x2,j,period,types):
    
    "this function returns the amount of financial help individuals will recieve"

    types = np.asarray(types, dtype=np.int64)
    grant_types = TYPE_GRANT[types - 1]
    transfer_types = TYPE_TRANSFER[types - 1]
    grants = expected_grants_vectorized(
        x1_new, j[:, 1], j[:, 2], auxiliary_financial_process["grant"],
        grant_type=grant_types,
    )
    transfers = expected_transfers_vectorized(
        x1_new, j[:, 1], j[:, 2], auxiliary_financial_process["transfer"],
        transfer_type=transfer_types,
    )
    h = transfers + grants

    return h

def tuition_agents(conterfactual,j):
    t = np.copy(j[:,1])
    if conterfactual == 2:
        t[:] = 0
        
    else:
        
        t[t==3] = 14000
        t[t==2] = 8000
        t[t==1] = 4000
    
    return t


#param_g = temp_param_g()
wage_0, params_wage, sigmas, param_prob_grad = load_all_parameters()
debt_range = get_debt_range()
budget_params = bs.load(raise_if_missing=False)
auxiliary_financial_process = None

def reload_budget_shock_params(raise_if_missing=True):
    """Reload the shared specification after re-estimating it in this process."""
    global budget_params
    budget_params = bs.load(raise_if_missing=raise_if_missing)
    return budget_params


def reload_auxiliary_financial_process():
    """Reload the typed grant/transfer process saved by the auxiliary EM."""
    global auxiliary_financial_process
    auxiliary_financial_process = load_auxiliary_financial_process(
        EST("auxiliary_em_results.npz")
    )
    return auxiliary_financial_process

# PROBLEMS I HAVE IDENETIFIED

# In the toymodel, debt level is tracked as the argmax of the optimal
# which means that is the position of the debt vector. This is only the case
# for individuals that study and hence get indebted. If you do a repayment plan,
# debt is now the actual value of the matrix. This inconsistency did not arise in
# an error since repayment was setting debt value to 0, which implies that
# it is the same value wether it is the argmax or the vector. I need to account for
# that now. 

