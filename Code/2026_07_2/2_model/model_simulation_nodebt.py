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
os.chdir(r"C:/Users/Sergi/Project/Real")
mu = 0
gamma = 0.57721
beta = 0.98
r = 0.05
T = 10 




# Parameters for the wage equations:
path = "C:/Users/Sergi/Dropbox/PhD/Projects/Papers/1_financial_constraints/Code/2024_09/2_model/output"
pathfunctions = "C:/Users/Sergi/Dropbox/PhD/Projects/Papers/1_financial_constraints/Model"

#---------------------------------------------------#
# Now simulate the data

# All agents start at the same period, but they are heterogeneous in 
# the invariant state space. I will start with 100 agents and then increase. 

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
    
    # Grants Probability
    
    pgrants_2 = np.load(f"{pathfunctions}/function_coefficients/prob_grants_twoyear.npy")[...,None].T
    pgrants_4 = np.load(f"{pathfunctions}/function_coefficients/prob_grants_fouryear.npy")[...,None].T
    
    param_prob_grants = np.concatenate((pgrants_2,pgrants_4),axis=0)
    
    # Parental Transfers
    
    param_fam = np.load(f"{pathfunctions}/function_coefficients/parental_transfers.npy")
    
    # Graduation Probability
    
    grad_2 = np.load(f"{pathfunctions}/function_coefficients/prob_grad_twoyear.npy")[...,None].T
    grad_4 = np.load(f"{pathfunctions}/function_coefficients/prob_grad_four.npy")[...,None].T
    grad_grad = np.load(f"{pathfunctions}/function_coefficients/prob_grad_grad.npy")[...,None].T
    
    param_prob_grad = [grad_2,grad_4,grad_grad]
    
    return wage_0, param_wage,sigmas,param_grants,param_prob_grants,param_fam,param_prob_grad

def initial_state():

        # The state space is composed by: 
        # Sex,Black,Hispanic, AFQT,Parental Income
        
        #data1 = np.random.binomial(3, 0.5, (n,2)).reshape(n,2) +1  # AFQT, Parental Income
        #data2 = np.random.binomial(1, 0.5, (n,2)).reshape(n,2)  # sex, african american
        
        #results = np.hstack((data1,data2))
        
        # now initialize states usin the real data at period 1
        
        x1 = np.load(f"{pathfunctions}/real_data/invariant_state_superfeasible_t1.npy") # no pubid
        n = np.shape(x1)[0]
        
        return x1,n
    
def get_types(x1,q,cohort):
    
    """
    This function draws types for each individuals based on their observed
    probability of belonigng to each type.
    """
    
    #types  = np.random.binomial(1, q[:,1], np.shape(q)[0]) +1
    types = np.load(f"{OUT('types')}/types_{cohort}.npy")
    return types
    
    
    
def initialize_states(q,cohort):
    
    x1_initial,n = initial_state()
    types = get_types(x1_initial,q,cohort)
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
    

    
    vjti = np.load(f"{pathout}/vjt_nodebt/{period}/vjt_t{period}_[{x1}]_em{types}_conter{conter}.npz")[f"vjt_t{period}_[{x1}]_{x2}"]
        

    
    return vjti


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

def fin_help(x1_new,x2,j,period):
    
    "this function returns the amount of financial help individuals will recieve"

    # help = b0 + x1_new +  4ydummy
    if j[1] == 1:
        x = np.append(x1_new,0)  # include 4ydummy
    elif j[1] == 2:
        x = np.append(x1_new,1) # include 4ydummy
    elif j[1] == 3: # For now assume is the same for simplicity
        x = np.append(x1_new,1) # include 4ydummy

    p_trans = x@param_fam
    grants =  x@param_grants
    h = np.exp(p_trans) + np.exp(grants)
    
    return h

def get_quadrature(deg,mu,j):
    
    ''' This function computes the range and weight of the Gauss-Hermite 
    quadrature, for a normal distribution. 
    
    deg -> number of points in the range (degree).
    mu -> mean of the variable to integrate.
    sigma -> st. deviation of the variable to integrate. 
    
    '''
    
    sigma = get_sigma(j,sigmas)  # get the variance of this choice
    
    [x,w] = scipy.special.roots_hermite(deg, mu=False)
    w = w*1/np.sqrt(np.pi)
    y = np.sqrt(2)*sigma*x+mu
    
    return y,w

@njit()
def tuition(j):
    
    if j[1] == 1:
        return 4000
    elif j[1] == 2:
        return 8000
    elif j[1] == 3:
        return 14000

def check_affordable(x1_new,x2,j,deg):
        
    e,weight = get_quadrature(deg, 0, j)
    # Get Financial Help and Tution
    
    h = fin_help(x1_new,x2,j,period=1)  # period does not matter!
    h_vis = np.repeat(h,np.shape(e)[0])
    t = tuition(j)
    
    # Get wage                          
    w = wage0_i(x1_new,x2)
    real_wage = np.exp(w+e)*(j[2]/2)*52*40 

    # Get consumption
    c = (h_vis-t+real_wage)
    
    if np.all(c > 0) :
        
        affordable = 1
    else:
        affordable = 0
        
    return affordable

def get_possible_choices_nodebt(x1,x2,deg):
    """
    This function returns the set of all possible choices considering if the
    choice is affordable
    """
    
    Jx  = get_possible_choices(x2)
    
    # Now for each of this choices loop to see if it is affordable
    
    affordable = np.ones(np.shape(Jx)[0])
    
    it = 0
    for j in Jx:
        
        if j[1] == 0:  # For sure will be affordable!
            
            pass
        else:
            affordable[it] = check_affordable(x1,x2,j,deg)
          
        it +=1 
        
    Jxfinal = Jx[affordable == 1]
    
    return Jxfinal

def get_expected_conditional_x(x,period,conter,types,maxdebt):
    
    # Loop over all individuals
    total_choices = get_total_choices()  # This should load the array with all possible choices
    payoff = np.zeros((np.shape(x)[0],np.shape(total_choices)[0])) # Total amount of possible choices
    # Generate column map
    column_map = np.repeat(np.nan,np.shape(x)[0]*np.shape(total_choices)[0]).reshape(np.shape(x)[0],np.shape(total_choices)[0])  # This array will hold how to map from columns in payoff, to the corresponding choice. As safety I will create it with nans
    
    x1_new = get_x1_new(x[:,1:5])
    
    for i in range(np.shape(x)[0]):
        xi = x[i,1:]   # get the first individual
    
        # First check which choices are feasible 
        
        x2 = xi[4:14]
        x1 = xi[:4]
        debt = xi[14]
        
        if conter == 0 :
        
            Jx  = get_possible_choices_nodebt(x1_new[i,:],x2,5)
        
        elif conter == 1:
            
            Jx = get_possible_choices(x2)

        # Perform the map
        
        idx = np.where( (total_choices==Jx[:,None]).all(-1) )[1]  # This tells which index in tota_choices corresponds to each choice in Jx
        
        idx = np.concatenate((idx[...,np.newaxis],np.arange(0,np.shape(idx)[0]).reshape(np.shape(idx)[0],1)),axis=1)  # This just includes the index of Jx as a column in the array. 

        column_map[i,idx[:,0]] = idx[:,1]  # This performs the match to this individual of the corresponding mapping.
        
        # load the vjts        
        vjt_jx = load_vjti(x1,x2,period,conter,types[i],maxdebt)
        
        column = 0  # This will track which choice is in the column. 
        for choice in range(np.shape(total_choices)[0]):
            if choice in list(idx[:,0]):
                payoff[i,column] = vjt_jx[debt,int(column_map[i,column])]  # Get the corresponding chioce with the corresponding debt level.
            else:
                payoff[i,column] = np.nan
            
            column +=1 # Identify next column!
            
    return payoff


def save_choice(choices,period,cohort,conter,maxdebt):
    
    """
    This function saves the simulatd choices depending on which conterfactual is
    """

    np.save(f'choice/choice_nodebt_t{period}_s{cohort}_conter{conter}',choices)
        
def save_welfare(choices,period,cohort,conter,maxdebt):


    np.save(f'welfare/w_nodebt_t{period}_s{cohort}_maxdebt{maxdebt}_conter{conter}',choices)


def load_epsilons(x,cohort,period):
    
    colnames = ["id","e"]
    
    epsilons = pd.DataFrame(np.load(f"epsilon/e_cohort{cohort}_period{period}.npy"),
                            ).rename(columns={0:"id"})
    
    xdf = pd.DataFrame(x).rename(columns={0:"id"})
    
    xdf = pd.merge(xdf,epsilons,on="id")
    
    epsilons_ordered = np.array(xdf)[:,16:]
    
    return epsilons_ordered

def compare_alternatives(x,period,cohort,conterfactual,types,maxdebt,get_welfare=1):
    
    # Compute the payoff
    
    payoff = get_expected_conditional_x(x,period,conterfactual,types,maxdebt)
    
    total_choices_size = np.shape(get_total_choices())[0]
    
    # Draw the epsilons
    
    #epsilons = np.random.gumbel(loc=0.0, scale=1.0, size=(np.shape(x)[0],total_choices_size))
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
        
        welfare = np.nanmax(evjt,axis=1)
        
        save_welfare(welfare,period,cohort,conterfactual,maxdebt)
    
    return choices


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
            sigmas[choice[0]-2]
            
        psi = np.random.normal(loc=mu, scale=sigma, size=1)
        
        real_wage[i] = np.exp((w + psi))*choice[2]*1/2*(40*52)
    
        periods[i] = period-x2[i,7]
        
    if conter == 0 :
        
        paid = np.maximum(0,np.minimum(real_wage-2000,(1/periods)*debt_range[debt]*(1+r)))
    
        debtnew = (1+r)*debt_range[debt] - paid
        
    elif conter == 1:
        
        discretionary_income = real_wage-2.25*15000
        
        paid = np.maximum(0,np.minimum(real_wage-2000,np.minimum(0.05*discretionary_income,(1/periods)*debt_range[debt]*(1+r))))
                
        debtnew = (1+r)*debt_range[debt] - paid
        
    debtnew = map_debt_position(debt_range, debtnew)
        
    return debtnew
        

        

def move_states_and_debt(sigma_u,x,choices,period,conterfactual,types,maxdebt,cohort):
    
    
    total_choices = get_total_choices()
    
    choices_original = total_choices[choices,:]
            
    x_new = np.concatenate((x[:,:5],move_state_agents(x[:,:5],x[:,5:15],choices_original,period,cohort),x[:,15][...,None]),axis=1).astype(np.int32)
    
    return x_new, types
    
def move_types(types,choices,educ_choices,not_educ_choices):
    
    types_educ = types[np.where( (choices[...,None]==educ_choices[:,None]).all(-1) )[1]]
    types_noteduc = types[np.where( (choices[...,None]==not_educ_choices[:,None]).all(-1) )[1]]
    typesnew = np.concatenate((types_noteduc,types_educ))
    return typesnew


def simulate_choices(cohort,sigma_u,conterfactual,q,maxdebt):
    
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
    
    # initialize states
    
    x, types = initialize_states(q,cohort)
    
    # loop over periods
    
    for period in range(1,T,1):
        print("Current period is", period)
        save_state(x,period,cohort,conterfactual,maxdebt)
        choices = compare_alternatives(x,period,cohort,conterfactual,types,maxdebt)
        x, types = move_states_and_debt(sigma_u,x,choices,period,conterfactual,types,maxdebt,cohort)
        if period == T-1:
            save_state(x,period+1,cohort,conterfactual,maxdebt)

def save_state(x,period,cohort,conterfactual,maxdebt):
    

    np.save(f"state/state_nodebt_t{period}_s{cohort}_conter{conterfactual}.npy",x)

        
        
def get_debt_range():
    
    debtrange1 = np.array([0,300,500,620,770,950])
    debtrange2 = np.linspace(1166,3500,16)
    debtrange3 = np.linspace(3720,8800,25)
    debtrange4 = np.linspace(9200,20000,25)
    debtrange5 = np.linspace(22700,100000,28)

    debt_range = np.concatenate((debtrange1,debtrange2,debtrange3,debtrange4,debtrange5))
    
    return debt_range
    
def get_conditional_agents(sigma_u,x1,x2,b,psi,j,period,conterfactual,types,maxdebt):
    
    "this function returns the conditional value function associated with each alternative"
    
    
    b1 = get_debt_range()
    
    # Map choice back to tuple
    
    total_choices = get_total_choices()
    
    j = total_choices[j,:]
            
    u = get_utility_agents(sigma_u,x1,x2,b,b1,psi,j,period,conterfactual,maxdebt)

    continuation = beta*VT_agents(x1,x2,b1,period,j,conterfactual,types,maxdebt)
        
    vjt = u + continuation

    return vjt

@njit()
def wage0_i(x1_new,x2_new):
    
    """ Different wage fucntion for numba"""
    
    param_wage = wage_0
    
    # This part puts together x1 and x2. x1 is time invariant, x2 is time variant. 
    x = np.hstack((x1_new[...,None].T,x2_new[:-1][...,None].T)) # do not include grad school coefficient
    
    w = x@param_wage.T
    return w


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
def move_state_agents(x1,x2,j,period,cohort):
    
    "this function moves the state space given your current state space and choice"
    
    z = np.copy(x2)
    x1_new = get_x1_new(x1[:,1:])
    gradall = np.load(f"grad_prob/gradall_c{cohort}_t{period}.npy")
    # Loop over all agents. For now I will do it brute force.
    
    for i in range(np.shape(x2)[0]):
        
        # First, check if the chioce could induce graduation! 
        
        if ((z[i,1] >=1) & (j[i,1] == 1) & (z[i,4] == 0)) | ((z[i,2]>=3) & (j[i,1] == 2) & (z[i,5] == 0)) | (j[i,1]==3)  :
            
            # It is the same as the other case, but it could get graduated with some probability
            
            if j[i,1] == 1:  # two-year choice
            
                z[i,1] = z[i,1] + 1  # plus one experiemce
                
                # Compute graduation probability
                
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
                    
                    z[i,4] = grad
                    
                    # Set mejor at graduation:
                        
                    z[i,8] = j[i,0]
                                       
                    # Chnge experience
                    
                    z[i,1] = 99
            
            elif j[i,1] == 2: # four year choice
            
                z[i,2] = z[i,2] + 1
                
                # Compute graduation probability
                
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
            evt = np.load(f"evt/{period+1}/evt_t{period+1}_[{x1i}]_em{types[space]}.npz")
        elif conter == 1: 
            evt = np.load(f"evt_conter/{period+1}/evt_t{period+1}_[{x1i}]_em{types[space]}_maxdebt{maxdebt}.npz")
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
            if c[row,col] < 2000:
                c[row,col] = 2000
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
def get_utility_agents(sigma_u,x1,x2,b,b1,psi,j,period,conterfactual,maxdebt):
    
    " this function returns the utility associated with a particular alternative"
    
    global debt_range
    
    r  = 0.05
    # Here I am making sure all the dimensions are correct, so that the final product is
    # (x2 X b) x b . This will allow to vectorize and obtain results without loops. 
    # Get expanded version of x1
    
    x1_new = get_x1_new(x1)
    h = fin_help_agents(x1_new,x2,j,period)
    
    full_part_time = j[:,2].copy()
    
    w = wage0(x1_new,x2)
    
    wage_shock = np.exp((w + psi[...,None]))*full_part_time[...,None]*1/2*(40*52)
    
    c = (h + wage_shock[:,0] -(1+r)*debt_range[np.array(b,dtype="int")]-tuition_agents(conterfactual,j))
    
    c =c[...,None]
    c = np.repeat(c,np.shape(b1)[0],axis=1)
    b2 = b1[...,None]
    b2 = np.repeat(b2.T,np.shape(c)[0],axis=0)
    c  = c + b2
    #c = check_consumption(c)

    u = 0.1*((0.00001*c)**(1-sigma_u)/(1-sigma_u))
    
    # Incorporate debt  rules! 
    
    u = get_debt_rules(c,u,b,maxdebt)

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

def get_debt_rules(c,u,b,maxdebt):
    """
    This function establishes very negative utilitie values for not feasible
    values of debt. The idea is: 
        - Consumption can't be negative
        - Should get at least as much debt as (1+r)debt

    """

    u[c<2000] = -100000
    
    if maxdebt == False:
        u = minimum_debt(u,b)
    elif maxdebt == True:
        u = minimum_debt_maxdebt(u,b)
    
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
def fin_help_agents(x1_new,x2,j,period):
    
    "this function returns the amount of financial help individuals will recieve"

    # Generate 4y or grad school dummy
    
    dum = np.zeros((np.shape(x1_new)[0]))
    dum[j[:,1]>1] = 1
    
    # put together
    x = np.hstack((x1_new,dum[...,None]))
    
    # perform the product.
    
    p_trans =  x@param_fam.T
    grants =  x@param_grants.T
    
    h = np.exp(p_trans) + np.exp(grants)

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
wage_0, params_wage,sigmas,param_grants,param_prob_grants,param_fam,param_prob_grad = load_all_parameters()
debt_range = get_debt_range()

#n=200000
#sigma_u = 1.4
#conterfactual = 0
#maxdebt = False
#q = np.load("estimates/em_q_typeff2.npy") 
#simulate_choices(1,sigma_u,conterfactual,q,maxdebt)
#--------------------------------------------#
# PROBLEMS I HAVE IDENETIFIED

# In the toymodel, debt level is tracked as the argmax of the optimal
# which means that is the position of the debt vector. This is only the case
# for individuals that study and hence get indebted. If you do a repayment plan,
# debt is now the actual value of the matrix. This inconsistency did not arise in
# an error since repayment was setting debt value to 0, which implies that
# it is the same value wether it is the argmax or the vector. I need to account for
# that now. 

