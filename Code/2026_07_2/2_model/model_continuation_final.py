# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:37:49 2024

@author: Sergi
"""

import pandas as pd
import scipy
import numpy as np
from numba import njit
import multiprocessing
import os
# set path
from config import DIR, OUT, INP, FUN, RDATA, CONT, EST
pathfunctions = DIR["MODEL_FUNCOEF"]
path = DIR["MODEL_REALDATA"]
pathcontinuation =  DIR["MODEL_CONTINUATION"]
pathcontfinal =  DIR["MODEL_CONTINUATION_FINAL"]
#os.chdir(r"C:/Users/Sergi/Project/Real")

T = 10
#-----------------------------------------------------------------------------#
# This code estimates the continuation value, wage and utility trajectories
# of individuals after period 29 of their life. 
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
#                         DEFINE THE FUNCTIONS
#-----------------------------------------------------------------------------#


@njit
def get_utility_trajectory(y,riskaversion):
    
    ''' This function computes the utility of each y
    given the riskaversion parameters'''
    
    
    inc = np.maximum(y,2000)
    
    #u = 0.0001*((0.00001*inc)**(1-riskaversion)/(1-riskaversion))
    u = 0.1*((0.00001*inc)**(1-riskaversion)/(1-riskaversion))
    return u



def get_debt_income(periods,debt,wage):
    
    '''
    This function computes the student loan payment and returns nexts periods
    debt and disposable income after payments. Notice the payment can't impply incomes
    lower than 2k, since would be cheating in the model. 
    '''
    
    global r
    
    paid = np.maximum(0,np.minimum(wage-2000,(1/periods)*debt*(1+r)))
    
    debtnew = (1+r)*debt - paid
    
    income = wage-paid
    
    return debtnew, income

def get_debt_incom_idr(forgiven,period,periods,debt,wage):
    
    '''
    This function comptues the student loan payment and returns next periods 
    debt and disposable income after payments under the SAVE plan. The rules are:
    
    -- Payments are at 5% of discerinary income
    -- Payments can't exceeed the 10y equivalent. 
    -- Payments are only made on the part of the income above 2.25 poverty line.
    -- Payments are forgiven after 10 years for debt on 10k and 1 year extra for
    each extra 1k on student loans. 
    
    On top of that, payments can't imply incomes lower than 2k since that would
    be cheating in the model. 
    
    '''
    global r
    
    discretionary_income = wage-2.25*15000
    
    paid = np.maximum(0,np.minimum(wage-2000,np.minimum(0.05*discretionary_income,(1/periods)*debt*(1+r))))
    
    income = wage-paid
    
    debtnew = (1+r)*debt - paid
    
    if period >= forgiven:
        
        debtnew = debtnew*0
    
    
    return debtnew, income
    

    
def trajectories_idr(df,initialdebt,periods,riskaversion,n):
    
    '''This function simulates the utility trajectories of n individuals that 
    graduate with some student debt and face an idr plan'''
    
    debt = np.ones(n)*initialdebt
    u = np.zeros((47-10,n))
    for period in range(10,47):
        # Draw unemployment and wage
        forgiven = initialdebt + 20
        data  = df[df["period"]==period]
        unemployment = np.random.binomial(1, data['unemployed'], n)
        wage = np.exp(np.random.normal(data['meanwage'],data['sdwage'], n))*52*40
        wage = wage*(1-unemployment)
        if periods > 0 :
            # get debt and income
            debt,income = get_debt_incom_idr(forgiven,period,periods,debt,wage)
            # get utility
            u[period-10,:] = get_utility_trajectory(income,riskaversion)
            
        else: 
            # get debt and income
            debt,income = get_debt_incom_idr(forgiven,period,1,debt,wage)
            # get utility
            u[period-10,:] = get_utility_trajectory(income,riskaversion)

        
        periods = periods-1
    
    u = np.mean(u,axis=1)
    
    return u


    
def trajectories(df,initialdebt,periods,riskaversion,n):
    
    '''This function simulates the utility trajectories of n
    individuals that graduate with some student debt and face some
    repayment periods'''
    
    debt = np.ones(n)*initialdebt
    u = np.zeros((47-10,n))
    for period in range(10,47):
        data = df[df["period"]==period]
        # Draw unemployment and wage
        unemployment = np.random.binomial(1, data['unemployed'], n)
        wage = np.exp(np.random.normal(data['meanwage'],data['sdwage'], n))*52*40
        wage = wage*(1-unemployment)
        if periods > 0 :
            # move debt
            debt,income = get_debt_income(periods,debt,wage)
            # get utility
            u[period-10,:] = get_utility_trajectory(income,riskaversion)
        else: 
            # move debt
            debt,income = get_debt_income(1,debt,wage)
            # get utility
            u[period-10,:] = get_utility_trajectory(income,riskaversion)
        
        periods = periods-1
    
    u = np.mean(u,axis=1)
    
    return u

def get_all_trajectories(df,initialdebt,periods,riskaversion,n):
    
    ''' This function gets all utility trajectories for some level of debt 
    and some repayment periods'''

    
    unew = np.array([])
    
    totalstates = len(df[df["period"]==10])
    
    for i in range(totalstates):
                
        index = df.iloc[i*37:(i+1)*37,:]
        
        u =  trajectories(index,initialdebt,periods,riskaversion,n)
        
        unew = np.concatenate((unew,u),axis=0)
        
    df["expected_u_debt"] = unew*df["beta"]
    
    
    expected = df.groupby(['sex','ethnicity','educmodel','majornlsy'])[['expected_u_debt']].sum()
    
    expected.to_csv(f"{pathcontinuation}/continuation_initialdebt{initialdebt}_period{periods}_sigma_u{riskaversion}.csv")
        
    #return expected


def get_all_trajectories_idr(df,initialdebt,periods,riskaversion,n):
    
    ''' This function gets all utility trajectories for some level of debt 
    and some repayment periods for an idr'''
    
    unew = np.array([])
    
    totalstates = len(df[df["period"]==10])
    
    for i in range(totalstates):
                
        index = df.iloc[i*37:(i+1)*37,:]
        
        u =  trajectories_idr(index,initialdebt,periods,riskaversion,n)
        
        unew = np.concatenate((unew,u),axis=0)
    
    df["expected_u_idr"] = unew*df["beta"]
    
    
    expected = df.groupby(['sex','ethnicity','educmodel','majornlsy'])[['expected_u_idr']].sum()
    
    expected.to_csv(f"{pathcontinuation}/continuation_idr_initialdebt{initialdebt}_period{periods}_sigma_u{riskaversion}.csv")

    #return expected

def get_debt_range():
    
    debtrange1 = np.array([0,300,500,620,770,950])
    debtrange2 = np.linspace(1166,3500,16)
    debtrange3 = np.linspace(3720,8800,25)
    debtrange4 = np.linspace(9200,20000,25)
    debtrange5 = np.linspace(22700,100000,28)

    debt_range = np.concatenate((debtrange1,debtrange2,debtrange3,debtrange4,debtrange5))
    
    return debt_range

def geteverything(df,initialdebt,periods,riskaversion,n,conter):

    if conter == 0:
        get_all_trajectories(df,initialdebt,periods,riskaversion,n)
    elif conter == 1:
        get_all_trajectories_idr(df,initialdebt,periods,riskaversion,n)
    
    
def loop_over_periods(sex,black,data,sigma_u,conter):
    
    results = []
    names =  []
    fields = 8
    for lastschool in range(0,T):
        for educ in range(0,5):
            
            if educ < 2:
                major = 0
                
                evt = data[(data["educmodel"]==educ) & (data["majornlsy"]==major)
                           & (data["last_educ_period"]==lastschool)]
                
                if conter == 1:
                    evt = np.array(evt["expected_u_idr"])
                else:
                    evt = np.array(evt["expected_u_debt"])
                names.append(f"con_last{lastschool}_educ{educ}_major{major}")
                results.append(evt)
                
            if educ == 2:
                major = 12
                evt = data[(data["educmodel"]==educ) & (data["majornlsy"]==major)
                           & (data["last_educ_period"]==lastschool)]
            
                if conter == 1:
                    evt = np.array(evt["expected_u_idr"])
                else:
                    evt = np.array(evt["expected_u_debt"])
                names.append(f"con_last{lastschool}_educ{educ}_major{major}")
                results.append(evt)
                
            if educ > 2:
                for major in range(1,fields+1):
                    evt = data[(data["educmodel"]==educ) & (data["majornlsy"]==major)
                               & (data["last_educ_period"]==lastschool)]
                    
                    if conter == 1:
                        evt = np.array(evt["expected_u_idr"])
                    else:
                        evt = np.array(evt["expected_u_debt"])
                    names.append(f"con_last{lastschool}_educ{educ}_major{major}")
                    results.append(evt)
        
    if conter == 0:
        np.savez_compressed(f"{pathcontfinal}/continuation_s{sex}_eth{black}_sigma{sigma_u}.npz",**{name:value for name,value in zip(names,results)})
    else:
        np.savez_compressed(f"{pathcontfinal}/continuation_conter_s{sex}_eth{black}_sigma{sigma_u}.npz",**{name:value for name,value in zip(names,results)})

    
def data_to_numpy(sigma_u,conter):
    
    """
    This will give the numpy structure to the data
    """
    
    # Get all unique x1 indiviuals
    data = pd.read_csv(f"{pathcontinuation}/all_continuations_c{conter}_sigma_u{sigma_u}.csv")
    for sex in range(0,2):
        for black in range(0,2):
            data2 = data[(data["sex"]==sex)&(data["ethnicity"]==black)]
            # not conterfactual:
            loop_over_periods(sex,black,data2,sigma_u,0)
            # conterfactual:
            loop_over_periods(sex,black,data2,sigma_u,1)


def give_model_format_plan(debt_range, riskaversion, conter):

    """
    This function saves all the continuation values and gives format of the model. 
    This is : save as x1, with all x2s as subarrays. It has to be a numpy array.
    """  
    first  = 0 
    for periods in range(0,T):
        for initialdebt in debt_range:

            if first == 0:
                if conter == 0: 
                    data = pd.read_csv(f"{pathcontinuation}/continuation_initialdebt{initialdebt}_period{periods}_sigma_u{riskaversion}.csv")
                elif conter == 1:
                    data = pd.read_csv(f"{pathcontinuation}/continuation_idr_initialdebt{initialdebt}_period{periods}_sigma_u{riskaversion}.csv")
                data["debt"] = initialdebt
                data["last_educ_period"] = periods
                first = 1
            else:
                if conter == 0: 
                    temp = pd.read_csv(f"{pathcontinuation}/continuation_initialdebt{initialdebt}_period{periods}_sigma_u{riskaversion}.csv")
                elif conter == 1:
                    temp = pd.read_csv(f"{pathcontinuation}/continuation_idr_initialdebt{initialdebt}_period{periods}_sigma_u{riskaversion}.csv")
                temp["debt"] = initialdebt
                temp["last_educ_period"] = periods
                data = pd.concat([data,temp])
                
    data.to_csv(f"{pathcontinuation}/all_continuations_c{conter}_sigma_u{riskaversion}.csv",index=False)

    
def give_model_format(debt_range,riskaversion):
    
    """
    This function saves all the continuation values and gives format of the model. 
    This is : save as x1, with all x2s as subarrays. It has to be a numpy array.
    """
    first  = 0 
    for periods in range(0,T):
        for initialdebt in debt_range:

            if first == 0:
                data = pd.read_csv(f"{pathcontinuation}/continuation_idr_initialdebt{initialdebt}_period{periods}_sigma_u{riskaversion}.csv")
                data2 = pd.read_csv(f"{pathcontinuation}/continuation_initialdebt{initialdebt}_period{periods}_sigma_u{riskaversion}.csv")
                data["expected_u_debt"] = data2["expected_u_debt"]
                data["debt"] = initialdebt
                data["last_educ_period"] = periods
                first = 1
            else: 
                temp = pd.read_csv(f"{pathcontinuation}/continuation_idr_initialdebt{initialdebt}_period{periods}_sigma_u{riskaversion}.csv")
                temp2 = pd.read_csv(f"{pathcontinuation}/continuation_initialdebt{initialdebt}_period{periods}_sigma_u{riskaversion}.csv")
                temp["expected_u_debt"] = temp2["expected_u_debt"]
                temp["debt"] = initialdebt
                temp["last_educ_period"] = periods
                data = pd.concat([data,temp])
                
    data.to_csv(f"{pathcontinuation}/all_continuations_sigma_u{riskaversion}.csv",index=False)
   
def load_data():
    
    df = pd.read_stata(f'{path}/wage_major_period.dta')

    df['majornlsy'] = df['majornlsy'].fillna(0)

    df['beta'] = 0.98**(df['period'].astype('int')-10)
    
    return df
    

             
#df = pd.read_stata(f'{path}/wage_major_period.dta')
#df['majornlsy'] = df['majornlsy'].fillna(0)
#df['beta'] = 0.98**(df['period'].astype('int')-10)
#debt_range = get_debt_range()               
#n = 100000
#riskaversion = 1.4  
#initialdebt  = 85000
#periods = 9
#r = 0.05
#geteverything(df,initialdebt,periods,riskaversion,n)
#debt_range = get_debt_range()               
#give_model_format(debt_range,riskaversion)
    
    
#-----------------------------------------------------------------------------#
#                                  EXECUTION
#-----------------------------------------------------------------------------#
r = 0.05
#%%
#if __name__ == '__main__':
    
    #df = load_data()
    #debt_range = get_debt_range()
    #n = 100000
    #riskaversion = 1.4 
    #args = [(df,initialdebt,lastperiod,riskaversion,n) for lastperiod in range(0,T) for initialdebt in debt_range]
    #pool_obj = multiprocessing.Pool(60)
    #results = pool_obj.starmap(geteverything, args)
    #pool_obj.close()

    # Now put everything together
    
    #give_model_format(debt_range,riskaversion)
    
    # And finally give numpy format
    
    #data_to_numpy(riskaversion)
    












