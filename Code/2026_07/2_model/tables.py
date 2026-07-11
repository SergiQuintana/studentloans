# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:17:17 2024

@author: Sergi
"""


import numpy as np
import os 

os.chdir(r"C:/Users/Sergi/Project/Real")

path = r"C:\Users\Sergi\Dropbox\PhD\Projects\Papers\1_financial_constraints\Code\2024_09\2_model\RealModel\output"
pathfunctions = "C:/Users/Sergi/Dropbox/PhD/Projects/Papers/1_financial_constraints/Model/"

T = 10
fields = 8
occupations = 8

#-----------------------------------------------------------------------------#

# Use necessary functions


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
    param_g_x1 = param_utility[:amount_x1].reshape((1+fields+occupation+1,9)).T
        
    # Now get the parameters for working full or part time. Those are:
    # 2 for associate, 2 for each field, 1 for each occupation, 2 for grad school. 
    # Total = 2+ 2*fields + occupations + 2
    
    amount_work = (2+2*fields+occupation+2)
    param_g_work = param_utility[amount_x1:amount_x1+amount_work]
    param_g_work_fields = param_g_work[:(fields+1)*2].reshape((1+fields),2).T
    param_g_work_occu = param_g_work[(fields+1)*2:(fields+1)*2 +occupation]
    param_g_work_grad = param_g_work[(fields+1)*2 +occupation:]
    
    param_g_work = np.zeros((2,1+fields+occupation+1))
    param_g_work[:,:fields+1] = param_g_work_fields
    param_g_work[1,fields+1:fields+1+occupation] =  param_g_work_occu
    param_g_work[:,-1] = param_g_work_grad
    
    
    #param_g_work = build_param_g_work(param_g_work,fields,occupation,size)
    
    # Now get the parameters for last choice. The possible last choices are: 
    # associate + fields + grad + occupations +home production(but home prodution has no coefficient)
    # (and as off now, grad school has no coefficient either!)
    
    amount_last = 1+ fields  + occupation
    param_g_last =  param_utility[amount_x1+amount_work:amount_x1+amount_work+amount_last]
    #param_g_last = build_param_g_last(param_g_last,fields,occupation,size)
    
    # Now get the parameters for the education status preferences effects
    # There are in total: no educ (base category), associate, fields, grad: 1+fields+1
    
    amount_educ = get_amount_educ()
    param_educ = param_utility[amount_x1+amount_work+amount_last:amount_x1+amount_work+amount_last+amount_educ]
    param_educ = get_param_educ(param_educ,fields,occupation,size)
    param_educ = param_educ[np.linspace(27,41,8).astype("int"),:].T
    
    
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
    #param_first = build_param_first(param_first,fields,occupation,size)
    
    
    table_educ, table_occ  = build_discrete_table(param_g_x1,param_g_work,param_g_last,param_educ,param_first)
    
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
    if em_type == 1:  # if it is the base category, set effects to 0!
        param_type[:,:] = 0

    utility_parameters = [param_g_x1,param_g_work,param_g_last,param_educ,param_period,param_period_work,param_first,param_exp,param_type]
    
    return table_educ, table_occ, param_educ


def build_discrete_table(param_g_x1,param_g_work,param_g_last,param_educ,param_first):
    
        params_first_time = np.zeros((4,fields+1))
        params_first_time[0,0] = param_first[0]
        params_first_time[:,1:] = param_first[1:-1][...,None]
    
        params_educ =  np.concatenate((param_g_x1[:,:fields+1],param_g_work[:,:fields+1],param_g_last[:fields+1][...,None].T),axis=0)   
        
        #params_educ = np.concatenate((params_educ,params_first_time),axis=0)
    
        params_occu = np.concatenate((param_g_x1[:,fields+1:-1],param_g_work[1,fields+1:-1][...,None].T,param_g_last[fields+1:][...,None].T),axis=0)
        
        params_grad = np.concatenate((param_g_x1[:,-1][...,None],param_g_work[:,-1][...,None]),axis=0)

        #params_grad = np.concatenate((params_grad,np.array(param_first[-1])[...,None][...,None]),axis=0)
        # Put the Constant the last one
        
        params_educ = np.concatenate((params_educ[1:,:],params_educ[0,:][...,None].T),axis=0)
        params_occu = np.concatenate((params_occu[1:,:],params_occu[0,:][...,None].T),axis=0)
        params_grad = np.concatenate((params_grad[1:,:],np.array([0])[...,None],params_grad[0,:][...,None].T),axis=0)
        
        params_educ = np.concatenate((params_educ,params_grad),axis=1) 

        return params_educ, params_occu
    
    
def table_education(coefficients: np.ndarray, std_errors: np.ndarray) -> str:
    """
    Converts two numpy matrices (coefficients and standard errors) into a LaTeX table format,
    with standard errors on a new row under each coefficient.
    
    :param coefficients: kxm numpy matrix where each row is a coefficient and each column is a regression.
    :param std_errors: kxm numpy matrix where each row is a standard error corresponding to the coefficient.
    :return: A string containing the LaTeX formatted table.
    """
    assert coefficients.shape == std_errors.shape, "The shape of coefficients and standard errors must match."
    
    k, m = coefficients.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
    
    # Create column names
    column_names =  "\\hline  \\\\  &  \\rot{Associate}  &   \\rot{Business}&    \\rot{STEM}&  \\rot{Undeclared} &  \\rot{Education} & \\rot{Social Sciences}&    \\rot{Humanities }&    \\rot{Health}&    \\rot{Other} & \\rot{Grad}  \\\\  \\hline \\\\"
    latex_table += column_names
    
    rowanmes = ["ParInc Q2", "ParInc Q3", "ParInc Q4", "Ability Q2", "Ability Q3", "Ability Q4",
                "Female", "Black", "Part-Time Work","Full-Time Work", "Switching Cost", "Constant"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${coefficients[i, j]:.2f}$" for j in range(m)]
        se_row = [f"(${std_errors[i, j]:.2f}$)" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"
        latex_table += " & " + " & ".join(se_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Regression coefficients and standard errors}\n\\end{table}"
    
    return latex_table


def table_education_complementarities(coefficients: np.ndarray, std_errors: np.ndarray) -> str:
    """
    Converts two numpy matrices (coefficients and standard errors) into a LaTeX table format,
    with standard errors on a new row under each coefficient.
    
    :param coefficients: kxm numpy matrix where each row is a coefficient and each column is a regression.
    :param std_errors: kxm numpy matrix where each row is a standard error corresponding to the coefficient.
    :return: A string containing the LaTeX formatted table.
    """
    assert coefficients.shape == std_errors.shape, "The shape of coefficients and standard errors must match."
    
    k, m = coefficients.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
    
    # Create column names
    column_names =  "\\hline  \\\\  &    \\rot{Business}&    \\rot{STEM}&  \\rot{Social Sciences} &  \\rot{Education} &   \\rot{Humanities }&    \\rot{Health}&    \\rot{Sales \\& Office} & \\rot{Production}   \\\\  \\hline \\\\"
    latex_table += column_names
    
    rowanmes = ["Associate Degree", "Business", "STEM", "Education", "Social Sciences",
                "Humanities","Health","Other","Graduate Degree"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${coefficients[i, j]:.2f}$" for j in range(m)]
        se_row = [f"(${std_errors[i, j]:.2f}$)" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"
        latex_table += " & " + " & ".join(se_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Regression coefficients and standard errors}\n\\end{table}"
    
    return latex_table


def table_occupation(coefficients: np.ndarray, std_errors: np.ndarray) -> str:
    """
    Converts two numpy matrices (coefficients and standard errors) into a LaTeX table format,
    with standard errors on a new row under each coefficient.
    
    :param coefficients: kxm numpy matrix where each row is a coefficient and each column is a regression.
    :param std_errors: kxm numpy matrix where each row is a standard error corresponding to the coefficient.
    :return: A string containing the LaTeX formatted table.
    """
    assert coefficients.shape == std_errors.shape, "The shape of coefficients and standard errors must match."
    
    k, m = coefficients.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
    
    # Create column names
    column_names =  "\\hline  \\\\  &    \\rot{Business}&    \\rot{STEM}&  \\rot{Social Sciences} &  \\rot{Education} &   \\rot{Humanities }&    \\rot{Health}&    \\rot{Sales \\& Office} & \\rot{Production}   \\\\  \\hline \\\\"
    latex_table += column_names
    
    rowanmes = ["ParInc Q2", "ParInc Q3", "ParInc Q4", "Ability Q2", "Ability Q3", "Ability Q4",
                "Female", "Black","Full-Time Work", "Switching Cost", "Constant"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${coefficients[i, j]:.2f}$" for j in range(m)]
        se_row = [f"(${std_errors[i, j]:.2f}$)" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"
        latex_table += " & " + " & ".join(se_row) + " \\\\\n"

    latex_table += "\\hline\n\\end{tabular}\n\\caption{Regression coefficients and standard errors}\n\\end{table}"
    
    return latex_table
  
def table_measures():
    
    m1 = np.load("estimates/aux_measure1.npz.npy")
    m2 = np.load("estimates/aux_measure2.npz.npy")
    
    se1 = np.load("estimates/aux_measure1_se.npz.npy")
    se2 = np.load("estimates/aux_measure2_se.npz.npy")
    
    ms = np.concatenate((m1[...,None],m2[...,None]),axis=1)
    se = np.concatenate((se1[...,None],se2[...,None]),axis=1)
    
    k, m = ms.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
    
    # Create column names
    column_names =  "\\hline  \\\\  &   Measure 1 & Measure 2   \\\\  \\hline \\\\"
    latex_table += column_names
    
    rowanmes = ["Constant","ParInc Q2", "ParInc Q3", "ParInc Q4", "Ability Q2", "Ability Q3", "Ability Q4",
                "Female", "Black","Type"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${ms[i, j]:.2f}$" for j in range(m)]
        se_row = [f"(${se[i, j]:.2f}$)" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"
        latex_table += " & " + " & ".join(se_row) + " \\\\\n"

    latex_table += "\\hline\n\\end{tabular}\n\\caption{Regression coefficients and standard errors}\n\\end{table}"
    
    return latex_table




def flow_payoff_table():
    
    
    table_educ, table_occ, param_educ = build_param_g(2,np.load("estimates/param_g.npy"))
    table_educse, table_occse, param_educse = build_param_g(2,np.load("estimates/se_it1_sigma1.4.npy"))
  
    tableeduc = table_education(table_educ,table_educse)
    tableocu  = table_occupation(table_occ,table_occse)
    tablecomplement = table_education_complementarities(param_educ, param_educse)
    
paramg = build_param_g(2,np.load("estimates/param_g.npy"))








