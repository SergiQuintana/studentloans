# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:26:45 2024

@author: Sergi
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

T = 10 

#-----------------------------------------------------------------------------#
# Read the Real Data

path = "C:/Users/Sergi/Dropbox/PhD/Projects/Papers/1_financial_constraints/Model/real_data"
figures= "C:/Users/Sergi/Dropbox/PhD/Projects/Papers/1_financial_constraints/Model/Figures"
datafigures = "C:/Users/Sergi/Dropbox/PhD/Projects/Papers/1_financial_constraints/Output/Figuresnew/data"
path_states =  "C:/Users/Sergi/Project/Real"
pathstata = "C:/Users/Sergi/Dropbox/PhD/Projects/Papers/1_financial_constraints/Output/data_to_stata"

# Cummulative share of graduate individuals by period

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


def load_real_data():
    
    
    real = pd.read_stata(f"{path}/realdata_tographs_allfields.dta")
    
    index = pd.DataFrame(np.load(f"{path}/feasible_index.npy"),columns=["PUBID"])
    
    index["nothing"] = 1
    
    real = real.set_index("PUBID").merge(index.set_index("PUBID"),
                                         how='inner',
                                         on="PUBID")
    
    del real["nothing"]
    
    return real
    
    
    
    
    
def load_data(datatype, samples, maxdebt):
    
    """This function loads simulated and conterfactual data and transforms
    it into a data frame like the stata files
    
    datatype = 1 -- Simulated
    datatype = 2 -- Conterfactual
    
    samples -- Amount of simulated samples to put together
    
    """
    # The problem is that there is no panel on reality, since in the simulaitons
    # I am not tracking individuals. For this reason, I will generate those specific
    # conterfacuals
    
    names = ["PUBID","parinc","ability","sex","ethnicity","exp","twoyear_exp","fouryear_exp","grad","twograd","fourgrad","gradgrad","last_educ","majorgrad","last_choice","debt"]


    if datatype == 1 :
        
        for sample in range(1,samples+1):
            data = pd.DataFrame(np.load(f"{path_states}/state/state_t1_s{sample}.npy"),columns=names)
            data["period"] = 1
            data["sample"] = sample
            data["gvalue"] = np.load(f"{path_states}/welfare/w_t1_s{sample}.npy")
            data["gepsi"] = np.load(f"{path_states}/welfare/wepsi_t1_s{sample}.npy")
            data["type"] = np.load(f"{path_states}/estimates/types_{sample}.npy")
            typeid = data[["PUBID","type"]]
            for t in range(2,T+1):
                state = pd.DataFrame(np.load(f"{path_states}/state/state_t{t}_s{sample}.npy"),columns=names)
                state["period"] = t
                state["sample"] = sample
                state = pd.merge(state,typeid,on="PUBID")
                if t == T:
                    state["gvalue"] =  np.nan
                else:
                    state["gvalue"] =  np.load(f"{path_states}/welfare/w_t{t}_s{sample}.npy")
                    state["gepsi"] =  np.load(f"{path_states}/welfare/wepsi_t{t}_s{sample}.npy")
                data = pd.concat([data,state])
            if sample >1:
                final = pd.concat([data,final])
            else:
                final = data
            
    elif datatype == 2:
        
        for sample in range(1,samples+1):
            data = pd.DataFrame(np.load(f"{path_states}/state/state_conter_t1_s{sample}_maxdebt{maxdebt}.npy"),columns=names)
            data["period"] = 1
            data["sample"] = sample
            data["gvalue"] = np.load(f"{path_states}/welfare/w_conter_t1_s{sample}_maxdebt{maxdebt}.npy")
            data["gepsi"] = np.load(f"{path_states}/welfare/wepsi_conter_t1_s{sample}_maxdebt{maxdebt}.npy")
            data["type"] = np.load(f"{path_states}/estimates/types_{sample}.npy")
            typeid = data[["PUBID","type"]]
            for t in range(2,T+1):
                state = pd.DataFrame( np.load(f"{path_states}/state/state_conter_t{t}_s{sample}_maxdebt{maxdebt}.npy"),columns=names)
                state["sample"] = sample
                state["period"] = t
                state = pd.merge(state,typeid,on="PUBID")
                if t == T:
                    state["gvalue"] =  np.nan
                else:
                    state["gvalue"] =  np.load(f"{path_states}/welfare/w_conter_t{t}_s{sample}_maxdebt{maxdebt}.npy")
                    state["gepsi"] =  np.load(f"{path_states}/welfare/wepsi_conter_t{t}_s{sample}_maxdebt{maxdebt}.npy")
            
                data = pd.concat([data,state])
            if sample >1:
                final = pd.concat([data,final])
            else:
                final = data
                
    elif datatype == 3: # conterfactual no debt
    
        for sample in range(1,samples+1):
            data = pd.DataFrame(np.load(f"{path_states}/state/state_nodebt_t1_s{sample}_conter{0}.npy"),columns=names)
            data["period"] = 1
            data["sample"] = sample
            data["welfare"] = np.load(f"{path_states}/welfare/w_nodebt_t1_s{sample}_maxdebt{True}_conter{0}.npy")
            data["type"] = np.load(f"{path_states}/estimates/types_{sample}.npy")
            typeid = data[["PUBID","type"]]
            for t in range(2,T+1):
                state = pd.DataFrame( np.load(f"{path_states}/state/state_nodebt_t{t}_s{sample}_conter{0}.npy"),columns=names)
                state["period"] = t
                state["sample"] = sample
                state = pd.merge(state,typeid,on="PUBID")
                if t == T:
                    state["welfare"] =  np.nan
                else:
                    state["welfare"] =  np.load(f"{path_states}/welfare/w_nodebt_t{t}_s{sample}_maxdebt{True}_conter{0}.npy")
                data = pd.concat([data,state])
            if sample >1:
                final = pd.concat([data,final])
            else:
                final = data
    
    
    
    elif datatype == 4: # conterfactual grants
    
        for sample in range(1,samples+1):
            data = pd.DataFrame(np.load(f"{path_states}/state/state_nodebt_t1_s{sample}_conter{1}.npy"),columns=names)
            data["period"] = 1
            data["sample"] = sample
            data["welfare"] = np.load(f"{path_states}/welfare/w_nodebt_t1_s{sample}_maxdebt{True}_conter{1}.npy")
            data["type"] = np.load(f"{path_states}/estimates/types_{sample}.npy")
            typeid = data[["PUBID","type"]]
            for t in range(2,T+1):
                state = pd.DataFrame( np.load(f"{path_states}/state/state_nodebt_t{t}_s{sample}_conter{1}.npy"),columns=names)
                state["sample"] = sample
                state["period"] = t
                state = pd.merge(state,typeid,on="PUBID")
                if t == T:
                    state["welfare"] =  np.nan
                else:
                    state["welfare"] =  np.load(f"{path_states}/welfare/w_nodebt_t{t}_s{sample}_maxdebt{True}_conter{1}.npy")
                data = pd.concat([data,state])
            if sample >1:
                final = pd.concat([data,final])
            else:
                final = data
                
    elif datatype == 5: # conterfactual stem
    
        for sample in range(1,samples+1):
            data = pd.DataFrame(np.load(f"{path_states}/state/state_stem_conter_t1_s{sample}_maxdebt{maxdebt}.npy"),columns=names)
            data["period"] = 1
            data["sample"] = sample
            data["welfare"] = np.load(f"{path_states}/welfare/w_stem_conter_t{1}_s{sample}_maxdebt{maxdebt}.npy")
            data["type"] = np.load(f"{path_states}/estimates/types_{sample}.npy")
            typeid = data[["PUBID","type"]]
            for t in range(2,T+1):
                state = pd.DataFrame( np.load(f"{path_states}/state/state_stem_conter_t{t}_s{sample}_maxdebt{maxdebt}.npy"),columns=names)
                state["sample"] = sample
                state["period"] = t
                state = pd.merge(state,typeid,on="PUBID")
                if t == T:
                    state["welfare"] =  np.nan
                else:
                    state["welfare"] =  np.load(f"{path_states}/welfare/w_stem_conter_t{t}_s{sample}_maxdebt{maxdebt}.npy")
                data = pd.concat([data,state])
            if sample >1:
                final = pd.concat([data,final])
            else:
                final = data
                
    elif datatype == 6 :
        
        for sample in range(1,samples+1):
            data = pd.DataFrame(np.load(f"{path_states}/state/state_debt_t1_s{sample}.npy"),columns=names)
            data["period"] = 1
            data["sample"] = sample
            data["type"] = np.load(f"{path_states}/estimates/types_{sample}.npy")
            typeid = data[["PUBID","type"]]
            for t in range(2,T+1):
                state = pd.DataFrame(np.load(f"{path_states}/state/state_debt_t{t}_s{sample}.npy"),columns=names)
                state["period"] = t
                state["sample"] = sample
                state = pd.merge(state,typeid,on="PUBID")
                data = pd.concat([data,state])
            if sample >1:
                final = pd.concat([data,final])
            else:
                final = data
                
    elif datatype == 7: #conter good debt
        
        for sample in range(1,samples+1):
            data = pd.DataFrame(np.load(f"{path_states}/state/state_debt_conter_t1_s{sample}_maxdebt{maxdebt}.npy"),columns=names)
            data["period"] = 1
            data["sample"] = sample
            data["type"] = np.load(f"{path_states}/estimates/types_{sample}.npy")
            typeid = data[["PUBID","type"]]
            for t in range(2,T+1):
                state = pd.DataFrame( np.load(f"{path_states}/state/state_debt_conter_t{t}_s{sample}_maxdebt{maxdebt}.npy"),columns=names)
                state["sample"] = sample
                state["period"] = t
                state = pd.merge(state,typeid,on="PUBID")

                data = pd.concat([data,state])
            if sample >1:
                final = pd.concat([data,final])
            else:
                final = data
    
    return final

def get_choices_index(choices):
    
    total_choices = get_total_choices()
    
    idx = np.where( (total_choices==choices[:,None]).all(-1) )[1]
    
    return idx

def load_choices_real():
    
    """
    This function loads choices from observed data
    """
    
    names = ["choices"]
    
    data = pd.DataFrame(get_choices_index(np.load(f"{path}/choice_t1.npy")),columns=names)
    
    data["period"] = 1
        
    for period in range(2,T):
            
        state = pd.DataFrame(get_choices_index(np.load(f"{path}/choice_t{period}.npy")),columns=names)
        state["period"] = period
        data = pd.concat([data,state])
            
    return data
    
def load_choices(datatype, samples, maxdebt):
    
    """This function loads simulated and conterfactual data and transforms
    it into a data frame like the stata files
    
    datatype = 1 -- Simulated
    datatype = 2 -- Conterfactual
    
    samples -- Amount of simulated samples to put together
    
    """
    # The problem is that there is no panel on reality, since in the simulaitons
    # I am not tracking individuals. For this reason, I will generate those specific
    # conterfacuals
    
    names = ["choices"]


    if datatype == 1 :
        
        for sample in range(1,samples+1):
            data = pd.DataFrame(np.load(f"{path_states}/choice/choice_t1_s{sample}.npy"),columns=names)
            data["period"] = 1
            for t in range(2,T):
                state = pd.DataFrame(np.load(f"{path_states}/choice/choice_t{t}_s{sample}.npy"),columns=names)
                state["period"] = t
                data = pd.concat([data,state])
            if sample > 1:
                final = pd.concat([data,final])
            else:
                final = data
            
    elif datatype == 2:
        
        for sample in range(1,samples+1):
            data = pd.DataFrame(np.load(f"{path_states}/choice/choice_conter_t1_s{sample}_maxdebt{maxdebt}.npy"),columns=names)
            data["period"] = 1
            for t in range(2,T):
                state = pd.DataFrame(np.load(f"{path_states}/choice/choice_conter_t{t}_s{sample}_maxdebt{maxdebt}.npy"),columns=names)
                state["period"] = t
                data = pd.concat([data,state])
            if sample > 1:
                final = pd.concat([data,final])
            else:
                final = data
                
                
    elif datatype == 3:
        
        for sample in range(1,samples+1):
            data = pd.DataFrame(np.load(f"{path_states}/choice/choice_nodebt_t{1}_s{sample}_conter{0}.npy"),columns=names)
            data["period"] = 1
            for t in range(2,T):
                state = pd.DataFrame(np.load(f"{path_states}/choice/choice_nodebt_t{t}_s{sample}_conter{0}.npy"),columns=names)
                state["period"] = t
                data = pd.concat([data,state])
            if sample > 1:
                final = pd.concat([data,final])
            else:
                final = data
                
    elif datatype == 4:
        
        for sample in range(1,samples+1):
            data = pd.DataFrame(np.load(f"{path_states}/choice/choice_nodebt_t{1}_s{sample}_conter{1}.npy"),columns=names)
            data["period"] = 1
            for t in range(2,T):
                state = pd.DataFrame(np.load(f"{path_states}/choice/choice_nodebt_t{t}_s{sample}_conter{1}.npy"),columns=names)
                state["period"] = t
                data = pd.concat([data,state])
            if sample > 1:
                final = pd.concat([data,final])
            else:
                final = data
                
    elif datatype == 5:
        
        for sample in range(1,samples+1):
            data = pd.DataFrame(np.load(f"{path_states}/choice/choice_stem_conter_t{1}_s{sample}_maxdebt{maxdebt}.npy"),columns=names)
            data["period"] = 1
            for t in range(2,T):
                state = pd.DataFrame(np.load(f"{path_states}/choice/choice_stem_conter_t{t}_s{sample}_maxdebt{maxdebt}.npy"),columns=names)
                state["period"] = t
                data = pd.concat([data,state])
            if sample > 1:
                final = pd.concat([data,final])
            else:
                final = data
                
    elif datatype == 6 :
        
        for sample in range(1,samples+1):
            data = pd.DataFrame(np.load(f"{path_states}/choice/choice_debt_t1_s{sample}.npy"),columns=names)
            data["period"] = 1
            for t in range(2,T):
                state = pd.DataFrame(np.load(f"{path_states}/choice/choice_debt_t{t}_s{sample}.npy"),columns=names)
                state["period"] = t
                data = pd.concat([data,state])
            if sample > 1:
                final = pd.concat([data,final])
            else:
                final = data
                
    elif datatype == 7:
        
        for sample in range(1,samples+1):
            data = pd.DataFrame(np.load(f"{path_states}/choice/choice_debt_conter_t1_s{sample}_maxdebt{maxdebt}.npy"),columns=names)
            data["period"] = 1
            for t in range(2,T):
                state = pd.DataFrame(np.load(f"{path_states}/choice/choice_debt_conter_t{t}_s{sample}_maxdebt{maxdebt}.npy"),columns=names)
                state["period"] = t
                data = pd.concat([data,state])
            if sample > 1:
                final = pd.concat([data,final])
            else:
                final = data
    
    return final
    


def get_shares(real,model,coltype,variable,level,graphtype=1):
    
    """This function computes the share of graduates each period,
    and saves the data to compute it with stata"""
    
    if graphtype == 1: 
        if coltype == 1:
        
            share_real = real.groupby(["period"])["twograd"].mean()
            share_model = model.groupby(["period"])["twograd"].mean()
        elif coltype == 2:
            
            share_real = real.groupby(["period"])["fourgrad"].mean()
            share_model = model.groupby(["period"])["fourgrad"].mean()
            
        elif coltype == 3:
            
            share_real = real.groupby(["period"])["gradgrad"].mean()
            share_model = model.groupby(["period"])["gradgrad"].mean()
        
        leg = ["Data","Model"]
        fig = plt.subplots()
        plt.plot(share_real)
        plt.plot(share_model)
        plt.ylabel("Share graduated")
        plt.xlabel("Period")
        plt.legend(leg,loc='upper center', bbox_to_anchor=(0.5,-0.2), ncol = 2)
        plt.savefig(f"{figures}/graduation_{coltype}.png",bbox_inches='tight')
        
        # save data
        share_real = share_real.rename("real")
        share_model = share_model.rename("model")
        tosave = pd.concat([share_real,share_model],axis=1)
        
        if variable == "None":
        
            tosave.to_stata(f"{datafigures}/graduation_{coltype}.dta",write_index=False)
            
        else:
            tosave.to_stata(f"{datafigures}/graduation_{coltype}_{variable}_{level}.dta",write_index=False)
        
    elif graphtype == 2:
        
        if coltype == 1:
        
            share_real = real.groupby(["period"])["twograd"].mean()
            share_model = model.groupby(["period"])["twograd"].mean()
            
        elif coltype == 2:
            
            share_real = real.groupby(["period"])["fourgrad"].mean()
            share_model = model.groupby(["period"])["fourgrad"].mean()
            
        elif coltype == 3:
            
            share_real = real.groupby(["period"])["gradgrad"].mean()
            share_model = model.groupby(["period"])["gradgrad"].mean()
        
        leg = ["Conterfactual","Baseline"]
        fig = plt.subplots()
        plt.plot(share_real)
        plt.plot(share_model)
        plt.ylabel("Share graduated")
        plt.xlabel("Period")
        plt.legend(leg,loc='upper center', bbox_to_anchor=(0.5,-0.2), ncol = 2)
        plt.savefig(f"{figures}/conterfactual_graduation_{coltype}.png",bbox_inches='tight')
        
        # save data
        share_real = share_real.rename("conterfactual")
        share_model = share_model.rename("model")
        tosave = pd.concat([share_real,share_model],axis=1)
        
        tosave.to_stata(f"{datafigures}/data/conterfactual_graduation_{coltype}.dta",write_index=False)
        
        
def get_entrance_shares(real,model,coltype,graphtype=1):
    
    """This function computes the share of graduates each period,
    and saves the data to compute it with stata"""
    
    if graphtype == 1: 
        if coltype == 1:
            
            real["entrance"] = 0 
            real.loc[(real["twoyear_exp"]==0) 
                     & (real["educ"] ==1) , "entrance"] = 1
            
            model["entrance"] = 0 
            model.loc[(model["twoyear_exp"]==0) 
                     & (model["educ"] ==1) , "entrance"] = 1
        
            share_real = real.groupby(["period"])["entrance"].mean()
            share_model = model.groupby(["period"])["entrance"].mean()
        elif coltype == 2:
            
            real["entrance"] = 0 
            real.loc[(real["fouryear_exp"]==0) 
                     & (real["educ"] ==2) , "entrance"] = 1
            
            model["entrance"] = 0 
            model.loc[(model["fouryear_exp"]==0) 
                     & (model["educ"] ==2) , "entrance"] = 1
        
            share_real = real.groupby(["period"])["entrance"].mean()
            share_model = model.groupby(["period"])["entrance"].mean()
            
        elif coltype == 3:
            
            share_real = real.groupby(["period"])["gradgrad"].mean()
            share_model = model.groupby(["period"])["gradgrad"].mean()
        
        leg = ["Data","Model"]
        fig = plt.subplots()
        plt.plot(share_real)
        plt.plot(share_model)
        plt.ylabel("Entrance Share")
        plt.xlabel("Period")
        plt.legend(leg,loc='upper center', bbox_to_anchor=(0.5,-0.2), ncol = 2)
        plt.savefig(f"{figures}/entrance_coltype{coltype}.png",bbox_inches='tight')
    
        
    elif graphtype == 2:
        
        if coltype == 1:
        
            real["entrance"] = 0 
            real.loc[(real["twoyear_exp"]==0) 
                     & (real["educ"] ==1) , "entrance"] = 1
            
            model["entrance"] = 0 
            model.loc[(model["twoyear_exp"]==0) 
                     & (model["educ"] ==1) , "entrance"] = 1
        
            share_real = real.groupby(["period"])["entrance"].mean()
            share_model = model.groupby(["period"])["entrance"].mean()
        elif coltype == 2:
            
            real["entrance"] = 0 
            real.loc[(real["fouryear_exp"]==0) 
                     & (real["educ"] ==2) , "entrance"] = 1
            
            model["entrance"] = 0 
            model.loc[(model["fouryear_exp"]==0) 
                     & (model["educ"] ==2) , "entrance"] = 1
        
            share_real = real.groupby(["period"])["entrance"].mean()
            share_model = model.groupby(["period"])["entrance"].mean()
            
        elif coltype == 3:
            
            share_real = real.groupby(["period"])["gradgrad"].mean()
            share_model = model.groupby(["period"])["gradgrad"].mean()
        
        leg = ["Conterfactual","Baseline"]
        fig = plt.subplots()
        plt.plot(share_real)
        plt.plot(share_model)
        plt.ylabel("Entrance Share")
        plt.xlabel("Period")
        plt.legend(leg,loc='upper center', bbox_to_anchor=(0.5,-0.2), ncol = 2)
        plt.savefig(f"{figures}/conterfactual_entrance_coltype{coltype}.png",bbox_inches='tight')
        
       
        
        
def get_shares_dropout(real,model,coltype,graphtype=1):
    
    """This function computes the share of graduates each period,
    and saves the data to compute it with stata"""
    
    
    if graphtype == 1: 

        real = real[real["period"]<T]
        
        if coltype == 1:
        
            real["2ydropout"] = 0 
            real.loc[(real["period_educ_last"]==(real["period"]-1))
                     &(real["twoyear_exp"]!=0) 
                     &(real["twograd"]==0) 
                     & (real["educ"]!=2) , "2ydropout"] = 1
            
            
            model["2ydropout"] = 0 
            model.loc[(model["last_educ"]==(model["period"]-1))
                     &(model["twoyear_exp"]!=0) 
                     &(model["twograd"]==0) 
                     & (model["educ"]!=2) , "2ydropout"] = 1
            
            share_real = real.groupby(["period"])["2ydropout"].mean()
            share_model = model.groupby(["period"])["2ydropout"].mean()
            
        elif coltype == 2:
            

            real["4ydropout"] = 0 
            real.loc[(real["period_educ_last"]==(real["period"]-1))
                     &(real["fouryear_exp"]!=0) 
                     &(real["fourgrad"]==0) 
                     & (real["educ"]!=2) , "4ydropout"] = 1
            
            
            model["4ydropout"] = 0 
            model.loc[(model["last_educ"]==(model["period"]-1))
                     &(model["fouryear_exp"]!=0) 
                     &(model["fourgrad"]==0) 
                     & (model["educ"]!=2) , "4ydropout"] = 1
            
            share_real = real.groupby(["period"])["4ydropout"].mean()
            share_model = model.groupby(["period"])["4ydropout"].mean()
            
        elif coltype == 3:
            
            share_real = real.groupby(["period"])["gradgrad"].mean()
            share_model = model.groupby(["period"])["gradgrad"].mean()
        
        leg = ["Data","Model"]
        fig = plt.subplots()
        plt.plot(share_real)
        plt.plot(share_model)
        plt.ylabel("Drop Out Share")
        plt.xlabel("Period")
        plt.legend(leg,loc='upper center', bbox_to_anchor=(0.5,-0.2), ncol = 2)
        plt.savefig(f"{figures}/dropout_coltype{coltype}.png",bbox_inches='tight')
        
        # save data
        #share_real = share_real.rename("real")
        #share_model = share_model.rename("model")
        #tosave = pd.concat([share_real,share_model],axis=1)
        
        #tosave.to_stata(f"{figures}/data/graduation_{coltype}.dta",write_index=False)
        
    elif graphtype == 2:
        
        if coltype == 1:
        
            real["2ydropout"] = 0 
            real.loc[(real["last_educ"]==(real["period"]-1))
                     &(real["twoyear_exp"]!=0) 
                     &(real["twograd"]==0) 
                     & (real["educ"]!=2) , "2ydropout"] = 1
            
            
            model["2ydropout"] = 0 
            model.loc[(model["last_educ"]==(model["period"]-1))
                     &(model["twoyear_exp"]!=0) 
                     &(model["twograd"]==0) 
                     & (model["educ"]!=2) , "2ydropout"] = 1
            
            share_real = real.groupby(["period"])["2ydropout"].mean()
            share_model = model.groupby(["period"])["2ydropout"].mean()
            
        elif coltype == 2:
            
            real["4ydropout"] = 0 
            real.loc[(real["last_educ"]==(real["period"]-1))
                     &(real["fouryear_exp"]!=0) 
                     &(real["fourgrad"]==0) 
                     & (real["educ"]!=2) , "4ydropout"] = 1
            
            model["4ydropout"] = 0 
            model.loc[(model["last_educ"]==(model["period"]-1))
                     &(model["fouryear_exp"]!=0) 
                     &(model["fourgrad"]==0) 
                     & (model["educ"]!=2) , "4ydropout"] = 1
            
            share_real = real.groupby(["period"])["4ydropout"].mean()
            share_model = model.groupby(["period"])["4ydropout"].mean()
            
        elif coltype == 3:
            
            share_real = real.groupby(["period"])["gradgrad"].mean()
            share_model = model.groupby(["period"])["gradgrad"].mean()
        
        leg = ["Conterfactual","Baseline"]
        fig = plt.subplots()
        plt.plot(share_real)
        plt.plot(share_model)
        plt.ylabel("Drop Out Share")
        plt.xlabel("Period")
        plt.legend(leg,loc='upper center', bbox_to_anchor=(0.5,-0.2), ncol = 2)
        plt.savefig(f"{figures}/conterfactual_dropout_coltype{coltype}.png",bbox_inches='tight')
        
        # save data
        #share_real = share_real.rename("conterfactual")
        #share_model = share_model.rename("model")
        #tosave = pd.concat([share_real,share_model],axis=1)
        
        #tosave.to_stata(f"{figures}/data/conterfactual_graduation_{coltype}.dta",write_index=False)
        pass


def get_effect_debt(simu,conter,conter_maxdebt,conter_nodebt,conter_grants):
    
    debtrange = get_debt_range()
    
    simu = simu[(simu["fourgrad"]==1)&(simu["period"]==simu["last_educ"]+1) &(simu["last_choice"]!=13)]
    conter = conter[(conter["fourgrad"]==1)&(conter["period"]==conter["last_educ"]+1) &(conter["last_choice"]!=13)]
    conter_maxdebt = conter_maxdebt[(conter_maxdebt["fourgrad"]==1)&(conter_maxdebt["period"]==conter_maxdebt["last_educ"]+1) &(conter_maxdebt["last_choice"]!=13)]
   
    
    simu["anydebt"] = 0
    simu["currentloans"] = debtrange[np.array(simu["debt"])]
    simu.loc[simu["currentloans"]>0, "anydebt"] = 1
    
    conter["anydebt"] = 0
    conter["currentloans"] = debtrange[np.array(conter["debt"])]
    conter.loc[conter["currentloans"]>0, "anydebt"] = 1
    
    
    conter_maxdebt["anydebt"] = 0
    conter_maxdebt["currentloans"] = debtrange[np.array(conter_maxdebt["debt"])]
    conter_maxdebt.loc[conter_maxdebt["currentloans"]>0, "anydebt"] = 1
    
    simugroupany = simu["anydebt"].mean()
    conterany = conter["anydebt"].mean()
    contermaxany = conter_maxdebt["anydebt"].mean()
    
    simugroup = simu[simu["anydebt"]==1]["currentloans"].mean()
    contergroup = conter[conter["anydebt"]==1]["currentloans"].mean()
    contermaxgroup = conter_maxdebt[conter_maxdebt["anydebt"]==1]["currentloans"].mean()
    
    sharedebt = pd.Series(np.array([simugroupany,conterany,contermaxany]),name="Share_Debt")
    avdebt = pd.Series(np.array([simugroup,contergroup,contermaxgroup]),name="Average_Debt")
    
    debtgroups = pd.concat((sharedebt,avdebt),axis=1)
    
    return debtgroups

    

def get_effect_category(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,category):
    
    
    share_simu = simu[simu["period"]==T-1][f"{category}"].mean()
    share_conter = conter[conter["period"]==T-1][f"{category}"].mean()
    share_conter_maxdebt = conter_maxdebt[conter_maxdebt["period"]==T-1][f"{category}"].mean()
    share_conter_nodebt = conter_nodebt[conter_nodebt["period"]==T-1][f"{category}"].mean()
    share_conter_grants = conter_grants[conter_grants["period"]==T-1][f"{category}"].mean()
    
    allshares = np.array([share_simu,share_conter,share_conter_maxdebt,share_conter_nodebt,share_conter_grants])
        
    allshares = pd.DataFrame(allshares,columns=[f"{category}"])
       
    
    return allshares


def get_conterfactual_effects(simu,conter,conter_maxdebt,conter_nodebt,conter_grants):
    
    """
    This function computes the casual effects of the different conterfactual policies on: 
        - 4y graduation
        - 2y graduation
        - Grad school graduation
        - Working While Enrolled
        - Welfare
        - Average Debt at Graduation
    The effect will be computed for different groups: 
        - all
        - females
        - black
        - parinc
        - afqt
        - parinc X afqt
    """    
    #----------------------         All Groups       -------------------------#
    # ------------------------------------------------------------------------#
    effect4 = get_effect_category(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"fourgrad")
    effect2 = get_effect_category(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"twograd")
    effectg = get_effect_category(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"gradgrad")
    effectw = get_effect_category(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"welfare")
    effectd = get_effect_debt(simu,conter,conter_maxdebt,conter_nodebt,conter_grants)
    effectwk = get_effect_work(simu,conter,conter_maxdebt,conter_nodebt,conter_grants)
    
    sharesallgroups = pd.concat((effect4,effect2,effectg,effectw,effectd),axis=1)
    
    # Save it
    sharesallgroups.to_stata(f"{datafigures}/effects_all.dta",write_index=False)
    
    #-------------------------------------------------------------------------#
    #----------------------           Female         -------------------------#
    # ------------------------------------------------------------------------#
    effect4 = get_effect_category_groups(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"fourgrad")
    effect2 = get_effect_category(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"twograd")
    effectg = get_effect_category(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"gradgrad")
    effectw = get_effect_category(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"welfare")
    
    #-------------------------------------------------------------------------#
    
    
    # 4 year graduation ------------------------------------------------------#
    
    share_simu = simu[simu["period"]==T-1].groupby(["parinc","ability"])["fourgrad"].mean()
    share_conter = conter[conter["period"]==T-1].groupby(["parinc","ability"])["fourgrad"].mean()
    share_conter_maxdebt = conter_maxdebt[conter_maxdebt["period"]==T-1].groupby(["parinc","ability"])["fourgrad"].mean()
    share_conter_nodebt = conter_nodebt[conter_nodebt["period"]==T-1].groupby(["parinc","ability"])["fourgrad"].mean()
    share_conter_grants = conter_grants[conter_grants["period"]==T-1].groupby(["parinc","ability"])["fourgrad"].mean()
    
    #-------------------------------------------------------------------------#
    
    # 2 year graduation ------------------------------------------------------#
    
    share_simu = simu[simu["period"]==T-1].groupby(["parinc","ability"])["fourgrad"].mean()
    share_conter = conter[conter["period"]==T-1].groupby(["parinc","ability"])["fourgrad"].mean()
    share_conter_maxdebt = conter_maxdebt[conter_maxdebt["period"]==T-1].groupby(["parinc","ability"])["fourgrad"].mean()
    share_conter_nodebt = conter_nodebt[conter_nodebt["period"]==T-1].groupby(["parinc","ability"])["fourgrad"].mean()
    share_conter_grants = conter_grants[conter_grants["period"]==T-1].groupby(["parinc","ability"])["fourgrad"].mean()
    
    #-------------------------------------------------------------------------#
    
    
    
    return effect, effect_maxdebt, effect_nodebt, effect_grants
    
def check_graduation_conterfactual(simu,conter):
    
    
    share_simu = simu[simu["period"]==T-1].groupby(["parinc","ability"])["fourgrad"].mean()
    share_conter = conter[conter["period"]==T-1].groupby(["parinc","ability"])["fourgrad"].mean()
    
    effect = (share_conter-share_simu) 
    
    return effect


def check_fields_conterfactual(simu,conter,typef):
    
    
    share_simu = simu[(simu["period"]==T-1)&(simu["fourgrad"]==1)].groupby(["majorgrad"])["fourgrad"].sum() / np.shape(simu[(simu["period"]==T-1)&(simu["fourgrad"]==1)])[0]
    share_conter = conter[(conter["period"]==T-1)&(conter["fourgrad"]==1)].groupby(["majorgrad"])["fourgrad"].sum() / np.shape(conter[(conter["period"]==T-1)&(conter["fourgrad"]==1)])[0]
    
    a = (share_conter-share_simu)/share_simu
    
    effect = pd.DataFrame(a.rename(f"{typef}"))
    
    effect.to_stata(f"{datafigures}/fields_effect_conter_{typef}.dta",write_index=False)
    
    return effect


def check_fields_conterfactual_groups(simu,conter,contermax,conternodebt,contergrants,group,typeff):
    
    simu  = simu[simu[f"{group}"]==typeff]
    conter  = conter[conter[f"{group}"]==typeff]
    contermax  = contermax[contermax[f"{group}"]==typeff]
    conternodebt  = conternodebt[conternodebt[f"{group}"]==typeff]
    contergrants  = contergrants[contergrants[f"{group}"]==typeff]
    
    share_simu = simu[(simu["period"]==T-1)&(simu["fourgrad"]==1)].groupby(["majorgrad"])["fourgrad"].sum() / np.shape(simu[(simu["period"]==T-1)&(simu["fourgrad"]==1)])[0]
    share_conter = conter[(conter["period"]==T-1)&(conter["fourgrad"]==1)].groupby(["majorgrad"])["fourgrad"].sum() / np.shape(conter[(conter["period"]==T-1)&(conter["fourgrad"]==1)])[0]   
    share_contermax = contermax[(contermax["period"]==T-1)&(contermax["fourgrad"]==1)].groupby(["majorgrad"])["fourgrad"].sum() / np.shape(contermax[(contermax["period"]==T-1)&(contermax["fourgrad"]==1)])[0]  
    share_conternodebt = conternodebt[(conternodebt["period"]==T-1)&(conternodebt["fourgrad"]==1)].groupby(["majorgrad"])["fourgrad"].sum() / np.shape(conternodebt[(conternodebt["period"]==T-1)&(conternodebt["fourgrad"]==1)])[0]   
    share_contergrants = contergrants[(contergrants["period"]==T-1)&(contergrants["fourgrad"]==1)].groupby(["majorgrad"])["fourgrad"].sum() / np.shape(contergrants[(contergrants["period"]==T-1)&(contergrants["fourgrad"]==1)])[0]

    share_simu = share_simu.rename("model")
    share_conter = share_conter.rename("conter")
    share_contermax = share_contermax.rename("conter_maxdebt")
    share_conternodebt = share_conternodebt.rename("conter_nodebt")
    share_contergrants = share_contergrants.rename("conter_grants")

    joint = pd.concat([share_simu,share_conter,share_contermax,share_conternodebt,share_contergrants],axis=1).reset_index()
    
    joint.to_stata(f"{datafigures}/fields_effect_conter_{group}{typeff}.dta",write_index=False)

    pass


def where_coming(simu, contermax, conter_nodebt,conter_grants):
    
    contermax.rename(columns={"majorgrad":"majorgrad_maxdebt",
                              "fouryear_exp":"fouryear_exp_maxdebt",
                              "twoyear_exp":"twoyear_exp_maxdebt",
                              "gradgrad":"gradgrad_maxdebt",
                              "twograd":"twograd_maxdebt",
                              "fourgrad":"fourgrad_maxdebt",
                              "gepsi":"gepsi_maxdebt",
                              "gvalue":"gvalue_maxdebt"},inplace=True)
    
    conter_nodebt.rename(columns={"majorgrad":"majorgrad_nodebt",
                              "fouryear_exp":"fouryear_exp_nodebt",
                              "twoyear_exp":"twoyear_exp_nodebt",
                              "gradgrad":"gradgrad_nodebt",
                              "twograd":"twograd_nodebt",
                              "fourgrad":"fourgrad_nodebt"},inplace=True)
    
    conter_grants.rename(columns={"majorgrad":"majorgrad_grants",
                              "fouryear_exp":"fouryear_exp_grants",
                              "twoyear_exp":"twoyear_exp_grants",
                              "gradgrad":"gradgrad_grants",
                              "twograd":"twograd_grants",
                              "fourgrad":"fourgrad_grants"},inplace=True)    
    
    simu = pd.merge(simu, contermax[["PUBID","period","sample","majorgrad_maxdebt",
                                     "fouryear_exp_maxdebt","twoyear_exp_maxdebt",
                                     "gradgrad_maxdebt","twograd_maxdebt",
                                     "fourgrad_maxdebt","gvalue_maxdebt",
                                     "gepsi_maxdebt"]], how="left", on = ["PUBID","period","sample"])
    simu = pd.merge(simu, conter_nodebt[["PUBID","period","sample","majorgrad_nodebt",
                                         "fouryear_exp_nodebt","twoyear_exp_nodebt",
                                         "gradgrad_nodebt","twograd_nodebt",
                                         "fourgrad_nodebt"]], how="left", on = ["PUBID","period","sample"])
    simu = pd.merge(simu, conter_grants[["PUBID","period","sample","majorgrad_grants",
                                         "fouryear_exp_grants","twoyear_exp_grants",
                                         "gradgrad_grants","twograd_grants",
                                         "fourgrad_grants"]], how="left", on = ["PUBID","period","sample"])
        
    temp = simu[simu["period"]==9]
    
    #grou = temp[temp["majorgrad_maxdebt"]==8].groupby(["majorgrad"])["majorgrad"].count() / (np.shape(temp[temp["majorgrad_maxdebt"]==8])[0])
    
    #grou = temp[temp["majorgrad"]==2].groupby(["majorgrad_maxdebt"])["majorgrad_maxdebt"].count() / (np.shape(temp[temp["majorgrad"]==2])[0])
    
    return simu

def merge_both(simu, contermax):
    
    simutemp = simu.copy()
    contertemp  = contermax.copy()
    
    a = get_total_choices()
        
    simutemp["field"] = a[simutemp["choices"],0]
    simutemp["educ"] =a[simutemp["choices"],1]
    simutemp["work"] = a[simutemp["choices"],2]
    
    contertemp["field"] = a[contertemp["choices"],0]
    contertemp["educ"] =a[contertemp["choices"],1]
    contertemp["work"] = a[contertemp["choices"],2]
    
    
    
    contertemp.rename(columns={"majorgrad":"majorgrad_maxdebt",
                              "fouryear_exp":"fouryear_exp_maxdebt",
                              "twoyear_exp":"twoyear_exp_maxdebt",
                              "gradgrad":"gradgrad_maxdebt",
                              "twograd":"twograd_maxdebt",
                              "fourgrad":"fourgrad_maxdebt",
                              "gepsi":"gepsi_maxdebt",
                              "gvalue":"gvalue_maxdebt",
                              "choices":"choices_maxdebt",
                              "field":"field_maxdebt",
                              "educ":"educ_maxdebt",
                              "work":"work_maxdebt"},inplace=True)
      
    
    simutemp = pd.merge(simutemp, contertemp[["PUBID","period","sample","majorgrad_maxdebt",
                                     "fouryear_exp_maxdebt","twoyear_exp_maxdebt",
                                     "gradgrad_maxdebt","twograd_maxdebt",
                                     "fourgrad_maxdebt","gvalue_maxdebt",
                                     "gepsi_maxdebt","field_maxdebt",
                                     "educ_maxdebt","work_maxdebt",
                                     "choices_maxdebt"]], how="left", on = ["PUBID","period","sample"])
    
    return simutemp


def merge_both_vis(simu, contermax):
    
    simutemp = simu.copy()
    contertemp  = contermax.copy()
    
    a = get_total_choices()
        
    simutemp["field"] = a[simutemp["choices"],0]
    simutemp["educ"] =a[simutemp["choices"],1]
    simutemp["work"] = a[simutemp["choices"],2]
    
    contertemp["field"] = a[contertemp["choices"],0]
    contertemp["educ"] =a[contertemp["choices"],1]
    contertemp["work"] = a[contertemp["choices"],2]
    
    
    
    contertemp.rename(columns={"majorgrad":"majorgrad_maxdebt",
                              "fouryear_exp":"fouryear_exp_maxdebt",
                              "twoyear_exp":"twoyear_exp_maxdebt",
                              "gradgrad":"gradgrad_maxdebt",
                              "twograd":"twograd_maxdebt",
                              "fourgrad":"fourgrad_maxdebt",
                              "choices":"choices_maxdebt",
                              "field":"field_maxdebt",
                              "educ":"educ_maxdebt",
                              "work":"work_maxdebt"},inplace=True)
      
    
    simutemp = pd.merge(simutemp, contertemp[["PUBID","period","sample","majorgrad_maxdebt",
                                     "fouryear_exp_maxdebt","twoyear_exp_maxdebt",
                                     "gradgrad_maxdebt","twograd_maxdebt",
                                     "fourgrad_maxdebt","field_maxdebt",
                                     "educ_maxdebt","work_maxdebt",
                                     "choices_maxdebt"]], how="left", on = ["PUBID","period","sample"])
    
    return simutemp

def table_with_gvalues(allsimu, allcontermax):
    
    data = merge_both(allsimu, allcontermax)
    #datavis = merge_both_vis(allsimu_good,allconter_good)
    data["never"] = 1 - np.minimum(data["fouryear_exp"] + data["twoyear_exp"],1)
    data["never_maxdebt"] = 1 - np.minimum(data["fouryear_exp_maxdebt"] + data["twoyear_exp_maxdebt"],1)
    
    data["educ_exp"] = data["fouryear_exp"] + data["twoyear_exp"]
    data["educ_exp_maxdebt"] = data["fouryear_exp_maxdebt"] + data["twoyear_exp_maxdebt"]

    
    data["drop"] = np.minimum(data["educ_exp"],1) - np.minimum(data["fourgrad"]+data["twograd"],1)
    data["drop_maxdebt"] = np.minimum(data["educ_exp_maxdebt"],1) - np.minimum(data["fourgrad_maxdebt"]+data["twograd_maxdebt"],1)

    
    data["maxtwograd"] = np.maximum(data["majorgrad"]-11,0)
    data["type2"] = data["type"] - 1
    
    data["type1"] = 1 - data["type2"]
    
    data["new"] = np.maximum(data["never"] - data["never_maxdebt"],0)
    
    
    data["switch"] = (data["fourgrad"]==1)  & (data["majorgrad"]!=data["majorgrad_maxdebt"])
    data["switch"] = data["switch"].astype("int")
    
    # include dummy for graduated individuals:

    data = pd.merge(data, data.groupby(["PUBID","sample"])["fourgrad"].max().reset_index(),
                    on = ["PUBID","sample"])
    
    # Find minimum period before graduatoin (graduation period)
    
    data = pd.merge(data, data[data["fourgrad_x"]==0].groupby(["PUBID","sample"])["period"].max().reset_index(),
                    on = ["PUBID","sample"])
    
       
    
    
    #temp = data
    data["nonwork"] = 0 
    data.loc[(data["educ"]==2) & (data["educ_maxdebt"]==2)
             & (data["work"]>0) & (data["work_maxdebt"]==0) , "nonwork"] = 1 

    data["lesswork"] = 0 
    data.loc[(data["educ"]==2) & (data["educ_maxdebt"]==2)
             & (data["work"]>data["work_maxdebt"]) , "lesswork"] = 1
    
    data["morework"] = 0 
    data.loc[(data["educ"]==2) & (data["educ_maxdebt"]==2)
             & (data["work"]<data["work_maxdebt"]) , "morework"] = 1
    
    data["samework"] = 0 
    data.loc[(data["educ"]==2) & (data["educ_maxdebt"]==2)
             & (data["work"]==data["work_maxdebt"]) , "samework"] = 1
    
    
    data["changefield"] = 0
    data.loc[(data["educ"]==2) & (data["educ_maxdebt"]==2)
             & (data["field"] != data["field_maxdebt"]), "changefield"] = 1
        
    temp = data[(data["fourgrad_y"]==1) & (data["period_x"]==data["period_y"])]

    #temp.to_stata(f"{pathstata}/analysis.dta",write_index=False)
    
    switch_work_decomposition = switchers_work_decomposition(data)
    enrollment_4years = enrollment_effect_decomp(data)
    labor_supply_change = get_labor_distribution(data)
    switchersfields = switchers_debt_decomposition(data)
    markovfields= big_switch_table(data)
    markovfieldspoor= big_switch_table(data[data["parinc"]==1])
    markovfieldsrich= big_switch_table(data[data["parinc"]==4])
    
    a = temp.groupby(["nonwork"])[["gepsi","gepsi_maxdebt"]].mean()
    a = temp.groupby(["changefield","work_maxdebt"])[["gepsi","gepsi_maxdebt"]].mean()
    a = temp[temp["changefield"]==1].groupby(["field_maxdebt"])[["gepsi","gepsi_maxdebt"]].mean()
    
    a = temp.groupby(["majorgrad_maxdebt"])[["gepsi","gepsi_maxdebt"]].mean()    
    
    a = temp[(temp["educ"]==2) & (temp["morework"]==1)
             &(temp["educ_maxdebt"]==2)
             & (temp["changefield"]==1)].groupby(["field_maxdebt"])[["gepsi","gepsi_maxdebt"]].mean() 
    
    a = temp[(temp["educ"]==2) & (temp["parinc"]==1)
         &(temp["educ_maxdebt"]==2) &(temp["changefield"]==1)
         &(temp["work_maxdebt"]==0)].groupby(["field"])[["gepsi","gepsi_maxdebt"]].mean() 

    a = temp[(temp["educ"]==2) & (temp["field"]==6)
         &(temp["educ_maxdebt"]==2) &(temp["changefield"]==1)
         &(temp["field_maxdebt"]!=3)&(temp["parinc"]==4)].groupby(["field_maxdebt"])[["gepsi","gepsi_maxdebt"]].mean() 


    a = temp[(temp["changefield"]==1)
             &(temp["parinc"]==1)].groupby(["field"])[["nonwork","lesswork"]].mean()


    a = temp[(temp["changefield"]==1)
             &(temp["parinc"]==4)].groupby(["field_maxdebt"])[["nonwork","lesswork"]].mean()

    
    a = temp[(temp["educ"]==2) & (temp["educ_maxdebt"]==2)
             & (temp["parinc"]==1)].groupby(["changefield"])[["nonwork","lesswork"]].mean()
    
    
    a = temp[(temp["educ"]==2) & (temp["educ_maxdebt"]==2)].groupby(["ability"])["changefield"].mean()

    a = temp[(temp["educ"]==2) & (temp["parinc"]==1)
             &(temp["educ_maxdebt"]==2) &(temp["changefield"]==1)
             &(temp["work_maxdebt"]==0)].groupby(["field_maxdebt"])[["gepsi","gepsi_maxdebt"]].mean() 

    a = temp[(temp["changefield"]==1) & (temp["nonwork"]==1)
             & (temp["parinc"]==1)].groupby(["field_maxdebt"])["field_maxdebt"].count()  
    
    a = temp[(temp["educ"]==2)
             & (temp["educ_maxdebt"]==2)
             & (temp["work"]==2)].groupby(["ability"])[["nonwork","lesswork"]].mean()
    
    a = temp[(temp["educ"]==2)
             & (temp["educ_maxdebt"]==2)
             & (temp["work"]>0)].groupby(["parinc"])[["lesswork"]].mean()
    
    a = temp[temp["educ"]==2].groupby(["parinc"])["nonwork"].mean()

    a = data[data["fourgrad"]==1].groupby(["majorgrad"])[["gvalue","gepsi"]].mean()
    aa = data[data["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])[["gvalue_maxdebt","gepsi_maxdebt"]].mean()

    a = temp[(temp["educ"]==2) & (temp["parinc"]==1)
             &(temp["period_x"]==6)
             &(temp["debt"]>0)].groupby(["changefield"])[["lesswork"]].mean()
    
    a = temp[(temp["educ"]==2)
             &(temp["period_x"]==6)
             &(temp["debt"]==0)].groupby(["parinc"])[["lesswork"]].mean()
    
    
def switchers_debt_field(data):
    
    df = data.copy()
    df["anydebt"] = 0
    df.loc[df["debt"]>0, "anydebt"] = 1 
    # identify first switch: 
        
    df["more_g"] = (df["gepsi_maxdebt"] > df["gepsi"]).astype("int")

    df = pd.merge(df, df[df["period_x"]==9][["PUBID","sample","more_g"]].reset_index(),
                    on = ["PUBID","sample"])
     
    df = pd.merge(df, df[df["changefield"]==1].groupby(["PUBID","sample"])["period_x"].min().reset_index(),
                    on = ["PUBID","sample"])
       
    df = df[df["period_x_x"]==df["period_x_y"]]

    # Table Switch In Switch Out
    
    same = df[(df["samework"]==1) & (df["field"]!=3)].groupby(["field"])[["more_g_x"]].mean()
    less = df[(df["lesswork"]==1) & (df["field"]!=3)].groupby(["field"])[["more_g_x"]].mean()
    more = df[(df["morework"]==1) & (df["field"]!=3)].groupby(["field"])[["more_g_x"]].mean()
    
    goingout = np.array(pd.concat([same,less,more],axis=1))
    
    same = df[(df["samework"]==1) & (df["field_maxdebt"]!=3)].groupby(["field_maxdebt"])[["more_g_x"]].mean()
    less = df[(df["lesswork"]==1) & (df["field_maxdebt"]!=3)].groupby(["field_maxdebt"])[["more_g_x"]].mean()
    more = df[(df["morework"]==1) & (df["field_maxdebt"]!=3)].groupby(["field_maxdebt"])[["more_g_x"]].mean()
    
    goinin= np.array(pd.concat([same,less,more],axis=1))
    
    table = np.concatenate((goingout,goinin),axis=1)
    
    k,m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
    
    column_names =  " & Same $ Less & More & Same & Less & More \\\\  \\hline \n"
    latex_table += column_names
    
    rowanmes = ["Business","STEM","Education","Social", "Humanities", "Health","Other"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i,j]:.2f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Non-Consumption Change}\n\\end{table}"
    
    
    return latex_table
    
    
def group_majors_fast(df,major):
    
    return df[df["majorgrad"]==major].groupby(["majorgrad_maxdebt"])["majorgrad_maxdebt"].count() / np.shape(df[df["majorgrad"]==major])[0]
    
def big_switch_table(data):
    
    df = data[(data["fourgrad_x"]==1) & (data["fourgrad_maxdebt"]==1)].copy()
    df = df[df["period_x"]==9]
    #df = df[df["parinc"]==4]
    
    business = group_majors_fast(df,1).T 
    stem = group_majors_fast(df,2)
    educ = group_majors_fast(df,4)
    soc = group_majors_fast(df,5)
    hum = group_majors_fast(df,6)
    hea = group_majors_fast(df,7)
    oth = group_majors_fast(df,8)
    
    table = np.array(pd.concat([business,stem,educ,soc,hum,hea,oth],axis=1)).T
    
    
    k,m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
    
    column_names =  " & \\rot{Business} & \\rot{STEM} & \\rot{Education} & \\rot{Social Sciences} & \\rot{Humanities} & \\rot{Health} & \\rot{Other}  \\\\  \\hline \n"
    latex_table += column_names
    
    rowanmes = ["Business","STEM","Education","Social", "Humanities", "Health","Other"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i,j]:.2f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Field Changes}\n\\end{table}"
    
    
    return latex_table
    
def drop_debt(data):
    
    df =data.copy()
    
    df["lasteduc"] = 0 
    df.loc[(df["last_educ"]==df["period_x"]-1)
           & (df["last_choice"]!=12), "lasteduc"] = 1
    
    
    
    df["tempdrop"] = 0 
    df.loc[(df["lasteduc"]==1) &
           (df["educ"]!=2) &
           (df["fourgrad_x"]==0), "tempdrop"] = 1

    a = df[df["lasteduc"]==1].groupby(["parinc"])["tempdrop"].mean()
    a = df[(df["lasteduc"]==1)
           & (df["debt"]==0)].groupby(["parinc"])["tempdrop"].mean()



def group_labor_educ(df):
    
    base1 = df[df["educ"]==1].groupby(["work"])["work"].count()/(np.shape(df[df["educ"]==1])[0])
    base2 = df[df["educ"]==2].groupby(["work"])["work"].count()/(np.shape(df[df["educ"]==2])[0])
    base3 = df[df["educ"]==3].groupby(["work"])["work"].count()/(np.shape(df[df["educ"]==3])[0])

    base = pd.concat([base1, base2, base3],axis=0)
    
    conter1 = df[df["educ_maxdebt"]==1].groupby(["work_maxdebt"])["work_maxdebt"].count()/(np.shape(df[df["educ_maxdebt"]==1])[0])
    conter2 = df[df["educ_maxdebt"]==2].groupby(["work_maxdebt"])["work_maxdebt"].count()/(np.shape(df[df["educ_maxdebt"]==2])[0])
    conter3 = df[df["educ_maxdebt"]==3].groupby(["work_maxdebt"])["work_maxdebt"].count()/(np.shape(df[df["educ_maxdebt"]==3])[0])

    conter = pd.concat([conter1, conter2, conter3],axis=0)
    
    change = (conter / base-1)*100
    
    return pd.concat([base,conter,change],axis=1)
    

def get_labor_distribution(data):
    
    df = data.copy()
    
    alldebt = group_labor_educ(df)
    nodebt = group_labor_educ(df[df["debt"]==0])
    debt = group_labor_educ(df[df["debt"]>0])
    
    table = np.array(pd.concat([alldebt,nodebt,debt],axis=1))
        
    k,m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
    
    column_names =  " & Base & Conter & Base & Conter & Base & Conter  \\\\  \\hline \n"
    latex_table += column_names
    
    rowanmes = ["Two-Year, No Work","Two-Year, Part-Time", "Two-Year, Full-Time",
                "Four-Year, No Work","Four-Year, Part-Time","Four-Year,Full-Time",
                "Grad School, No Work","Grad School, Part-Time","Grad School, Full-Time"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i,j]:.2f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Labor Supply Distribution While Enrolled}\n\\end{table}"
    
    
    return latex_table

def get_gradschool(data):
    
    df = data.copy()
    df = df[df["period_x"]==9]
    
    gradbase = getgroup(df,"gradgrad")
    gradconter = getgroup(df,"gradgrad_maxdebt")
    change = (gradconter/gradbase -1)*100
    # grad new
    
    df["evergrad"] = 0
    df.loc[(df["grad"]>0) , "evergrad"] = 1
    
    df["nevergrad"] = 1-df["evergrad"]
    
    df["dropgrad"] = 0
    df.loc[(df["evergrad"]==1) &
           (df["gradgrad"]==0), "dropgrad"] = 1
    
    previousdrop = getgroup(df[(df["gradgrad_maxdebt"]==1)
           & (df["gradgrad"]==0)],"dropgrad")
    
    previousgain = change*previousdrop
    
    newgrad = getgroup(df[(df["gradgrad_maxdebt"]==1)
           & (df["gradgrad"]==0)],"nevergrad")
    
    newgain = change*newgrad
    
    # previous grad not grad anymore
    
    previousgrad = 1 - getgroup(df[(df["gradgrad"]==1)],"gradgrad_maxdebt")
    
    
    table = pd.concat([gradbase,gradconter,change,
                       previousdrop,previousgain,
                       newgrad,newgain],axis=1)
    
def getgroup(data,variable):
    
    parinc = data.groupby(["parinc"])[f"{variable}"].mean()
    ability = data.groupby(["ability"])[f"{variable}"].mean()
    sex = data.groupby(["sex"])[f"{variable}"].mean()
    race = data.groupby(["ethnicity"])[f"{variable}"].mean()
    
    return pd.concat([parinc,ability,sex,race],axis=0)
    


def overall_effect(data):
    
    # works with the data that has 10 periods!
    
    df = data.copy()
    df = df[df["period"]==10]
  
    
    df["ever4_maxdebt"] = 0 
    df.loc[(df["fouryear_exp_maxdebt"]>0) , "ever4_maxdebt"] = 1
    
    df["ever4"] = 0 
    df.loc[(df["fouryear_exp"]>0) , "ever4"] = 1
    
    df["ever2_maxdebt"] = 0 
    df.loc[(df["twoyear_exp_maxdebt"]>0) , "ever2_maxdebt"] = 1
    
    df["ever2"] = 0 
    df.loc[(df["twoyear_exp"]>0) , "ever2"] = 1
    
    
    
    # Two Year Enrollment
    base = df["ever2"].mean()
    count = df["ever2_maxdebt"].mean()
    ppt = count - base 
    per = (count/base -1)*100
    
    twoyeare = np.array([base,count,ppt,per])
    
    # Four year enrollment 
    base = df["ever4"].mean()
    count = df["ever4_maxdebt"].mean()
    ppt = count - base 
    per = (count/base -1)*100
    
    fouryeare = np.array([base,count,ppt,per])
    
    # Two Year Graduation
    base = df["twograd"].mean()
    count = df["twograd_maxdebt"].mean()
    ppt = count - base 
    per = (count/base -1)*100
    twoyearg = np.array([base,count,ppt,per])
    
    # Four Year Graduation
    
    base = df["fourgrad"].mean()
    count = df["fourgrad_maxdebt"].mean()
    ppt = count - base 
    per = (count/base -1)*100
    fouryearg = np.array([base,count,ppt,per])
    
    # Grad School Graduation
    
    base = df["gradgrad"].mean()
    count = df["gradgrad_maxdebt"].mean()
    ppt = count - base 
    per = (count/base -1)*100
    gradg = np.array([base,count,ppt,per])
    


def enrollment_effect_decomp(data):
    
    """ This function computes the table for new enrolled individuals,
    and who drops out, who graduates and  etc. 
    """
    df = data.copy()
    df = df[df["period_x"]==9]
  
    
    df["ever4"] = 0 
    df.loc[(df["fouryear_exp_maxdebt"]>0) , "ever4"] = 1
    
    #df["ever2"] = 0 
    #df.loc[(df["twoyear_exp_maxdebt"]>0) &
    #       (df["fouryear_exp_maxdebt"]==0) , "ever2"] = 1
    
    df["drop4"] = 0
    df.loc[(df["ever4"]==1) &
           (df["fourgrad_maxdebt"]==0), "drop4"] = 1
    
    #df["drop2"] = 0
    #df.loc[(df["ever2"]==1) &
    #       (df["twograd_maxdebt"]==0)
    #       & (df["fourgrad_maxdebt"]==0), "drop2"] = 1
    
    # who enrolls
    enrollment = np.array(getgroup(df[df["never"]==1],"ever4"))
    enrollmentype = np.array(getgroup(df[(df["never"]==1) & (df["ever4"]==1)],"type2"))
    
    drop = np.array(getgroup(df[(df["never"]==1)&(df["ever4"]==1)],"drop4"))
    droptype = np.array(getgroup(df[(df["never"]==1)&(df["ever4"]==1) & (df["drop4"]==1)],"type2"))

    grad = np.array(getgroup(df[(df["never"]==1)&(df["ever4"]==1)],"fourgrad_maxdebt"))
    gradtype = np.array(getgroup(df[(df["never"]==1)&(df["ever4"]==1) & (df["fourgrad_maxdebt"]==1)],"type2"))

        
    table = np.concatenate((enrollment[...,None],enrollmentype[...,None],
                            drop[...,None],droptype[...,None],
                            grad[...,None],gradtype[...,None]),axis=1)
    
    k,m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
    
    column_names =  " & \multicolumn{2}{c}{Enrollment} & \multicolumn{2}{c}{Drop Out} & \multicolumn{2}{c}{Graduate}  \\\\  \\hline \n"
    latex_table += column_names
    
    rowanmes = ["ParInc Q1","ParInc Q2", "ParInc Q3", "ParInc Q4",
                "AFQT Q1", "AFQT Q2", "AFQT Q3", "AFQT Q4",
                "Male","Female","Non Black","Black"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i,j]:.2f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Enrollment and paths}\n\\end{table}"
    
    
    return latex_table
    

def group_dropout4(data,level):
    
    
    groupab = data.groupby(["ability"])[f"{level}"].mean()
    grouppar = data.groupby(["parinc"])[f"{level}"].mean()
    groupsex = data.groupby(["sex"])[f"{level}"].mean()
    groupblack = data.groupby(["ethnicity"])[f"{level}"].mean()
    
    return np.array(pd.concat([grouppar,groupab, groupsex, groupblack],axis=0))
    
    
    
    


def table_drop_decomposition(data):
    
    df = data.copy()
    df = df[df["period"]==9]
    
    df["ever4"] = 0 
    df.loc[(df["fouryear_exp"]>0) , "ever4"] = 1
    
    df["ever4_maxdebt"] = 0 
    df.loc[(df["fouryear_exp_maxdebt"]>0) , "ever4_maxdebt"] = 1

    df["drop4"] = np.minimum(data["fouryear_exp"],1) - data["fourgrad"]
    df["drop4"] = np.maximum(df["drop4"] - df["twograd"],0)
    
    real["drop4"] = 0
    real.loc[(real["fouryear_exp"]>0) &
           (real["fourgrad"]==0), "drop4"] = 1
    
    real["drop"] = np.minimum(real["fouryear_exp"],1) - real["fourgrad"]
    real["drop"] = np.maximum(real["drop"] - real["twograd"],0)
    
    df["drop4"] = 0
    df.loc[(df["ever4"]==1) &
           (df["fourgrad"]==0), "drop4"] = 1
    
    df["drop4_maxdebt"] = 0
    df.loc[(df["ever4_maxdebt"]==1) &
           (df["fourgrad_maxdebt"]==0), "drop4_maxdebt"] = 1
    
    
    # I need to identify drop out among new enrolled,
    # and drop out amoung preivous enrolled. 
    
    # Overall Effect
    
    df[(df["fouryear_exp_maxdebt"]>0)]["drop4_maxdebt"].mean()
    df[(df["fouryear_exp"]>0)]["drop4"].mean()
    
    # Real
    realdrop = group_dropout4(real[(real["fouryear_exp"]>0)
                                   & (real["period"]==9)],"drop4")
    simudrop = group_dropout4(df[(df["fouryear_exp"]>0)],"drop4")
    
    conterdrop = group_dropout4(df[(df["fouryear_exp_maxdebt"]>0)],"drop4_maxdebt")
    
    # drop out previous people
    previousdrop = group_dropout4(df[(df["fouryear_exp"]>0)],"drop4_maxdebt")
    
    # drop out new people
    
    newdrop = group_dropout4(df[(df["ever4_maxdebt"]>0)
                                &(df["never"]==1)],"drop4_maxdebt")
    
    # drop out previous twoyear
    
    droptwo = group_dropout4(df[(df["ever4_maxdebt"]>0)
                                &(df["fouryear_exp"]==0)
                                &(df["twoyear_exp"]>0)],"drop4_maxdebt")
    
    table = np.concatenate((realdrop[...,None],simudrop[...,None],
                            conterdrop[...,None],previousdrop[...,None],
                            newdrop[...,None]),axis=1)
    
    k,m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
    
    column_names =  " & Data & Baseline & SAVE & Previous Enrolled & New Enrolled  \\\\  \\hline \n"
    latex_table += column_names
    
    rowanmes = ["ParInc Q1","ParInc Q2", "ParInc Q3", "ParInc Q4",
                "AFQT Q1", "AFQT Q2", "AFQT Q3", "AFQT Q4",
                "Male","Female","Non Black","Black"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i,j]:.2f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Drop Out Decomposition}\n\\end{table}"
    
    
    return table
    

    
    

        
    
def switchers_work_decomposition(temp):
    
    temp = data[data["debt"]==0]
    temp[temp["changefield"]==1][["samework","lesswork","morework"]].mean()
    parinc = temp[temp["changefield"]==1].groupby(["parinc"])[["samework","lesswork","morework"]].mean()
    ability = temp[temp["changefield"]==1].groupby(["ability"])[["samework","lesswork","morework"]].mean()
    sex = temp[temp["changefield"]==1].groupby(["sex"])[["samework","lesswork","morework"]].mean()
    race = temp[temp["changefield"]==1].groupby(["ethnicity"])[["samework","lesswork","morework"]].mean()

    tablenodebt =np.array( pd.concat([parinc,ability,sex,race],axis=0))
    
    temp = data[data["debt"]>0]
    temp[temp["changefield"]==1][["samework","lesswork","morework"]].mean()
    parinc = temp[temp["changefield"]==1].groupby(["parinc"])[["samework","lesswork","morework"]].mean()
    ability = temp[temp["changefield"]==1].groupby(["ability"])[["samework","lesswork","morework"]].mean()
    sex = temp[temp["changefield"]==1].groupby(["sex"])[["samework","lesswork","morework"]].mean()
    race = temp[temp["changefield"]==1].groupby(["ethnicity"])[["samework","lesswork","morework"]].mean()

    tabledebt =np.array( pd.concat([parinc,ability,sex,race],axis=0))
    
    table = np.concatenate((tablenodebt,tabledebt),axis=1)
    
    k,m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
    
    column_names =  " & Same & Less & More  & Same & Less & More   \\\\  \\hline \n"
    latex_table += column_names
    
    rowanmes = ["ParInc Q1","ParInc Q2", "ParInc Q3", "ParInc Q4",
                "AFQT Q1", "AFQT Q2", "AFQT Q3", "AFQT Q4",
                "Male","Female","Non Black","Black"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i,j]:.2f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Changes Across Work}\n\\end{table}"
    
    
    return latex_table

def switchers_debt_decomposition(data):
    
    df = data.copy()
    df = df[(df["fourgrad_x"]==1)& (df["period_x"]==9)]
    
    df["anydebt"] = 0
    df.loc[df["debt"]>0,"anydebt"] = 1
    
    allswitch = group_dropout4(df,"switch")
    switchdebt = group_dropout4(df[df["debt"]>0],"switch")
    switchnodebt = group_dropout4(df[df["debt"]==0],"switch")
    change = (switchdebt / switchnodebt-1)*100
    
    a = group_dropout4(df[df["grad"]==0],"anydebt")
    
    
    table = np.concatenate((switchnodebt[...,None],
                        switchdebt[...,None],
                        change[...,None]),axis=1)
    
    
    k,m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
    
    column_names =  " & Not Indebted & Indebted &   \\\\  \\hline \n"
    latex_table += column_names
    
    rowanmes = ["ParInc Q1","ParInc Q2", "ParInc Q3", "ParInc Q4",
                "AFQT Q1", "AFQT Q2", "AFQT Q3", "AFQT Q4",
                "Male","Female","Non Black","Black"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i,j]:.2f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Differences in Switching Field Behavior}\n\\end{table}"
    
    
    return latex_table

def get_decomposition_changefield(temp):
        
    #temp = data
    
    temp= data.copy()
        
    original = temp[temp["educ"]==2].groupby(["field"])["field"].count() / np.shape(temp[temp["educ"]==2])[0]
    new = temp[temp["educ_maxdebt"]==2].groupby(["field_maxdebt"])["field_maxdebt"].count() / np.shape(temp[temp["educ_maxdebt"]==2])[0]
    
    changers = temp[(temp["changefield"]==1)
                    & (temp["parinc"]==4)
                    & (temp["morework"]==1)].groupby(["field_maxdebt"])["field_maxdebt"].count() / np.shape(temp[(temp["changefield"]==1)& (temp["parinc"]==4)& (temp["lesswork"]==1)])[0]
    
    originaltemp = temp[temp["fourgrad_x"]==1].groupby(["majorgrad"])["majorgrad"].count()
    
    #switch
    newswitch = temp[temp["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])["switch"].sum() 
    newleave = temp[temp["fourgrad_x"]==1].groupby(["majorgrad"])["switch"].sum()
    netswitch = newswitch - newleave
   

    originaltemp = originaltemp + netswitch
    originaltempshare = originaltemp / np.sum(originaltemp)
    # Percentage of each field that are new enrolled
    
    newenr = temp[temp["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])["new"].sum() 
    
    # percentage that are previous dropouts
    newdrops = temp[temp["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])["drop"].sum() 
    
        # percentage that are coming from two.year schools
    
    newtwo = temp[temp["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])["maxtwograd"].sum()
    
    
    switchweight = netswitch.copy()
    switchweight[switchweight<0] = switchweight*-1
    weights = pd.concat([switchweight,newenr,newdrops,newtwo],axis=1)
    weights = np.array(weights) /np.array(weights.sum(axis=1))[...,None]
    
    gain = (new - original)*100
    netswitchgood = ((originaltemp+netswitch)/np.sum(originaltemp+netswitch) - original)*100 
    newenr = ((originaltemp+newenr)/np.sum(originaltemp+newenr) - original)*100
    newdrops = ((originaltemp+newdrops)/np.sum(originaltemp+newdrops) - original)*100
    newtwo = ((originaltemp+newtwo)/np.sum(originaltemp+newtwo) - original)*100 
    
    
    table = np.array(pd.concat([gain,netswitchgood,newenr,newdrops,newtwo],axis=1))  
    table[:,1:] = table[:,1:]*weights
    
    table = table/table[:,0][...,None]
    
    table[:,1] = table[:,1] -1 
    table = table[:,1:]
    table[:,2] = table[:,2]*-1

    k,m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
    
    
    column_names =  " & Change & In & Out & Net & New Enrolled & Previous Drop Out & Previous Two  \\\\  \\hline \n"
    latex_table += column_names
    
    rowanmes = ["\hspace{13mm} Business","\hspace{13mm} STEM","\hspace{13mm} Education",
                "\hspace{13mm} Social Sciences","\hspace{13mm} Humanities",
                "\hspace{13mm} Health","\hspace{13mm} Other"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i,j]:.2f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Changes Across Fields}\n\\end{table}"
    
    
    return latex_table
    


def table_movement_conter(simu, contermax, conter_nodebt, conter_grants):
    
    data =  where_coming(simu, contermax, conter_nodebt,conter_grants)
    
    data["never"] = 1 - np.minimum(data["fouryear_exp"] + data["twoyear_exp"],1)
    data["never_maxdebt"] = 1 - np.minimum(data["fouryear_exp_maxdebt"] + data["twoyear_exp_maxdebt"],1)
    data["never_nodebt"] = 1 - np.minimum(data["fouryear_exp_nodebt"] + data["twoyear_exp_nodebt"],1)
    
    data["educ_exp"] = data["fouryear_exp"] + data["twoyear_exp"]
    data["educ_exp_maxdebt"] = data["fouryear_exp_maxdebt"] + data["twoyear_exp_maxdebt"]
    data["educ_exp_nodebt"] = data["fouryear_exp_nodebt"] + data["twoyear_exp_nodebt"]

    
    data["drop"] = np.minimum(data["educ_exp"],1) - np.minimum(data["fourgrad"]+data["twograd"],1)
    data["drop_maxdebt"] = np.minimum(data["educ_exp_maxdebt"],1) - np.minimum(data["fourgrad_maxdebt"]+data["twograd_maxdebt"],1)
    data["drop_nodebt"] = np.minimum(data["educ_exp_nodebt"],1) - np.minimum(data["fourgrad_nodebt"]+data["twograd_nodebt"],1)

    data = data[data["period"]==9]
    
    data["maxtwograd"] = np.maximum(data["majorgrad"]-11,0)
    data["maxtwograd_nodebt"] = np.maximum(data["majorgrad_nodebt"]-11,0)
    data["type2"] = data["type"] - 1
    
    data["type1"] = 1 - data["type2"]
    
    data["new"] = np.maximum(data["never"] - data["never_maxdebt"],0)
    
    
    
    a = data.groupby(["new"])["type1"].mean()
    a = data[data["new"]==1].groupby(["fourgrad_maxdebt"])["type1"].mean()
    
    
    data["switch"] = (data["fourgrad"]==1)  & (data["majorgrad"]!=data["majorgrad_maxdebt"])
    data["switch"] = data["switch"].astype("int")
    
    data["switch_nodebt"] = (data["fourgrad_nodebt"]==1)  & (data["majorgrad_nodebt"]!=data["majorgrad"])
    data["switch_nodebt"] = data["switch_nodebt"].astype("int")
    
    
    
    overall = data[data["fourgrad"]==1].groupby(["majorgrad"])["fourgrad"].count() / (np.shape(data[data["fourgrad"]==1])[0])
    newdist = data[data["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])["fourgrad_maxdebt"].count() / (np.shape(data[data["fourgrad_maxdebt"]==1])[0])
    
    a = data[data["majorgrad"]==2].groupby(["majorgrad_maxdebt"])["majorgrad_maxdebt"].count() / (np.shape(data[data["majorgrad"]==2])[0])
    
    
    a = np.array(newdist) / np.array(overall) 
    
    table_movement = latex_table_movements(data)
    
    table_wherefourgrad = latex_table_newgrads(data)
    
    table_decomposition = latex_table_newgrads_decomposition(data)
    table_decomposition2 = latex_table_newgrads_decomposition2(data)
    table_decom_enroll =  latex_table_newenrolled_decomposition(data)
    
    table_type1_dist = latex_table_type1newenrolled(data)
    
    table_elasticities = get_switchers(data)
    
    table_fieldsparinc1 = get_decomposition(data[data["parinc"]==1])
    table_fieldsparinc4 = get_decomposition(data[data["parinc"]==4])
    
    table_fielddist = table_field_dist(data)
    
    
    # Now set no debt as baseline and introduce loans
    
    table_decomnodebt = latex_table_newgrads_decomposition_nodebt(data)
    
    table_elasticitiesnodebt = get_switchers_nodebt(data)
    
    table_fielddistno = table_field_distnodebt(data)
    

def newwelfare_(data):
    
    temp = data[(data["switch"]==1)&(data["parinc"]==1)]
    
    a = temp[temp["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])[["gepsi_maxdebt"]].mean()
    aa = temp.groupby(["majorgrad"])[["gepsi"]].mean()
    
def where_coming_vis(data):
    
    temp = data[data["switch"]==1]
    
    temp = temp[temp["majorgrad_maxdebt"]==4]
    
    a = temp[temp["parinc"]==4].groupby(["majorgrad"])["majorgrad"].count() / np.shape(temp[temp["parinc"]==4])[0]
    
def table_field_distnodebt(data):
    
    old = get_fields_dist(data[data["fourgrad_nodebt"]==1],"majorgrad_nodebt")
    new = get_fields_dist(data[data["fourgrad"]==1],"majorgrad")

    table = np.array(new) / np.array(old)
    
    k,m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
    
        
    rowanmes = ["Overall","ParInc Q1","ParInc Q2", "ParInc Q3", "ParInc Q4",
                "AFQT Q1", "AFQT Q2", "AFQT Q3", "AFQT Q4",
                "Male","Female","Non Black","Black","Low Schooling","High Schooling"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i,j]:.2f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Changes Across Fields}\n\\end{table}"
    
    
    return latex_table

def table_field_dist(data):
    
    old = get_fields_dist(data[data["fourgrad"]==1],"majorgrad")
    new = get_fields_dist(data[data["fourgrad_maxdebt"]==1],"majorgrad_maxdebt")

    table = (np.array(new) / np.array(old)-1)*100
    
    k,m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
    
        
    rowanmes = ["Overall","ParInc Q1","ParInc Q2", "ParInc Q3", "ParInc Q4",
                "AFQT Q1", "AFQT Q2", "AFQT Q3", "AFQT Q4",
                "Male","Female","Non Black","Black","Low Schooling","High Schooling"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i,j]:.2f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Changes Across Fields}\n\\end{table}"
    
    
    return latex_table

def table_field_dist_demographic(data):
    
    old = get_fields_dist(data[data["fourgrad"]==1],"majorgrad")
    new = get_fields_dist(data[data["fourgrad_maxdebt"]==1],"majorgrad_maxdebt")

    table = (np.array(new) / np.array(old)-1)*100
    
    k,m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
    
        
    rowanmes = ["Overall","ParInc Q1","ParInc Q2", "ParInc Q3", "ParInc Q4",
                "AFQT Q1", "AFQT Q2", "AFQT Q3", "AFQT Q4",
                "Male","Female","Non Black","Black","Low Schooling","High Schooling"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i,j]:.2f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Changes Across Fields}\n\\end{table}"
    
    
    return latex_table
    
    

def get_fields_dist(temp,group):
    
    allgroups = temp.groupby([f"{group}"])[f"{group}"].count() / np.shape(temp)[0]
    
    parinc1 = temp[temp["parinc"]==1].groupby([f"{group}"])[f"{group}"].count() / np.shape(temp[temp["parinc"]==1])[0]
    parinc2 = temp[temp["parinc"]==2].groupby([f"{group}"])[f"{group}"].count() / np.shape(temp[temp["parinc"]==2])[0]
    parinc3 = temp[temp["parinc"]==3].groupby([f"{group}"])[f"{group}"].count() / np.shape(temp[temp["parinc"]==3])[0]
    parinc4 = temp[temp["parinc"]==4].groupby([f"{group}"])[f"{group}"].count() / np.shape(temp[temp["parinc"]==4])[0]
    
    parinc = pd.concat([parinc1,parinc2,parinc3,parinc4],axis=1)
    
    
    ability1 = temp[temp["ability"]==1].groupby([f"{group}"])[f"{group}"].count() / np.shape(temp[temp["ability"]==1])[0]
    ability2 = temp[temp["ability"]==2].groupby([f"{group}"])[f"{group}"].count() / np.shape(temp[temp["ability"]==2])[0]
    ability3 = temp[temp["ability"]==3].groupby([f"{group}"])[f"{group}"].count() / np.shape(temp[temp["ability"]==3])[0]
    ability4 = temp[temp["ability"]==4].groupby([f"{group}"])[f"{group}"].count() / np.shape(temp[temp["ability"]==4])[0]

    ability = pd.concat([ability1,ability2,ability3,ability4],axis=1)
    
    sex0 = temp[temp["sex"]==0].groupby([f"{group}"])[f"{group}"].count() / np.shape(temp[temp["sex"]==0])[0]
    sex1 = temp[temp["sex"]==1].groupby([f"{group}"])[f"{group}"].count() / np.shape(temp[temp["sex"]==1])[0]

    sex = pd.concat([sex0,sex1],axis=1)
    
    race0 = temp[temp["ethnicity"]==0].groupby([f"{group}"])[f"{group}"].count() / np.shape(temp[temp["ethnicity"]==0])[0]
    race1 = temp[temp["ethnicity"]==1].groupby([f"{group}"])[f"{group}"].count() / np.shape(temp[temp["ethnicity"]==1])[0]

    race = pd.concat([race0,race1],axis=1)
    
    type0 = temp[temp["type"]==1].groupby([f"{group}"])[f"{group}"].count() / np.shape(temp[temp["type"]==1])[0]
    type1 = temp[temp["type"]==2].groupby([f"{group}"])[f"{group}"].count() / np.shape(temp[temp["type"]==2])[0]

    types = pd.concat([type0,type1],axis=1)
    
    
    return np.array(pd.concat([allgroups,parinc,ability,sex,race,types],axis=1)).T



def movers(data):
    
    temp = data[data["parinc"]==1]
    
    a = temp[temp["majorgrad"]==5].groupby(["majorgrad_maxdebt"])["majorgrad_maxdebt"].count()  / np.shape(temp[temp["majorgrad"]==5])[0]
    

    

def get_decomposition_pp(temp):
        
    #temp = data
        
    original = temp[temp["fourgrad"]==1].groupby(["majorgrad"])["majorgrad"].count() / np.shape(temp[temp["fourgrad"]==1])[0]
    new = temp[temp["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])["majorgrad_maxdebt"].count() / np.shape(temp[temp["fourgrad_maxdebt"]==1])[0]
    
    originaltemp = temp[temp["fourgrad"]==1].groupby(["majorgrad"])["majorgrad"].count()
    
    #switch
    newswitch = temp[temp["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])["switch"].sum() 
    newleave = temp[temp["fourgrad"]==1].groupby(["majorgrad"])["switch"].sum()
    netswitch = newswitch - newleave
   

    originaltemp = originaltemp + netswitch
    originaltempshare = originaltemp / np.sum(originaltemp)
    # Percentage of each field that are new enrolled
    
    newenr = temp[temp["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])["new"].sum() 
    
    # percentage that are previous dropouts
    newdrops = temp[temp["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])["drop"].sum() 
    
        # percentage that are coming from two.year schools
    
    newtwo = temp[temp["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])["maxtwograd"].sum()
    
    
    switchweight = netswitch.copy()
    switchweight[switchweight<0] = switchweight*-1
    weights = pd.concat([switchweight,newenr,newdrops,newtwo],axis=1)
    weights = np.array(weights) /np.array(weights.sum(axis=1))[...,None]
    
    gain = (new - original)*100
    netswitchgood = ((originaltemp+netswitch)/np.sum(originaltemp+netswitch) - original)*100 
    newenr = ((originaltemp+newenr)/np.sum(originaltemp+newenr) - original)*100
    newdrops = ((originaltemp+newdrops)/np.sum(originaltemp+newdrops) - original)*100
    newtwo = ((originaltemp+newtwo)/np.sum(originaltemp+newtwo) - original)*100 
    
    
    table = np.array(pd.concat([gain,netswitchgood,newenr,newdrops,newtwo],axis=1))  
    table[:,1:] = table[:,1:]*weights
    
    table = table/table[:,0][...,None]
    
    table[:,1] = table[:,1] -1 
    table = table[:,1:]
    table[:,2] = table[:,2]*-1

    k,m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
    
    
    column_names =  " & Change & In & Out & Net & New Enrolled & Previous Drop Out & Previous Two  \\\\  \\hline \n"
    latex_table += column_names
    
    rowanmes = ["\hspace{13mm} Business","\hspace{13mm} STEM","\hspace{13mm} Education",
                "\hspace{13mm} Social Sciences","\hspace{13mm} Humanities",
                "\hspace{13mm} Health","\hspace{13mm} Other"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i,j]:.2f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Changes Across Fields}\n\\end{table}"
    
    
    return latex_table





def get_decomposition(temp):
        
    #temp = data
        
    original = temp[temp["fourgrad"]==1].groupby(["majorgrad"])["majorgrad"].count()
    new = temp[temp["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])["majorgrad_maxdebt"].count()
    
    # Percentage of each field that are new enrolled
    
    newenr = temp[temp["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])["new"].sum()
    
    # percentage that are previous dropouts
    newdrops = temp[temp["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])["drop"].sum()

    # percentage that are switchers
    
    newswitch = temp[temp["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])["switch"].sum()

    # percentage that are coming from two.year schools
    
    newtwo = temp[temp["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])["maxtwograd"].sum()
    
    # percentage that leave
    
    newleave = temp[temp["fourgrad"]==1].groupby(["majorgrad"])["switch"].sum()
    
    netswitch = newswitch - newleave

    table = np.array(pd.concat([original,new,newswitch,newleave,netswitch,newenr,newdrops,newtwo],axis=1))  
    
    table = table/table[:,0][...,None]
    
    table[:,1] = table[:,1] -1 
    table = table[:,1:]
    table[:,2] = table[:,2]*-1

    k,m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
    
    
    column_names =  " & Change & In & Out & Net & New Enrolled & Previous Drop Out & Previous Two  \\\\  \\hline \n"
    latex_table += column_names
    
    rowanmes = ["\hspace{13mm} Business","\hspace{13mm} STEM","\hspace{13mm} Education",
                "\hspace{13mm} Social Sciences","\hspace{13mm} Humanities",
                "\hspace{13mm} Health","\hspace{13mm} Other"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i,j]:.2f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Changes Across Fields}\n\\end{table}"
    
    
    return latex_table

    
def get_decomposition_old(data):
    
    temp = data[data["parinc"]==1]
    
    original = temp[temp["fourgrad"]==1].groupby(["majorgrad"])["majorgrad"].count()/np.shape(temp[temp["fourgrad"]==1])[0]
    new = temp[temp["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])["majorgrad_maxdebt"].count()/np.shape(temp[temp["fourgrad_maxdebt"]==1])[0]
    change = np.array(new)/np.array(original)
    
    # Percentage of each field that are new enrolled
    
    newenr = temp[temp["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])["new"].mean()
    
    # percentage that are previous dropouts
    newdrops = temp[temp["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])["drop"].mean()

    # percentage that are switchers
    
    newswitch = temp[temp["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])["switch"].mean()

    # percentage that are coming from two.year schools
    
    newtwo = temp[temp["fourgrad_maxdebt"]==1].groupby(["majorgrad_maxdebt"])["maxtwograd"].mean()
    
    # percentage that leave
    
    newleave = temp[temp["fourgrad"]==1].groupby(["majorgrad"])["switch"].mean()

    newgroup = pd.concat([newenr,newdrops,newswitch,newtwo,newleave],axis=1)
    
    
def where_swithchers_going(data):
    
    
    temp = data[(data["switch"]==1)&(data["fourgrad_maxdebt"]==1)]
    
    switchers = get_fields_dist(temp,"majorgrad_maxdebt")
    
    
    
def fields_with_switchers(data):
    
    temp = data[data["fourgrad"]==1]
    
    a = temp[temp["ability"]==1].groupby(["majorgrad"])["switch"].mean()
    
def get_newdistfields(data):
    
    new = data[data["new"]==1]
    
    newgrad = new[(new["fourgrad_maxdebt"]==1)]
    
    a = newgrad[(newgrad["parinc"]==1)].groupby(["majorgrad_maxdebt"])["majorgrad_maxdebt"].count() / np.shape(newgrad[(newgrad["parinc"]==1)])[0]
    a = data[(data["parinc"]==1) & (data["fourgrad"]==1)].groupby(["majorgrad"])["majorgrad"].count() / np.shape(data[(data["parinc"]==1) & (data["fourgrad"]==1)])[0]

def get_switchers_nodebt(data):
    
    parinc = data[data["fourgrad_nodebt"]==1].groupby(["parinc"])["switch_nodebt"].mean()
    ability = data[data["fourgrad_nodebt"]==1].groupby(["ability"])["switch_nodebt"].mean()
    sex = data[data["fourgrad_nodebt"]==1].groupby(["sex"])["switch_nodebt"].mean()
    race = data[data["fourgrad_nodebt"]==1].groupby(["ethnicity"])["switch_nodebt"].mean()
    type1 = data[data["fourgrad_nodebt"]==1].groupby(["type"])["switch_nodebt"].mean()
    
    switch = np.array(pd.concat([parinc,ability,sex,race,type1],axis=0))
    
    parinc = data[data["drop_nodebt"]==1].groupby(["parinc"])["fourgrad"].mean()
    ability = data[data["drop_nodebt"]==1].groupby(["ability"])["fourgrad"].mean()
    sex = data[data["drop_nodebt"]==1].groupby(["sex"])["fourgrad"].mean()
    race = data[data["drop_nodebt"]==1].groupby(["ethnicity"])["fourgrad"].mean()
    type1 = data[data["drop_nodebt"]==1].groupby(["type"])["fourgrad"].mean()
    
    drop = np.array(pd.concat([parinc,ability,sex,race,type1],axis=0))
    
    parinc = data[data["never_nodebt"]==1].groupby(["parinc"])["never"].mean()
    ability = data[data["never_nodebt"]==1].groupby(["ability"])["never"].mean()
    sex = data[data["never_nodebt"]==1].groupby(["sex"])["never"].mean()
    race = data[data["never_nodebt"]==1].groupby(["ethnicity"])["never"].mean()
    type1 = data[data["never_nodebt"]==1].groupby(["type"])["never"].mean()
    
    enrl = 1 - np.array(pd.concat([parinc,ability,sex,race,type1],axis=0))

    parinc = data[data["never_nodebt"]==1].groupby(["parinc"])["fourgrad"].mean()
    ability = data[data["never_nodebt"]==1].groupby(["ability"])["fourgrad"].mean()
    sex = data[data["never_nodebt"]==1].groupby(["sex"])["fourgrad"].mean()
    race = data[data["never_nodebt"]==1].groupby(["ethnicity"])["fourgrad"].mean()
    type1 = data[data["never_nodebt"]==1].groupby(["type"])["fourgrad"].mean()
    
    grad = np.array(pd.concat([parinc,ability,sex,race,type1],axis=0))

    parinc = data[data["maxtwograd_nodebt"]==1].groupby(["parinc"])["fourgrad"].mean()
    ability = data[data["maxtwograd_nodebt"]==1].groupby(["ability"])["fourgrad"].mean()
    sex = data[data["maxtwograd_nodebt"]==1].groupby(["sex"])["fourgrad"].mean()
    race = data[data["maxtwograd_nodebt"]==1].groupby(["ethnicity"])["fourgrad"].mean()
    type1 = data[data["maxtwograd_nodebt"]==1].groupby(["type"])["fourgrad"].mean()
    
    two = np.array(pd.concat([parinc,ability,sex,race,type1],axis=0))

    table = np.concatenate((enrl[...,None],drop[...,None],
                            switch[...,None],two[...,None]),axis=1)
    
    k,m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
        
    column_names =  " & Never Enrolled & Drop Out & Switch & Two-Year  \\\\  \\hline \n"
    latex_table += column_names
    
    rowanmes = ["ParInc Q1","ParInc Q2", "ParInc Q3", "ParInc Q4",
                "AFQT Q1", "AFQT Q2", "AFQT Q3", "AFQT Q4",
                "Male","Female","Non Black","Black","Low Schooling","High Schooling"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i,j]:.2f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Elasticities to SAVE}\n\\end{table}"
    
    
    return latex_table
    
def get_switchers(data):
    
    parinc = data[data["fourgrad"]==1].groupby(["parinc"])["switch"].mean()
    ability = data[data["fourgrad"]==1].groupby(["ability"])["switch"].mean()
    sex = data[data["fourgrad"]==1].groupby(["sex"])["switch"].mean()
    race = data[data["fourgrad"]==1].groupby(["ethnicity"])["switch"].mean()
    type1 = data[data["fourgrad"]==1].groupby(["type"])["switch"].mean()
    
    switch = np.array(pd.concat([parinc,ability,sex,race,type1],axis=0))
    
    parinc = data[data["drop"]==1].groupby(["parinc"])["fourgrad_maxdebt"].mean()
    ability = data[data["drop"]==1].groupby(["ability"])["fourgrad_maxdebt"].mean()
    sex = data[data["drop"]==1].groupby(["sex"])["fourgrad_maxdebt"].mean()
    race = data[data["drop"]==1].groupby(["ethnicity"])["fourgrad_maxdebt"].mean()
    type1 = data[data["drop"]==1].groupby(["type"])["fourgrad_maxdebt"].mean()
    
    drop = np.array(pd.concat([parinc,ability,sex,race,type1],axis=0))
    
    parinc = data[data["never"]==1].groupby(["parinc"])["never_maxdebt"].mean()
    ability = data[data["never"]==1].groupby(["ability"])["never_maxdebt"].mean()
    sex = data[data["never"]==1].groupby(["sex"])["never_maxdebt"].mean()
    race = data[data["never"]==1].groupby(["ethnicity"])["never_maxdebt"].mean()
    type1 = data[data["never"]==1].groupby(["type"])["never_maxdebt"].mean()
    
    enrl = 1 - np.array(pd.concat([parinc,ability,sex,race,type1],axis=0))

    parinc = data[data["never"]==1].groupby(["parinc"])["fourgrad_maxdebt"].mean()
    ability = data[data["never"]==1].groupby(["ability"])["fourgrad_maxdebt"].mean()
    sex = data[data["never"]==1].groupby(["sex"])["fourgrad_maxdebt"].mean()
    race = data[data["never"]==1].groupby(["ethnicity"])["fourgrad_maxdebt"].mean()
    type1 = data[data["never"]==1].groupby(["type"])["fourgrad_maxdebt"].mean()
    
    grad = np.array(pd.concat([parinc,ability,sex,race,type1],axis=0))

    parinc = data[data["maxtwograd"]==1].groupby(["parinc"])["fourgrad_maxdebt"].mean()
    ability = data[data["maxtwograd"]==1].groupby(["ability"])["fourgrad_maxdebt"].mean()
    sex = data[data["maxtwograd"]==1].groupby(["sex"])["fourgrad_maxdebt"].mean()
    race = data[data["maxtwograd"]==1].groupby(["ethnicity"])["fourgrad_maxdebt"].mean()
    type1 = data[data["maxtwograd"]==1].groupby(["type"])["fourgrad_maxdebt"].mean()
    
    two = np.array(pd.concat([parinc,ability,sex,race,type1],axis=0))

    table = np.concatenate((enrl[...,None],drop[...,None],
                            switch[...,None],two[...,None]),axis=1)
    
    k,m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
        
    column_names =  " & Never Enrolled & Drop Out & Switch & Two-Year  \\\\  \\hline \n"
    latex_table += column_names
    
    rowanmes = ["ParInc Q1","ParInc Q2", "ParInc Q3", "ParInc Q4",
                "AFQT Q1", "AFQT Q2", "AFQT Q3", "AFQT Q4",
                "Male","Female","Non Black","Black","Low Schooling","High Schooling"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i,j]:.2f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Elasticities to SAVE}\n\\end{table}"
    
    
    return latex_table
    
    
def get_groups(new):
    
    parinc = new.groupby(["parinc"])[["parinc"]].count() /(np.shape(new)[0])
    ability = new.groupby(["ability"])[["ability"]].count() /(np.shape(new)[0])
    sex = new.groupby(["sex"])[["sex"]].count() /(np.shape(new)[0])
    race = new.groupby(["ethnicity"])[["ethnicity"]].count() /(np.shape(new)[0])
    type1 = new.groupby(["type1"])[["type1"]].count() /(np.shape(new)[0])
    
    return pd.concat([parinc,ability,sex,race,type1],axis=0).sum(axis=1)
    

def get_groups_type1(new):
    
    parinc = new.groupby(["parinc"])[["type1"]].mean()
    ability = new.groupby(["ability"])[["type1"]].mean()
    sex = new.groupby(["sex"])[["type1"]].mean()
    race = new.groupby(["ethnicity"])[["type1"]].mean()
    
    
    return np.append(np.array(pd.concat([parinc,ability,sex,race],axis=0).sum(axis=1))[...,None],new["type1"].mean())
    


def latex_table_type1newenrolled(data):
    
    new = data[data["new"]==1]
    
    datall = get_groups_type1(new)
    datadrop = get_groups_type1(new[new["drop_maxdebt"]==1])
    datatwo = get_groups_type1(new[new["twograd_maxdebt"]==1])
    datafour = get_groups_type1(new[new["fourgrad_maxdebt"]==1])
    #datagrad = get_groups_type1(new[new["gradgrad_maxdebt"]==1])
    
    table = np.array(np.concatenate((datall[...,None],datadrop[...,None],datatwo[...,None],
                            datafour[...,None]),axis=1))
    
    k, m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
        
    column_names =  " & All & Drop Out & Two-Year Grad & Four-Year Grad   \\\\  \\hline "
    latex_table += column_names
    
    rowanmes = ["ParInc Q1","ParInc Q2", "ParInc Q3", "ParInc Q4",
                "AFQT Q1", "AFQT Q2", "AFQT Q3", "AFQT Q4",
                "Male","Female","Non Black","Black","All"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i, j]:.2f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Decomposition of New Graduates}\n\\end{table}"
    
    
    return latex_table
    

def latex_table_newenrolled_decomposition(data):
    
    new = data[data["new"]==1]
    
    datall = get_groups(new)
    datadrop = get_groups(new[new["drop_maxdebt"]==1])
    datatwo = get_groups(new[new["twograd_maxdebt"]==1])
    datafour = get_groups(new[new["fourgrad_maxdebt"]==1])
    #datagrad = get_groups(new[new["gradgrad_maxdebt"]==1])
    
    table = np.array(pd.concat([datall,datadrop,datatwo,datafour],axis=1))
    
    k, m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
        
    column_names =  " & All & Drop Out & Two-Year Grad & Four-Year Grad   \\\\  \\hline "
    latex_table += column_names
    
    rowanmes = ["ParInc Q1","ParInc Q2", "ParInc Q3", "ParInc Q4",
                "AFQT Q1", "AFQT Q2", "AFQT Q3", "AFQT Q4",
                "Male","Female","Non Black","Black","High Schooling","Low Schooling"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i, j]:.2f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Decomposition of New Graduates}\n\\end{table}"
    
    
    return latex_table


    
def latex_table_newgrads_decomposition_nodebt(data):
    
    parincgrad = data.groupby(["parinc"])[["fourgrad","fourgrad_nodebt"]].mean()
    parincgrad["share"] = parincgrad["fourgrad"] / parincgrad["fourgrad_nodebt"] - 1
    
    abilitygrad = data.groupby(["ability"])[["fourgrad","fourgrad_nodebt"]].mean()
    abilitygrad["share"] = abilitygrad["fourgrad"] / abilitygrad["fourgrad_nodebt"] - 1
    
    sexgrad = data.groupby(["sex"])[["fourgrad","fourgrad_nodebt"]].mean()
    sexgrad["share"] = sexgrad["fourgrad"] / sexgrad["fourgrad_nodebt"] - 1
    
    
    racegrad = data.groupby(["ethnicity"])[["fourgrad","fourgrad_nodebt"]].mean()
    racegrad["share"] = racegrad["fourgrad"] / racegrad["fourgrad_nodebt"] - 1
    
    graduates = np.array(pd.concat([parincgrad["share"],abilitygrad["share"],
                           sexgrad["share"],racegrad["share"]],axis=0))

    parinc = data[(data["fourgrad_nodebt"]==0) & (data["fourgrad"]==1)].groupby(["parinc"])[["never_nodebt","drop_nodebt","twograd_nodebt"]].mean()
    ability = data[(data["fourgrad_nodebt"]==0) & (data["fourgrad"]==1)].groupby(["ability"])[["never_nodebt","drop_nodebt","twograd_nodebt"]].mean()
    sex = data[(data["fourgrad_nodebt"]==0) & (data["fourgrad"]==1)].groupby(["sex"])[["never_nodebt","drop_nodebt","twograd_nodebt"]].mean()
    race = data[(data["fourgrad_nodebt"]==0) & (data["fourgrad"]==1)].groupby(["ethnicity"])[["never_nodebt","drop_nodebt","twograd_nodebt"]].mean()
    
    table = np.array(pd.concat([parinc,ability,sex,race],axis=0))
    
    table = table*graduates[...,None]
    
    table = np.concatenate([graduates[...,None],table],axis=1)
    
    table = table*100

    k, m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
        
    column_names =  " & Change & Never & Drop Out & Two Year   \\\\  \\hline "
    latex_table += column_names
    
    rowanmes = ["ParInc Q1","ParInc Q2", "ParInc Q3", "ParInc Q4",
                "AFQT Q1", "AFQT Q2", "AFQT Q3", "AFQT Q4",
                "Male","Female","Non Black","Black"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i, j]:.2f}\%$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Decomposition of New Graduates}\n\\end{table}"
    
    
    return latex_table 

def latex_table_newgrads_decomposition(data):
    
    parincgrad = data.groupby(["parinc"])[["fourgrad","fourgrad_maxdebt"]].mean()
    parincgrad["share"] = parincgrad["fourgrad_maxdebt"] / parincgrad["fourgrad"] - 1
    
    abilitygrad = data.groupby(["ability"])[["fourgrad","fourgrad_maxdebt"]].mean()
    abilitygrad["share"] = abilitygrad["fourgrad_maxdebt"] / abilitygrad["fourgrad"] - 1
    
    sexgrad = data.groupby(["sex"])[["fourgrad","fourgrad_maxdebt"]].mean()
    sexgrad["share"] = sexgrad["fourgrad_maxdebt"] / sexgrad["fourgrad"] - 1
    
    
    racegrad = data.groupby(["ethnicity"])[["fourgrad","fourgrad_maxdebt"]].mean()
    racegrad["share"] = racegrad["fourgrad_maxdebt"] / racegrad["fourgrad"] - 1
    
    gradbase=  np.array(pd.concat([parincgrad["fourgrad"],abilitygrad["fourgrad"],
                           sexgrad["fourgrad"],racegrad["fourgrad"]],axis=0))
    gradconter=  np.array(pd.concat([parincgrad["fourgrad_maxdebt"],abilitygrad["fourgrad_maxdebt"],
                           sexgrad["fourgrad_maxdebt"],racegrad["fourgrad_maxdebt"]],axis=0))
    graduates = np.array(pd.concat([parincgrad["share"],abilitygrad["share"],
                           sexgrad["share"],racegrad["share"]],axis=0))

    parinc = data[(data["fourgrad"]==0) & (data["fourgrad_maxdebt"]==1)].groupby(["parinc"])[["never","drop","twograd"]].mean()
    ability = data[(data["fourgrad"]==0) & (data["fourgrad_maxdebt"]==1)].groupby(["ability"])[["never","drop","twograd"]].mean()
    sex = data[(data["fourgrad"]==0) & (data["fourgrad_maxdebt"]==1)].groupby(["sex"])[["never","drop","twograd"]].mean()
    race = data[(data["fourgrad"]==0) & (data["fourgrad_maxdebt"]==1)].groupby(["ethnicity"])[["never","drop","twograd"]].mean()
    
    table = np.array(pd.concat([parinc,ability,sex,race],axis=0))
    shares = table.copy()
    
    table = table*graduates[...,None]
    
    table = np.concatenate([graduates[...,None],table],axis=1)
    
    table = table*100
    
    table2 = np.zeros((np.shape(table)[0],9))
    
    table2[:,0] = gradbase
    table2[:,1] = gradconter 
    table2[:,2] = table[:,0]
    table2[:,3] = shares[:,0]
    table2[:,4] = table[:,1]
    table2[:,5] = shares[:,1]
    table2[:,6] = table[:,2]
    table2[:,7] = shares[:,2]
    table2[:,8] = table[:,3]    
    
    table = table2

    k, m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
        
    column_names =  " & Base & Conter &  & Never & c & Drop Out & c & Two Year & c   \\\\  \\hline "
    latex_table += column_names
    
    rowanmes = ["ParInc Q1","ParInc Q2", "ParInc Q3", "ParInc Q4",
                "AFQT Q1", "AFQT Q2", "AFQT Q3", "AFQT Q4",
                "Male","Female","Non Black","Black"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i, j]:.2f}\%$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Decomposition of New Graduates}\n\\end{table}"
    
    
    return latex_table  

def latex_table_newgrads_decomposition2(data):
    
    parincgrad = data.groupby(["parinc"])[["twograd","twograd_maxdebt"]].mean()
    parincgrad["share"] = parincgrad["twograd_maxdebt"] / parincgrad["twograd"] - 1
    
    abilitygrad = data.groupby(["ability"])[["twograd","twograd_maxdebt"]].mean()
    abilitygrad["share"] = abilitygrad["twograd_maxdebt"] / abilitygrad["twograd"] - 1
    
    sexgrad = data.groupby(["sex"])[["twograd","twograd_maxdebt"]].mean()
    sexgrad["share"] = sexgrad["twograd_maxdebt"] / sexgrad["twograd"] - 1
    
    
    racegrad = data.groupby(["ethnicity"])[["twograd","twograd_maxdebt"]].mean()
    racegrad["share"] = racegrad["twograd_maxdebt"] / racegrad["twograd"] - 1
    
    gradbase=  np.array(pd.concat([parincgrad["twograd"],abilitygrad["twograd"],
                           sexgrad["twograd"],racegrad["twograd"]],axis=0))
    gradconter=  np.array(pd.concat([parincgrad["twograd_maxdebt"],abilitygrad["twograd_maxdebt"],
                           sexgrad["twograd_maxdebt"],racegrad["twograd_maxdebt"]],axis=0))
    graduates = np.array(pd.concat([parincgrad["share"],abilitygrad["share"],
                           sexgrad["share"],racegrad["share"]],axis=0))

    parinc = data[(data["twograd"]==0) & (data["twograd_maxdebt"]==1)].groupby(["parinc"])[["never","drop","fourgrad","fourgrad_maxdebt"]].mean()
    ability = data[(data["twograd"]==0) & (data["twograd_maxdebt"]==1)].groupby(["ability"])[["never","drop","fourgrad","fourgrad_maxdebt"]].mean()
    sex = data[(data["twograd"]==0) & (data["twograd_maxdebt"]==1)].groupby(["sex"])[["never","drop","fourgrad","fourgrad_maxdebt"]].mean()
    race = data[(data["twograd"]==0) & (data["twograd_maxdebt"]==1)].groupby(["ethnicity"])[["never","drop","fourgrad","fourgrad_maxdebt"]].mean()
    
    table = np.array(pd.concat([parinc,ability,sex,race],axis=0))
    shares = table.copy()
    
    table = table[:,:3]*graduates[...,None]
    
    table = np.concatenate([graduates[...,None],table],axis=1)
    
    table = table*100
    
    table2 = np.zeros((np.shape(table)[0],9))
    
    table2[:,0] = gradbase
    table2[:,1] = gradconter 
    table2[:,2] = table[:,0]
    table2[:,3] = shares[:,0]
    table2[:,4] = table[:,1]
    table2[:,5] = shares[:,1]
    table2[:,6] = table[:,2]
    table2[:,7] = shares[:,2]
    table2[:,8] = table[:,3]    
    
    table = table2

    k, m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
        
    column_names =  " & Base & Conter &  & Never & c & Drop Out & c & Two Year & c   \\\\  \\hline "
    latex_table += column_names
    
    rowanmes = ["ParInc Q1","ParInc Q2", "ParInc Q3", "ParInc Q4",
                "AFQT Q1", "AFQT Q2", "AFQT Q3", "AFQT Q4",
                "Male","Female","Non Black","Black"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i, j]:.2f}\%$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Decomposition of New Graduates}\n\\end{table}"
    
    
    return latex_table


def latex_table_newgrads(data):


    parinc = data[(data["fourgrad"]==0) & (data["fourgrad_maxdebt"]==1)].groupby(["parinc"])[["never","drop","twograd"]].mean()
    ability = data[(data["fourgrad"]==0) & (data["fourgrad_maxdebt"]==1)].groupby(["ability"])[["never","drop","twograd"]].mean()
    sex = data[(data["fourgrad"]==0) & (data["fourgrad_maxdebt"]==1)].groupby(["sex"])[["never","drop","twograd"]].mean()
    race = data[(data["fourgrad"]==0) & (data["fourgrad_maxdebt"]==1)].groupby(["ethnicity"])[["never","drop","twograd"]].mean()
    
    table = np.array(pd.concat([parinc,ability,sex,race],axis=0))

    k, m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
        
    column_names =  "  & Never & Drop Out & Two Year   \\\\  \\hline "
    latex_table += column_names
    
    rowanmes = ["ParInc Q1","ParInc Q2", "ParInc Q3", "ParInc Q4",
                "AFQT Q1", "AFQT Q2", "AFQT Q3", "AFQT Q4",
                "Male","Female","Non Black","Black"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i, j]:.3f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Origin of New Graduates}\n\\end{table}"
    
    
    return latex_table  

    
def latex_table_movements(data):
    
    never = data[data["never"]==1][["never_maxdebt","drop_maxdebt",
                                    "twograd_maxdebt","fourgrad_maxdebt",
                                    "gradgrad_maxdebt"]].mean()
    
    drop = data[data["drop"]==1][["never_maxdebt","drop_maxdebt",
                                    "twograd_maxdebt","fourgrad_maxdebt",
                                    "gradgrad_maxdebt"]].mean()
    
    two = data[data["twograd"]==1][["never_maxdebt","drop_maxdebt",
                                    "twograd_maxdebt","fourgrad_maxdebt",
                                    "gradgrad_maxdebt"]].mean()
    
    four = data[data["fourgrad"]==1][["never_maxdebt","drop_maxdebt",
                                    "twograd_maxdebt","fourgrad_maxdebt",
                                    "gradgrad_maxdebt"]].mean()
    
    grad = data[data["gradgrad"]==1][["never_maxdebt","drop_maxdebt",
                                    "twograd_maxdebt","fourgrad_maxdebt",
                                    "gradgrad_maxdebt"]].mean()
    
    table = np.array(pd.concat([never,drop,two,four,grad],axis=1)).T
    
    k, m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
        
    column_names =  "  & Never &  Drop Out  & Two-Year Grad &  Four-Year Grad &  Grad   \\\\  \\hline "
    latex_table += column_names
    
    rowanmes = ["Never","Drop Out","Two-Year Grad","Four-Year Grad","Grad"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i, j]:.3f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Education Level After SAVE}\n\\end{table}"
    
    
    return latex_table  

    


def check_fields_conterfactual_groups_total(simu,conter,contermax,conternodebt,contergrants,group,typeff):
    
    simu  = simu[simu[f"{group}"]==typeff]
    conter  = conter[conter[f"{group}"]==typeff]
    contermax  = contermax[contermax[f"{group}"]==typeff]
    conternodebt  = conternodebt[conternodebt[f"{group}"]==typeff]
    contergrants  = contergrants[contergrants[f"{group}"]==typeff]
    
    share_simu = simu[(simu["period"]==T-1)].groupby(["majorgrad"])["fourgrad"].sum() / np.shape(simu[(simu["period"]==T-1)])[0]
    share_conter = conter[(conter["period"]==T-1)].groupby(["majorgrad"])["fourgrad"].sum() / np.shape(conter[(conter["period"]==T-1)])[0]   
    share_contermax = contermax[(contermax["period"]==T-1)].groupby(["majorgrad"])["fourgrad"].sum() / np.shape(contermax[(contermax["period"]==T-1)])[0]  
    share_conternodebt = conternodebt[(conternodebt["period"]==T-1)].groupby(["majorgrad"])["fourgrad"].sum() / np.shape(conternodebt[(conternodebt["period"]==T-1)])[0]   
    share_contergrants = contergrants[(contergrants["period"]==T-1)].groupby(["majorgrad"])["fourgrad"].sum() / np.shape(contergrants[(contergrants["period"]==T-1)])[0]

    share_simu = share_simu.rename("model")
    share_conter = share_conter.rename("conter")
    share_contermax = share_contermax.rename("conter_maxdebt")
    share_conternodebt = share_conternodebt.rename("conter_nodebt")
    share_contergrants = share_contergrants.rename("conter_grants")

    joint = pd.concat([share_simu,share_conter,share_contermax,share_conternodebt,share_contergrants],axis=1).reset_index()
    
    joint.to_stata(f"{datafigures}/fields_effect_all_conter_{group}{typeff}.dta",write_index=False)

    pass


def check_fields_conterfactual_all(simu,conter,contermax,conternodebt,contergrants):
    
    
    share_simu = simu[(simu["period"]==T-1)&(simu["fourgrad"]==1)].groupby(["parinc","ability","sex","ethnicity","majorgrad"])["fourgrad"].sum().rename("model")
    share_conter = conter[(conter["period"]==T-1)&(conter["fourgrad"]==1)].groupby(["parinc","ability","sex","ethnicity","majorgrad"])["fourgrad"].sum().rename("conter")
    share_contermax = contermax[(contermax["period"]==T-1)&(contermax["fourgrad"]==1)].groupby(["parinc","ability","sex","ethnicity","majorgrad"])["fourgrad"].sum().rename("conter_maxdebt")
    share_conternodebt = conternodebt[(conternodebt["period"]==T-1)&(conternodebt["fourgrad"]==1)].groupby(["parinc","ability","sex","ethnicity","majorgrad"])["fourgrad"].sum().rename("conter_nodebt")
    share_contergrants = contergrants[(contergrants["period"]==T-1)&(contergrants["fourgrad"]==1)].groupby(["parinc","ability","sex","ethnicity","majorgrad"])["fourgrad"].sum().rename("conter_grants")

    joint = pd.concat([share_simu,share_conter,share_contermax,share_conternodebt,share_contergrants],axis=1).reset_index()
    
    joint.to_stata(f"{datafigures}/fields_effect_conter_all.dta",write_index=False)

    return effect
 
    
    
    
def get_distribution(period,real,simu,coltype,graphtype,name):
    
    """This function computes the distribution of majors at period 10. """
    
    if coltype == 1: 
        
        dist_real = real[(real["period"]==period)& (real["twograd"]==1)].groupby(["majorgrad"])["majorgrad"].count() / real["twograd"][real["period"]==period].sum()
        dist_simu = simu[(simu["period"]==period)& (simu["twograd"]==1)].groupby(["majorgrad"])["majorgrad"].count() / simu["twograd"][simu["period"]==period].sum()
    
    elif coltype == 2: 
        
        dist_real = real[(real["period"]==period)& (real["fourgrad"]==1)].groupby(["majorgrad"])["majorgrad"].count() / real["fourgrad"][real["period"]==period].sum()
        dist_simu = simu[(simu["period"]==period)& (simu["fourgrad"]==1)].groupby(["majorgrad"])["majorgrad"].count() / simu["fourgrad"][simu["period"]==period].sum()

        
    width = 0.4

    positions1 = np.arange(1,14,2)-width
    positions2 = np.arange(1,14,2)+width
    positions3 = np.arange(1,14,2)

    names = ["Bus","STEM","Educ","Social","Huamn","Health","Other"]
    
    if graphtype == 1:
        leg = ["Data","Model"]
    elif graphtype == 2:
        leg = ["Baseline","Conterfactual"]

    fig, ax = plt.subplots()
    axes = plt.gca()
    axes.yaxis.grid(zorder=0)
    ax.bar(positions1,dist_real,width=width*2, zorder=3,edgecolor="black")
    ax.bar(positions2,dist_simu,width=width*2, zorder=3,edgecolor="black")
    ax.set_xticks(positions3, labels=names,rotation = 45)
    plt.ylabel("Share of each field")
    plt.legend(leg,loc='upper center', bbox_to_anchor=(0.5,-0.2), ncol = 2)
    plt.savefig(f"{figures}/{name}.png",bbox_inches='tight')
    
    
    #save data
    dist_real = dist_real.rename("data")
    dist_simu = dist_simu.rename("model")
    tosave = pd.concat([dist_real,dist_simu],axis=1)
   
    tosave.to_stata(f"{datafigures}/distribution_fields_{name}.dta",write_index=False)
    
    
def get_choice_distribution(data,period):
    
    """
    This function gets the choice distribution of a spefic period
    """
    
    data = data[data["period"]==period]
    
    dist = data.groupby(["choices"])["choices"].count() / np.shape(data)[0]
    
    # MAke sure to inlcude the 0s
    
    zeros = pd.DataFrame(np.zeros(47))
    zeros["dist"] = dist
    zeros.replace(0, np.nan, inplace=True) 
    
    return zeros["dist"]

def get_choice_distribution_expanded(data,period,group):
    
    a = get_total_choices()
    
    data = data[data["period"]==period].copy()
    
    data["field"] = a[data["choices"],0]
    data["educ"] =a[data["choices"],1]
    data["work"] = a[data["choices"],2]
    
    if group == "fields":
        
        dist = data.groupby(["field","educ"])["field"].count() / np.shape(data)[0]
    
    elif group == "work":
        
        dist = data.groupby(["work","educ"])["work"].count() / np.shape(data)[0]
        
    return dist

def evolution_choices_expanded(real,simu,group):
    
    """
    This function plots the evolution of the first field 
    """
    real_data = np.zeros(T-1)
    simu_data = np.zeros(T-1)
    if group == "fields":
        for educ in range(0,3):
            if educ == 0 :
                listfields = [1,2,3,6,7,8,9,10]
            elif educ == 1:
                listfields = [12]
            elif educ == 2: 
                listfields = [1,2,3,4,5,6,7,8]
                
            for field in listfields:
            
                for period in range(1,T):
                    realdist = get_choice_distribution_expanded(real,period,group)
                    simudist = get_choice_distribution_expanded(simu,period,group)
                    try:
                        real_data[period-1] = np.sum(realdist.loc[(field,educ)])
                    except:
                        real_data[period-1] = 0
                    try:
                        simu_data[period-1] = np.sum(simudist.loc[(field,educ)])
                    except:
                        simu_data[period-1] = 0 
    
                leg = ["Data","Model"]
                fig = plt.subplots()
                plt.plot(real_data)
                plt.plot(simu_data)
                plt.ylabel(f"Share choosing {field}")
                plt.xlabel("Period")
                plt.legend(leg,loc='upper center', bbox_to_anchor=(0.5,-0.2), ncol = 2)
                plt.savefig(f"{figures}/distribution_choice_field{field}_educ{educ}.png",bbox_inches='tight')
                
    if group == "work":
        
        for educ in range(0,3):
            
            for work in range(0,3):
                
                for period in range(1,T):
                    realdist = get_choice_distribution_expanded(real,period,group)
                    simudist = get_choice_distribution_expanded(simu,period,group)
                        
                    real_data[period-1] = np.sum(realdist.loc[(work,educ)])
                    simu_data[period-1] = np.sum(simudist.loc[(work,educ)])
    
                leg = ["Data","Model"]
                fig = plt.subplots()
                plt.plot(real_data)
                plt.plot(simu_data)
                plt.ylabel(f"Share choosing to work {work}")
                plt.xlabel("Period")
                plt.legend(leg,loc='upper center', bbox_to_anchor=(0.5,-0.2), ncol = 2)
                plt.savefig(f"{figures}/distribution_choice_work{work}_educ{educ}.png",bbox_inches='tight')
                
                
        
        

    

def evolution_choices(real,simu,choice):
    
    """
    This function plots the evolution of the first field 
    """
    real_data = np.zeros(T-1)
    simu_data = np.zeros(T-1)

    for period in range(1,T):
        real_data[period-1] = np.sum(get_choice_distribution(real,period)[choice])
        simu_data[period-1] = np.sum(get_choice_distribution(simu,period)[choice])
    
    leg = ["Data","Model"]
    fig = plt.subplots()
    plt.plot(real_data)
    plt.plot(simu_data)
    plt.ylabel(f"Share choosing {choice}")
    plt.xlabel("Period")
    plt.legend(leg,loc='upper center', bbox_to_anchor=(0.5,-0.2), ncol = 2)
    plt.savefig(f"{figures}/distribution_choice{choice}.png",bbox_inches='tight')

def get_report(real,simu):
    
    """
    This function produces many figures regarding the choice distribution. 
    """
    for choice in range(50):
    
        evolution_choices(real,simu,choice)
        
def get_report_expanded(real,simu):
    pass
    
    
    
    
    

def state_distribution(data,period,coltype):
    
    """
    This function computes the distribution of individuals at a particular state and period
    """
    
    dfperiod = data[data["period"]==period]
    if coltype == 1:
        test = dfperiod.groupby(["twoyear_exp"])["twoyear_exp"].count()/np.shape(dfperiod)[0]
        
    elif coltype == 2:
        test = dfperiod.groupby(["fouryear_exp"])["fouryear_exp"].count()/np.shape(dfperiod)[0]
        
    return test

def get_debt_range():
    
    debtrange1 = np.array([0,300,500,620,770,950])
    debtrange2 = np.linspace(1166,3500,16)
    debtrange3 = np.linspace(3720,8800,25)
    debtrange4 = np.linspace(9200,20000,25)
    debtrange5 = np.linspace(22700,100000,28)

    debt_range = np.concatenate((debtrange1,debtrange2,debtrange3,debtrange4,debtrange5))
    
    return debt_range



def work_with_debt(real,simu):
    
    debtrange = get_debt_range()
    
    a = get_total_choices()
    
    simu = allsimu.copy()
    real = allreal.copy()
    
    simu["field"] = a[simu["choices"],0]
    simu["educ"] =a[simu["choices"],1]
    simu["work"] = a[simu["choices"],2]

    real["anydebt"] = 0
    real.loc[real["currentloans"]>0, "anydebt"] = 1
    
    
    simu["anydebt"] = 0
    simu["currentloans"] = debtrange[np.array(simu["debt"])]
    simu.loc[simu["currentloans"]>0, "anydebt"] = 1
    
    realgroupany = real[(real["period"]==2)&(real["fouryear_exp"]==1)].groupby(["parinc","ability"])["anydebt"].mean()
    simugroupany = simu[(simu["fouryear_exp"]==3)&(simu["period"]==4)].groupby(["parinc","ability"])["anydebt"].mean()
    realgroup = real[(real["period"]==2)&(real["anydebt"]==1)&(real["fouryear_exp"]==1)].groupby(["parinc","ability"])["currentloans"].mean()
    simugroup = simu[(simu["fouryear_exp"]==4)&(simu["period"]==5)&(simu["anydebt"]==1)].groupby(["parinc","ability"])["currentloans"].mean()
    
    realdebt = pd.concat((realgroupany,realgroup),axis=1)
    simudebt = pd.concat((simugroupany,simugroup),axis=1)

    
    
def get_debt_new(real,simu):
    
    """
    Input simu_good!!!
    """
    
    debtrange = get_debt_range()
    
    real = real[(real["fourgrad"]==1)&(real["period"]==real["period_educ_last"]+1)  &(real["lastchoice"]!=13)]
    simu = simu[(simu["fourgrad"]==1)&(simu["period"]==simu["last_educ"]+1) &(simu["last_choice"]!=13)]
    
    real["anydebt"] = 0
    real.loc[real["currentloans"]>1700, "anydebt"] = 1
    
    
    simu["anydebt"] = 0
    simu["currentloans"] = debtrange[np.array(simu["debt"])]
    simu.loc[simu["debt"]>10, "anydebt"] = 1

    
    realgroupany = group_dropout4(real, "anydebt")
    simugroupany = group_dropout4(simu, "anydebt")
    realgroup = group_dropout4(real[real["anydebt"]==1], "currentloans")
    simugroup = group_dropout4(simu[simu["anydebt"]==1], "currentloans")
    
    realdebt = np.concatenate((realgroupany[...,None],realgroup[...,None]),axis=1)
    simudebt = np.concatenate((simugroupany[...,None],simugroup[...,None]),axis=1)
    

def get_debt_new_conter(simu,conter):
    
    """
    Input simu_good and conter_good
    
    """    
    
    debtrange = get_debt_range()
    
    conter = conter[(conter["fourgrad"]==1)&
                    (conter["period"]==conter["last_educ"]+1)  &
                    (conter["last_choice"]!=13)]
    simu = simu[(simu["fourgrad"]==1)&
                (simu["period"]==simu["last_educ"]+1) &
                (simu["last_choice"]!=13)]
    
    
    simu["anydebt"] = 0
    simu["currentloans"] = debtrange[np.array(simu["debt"])]
    simu.loc[simu["debt"]>10, "anydebt"] = 1
    
    conter["anydebt"] = 0
    conter["currentloans"] = debtrange[np.array(conter["debt"])]
    conter.loc[conter["debt"]>10, "anydebt"] = 1
    
    
    # overall effect
    
    conter[conter["anydebt"]==1]["currentloans"].mean()

    
    contergroupany = group_dropout4(conter, "anydebt")
    simugroupany = group_dropout4(simu, "anydebt")
    contergroup = group_dropout4(conter[conter["anydebt"]==1], "currentloans")
    simugroup = group_dropout4(simu[simu["anydebt"]==1], "currentloans")
    
    conterdebt = np.concatenate((contergroupany[...,None],contergroup[...,None]),axis=1)
    simudebt = np.concatenate((simugroupany[...,None],simugroup[...,None]),axis=1)
    
    table = np.zeros((np.shape(simudebt)[0],6))
    
    table[:,0] = simudebt[:,0]
    table[:,1] = conterdebt[:,0]
    table[:,2] = (conterdebt[:,0]/simudebt[:,0] -1)*100
    table[:,3] = simudebt[:,1]
    table[:,4] = conterdebt[:,1]
    table[:,5] = (conterdebt[:,1]/simudebt[:,1] -1)*100
    
    k, m = table.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
        
    column_names =  "  & Base &  Conter &  & Base & Conter &   \\\\  \\hline "
    latex_table += column_names
    
    rowanmes = ["ParInc Q1","ParInc Q2", "ParInc Q3", "ParInc Q4","AFQT Q1", "AFQT Q2", 
                "AFQT Q3", "AFQT Q4","Male", "Female", "Non-Black","Black"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${table[i, j]:.3f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Debt Changes}\n\\end{table}"
    
    
    return latex_table
    

def get_debt_descriptives(real,simu):
    
    """
    This function gets the average debt and share of indebtedness at last period
    """
    
    debtrange = get_debt_range()
    
    real = real[(real["fourgrad"]==1)&(real["period"]==real["period_educ_last"]+1)  &(real["lastchoice"]!=13)]
    simu = simu[(simu["fourgrad"]==1)&(simu["period"]==simu["last_educ"]+1) &(simu["last_choice"]!=13)]
    
    real["anydebt"] = 0
    real.loc[real["currentloans"]>0, "anydebt"] = 1
    
    
    simu["anydebt"] = 0
    simu["currentloans"] = debtrange[np.array(simu["debt"])]
    simu.loc[simu["currentloans"]>0, "anydebt"] = 1
    
    realgroupany = real.groupby(["parinc","ability"])["anydebt"].mean()
    simugroupany = simu.groupby(["parinc","ability"])["anydebt"].mean()
    realgroup = real[real["anydebt"]==1].groupby(["parinc","ability"])["currentloans"].mean()
    simugroup = simu[simu["anydebt"]==1].groupby(["parinc","ability"])["currentloans"].mean()
    
    realdebt = pd.concat((realgroupany,realgroup),axis=1)
    simudebt = pd.concat((simugroupany,simugroup),axis=1)
    return realdebt,simudebt


def get_experience_distribution(allreal,allsimu,coltype):
    
    arealdist = state_distribution(allreal,period,coltype)
    asimudist = state_distribution(allsimu,period,coltype)
    
    # Put them together based on index
    
    joint = pd.concat([arealdist,asimudist],axis=1)
    
    
def get_occupation_distribution(real,simu,humancapital):
    
    '''
    Get share of individuals working at a specific occupation for 
    different human capital levels.
    
    '''

    
    a = get_total_choices()
    
    if humancapital != 99 :
    
        real = real[real["majorgrad"]==humancapital]
        simu = simu[simu["majorgrad"]==humancapital]
    
    elif humancapital ==  99:
        
        real = real[real["fourgrad"]==1]
        simu = simu[simu["fourgrad"]==1]
        
    
    real["field"] = a[real["choices"],0]
    real["educ"] =a[real["choices"],1]
    real["work"] = a[real["choices"],2]
    
    simu["field"] = a[simu["choices"],0]
    simu["educ"] =a[simu["choices"],1]
    simu["work"] = a[simu["choices"],2]
    
    realdist = real[(real["work"]>0)&(real["educ"]==0)].groupby(["field"])["field"].count() / np.shape(real[(real["work"]>0)&(real["educ"]==0)])[0]
    simudist = simu[(simu["work"]>0)&(simu["educ"]==0)].groupby(["field"])["field"].count() / np.shape(simu[(simu["work"]>0)&(simu["educ"]==0)])[0]
    
    return realdist, simudist 


def get_occupation_report(real,simu):
    pass 
    
    
    
    
def get_welfare_distribution(simu,conter):
    
    '''
    This function computes welfare difference between 10yr and IDR
    '''

    simuwelfare = simu.groupby(["parinc","ability","sex","ethnicity"])["welfare"].sum()
    simuwelfare = simuwelfare.rename('Baseline')
    conterwelfare = conter.groupby(["parinc","ability","sex","ethnicity"])["welfare"].sum()
    conterwelfare = conterwelfare.rename('Counterfactual')
    
    joint = pd.concat((simuwelfare,conterwelfare),axis=1)
    
    joint["Percentage"] = (joint["Baseline"]-joint["Counterfactual"])/joint["Baseline"]
    
    return joint

def group_graduation(data,level):
    
    data = data[(data["period"]==T-1)]
    groupab = data.groupby(["ability"])[f"{level}"].mean()
    grouppar = data.groupby(["parinc"])[f"{level}"].mean()
    groupsex = data.groupby(["sex"])[f"{level}"].mean()
    groupblack = data.groupby(["ethnicity"])[f"{level}"].mean()
    
    return groupab, grouppar, groupsex, groupblack



def get_table_distribution(simu,contermax,conternodebt,contergrants):
    
    # Baseline
    
    simuab2, simupar2, simusex2, simurace2 = group_graduation(simu,"twograd")
    simuab4, simupar4, simusex4, simurace4 = group_graduation(simu,"fourgrad")
    simuabG, simuparG, simusexG, simuraceG = group_graduation(simu,"gradgrad")
    
    simu2 = pd.concat([simupar2,simuab2,simusex2,simurace2],axis=0)
    simu4 = pd.concat([simupar4,simuab4,simusex4,simurace4],axis=0)
    simuG = pd.concat([simuparG,simuabG,simusexG,simuraceG],axis=0)
    
    # IDR Max Debt
    
    maxab2, maxpar2, maxsex2, maxrace2 = group_graduation(contermax,"twograd")
    maxab4, maxpar4, maxsex4, maxrace4 = group_graduation(contermax,"fourgrad")
    maxabG, maxparG, maxsexG, maxraceG = group_graduation(contermax,"gradgrad")
    
    max2 = pd.concat([maxpar2,maxab2,maxsex2,maxrace2],axis=0)
    max4 = pd.concat([maxpar4,maxab4,maxsex4,maxrace4],axis=0)
    maxG = pd.concat([maxparG,maxabG,maxsexG,maxraceG],axis=0)
    
    # Conter No Debt
    
    nodebtab2, nodebtpar2, nodebtsex2, nodebtrace2 = group_graduation(conternodebt,"twograd")
    nodebtab4, nodebtpar4, nodebtsex4, nodebtrace4 = group_graduation(conternodebt,"fourgrad")
    nodebtabG, nodebtparG, nodebtsexG, nodebtraceG = group_graduation(conternodebt,"gradgrad")
    
    nodebt2 = pd.concat([nodebtpar2, nodebtab2, nodebtsex2, nodebtrace2],axis=0)
    nodebt4 = pd.concat([nodebtpar4, nodebtab4, nodebtsex4, nodebtrace4],axis=0)
    nodebtG = pd.concat([nodebtparG, nodebtabG, nodebtsexG, nodebtraceG],axis=0)
    
    
    # Conter Grants
    
    grantab2, grantpar2, grantsex2, grantrace2 = group_graduation(contergrants,"twograd")
    grantab4, grantpar4, grantsex4, grantrace4 = group_graduation(contergrants,"fourgrad")
    grantabG, grantparG, grantsexG, grantraceG = group_graduation(contergrants,"gradgrad")
    
    grants2 = pd.concat([grantpar2, grantab2, grantsex2, grantrace2],axis=0)
    grants4 = pd.concat([grantpar4, grantab4, grantsex4, grantrace4],axis=0)
    grantsG = pd.concat([grantparG, grantabG, grantsexG, grantraceG],axis=0)
    
    twoyear = pd.concat([simu2, max2, nodebt2, grants2],axis=1)
    fouryear = pd.concat([simu4, max4, nodebt4, grants4],axis=1)
    grad = pd.concat([simuG, maxG, nodebtG, grantsG],axis=1)
    
    table2 = latex_table_grad(np.array(twoyear))
    table4 = latex_table_grad(np.array(fouryear))
    tableG = latex_table_grad(np.array(grad))
    
    
def group_dropout(data,level):
    
    if level == "fourgrad":
        data = data[data["period"]==T-1]
        data["drop"] = np.minimum(data["fouryear_exp"],1) - data["fourgrad"]
        data["drop"] = np.maximum(data["drop"] - data["twograd"],0)
        data = data[data["fouryear_exp"]>0]
    else:
        data = data[data["period"]==T-1]
        data["drop"] = np.minimum(data["twoyear_exp"],1) - data["twograd"] 
        data["drop"] = np.maximum(data["drop"] - data["fourgrad"],0)
        data["drop"] = np.maximum(data["drop"] - np.minimum(data["fouryear_exp"],1),0)
        data = data[data["twoyear_exp"]>0]
        data = data[data["fouryear_exp"]==0]
        
        
    
    groupab = data.groupby(["ability"])["drop"].mean()
    grouppar = data.groupby(["parinc"])["drop"].mean()
    groupsex = data.groupby(["sex"])["drop"].mean()
    groupblack = data.groupby(["ethnicity"])["drop"].mean()
    
    return groupab, grouppar, groupsex, groupblack

def group_dropout_all(data):
    
    data = data[data["period"]==T-1]
    data["educ_exp"] = data["fouryear_exp"] + data["twoyear_exp"]
    data["drop"] = np.minimum(data["educ_exp"],1) - np.minimum(data["fourgrad"]+data["twograd"],1)
    data = data[(data["educ_exp"]>0)]

    
    groupab = data.groupby(["ability"])["drop"].mean()
    grouppar = data.groupby(["parinc"])["drop"].mean()
    groupsex = data.groupby(["sex"])["drop"].mean()
    groupblack = data.groupby(["ethnicity"])["drop"].mean()
    
    return groupab, grouppar, groupsex, groupblack
    
def get_table_distribution_dropout(real,simu,contermax,conternodebt,contergrants):
    
    realab2, realpar2, realsex2, realrace2 = group_dropout(real,"twograd")
    realab4, realpar4, realsex4, realrace4 = group_dropout(real,"fourgrad")
    realab, realpar, realsex, realrace = group_dropout_all(real)
    
    real2 = pd.concat([realpar2,realab2,realsex2,realrace2],axis=0)
    real4 = pd.concat([realpar4,realab4,realsex4,realrace4],axis=0)
    realall = pd.concat([realpar,realab,realsex,realrace],axis=0)
    
    simuab2, simupar2, simusex2, simurace2 = group_dropout(simu,"twograd")
    simuab4, simupar4, simusex4, simurace4 = group_dropout(simu,"fourgrad")
    simuab, simupar, simusex, simurace = group_dropout_all(simu)
    
    simu2 = pd.concat([simupar2,simuab2,simusex2,simurace2],axis=0)
    simu4 = pd.concat([simupar4,simuab4,simusex4,simurace4],axis=0)
    simuall = pd.concat([simupar,simuab,simusex,simurace],axis=0)
    
    simuab2, simupar2, simusex2, simurace2 = group_dropout_all(simu)
    simuab4, simupar4, simusex4, simurace4 = group_dropout_all(simu)
    
    simu2all = pd.concat([simupar2,simuab2,simusex2,simurace2],axis=0)
    simu4all = pd.concat([simupar4,simuab4,simusex4,simurace4],axis=0)
    
    # IDR Max Debt
    
    maxab2, maxpar2, maxsex2, maxrace2 = group_dropout(contermax,"twograd")
    maxab4, maxpar4, maxsex4, maxrace4 = group_dropout(contermax,"fourgrad")
    
    max2 = pd.concat([maxpar2,maxab2,maxsex2,maxrace2],axis=0)
    max4 = pd.concat([maxpar4,maxab4,maxsex4,maxrace4],axis=0)
    
    # Conter No Debt
    
    nodebtab2, nodebtpar2, nodebtsex2, nodebtrace2 = group_dropout(conternodebt,"twograd")
    nodebtab4, nodebtpar4, nodebtsex4, nodebtrace4 = group_dropout(conternodebt,"fourgrad")
    
    nodebt2 = pd.concat([nodebtpar2, nodebtab2, nodebtsex2, nodebtrace2],axis=0)
    nodebt4 = pd.concat([nodebtpar4, nodebtab4, nodebtsex4, nodebtrace4],axis=0)
    
    
    # Conter Grants
    
    grantab2, grantpar2, grantsex2, grantrace2 = group_dropout(contergrants,"twograd")
    grantab4, grantpar4, grantsex4, grantrace4 = group_dropout(contergrants,"fourgrad")
    
    grants2 = pd.concat([grantpar2, grantab2, grantsex2, grantrace2],axis=0)
    grants4 = pd.concat([grantpar4, grantab4, grantsex4, grantrace4],axis=0)
    
    twoyear = pd.concat([simu2, max2, nodebt2, grants2],axis=1)
    fouryear = pd.concat([real4,simu4, max4, nodebt4, grants4],axis=1)
    fitdrop = pd.concat([real2,simu2, real4,simu4,realall,simuall],axis=1)
    
    #table2 = latex_table_grad(np.array(twoyear))
    table4 = latex_table_drop(np.array(fouryear))
    
    
def latex_table_drop(data):
    
    
    k, m = data.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
        
    column_names =  "  & Data &  Baseline  & SAVE & No Debt &   Grants   \\\\  \\hline "
    latex_table += column_names
    
    rowanmes = ["ParInc Q1","ParInc Q2", "ParInc Q3", "ParInc Q4","AFQT Q1", "AFQT Q2", 
                "AFQT Q3", "AFQT Q4","Male", "Female", "Non-Black","Black"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${data[i, j]:.3f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Regression coefficients and standard errors}\n\\end{table}"
    
    
    return latex_table
    
def latex_table_grad(data):
    
    
    k, m = data.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
        
    column_names =  "  &  Baseline  & SAVE & No Debt &   Grants   \\\\  \\hline "
    latex_table += column_names
    
    rowanmes = ["ParInc Q1","ParInc Q2", "ParInc Q3", "ParInc Q4","AFQT Q1", "AFQT Q2", 
                "AFQT Q3", "AFQT Q4","Male", "Female", "Non-Black","Black"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${data[i, j]:.3f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Regression coefficients and standard errors}\n\\end{table}"
    
    
    return latex_table

def group_work_enrolled(data):
    
    a = get_total_choices()

    data["field"] = a[data["choices"],0] 
    data["educ"] =a[data["choices"],1]
    data["work"] = a[data["choices"],2]
    
    
    data = data[data["educ"]==2]
    
    data["nowork"] = 0 
    data["partime"] = 0 
    data["fulltime"] = 0
    data.loc[data["work"]==0,"nowork"] = 1
    data.loc[data["work"]==1,"partime"] = 1
    data.loc[data["work"]==2,"fulltime"] = 1
    
    cpar  = data.groupby(["parinc"])[["nowork","partime","fulltime"]].mean()
    cab   = data.groupby(["ability"])[["nowork","partime","fulltime"]].mean()
    csex  = data.groupby(["sex"])[["nowork","partime","fulltime"]].mean()
    crace = data.groupby(["ethnicity"])[["nowork","partime","fulltime"]].mean()
    
    c = pd.concat([cpar,cab,csex,crace],axis=0)

    
    return c

def get_table_distribution_working_enrolled(simu,contermax):
    

    simu    = group_work_enrolled(simu)
    conter = group_work_enrolled(contermax)    
   
    table = pd.concat([simu["nowork"], conter["nowork"],
                          simu["partime"],conter["partime"],
                          simu["fulltime"],conter["fulltime"]],axis=1)
    
    table = latex_table_work(np.array(table))
    
    
def latex_table_work(data):
    
    
    k, m = data.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
        
    column_names =  "  & Baseline &  SAVE  & Baseline &  SAVE &  Baseline &  SAVE   \\\\  \\hline "
    latex_table += column_names
    
    rowanmes = ["ParInc Q1","ParInc Q2", "ParInc Q3", "ParInc Q4","AFQT Q1", "AFQT Q2", 
                "AFQT Q3", "AFQT Q4","Male", "Female", "Non-Black","Black"]
    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${data[i, j]:.3f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Regression coefficients and standard errors}\n\\end{table}"
    
    
    return latex_table   
    
    
def graduate_distribution(real,simu,level):
    
    """
    This function computes the graduate distribution at the terminal peirod
    """
    
    #  For all
    real = real[(real["period"]==T-1)]
    groupreal = real.groupby(["parinc","ability"])[f"{level}"].mean()
    groupreal = groupreal.rename("real")
    simu = simu[(simu["period"]==T-1)]
    groupsimu = simu.groupby(["parinc","ability"])[f"{level}"].mean()
    groupsimu = groupsimu.rename("simu")
    
    group = pd.concat((groupreal,groupsimu),axis=1)
    
    # For parental income
    
    groupreal = real.groupby(["parinc"])[f"{level}"].mean()
    groupreal = groupreal.rename("real")
    groupsimu = simu.groupby(["parinc"])[f"{level}"].mean()
    groupsimu = groupsimu.rename("simu")
    
    groupinc = pd.concat((groupreal,groupsimu),axis=1)
    # For ability
    
    groupreal = real.groupby(["ability"])[f"{level}"].mean()
    groupreal = groupreal.rename("real")
    groupsimu = simu.groupby(["ability"])[f"{level}"].mean()
    groupsimu = groupsimu.rename("simu")
    
    groupab = pd.concat((groupreal,groupsimu),axis=1)
    
    # Save to do graphs. 
    
    group.to_stata(f"{datafigures}/grad_dist_all_{level}.dta",write_index=False)
    groupinc.to_stata(f"{datafigures}/grad_dist_inc_{level}.dta",write_index=False)
    groupab.to_stata(f"{datafigures}/grad_dist_ab_{level}.dta",write_index=False)
    
    return group, groupinc, groupab


def get_debt_graduation(real,simu):
    
    realgrad = real[(real["fourgrad"]==1) & (real["period"]==real["period_educ_last"]+1)  &(real["lastchoice"]!=13)]
    simugrad = simu[(simu["fourgrad"]==1) & (simu["period"]==simu["last_educ"]+1)  &(simu["last_choice"]!=13)]
    
    realdebt = realgrad[realgrad["currentloans"]>0].groupby(["parinc","ability"])["currentloans"].mean()
    simudebt = simugrad[simugrad["debt"]>0].groupby(["parinc","ability"])["debt"].mean()
    return realdebt, simudebt 

def choices_over_experience(real,simu):
    
    for exp in range(0,5):
        
        get_distribution_choice_field(real[real["fouryear_exp"] == exp],simu[simu["fouryear_exp"] == exp],exp,1)
        
    pass

def choices_over_period(real,simu):
    
    for period in range(1,T):
        
        get_distribution_choice_field_period(real[real["period"] == period],simu[simu["period"] == period],period,1)
        
    pass

def get_distribution_choice_field_period(real,simu,exp,graphtype):
    
    """This function computes the distribution of majors at period 10. """
    
    a = get_total_choices()

    simu["field"] = a[simu["choices"],0] 
    simu["educ"] =a[simu["choices"],1]
    simu["work"] = a[simu["choices"],2]
    

    dist_real = real[(real["educ"]==2)&(real["field"]!=3)].groupby(["field"])["field"].count() / np.shape(real[real["educ"]==2])[0]
    dist_simu = simu[(simu["educ"]==2)&(simu["field"]!=3)].groupby(["field"])["field"].count() / np.shape(simu[simu["educ"]==2 ])[0]


    width = 0.4
       
    positions1 = np.arange(1,14,2)-width
    positions2 = np.arange(1,14,2)+width
    positions3 = np.arange(1,14,2)
    names = ["Bus","STEM","Educ","Social","Huamn","Health","Other"]
    
    
  
    if graphtype == 1:
        leg = ["Data","Model"]
    elif graphtype == 2:
        leg = ["Baseline","Conterfactual"]

    fig, ax = plt.subplots()
    axes = plt.gca()
    axes.yaxis.grid(zorder=0)
    ax.bar(positions1,dist_real,width=width*2, zorder=3,edgecolor="black")
    ax.bar(positions2,dist_simu,width=width*2, zorder=3,edgecolor="black")
    ax.set_xticks(positions3, labels=names,rotation = 45)
    plt.ylabel("Share of each field")
    plt.legend(leg,loc='upper center', bbox_to_anchor=(0.5,-0.2), ncol = 2)
    plt.savefig(f"{figures}/share_each_field_{graphtype}_period{exp}.png",bbox_inches='tight')

def get_distribution_choice_field(real,simu,exp,graphtype):
    
    """This function computes the distribution of majors at period 10. """
    
    a = get_total_choices()

    simu["field"] = a[simu["choices"],0] 
    simu["educ"] =a[simu["choices"],1]
    simu["work"] = a[simu["choices"],2]
    

    dist_real = real[(real["educ"]==2)].groupby(["field"])["field"].count() / np.shape(real[real["educ"]==2])[0]
    dist_simu = simu[(simu["educ"]==2)].groupby(["field"])["field"].count() / np.shape(simu[simu["educ"]==2 ])[0]

        
    width = 0.4
    if (exp < 2) | (exp == 99):
        positions1 = np.arange(1,16,2)-width
        positions2 = np.arange(1,16,2)+width
        positions3 = np.arange(1,16,2)
    
        names = ["Bus","STEM","Undeclared","Educ","Social","Huamn","Health","Other"]
    elif (exp>=2) & (exp!=99):
        positions1 = np.arange(1,14,2)-width
        positions2 = np.arange(1,14,2)+width
        positions3 = np.arange(1,14,2)
    
        names = ["Bus","STEM","Educ","Social","Huamn","Health","Other"]
    
    if graphtype == 1:
        leg = ["Data","Model"]
    elif graphtype == 2:
        leg = ["Baseline","Conterfactual"]

    fig, ax = plt.subplots()
    axes = plt.gca()
    axes.yaxis.grid(zorder=0)
    ax.bar(positions1,dist_real,width=width*2, zorder=3,edgecolor="black")
    ax.bar(positions2,dist_simu,width=width*2, zorder=3,edgecolor="black")
    ax.set_xticks(positions3, labels=names,rotation = 45)
    plt.ylabel("Share of each field")
    plt.legend(leg,loc='upper center', bbox_to_anchor=(0.5,-0.2), ncol = 2)
    plt.savefig(f"{figures}/share_each_field_{graphtype}_experience{exp}.png",bbox_inches='tight')

def get_choices_general(real,simu,period):
    
    """This function computes the distribution of majors at period 10. """
    
    a = get_total_choices()
    
    real = real[real["period"]==period]
    simu = simu[simu["period"]==period]

    simu["field"] = a[simu["choices"],0] 
    simu["educ"] =a[simu["choices"],1]
    simu["work"] = a[simu["choices"],2]
    

    dist_real = real[real["educ"]!=3].groupby(["educ","work"])["educ"].count() / np.shape(real)[0]
    dist_simu = simu[simu["educ"]!=3].groupby(["educ","work"])["educ"].count() / np.shape(simu)[0]


    return dist_real, dist_simu

def get_choices_general_vis(real,simu):
    
    """This function computes the distribution of majors at period 10. """
    
    a = get_total_choices()
    

    simu["field"] = a[simu["choices"],0] 
    simu["educ"] =a[simu["choices"],1]
    simu["work"] = a[simu["choices"],2]
    

    dist_real = real.groupby(["educ","work"])["educ"].count() / np.shape(real)[0]
    dist_simu = simu.groupby(["educ","work"])["educ"].count() / np.shape(simu)[0]


    return dist_real, dist_simu

def table_choices(real,simu):
    
    rall, sall = get_choices_general_vis(real,simu)
    rall1, sall1 = get_choices_general_vis(real[real["period"]==1],simu[simu["period"]==1])
    rall9, sall9 = get_choices_general_vis(real[real["period"]==9],simu[simu["period"]==9])
    
    sall1 = np.concatenate((sall1,np.zeros(3)))
    rall1 = np.concatenate((rall1,np.zeros(3)))
    rall = np.array(rall)
    sall = np.array(sall)
    sall9 = np.array(sall9)
    rall9 = np.array(rall9)
    
    realdist = np.concatenate((rall[...,None],sall[...,None],rall1[...,None],
                               sall1[...,None],rall9[...,None],sall9[...,None]),axis=1)
    
     
    k, m = realdist.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
    
    # Create column names
    column_names =  "\\hline  \\\\  &   All Periods & Period 1 & Period 2   \\\\  \\hline \\\\"
    latex_table += column_names
    
    rowanmes = ["Not Educ, Not Work","Not Educ, Part-Time", "Not Educ, Full-Time", "Two-Year,No-Work",
                "Two-Year,Part-Time", "Two-Year Full-Time","Four-Year, No Work", "Four-Year, Part-Time",
                "Four-Year,Full-Time", "Grad Sch, No Work", "Grad Sch, Part-Time", "Grad Sch, Full-Time"]    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${realdist[i, j]:.2f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    latex_table += "\\hline\n\\end{tabular}\n\\caption{Regression coefficients and standard errors}\n\\end{table}"
    
def table_work_conter(data):
        
     
    k, m = data.shape  # Get the dimensions
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * m + "}\n"
    latex_table += "\\hline\n"
    
    # Create column names
    column_names =  "\\hline  \\\\  &  Baseline & SAVE & No Debt & Grants   \\\\  \\hline \\\\"
    latex_table += column_names
    
    rowanmes = ["Not Educ, Not Work","Not Educ, Part-Time", "Not Educ, Full-Time", "Two-Year,No-Work",
                "Two-Year,Part-Time", "Two-Year Full-Time","Four-Year, No Work", "Four-Year, Part-Time",
                "Four-Year,Full-Time", "Grad Sch, No Work", "Grad Sch, Part-Time", "Grad Sch, Full-Time"]    
    # Fill in the table rows (coefficients and standard errors)
    for i in range(k):
        coef_row = [f"${data[i, j]:.2f}$" for j in range(m)]
        latex_table += f"{rowanmes[i]} & " + " & ".join(coef_row) + " \\\\\n"

    latex_table += "\\hline\n\\end{tabular}\n\\caption{Regression coefficients and standard errors}\n\\end{table}"
    
    return latex_table

def get_table_fields(simu,conter):
    
    simu = simu[simu["parinc"]==1]
    conter = conter[conter["parinc"]==1]
    
    simuc   = simu[simu["fourgrad"]==1]
    conterc = conter[conter["fourgrad"]==1]
    
    simuc   = simuc.groupby(["majorgrad"])["fourgrad"].count() / (np.shape(simuc)[0])
    conterc = conterc.groupby(["majorgrad"])["fourgrad"].count() / (np.shape(conterc)[0])
    
    simuu   = simu.groupby(["majorgrad"])["fourgrad"].count() / (np.shape(simu)[0])
    conteru = conter.groupby(["majorgrad"])["fourgrad"].count() / (np.shape(conter)[0])
    
    dist = pd.concat([simuc,conterc,simuu,conteru],axis=1)
    
def get_choices_general_educ(real,simu,educ):
    
    """This function computes the distribution of majors at period 10. """
    
    a = get_total_choices()

    simu["field"] = a[simu["choices"],0] 
    simu["educ"] =a[simu["choices"],1]
    simu["work"] = a[simu["choices"],2]
    
    real = real[real["educ"]==educ]
    simu = simu[simu["educ"]==educ]
    

    share_real = real.groupby(["educ","work"])["educ"].count() / np.shape(real)[0]
    share_model = simu.groupby(["educ","work"])["educ"].count() / np.shape(simu)[0]
    
    share_real = share_real.rename("real")
    share_model = share_model.rename("model")
    tosave = pd.concat([share_real,share_model],axis=1)
    
    tosave.to_stata(f"{datafigures}/work_distribution_educ{educ}.dta",write_index=False)

    return share_real, share_model

def get_choices_general_educ_conter(simu,conter,conter_max,conter_nodebt,conter_grants,educ):
    
    """This function computes the distribution of majors at period 10. """
    
    a = get_total_choices()

    simu["field"] = a[simu["choices"],0] 
    simu["educ"] =a[simu["choices"],1]
    simu["work"] = a[simu["choices"],2]
    
    conter["field"] = a[conter["choices"],0] 
    conter["educ"] =a[conter["choices"],1]
    conter["work"] = a[conter["choices"],2]
    
    conter_max["field"] = a[conter_max["choices"],0] 
    conter_max["educ"] =a[conter_max["choices"],1]
    conter_max["work"] = a[conter_max["choices"],2]
    
    
    conter_nodebt["field"] = a[conter_nodebt["choices"],0] 
    conter_nodebt["educ"] =a[conter_nodebt["choices"],1]
    conter_nodebt["work"] = a[conter_nodebt["choices"],2]
    
    
    conter_grants["field"] = a[conter_grants["choices"],0] 
    conter_grants["educ"] =a[conter_grants["choices"],1]
    conter_grants["work"] = a[conter_grants["choices"],2]
    
    simu = simu[simu["educ"]==educ]
    conter = conter[conter["educ"]==educ]
    conter_max = conter_max[conter_max["educ"]==educ]
    conter_nodebt = conter_nodebt[conter_nodebt["educ"]==educ]
    conter_grants = conter_grants[conter_grants["educ"]==educ]
    

    share_model = simu.groupby(["educ","work"])["educ"].count() / np.shape(simu)[0]
    share_conter = conter.groupby(["educ","work"])["educ"].count() / np.shape(conter)[0]
    share_conter_max = conter_max.groupby(["educ","work"])["educ"].count() / np.shape(conter_max)[0]
    share_conter_nodebt = conter_nodebt.groupby(["educ","work"])["educ"].count() / np.shape(conter_nodebt)[0]
    share_conter_grants = conter_grants.groupby(["educ","work"])["educ"].count() / np.shape(conter_grants)[0]
    
    share_model = share_model.rename("model")
    share_conter = share_conter.rename("conter")
    share_conter_max = share_conter_max.rename("conter_max")
    share_conter_nodebt = share_conter_nodebt.rename("conter_nodebt")
    share_conter_grants = share_conter_grants.rename("conter_grants")
    tosave = pd.concat([share_model,share_conter,share_conter_max,share_conter_nodebt,share_conter_grants],axis=1)
    
    tosave.to_stata(f"{datafigures}/work_distribution_conterfactual_educ{educ}.dta",write_index=False)

    return pd.concat([share_model,share_conter_max,share_conter_nodebt,share_conter_grants],axis=1)

def get_choices_general_educ_table(simu,conter,conter_max,conter_nodebt,conter_grants):
    
    """This function computes the distribution of majors at period 10. """
    
    table0 = get_choices_general_educ_conter(simu,conter,conter_max,conter_nodebt,conter_grants,0)   
    table1 = get_choices_general_educ_conter(simu,conter,conter_max,conter_nodebt,conter_grants,1)   
    table2 = get_choices_general_educ_conter(simu,conter,conter_max,conter_nodebt,conter_grants,2)   
    table3 = get_choices_general_educ_conter(simu,conter,conter_max,conter_nodebt,conter_grants,3)   

    totable = pd.concat([table0,table1,table2,table3],axis=0)
    
    table0 = get_choices_general_educ_conter(simu[simu["debt"]==0],
                                             conter,conter_max,conter_nodebt,conter_grants,0)   
    table1 = get_choices_general_educ_conter(simu,conter,conter_max,conter_nodebt,conter_grants,1)   
    table2 = get_choices_general_educ_conter(simu,conter,conter_max,conter_nodebt,conter_grants,2)   
    table3 = get_choices_general_educ_conter(simu,conter,conter_max,conter_nodebt,conter_grants,3)   

    
    table = table_work_conter(np.array(totable))
    
    return table


def table_debt(real,simu):
    
    
    realdebt,simudebt = get_debt_descriptives(real,simu)
    
    
    rows = []
    abilities = simudebt.index.get_level_values('ability').unique()
    parental_incomes = simudebt.index.get_level_values('parinc').unique()

    # Iterate through abilities
    for ability in abilities:
        # Row for share_indebted (A and B)
        share_row = [f"Q{ability} "r"\textit{Share}"]
        for parental_income in parental_incomes:
            share_A = realdebt.loc[(parental_income, ability), 'anydebt']
            share_B = simudebt.loc[(parental_income, ability), 'anydebt']
            share_row.append(f"{share_A:.2f} & {share_B:.2f}")
        rows.append(" & ".join(share_row) + " \\\\")

        # Row for totaldebt (A and B)
        debt_row = [r"\hspace{5mm} \textit{Avg}"]
        for parental_income in parental_incomes:
            debt_A = round(realdebt.loc[(parental_income, ability), 'currentloans'],2)
            debt_B = round(simudebt.loc[(parental_income, ability), 'currentloans'],2)
            debt_row.append(f"{debt_A:.0f} & {debt_B:.0f}")
        rows.append(" & ".join(debt_row) + " \\\\[2pt] \hline ")

    # Header for the LaTeX table
    header = """\\begin{tabular}{c|cc|cc|cc|cc}
        \hline\hline \\\\
        &  & \multicolumn{6}{c}{Parental Income} & \\\\ \hline  \\\\
            & \multicolumn{2}{c}{Q1} \\vline & \multicolumn{2}{c}{Q2} \\vline & \multicolumn{2}{c}{Q3} \\vline & \multicolumn{2}{c}{Q4} \\\\
    Ability & Data & Model & Data & Model & Data & Model & Data & Model \\\\[2pt]
    \\hline
    """
    
    # Combine header and rows
    latex_table = header + "\n".join(rows) + "\n \hline \n \end{tabular}"


    print(latex_table)
    
    
    
    
    
    

def merge_data(state,choices,real=0):
    
    """
    This function puts together the state and choices data 
    """
    
    if real == 0: 
        
        state = state[state["period"]<T]
        choices = choices[choices["period"]<T] 
        
        state["choices"] = np.array(choices["choices"])
    
    if real == 1: 
        
        state["choices"] = get_choices_index(np.array(state[["field","educ","work"]]))
        
    
    return state


def load_epsi():
    
    epsi = r"C:\Users\Sergi\Dropbox\PhD\Projects\Papers\1_financial_constraints\Output\SimuSameEpsi"
    names = ["PUBID","parinc","ability","sex","ethnicity","exp","twoyear_exp","fouryear_exp","grad","twograd","fourgrad","gradgrad","last_educ","majorgrad","last_choice","debt","period","sample","welfare"]

    simu = pd.DataFrame(np.load(f"{epsi}/simu.npy"),columns=names)
    contermax = pd.DataFrame(np.load(f"{epsi}/conter_maxdebt.npy"),columns=names)
    conter_nodebt = pd.DataFrame(np.load(f"{epsi}/conter_nodebt.npy"),columns=names)
    conter_grants = pd.DataFrame(np.load(f"{epsi}/conter_grants.npy"),columns=names)
    
    return  simu, contermax, conter_nodebt,conter_grants

#-----------------------------------------------------------------------------#
# Load data

# Real
samples = 30
real = load_real_data()
simu = load_data(1,samples,True) 
conter = load_data(2,samples,False)
conter_maxdebt = load_data(2,samples, True) 
conter_nodebt = load_data(3,samples,True)
conter_grants = load_data(4,samples,True) 
conter_stem = load_data(5,samples,True) 
simu_good = load_data(6,samples,True)
conter_good = load_data(7,samples,True)
#cont = load_data(2)
get_shares(real,simu,2,"None",0)
get_shares(real,simu,1,"None",0) 
get_shares(real,simu,3,"None",0)

#simu, contermax, conter_nodebt,conter_grants = load_epsi()

#%% 

real_choices = load_choices_real()
simu_choices = load_choices(1,samples,True)
conter_choices = load_choices(2,samples,False)
conter_choices_max = load_choices(2,samples,True)
choices_nodebt = load_choices(3,samples,True)
choices_grants = load_choices(4,samples,True)
choices_stem = load_choices(5,samples,True)
choices_simu_good = load_choices(6,samples,True)
choices_conter_good = load_choices(7,samples,True)

period = 1
csimu = get_choice_distribution(simu_choices,period)
ccont = get_choice_distribution(conter_choices,period)
creal = get_choice_distribution(real_choices,period)

get_distribution(T-1,real,simu,2,1,"modelfit")
get_distribution(T-1,conter,simu,2,1,"conterfactual")
#get_distribution(conter,simu,2,1,"conter")

#evolution_choices_expanded(real_choices,simu_choices,"fields")
#evolution_choices_expanded(real_choices,simu_choices,"work")

#get_report(real_choices,simu_choices)
#get_distribution(real,cont,2,2,"conterfactual")

# graduate distributions
gall, ginc, gab = graduate_distribution(real,simu,"fourgrad")
gall, ginc, gab = graduate_distribution(real,simu,"twograd")

# put data together
allreal = merge_data(real,real_choices,1) 
allsimu = merge_data(simu,simu_choices) 
allconter = merge_data(conter,conter_choices)
allcontermax = merge_data(conter_maxdebt,conter_choices_max)
allconter_nodebt = merge_data(conter_nodebt,choices_nodebt)
allconter_grants = merge_data(conter_grants,choices_grants)
allconter_stem = merge_data(conter_stem,choices_stem)
allsimu_good = merge_data(simu_good,choices_simu_good)
allconter_good = merge_data(conter_good,choices_conter_good)

get_distribution_choice_field(allreal,allsimu,99,1)
period = T-1
arealdist = state_distribution(allreal,period,2)
asimudist = state_distribution(allsimu,period,2)

realdebt,simudebt = get_debt_descriptives(real,simu)
realdebt,conterdebt = get_debt_descriptives(real,conter)

creal,csimu = get_choices_general(allreal,allsimu,1)
creal,cconter = get_choices_general(allreal,allconter,1)


creal,csimu = get_choices_general_educ(allreal,allsimu,0)
creal,csimu = get_choices_general_educ(allreal,allsimu,1)
creal,csimu = get_choices_general_educ(allreal,allsimu,2)
creal,csimu = get_choices_general_educ(allreal,allsimu,3)

table = get_choices_general_educ_table(allsimu,allconter,allcontermax,allconter_nodebt,allconter_grants)


get_choices_general_educ_conter(allsimu,allconter,allcontermax,allconter_nodebt,allconter_grants,1)
get_choices_general_educ_conter(allsimu,allconter,allcontermax,allconter_nodebt,allconter_grants,2)
get_choices_general_educ_conter(allsimu,allconter,allcontermax,allconter_nodebt,allconter_grants,3)

effect = check_graduation_conterfactual(simu, conter)
effectfields = check_fields_conterfactual(simu, conter,"conter")
effectfields = check_fields_conterfactual(simu, conter_maxdebt,"maxdebt")
effectfields = check_fields_conterfactual(simu, conter_nodebt,"nodebt")
effectfields = check_fields_conterfactual(simu, conter_grants,"grants")

check_fields_conterfactual_groups(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"parinc",1)
check_fields_conterfactual_groups(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"parinc",2)
check_fields_conterfactual_groups(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"parinc",3)
check_fields_conterfactual_groups(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"parinc",4)

check_fields_conterfactual_groups(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"ability",1)
check_fields_conterfactual_groups(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"ability",2)
check_fields_conterfactual_groups(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"ability",3)
check_fields_conterfactual_groups(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"ability",4)

check_fields_conterfactual_groups(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"sex",0)
check_fields_conterfactual_groups(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"sex",1)

check_fields_conterfactual_groups(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"ethnicity",0)
check_fields_conterfactual_groups(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"ethnicity",1)

check_fields_conterfactual_groups_total(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"parinc",1)
check_fields_conterfactual_groups_total(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"parinc",2)
check_fields_conterfactual_groups_total(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"parinc",3)
check_fields_conterfactual_groups_total(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"parinc",4)

check_fields_conterfactual_groups_total(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"ability",1)
check_fields_conterfactual_groups_total(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"ability",2)
check_fields_conterfactual_groups_total(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"ability",3)
check_fields_conterfactual_groups_total(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"ability",4)

check_fields_conterfactual_groups_total(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"sex",0)
check_fields_conterfactual_groups_total(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"sex",1)

check_fields_conterfactual_groups_total(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"ethnicity",0)
check_fields_conterfactual_groups_total(simu,conter,conter_maxdebt,conter_nodebt,conter_grants,"ethnicity",1)

effectfieldsall = check_fields_conterfactual_all(simu, conter)
effect_nodebt  = check_graduation_conterfactual(simu,conter_nodebt) 
effect_nodebt_vis = check_graduation_conterfactual(conter,conter_nodebt) 

get_shares_dropout(allreal,allsimu,1)
get_shares_dropout(allreal,allsimu,2)
get_entrance_shares(allreal,allsimu,1)
get_entrance_shares(allreal,allsimu,2)

occureal, occusimu = get_occupation_distribution(allreal,allsimu,99)
occureal, occuconter = get_occupation_distribution(allreal,allconter,99)


occureal, occusimu = get_occupation_distribution(allreal,allsimu,5)
occureal, occuconter = get_occupation_distribution(allreal,allconter,5)

welfare = get_welfare_distribution(simu,conter)

#%% Get all the data for ability, parinc, sex and race

for level in range(1,5):
    
    get_shares(real[real["ability"]==level],simu[simu["ability"]==level],2,"ability",level) 
    get_shares(real[real["ability"]==level],simu[simu["ability"]==level],1,"ability",level) 
    get_shares(real[real["ability"]==level],simu[simu["ability"]==level],3,"ability",level)
    
    get_shares(real[real["parinc"]==level],simu[simu["parinc"]==level],2,"parinc",level) 
    get_shares(real[real["parinc"]==level],simu[simu["parinc"]==level],1,"parinc",level) 
    get_shares(real[real["parinc"]==level],simu[simu["parinc"]==level],3,"parinc",level)
    

for level in range(0,2):
    
    get_shares(real[real["sex"]==level],simu[simu["sex"]==level],2,"sex",level) 
    get_shares(real[real["sex"]==level],simu[simu["sex"]==level],1,"sex",level) 
    get_shares(real[real["sex"]==level],simu[simu["sex"]==level],3,"sex",level)
    
    get_shares(real[real["ethnicity"]==level],simu[simu["ethnicity"]==level],2,"ethnicity",level) 
    get_shares(real[real["ethnicity"]==level],simu[simu["ethnicity"]==level],1,"ethnicity",level) 
    get_shares(real[real["ethnicity"]==level],simu[simu["ethnicity"]==level],3,"ethnicity",level)
    

#%%
# Generate choice distributions for different profiles:
level = 4 
period = T-1 
evolution_choices_expanded(allreal[allreal["ability"]==level],allsimu[allsimu["ability"]==level],"fields")
evolution_choices_expanded(allreal[allreal["ability"]==level],allsimu[allsimu["ability"]==level],"work")
get_shares(real[real["ability"]==level],simu[simu["ability"]==level],2) 
get_shares(real[real["ability"]==level],simu[simu["ability"]==level],1) 
get_shares(real[real["ability"]==level],simu[simu["ability"]==level],3) 
get_shares(conter[conter["ability"]==level],simu[simu["ability"]==level],2,2) 
get_shares(conter[conter["ability"]==level],simu[simu["ability"]==level],1,2) 
get_shares(conter[conter["ability"]==level],simu[simu["ability"]==level],3,2) 
get_distribution(period,real[real["ability"]==level],simu[simu["ability"]==level],2,1,"modelfit")
get_distribution(period,conter[conter["ability"]==level],simu[simu["ability"]==level],2,1,"conter")
arealdist = state_distribution(allreal[allreal["ability"]==level],period,2)   
asimudist = state_distribution(allsimu[allsimu["ability"]==level],period,2)  
choices_over_experience(allreal[allreal["ability"]==level],allsimu[allsimu["ability"]==level])
choices_over_period(allreal[allreal["ability"]==level],allsimu[allsimu["ability"]==level])

get_distribution_choice_field(allreal[allreal["ability"]==level],allsimu[allsimu["ability"]==level],99,1) 
#aconterdist = state_distribution(allconter[allconter["parinc"]==level],period,2)
creal,csimu = get_choices_general(allreal[allreal["ability"]==level],allsimu[allsimu["ability"]==level],9) 
print(np.sum(creal.loc[(2,)]),np.sum(csimu.loc[(2,)]))
effectfields = check_fields_conterfactual(simu[simu["ability"]==level], conter[conter["ability"]==level])

get_shares_dropout(allreal[allreal["ability"]==level],allsimu[allsimu["ability"]==level],1,graphtype=1)
get_shares_dropout(allreal[allreal["ability"]==level],allsimu[allsimu["ability"]==level],2,graphtype=1)

get_entrance_shares(allreal[allreal["ability"]==level],allsimu[allsimu["ability"]==level],1,graphtype=1)
get_entrance_shares(allreal[allreal["ability"]==level],allsimu[allsimu["ability"]==level],2,graphtype=1)

#%%
# Generate choice distributions for different profiles: 
level = 4 
period = T-1 
evolution_choices_expanded(allreal[allreal["parinc"]==level],allsimu[allsimu["parinc"]==level],"fields")
evolution_choices_expanded(allreal[allreal["parinc"]==level],allsimu[allsimu["parinc"]==level],"work")
get_shares(real[real["parinc"]==level],simu[simu["parinc"]==level],2) 
get_shares(real[real["parinc"]==level],simu[simu["parinc"]==level],1) 
get_shares(real[real["parinc"]==level],simu[simu["parinc"]==level],3) 
get_shares(conter[conter["parinc"]==level],simu[simu["parinc"]==level],2,2) 
get_shares(conter[conter["parinc"]==level],simu[simu["parinc"]==level],1,2) 
get_shares(conter[conter["parinc"]==level],simu[simu["parinc"]==level],3,2) 
get_distribution(period,real[real["parinc"]==level],simu[simu["parinc"]==level],2,1,"modelfit")
get_distribution(period,conter[conter["parinc"]==level],simu[simu["parinc"]==level],2,1,"conter")
arealdist = state_distribution(allreal[allreal["parinc"]==level],period,2)   
asimudist = state_distribution(allsimu[allsimu["parinc"]==level],period,2)  
choices_over_experience(allreal[allreal["parinc"]==level],allsimu[allsimu["parinc"]==level])
choices_over_period(allreal[allreal["parinc"]==level],allsimu[allsimu["parinc"]==level])

get_distribution_choice_field(allreal[allreal["parinc"]==level],allsimu[allsimu["parinc"]==level],99,1) 
#aconterdist = state_distribution(allconter[allconter["parinc"]==level],period,2)
creal,csimu = get_choices_general(allreal[allreal["parinc"]==level],allsimu[allsimu["parinc"]==level],6) 
effectfields = check_fields_conterfactual(simu[simu["parinc"]==level], conter[conter["parinc"]==level])

get_shares_dropout(allreal[allreal["parinc"]==level],allsimu[allsimu["parinc"]==level],1,graphtype=1)
get_shares_dropout(allreal[allreal["parinc"]==level],allsimu[allsimu["parinc"]==level],2,graphtype=1)
#%% 
# Analyze ovegraduation in twoyear
aarealdist = state_distribution(allreal,period,1)
aasimudist = state_distribution(allsimu,period,1)
level = 1  
period = 7
arealdist = state_distribution(allreal[allreal["parinc"]==level],period,1)
asimudist = state_distribution(allsimu[allsimu["parinc"]==level],period,1)