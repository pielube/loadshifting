
"""First results"""

import numpy as np
import pandas as pd
import copy
import strobe
import ramp
import json
from preprocess import ProcebarExtractor,HouseholdMembers, HouseholdMembers_real
import os
import pickle
<<<<<<< Updated upstream
from joblib import Memory
<<<<<<< Updated upstream
=======
>>>>>>> Stashed changes

import defaults
import copy 

<<<<<<< Updated upstream
memory = Memory(__location__ + '/cache/', verbose=1)
<<<<<<< Updated upstream

#@memory.cache
def compute_demand(inputss,N,members= None,thermal_parameters=None):
=======
@memory.cache






def compute_demand(inputs,N,members= None,thermal_parameters=None, factor_gain_sim=None, correction = False):
>>>>>>> Stashed changes
=======

def compute_demand(inputs,N,members= None,thermal_parameters=None, factor_gain_sim=None):
>>>>>>> Stashed changes
=======
import defaults
import copy 


def compute_demand(inputs,N,members= None,thermal_parameters=None, factor_gain_sim = None):
>>>>>>> Stashed changes
    '''
    Function that generates the stochastic time series for
    - The occupancy profiles
    - The household electrical demand
    - The heat pump demand
    - The DHW demand
    - The Electric vehicle charging profile

    Parameters
    ----------
    inputs : dict
        Dictionary with the simulation inputs.
    N : int
        Number of stochastic simulations to be run.

    Returns
    -------
    out : dict
        Contains, for each simulation, the demand curves, the occupancy profile and the input data.

    '''
    
    out = {'results':[],'occupancy':[],'input_data':[]}
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream

    for jj in range(N):          # run the simulation N times and append the results to the list
        inputs=inputss.copy()
=======
=======
>>>>>>> Stashed changes
    if factor_gain_sim is not None :
        TAAA=copy.deepcopy(factor_gain_sim)
        factor_gain_simu={}
    
<<<<<<< Updated upstream
    for jj in range(N):    # run the simulation N times and append the results to the list
        if factor_gain_sim is not None :
            factor_gain=copy.deepcopy(TAAA)

>>>>>>> Stashed changes
        # People living in the dwelling
        # taken from strobe list
        print ('Il y a {} membres dans la maison'.format(inputs["members"]))
        if inputs["members"] is not None:
            newcase=inputs["members"]["child"]*['U12']
            newcase.extend( HouseholdMembers_real(inputs["members"]))
            inputs["members"] = newcase
            print(inputs["members"])
        else:
            
            inputs['members'] = HouseholdMembers(inputs['HP']['dwelling_type'])
              
=======
    
    if factor_gain_sim is not None :
        TAAA=copy.deepcopy(factor_gain_sim)
        factor_gain_simu={}
    members_test =[]
    for jj in range(N):          # run the simulation N times and append the results to the list
        
        if factor_gain_sim is not None :
            factor_gain=copy.deepcopy(TAAA)
            
=======
    members_test =[]
    for jj in range(N):          # run the simulation N times and append the results to the list
        if factor_gain_sim is not None :
            factor_gain=copy.deepcopy(TAAA)
>>>>>>> Stashed changes
        # People living in the dwelling, taken as input or from Strobe's list
        if members is not None:
            inputs['members'] = HouseholdMembers(members)
        else:
            x = inputs['members']
            inputs['members'] = HouseholdMembers(inputs['members'])
        members_test.append(inputs['members'])
>>>>>>> Stashed changes
        # Thermal parameters of the dwelling
        # Taken from Procebar .xls files
        if thermal_parameters is not None:
            inputs['HP'] = {**inputs['HP'],**thermal_parameters}
        else:
            procebinp = ProcebarExtractor(inputs['HP']['dwelling_type'],True)
            inputs['HP'] = {**inputs['HP'],**procebinp}
        
        """
        Running the models
        """
        
        # Strobe
        # House thermal model + HP
        # DHW
        result,textoutput = strobe.simulate_scenarios(1,inputs)
        
        n_scen = 0 # Working only with the first scenario
        
<<<<<<< Updated upstream
        # RAMP-mobility
=======
        occ = np.array(result['occupancy'][n_scen])
        occupancy_10min = (occ==1).sum(axis=0) # when occupancy==1, the person is in the house and not sleeping
        occupancy_10min = (occupancy_10min>0)  # if there is at least one person awake in the house
        occupancy_10min = pd.Series(data=occupancy_10min, index = index10min)
        occupancy_1min = occupancy_10min.reindex(index1min,method='nearest')
        Qintgains = result['InternalGains'][n_scen]
        n1min = len(result['InternalGains'][n_scen])
        
        ### House heating ###
        
        Tset = np.full(n1min,defaults.T_sp_low) + np.full(n1min,defaults.T_sp_occ-defaults.T_sp_low) * occupancy_1min
        ts=1/60
        # Heat pump sizing
        if inputs['HP']['HeatPumpThermalPower'] is not None:
            QheatHP = inputs['HP']['HeatPumpThermalPower']
        else:
            QheatHP = HPSizing(inputs,defaults.fracmaxP) # W
            
        Qheat,Tin_heat = HouseHeating(inputs,QheatHP,Tset,Qintgains,temp,irr,n1min,defaults.heatseas_st,defaults.heatseas_end,ts)
        
        Eheat = np.zeros((1,n1min)) 
        for i in range(n1min):
            COP = COP_Tamb(temp[i])
            Eheat[0,i] = Qheat[i]/COP # W
        
        # result['HeatPumpPower2'][n_scen] = Eheat
        result['HeatPumpPower'] = Eheat
        
        
        ### RAMP-mobility ###
        
>>>>>>> Stashed changes
        if inputs['EV']['loadshift']:
            result_ramp = ramp.EVCharging(inputs, result['occupancy'][n_scen])
            res_ramp_charge_home = result_ramp['charge_profile_home']
            inputs['EV']['MainDriver'] = result_ramp['main_driver']
        else:
            res_ramp_charge_home = pd.DataFrame()
    
        inputs['members'] = x
        """
        Creating dataframe with the results
        """
        
        n_steps = np.size(result['StaticLoad'][n_scen,:])
        index = pd.date_range(start='2019-08-01 00:00', periods=n_steps, freq='1min')
        
        index_10min = pd.date_range(start='2019-08-01 00:00', periods=len(result['occupancy'][n_scen][0]), freq='10min')
        
<<<<<<< Updated upstream
        # Dataframe of demands
=======
        #  Dataframe of demands
>>>>>>> Stashed changes
        df = pd.DataFrame(index=index,columns=['StaticLoad','TumbleDryer','DishWasher','WashingMachine','DomesticHotWater','HeatPumpPower','EVCharging','InternalGains'],dtype=object)
        
        res_ramp_charge_home.loc[df.index[-1],'EVCharging']=0
        
        for key in df.columns:
            if key in result:
                df[key] = result[key][n_scen,:]
            elif key in res_ramp_charge_home:
                df[key] = res_ramp_charge_home[key]* 1000
            else : 
                df[key] = 0
<<<<<<< Updated upstream
        if factor_gain_sim is not None :
            for key in ['StaticLoad','TumbleDryer','DishWasher','WashingMachine','DomesticHotWater','HeatPumpPower']:
<<<<<<< Updated upstream
                factor_gain[key]['factor'].append((df[key].sum()/60000)/factor_gain[key]['value'])
        
            factor_gain_simu['simulation {}'.format(jj)] = factor_gain
            
        #Correction factor 
        
        if correction == True :
            for key in df.columns :
                df[key]=df[key]*factor_gain[key]['factor'][jj]
        
=======
                factor_gain[key]['factor']=(df[key].sum()/60000)/factor_gain[key]['value']
        
            factor_gain_simu['simulation {}'.format(jj)] = factor_gain
        
       
                
>>>>>>> Stashed changes
=======
                
        #  Correction factor
        if factor_gain_sim is not None :
            for key in ['StaticLoad','TumbleDryer','DishWasher','WashingMachine','DomesticHotWater','HeatPumpPower']:

                factor_gain[key]['factor'].append((df[key].sum()/60000)/factor_gain[key]['value'])
        
            factor_gain_simu['simulation {}'.format(jj)] = factor_gain
>>>>>>> Stashed changes
        # Dataframe with the occupancy data
        occupancy = pd.DataFrame(index=index_10min)
        for i,m in enumerate(result['members'][n_scen]):
            membername = str(i) + '-' + m 
            occupancy[membername] = result['occupancy'][n_scen][i]
        inputs['members'] = x
        
        out['results'].append(df)
        out['occupancy'].append(occupancy)
<<<<<<< Updated upstream
        out['input_data'].append(inputs)    
    if factor_gain_sim is not None :
<<<<<<< Updated upstream
        out['factor gain'] = factor_gain_simu    
    return out
=======
        out['factor gain'] = factor_gain_simu  
    return (out, members_test)
>>>>>>> Stashed changes
=======
        out['input_data'].append(inputs) 
        out['factor gain'] = factor_gain_simu 

    return (out)
>>>>>>> Stashed changes


if __name__ == "__main__":
    """
    Testing the main function and saving the results
    """
    path = './inputs'
    names = ['4f']# ['1f','2f','4f']
    
    for name in names:            # For each house type
        
        """
        Loading inputs
        """
        filename = name+'.json'
        file = os.path.join(path,filename)
        with open(file) as f:
          inputs = json.load(f)
        N = 1
      
        out, members = compute_demand(inputs,N)
        print(members) 
        """
        Saving results, occupancy, and inputs
        """
        
        path = './simulations'
        if not os.path.exists(path):
            os.makedirs(path)
            
        file = os.path.join(path,name+'.pkl')
        with open(file, 'wb') as b:
            pickle.dump(out['results'],b)
            
        file = os.path.join(path,name+'_occ.pkl')
        with open(file, 'wb') as b:
            pickle.dump(out['occupancy'],b)
            
        file = os.path.join(path,name+'_inputs.pkl')
        with open(file, 'wb') as b:
            pickle.dump(out['input_data'],b)    









        