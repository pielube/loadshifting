
"""First results"""

import numpy as np
import pandas as pd
import strobe
import ramp
import json
from preprocess import ProcebarExtractor,HouseholdMembers
import os
import pickle
from joblib import Memory
memory = Memory('./cache/', verbose=1)


@memory.cache
def compute_demand(inputs,N,members= None,thermal_parameters=None):
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

    for jj in range(N):          # run the simulation N times and append the results to the list
    
        # People living in the dwelling
        # taken from strobe list
        if members is not None:
            inputs['members'] = members
        else:
            inputs['members'] = HouseholdMembers(inputs['HP']['dwelling_type'])
               
        # Thermal parameters of the dwelling
        # Taken from Procebar .xls files
        if thermal_parameters is not None:
            inputs['HP'] = {**thermal_parameters,**procebinp}
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
        
        # RAMP-mobility
        if inputs['EV']['loadshift']:
            result_ramp = ramp.EVCharging(inputs, result['occupancy'][n_scen])
        else:
            result_ramp=pd.DataFrame()
    
        """
        Creating dataframe with the results
        """
        
        n_steps = np.size(result['StaticLoad'][n_scen,:])
        index = pd.date_range(start='2015-01-01 00:00', periods=n_steps, freq='1min')
        
        index_10min = pd.date_range(start='2015-01-01 00:00', periods=len(result['occupancy'][n_scen][0]), freq='10min')
        
        # Dataframe of demands
        df = pd.DataFrame(index=index,columns=['StaticLoad','TumbleDryer','DishWasher','WashingMachine','DomesticHotWater','HeatPumpPower','EVCharging','InternalGains'])
        result_ramp.loc[df.index[-1],'EVCharging']=0
        
        for key in df.columns:
            if key in result:
                df[key] = result[key][n_scen,:]
            elif key in result_ramp:
                df[key] = result_ramp[key]* 1000
            else:
                df[key] = 0
        # Dataframe with the 
        occupancy = pd.DataFrame(index=index_10min)
        for i,m in enumerate(result['members']):
            membername = str(i) + '-' + m 
            occupancy[membername] = result['occupancy'][n_scen][i]
        
        out['results'].append(df)
        out['occupancy'].append(occupancy)
        out['input_data'].append(inputs)    

    return out


if __name__ == "__main__":
    """
    Testing the main function and saving the results
    """
    path = r'./inputs'
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
      
        out = compute_demand(inputs,N)
            
        """
        Saving results, occupancy, and inputs
        """
        
        path = r'./simulations'
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









        