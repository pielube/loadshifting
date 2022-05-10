
"""First results"""

import numpy as np
import pandas as pd
import strobe
import ramp
import json
from preprocess import ProcebarExtractor,HouseholdMembers, HouseholdMembers_real
import os
import pickle
from joblib import Memory

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

memory = Memory(__location__ + '/cache/', verbose=1)

@memory.cache
def compute_demand(inputss,N,members= None,thermal_parameters=None):
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
        inputs=inputss.copy()
        # People living in the dwelling
        # taken from strobe list
        print ('Il y a {} membres dans la maison'.format(inputs["members"]))
        if inputs["members"] is not None:
            newcase=inputs["members"]["child"]*['School']
            newcase.extend( HouseholdMembers_real(inputs["members"]))
            inputs["members"] = newcase
            print(inputs["members"])
        else:
            
            inputs['members'] = HouseholdMembers(inputs['HP']['dwelling_type'])
              
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
        
        # RAMP-mobility
        if inputs['EV']['loadshift']:
            result_ramp = ramp.EVCharging(inputs, result['occupancy'][n_scen])
            res_ramp_charge_home = result_ramp['charge_profile_home']
            inputs['EV']['MainDriver'] = result_ramp['main_driver']
        else:
            res_ramp_charge_home = pd.DataFrame()
    
        """
        Creating dataframe with the results
        """
        
        n_steps = np.size(result['StaticLoad'][n_scen,:])
        index = pd.date_range(start='2019-08-01 00:00', periods=n_steps, freq='1min')
        
        index_10min = pd.date_range(start='2019-08-01 00:00', periods=len(result['occupancy'][n_scen][0]), freq='10min')
        
        # Dataframe of demands
        df = pd.DataFrame(index=index,columns=['StaticLoad','TumbleDryer','DishWasher','WashingMachine','DomesticHotWater','HeatPumpPower','EVCharging','InternalGains'],dtype=object)
        
        res_ramp_charge_home.loc[df.index[-1],'EVCharging']=0
        
        for key in df.columns:
            if key in result:
                df[key] = result[key][n_scen,:]
            elif key in res_ramp_charge_home:
                df[key] = res_ramp_charge_home[key]* 1000
            else:
                df[key] = 0
        # Dataframe with the occupancy data
        occupancy = pd.DataFrame(index=index_10min)
        for i,m in enumerate(result['members'][n_scen]):
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
      
        out = compute_demand(inputs,N)
            
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









        