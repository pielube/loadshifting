
"""First results"""

import numpy as np
import pandas as pd
import strobe
import ramp
import json
import time
from preprocess import ProcebarExtractor,HouseholdMembers
import os
import pickle


for ii in range(1):
    
    """
    Loading inputs
    """
    
    path = r'./inputs'
    names = ['4f']# ['1f','2f','4f']
    name = names[ii]+'.json'
    file = os.path.join(path,name)
    with open(file) as f:
      inputs = json.load(f)
  
    # People living in the dwelling
    # Taken from StRoBe list
    cond1 = 'members' not in inputs
    cond2 = 'members' in inputs and inputs['members'] == None
    if cond1 or cond2:
        inputs['members'] = HouseholdMembers(inputs['HP']['dwelling_type'])
    
    results = []
    occupancy = []
    input_data = []

    for jj in range(1):
               
        # Thermal parameters of the dwelling
        # Taken from Procebar .xls files
        procebinp = ProcebarExtractor(inputs['HP']['dwelling_type'],True)
        inputs['HP'] = {**inputs['HP'],**procebinp}
        
        start_time = time.time()
    
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
        
        # Dataframe of original results
        
        df = pd.DataFrame(index=index,columns=['StaticLoad','TumbleDryer','DishWasher','WashingMachine','DomesticHotWater','HeatPumpPower','EVCharging','InternalGains'])
        result_ramp.loc[df.index[-1],'EVCharging']=0
        
        for key in df.columns:
            if key in result:
                df[key] = result[key][n_scen,:]
            elif key in result_ramp:
                df[key] = result_ramp[key]* 1000
            else:
                df[key] = 0
        
        exectime = (time.time() - start_time)/60.
        print('Simulation: '+names[ii]+' Run: '+str(jj+1))
        print(' It took {:.1f} minutes'.format(exectime))
        
        results.append(df)
        occupancy.append(result['occupancy'][n_scen])
        input_data.append(inputs)
        
    """
    Saving results, occupancy, and inputs
    """
    
    path = r'./simulations'
    if not os.path.exists(path):
        os.makedirs(path)
        
    name = names[ii]+'.pkl'
    file = os.path.join(path,name)
    with open(file, 'wb') as b:
        pickle.dump(results,b)
        
    name = names[ii]+'_occ.pkl'
    file = os.path.join(path,name)
    with open(file, 'wb') as b:
        pickle.dump(occupancy,b)
        
    name = names[ii]+'_inputs.pkl'
    file = os.path.join(path,name)
    with open(file, 'wb') as b:
        pickle.dump(input_data,b)    









        