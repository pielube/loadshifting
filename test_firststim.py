
"""First results"""

import numpy as np
import pandas as pd
import strobe
import ramp
import json
import time
from preprocess import ProcebarExtractor,HouseholdMembers
import os



for ii in range(3):
    
    """
    Loading inputs
    """
    
    path = r'.\inputs\firstsim'
    names = ['1f_machine0','2f_machine0','4f_machine0']
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

    # Thermal parameters of the dwelling
    # Taken from Procebar .xls files
    
    procebinp = ProcebarExtractor(inputs['HP']['dwelling_type'],True)
    inputs['HP'] = {**inputs['HP'],**procebinp}  
  
    for jj in range(10):
        
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
        
        df = pd.DataFrame(index=index,columns=['StaticLoad','TumbleDryer','DishWasher','WashingMachine','DomesticHotWater','HeatPumpPower','EVCharging'])
        result_ramp.loc[df.index[-1],'EVCharging']=0
        
        for key in df.columns:
            if key in result:
                df[key] = result[key][n_scen,:]
            elif key in result_ramp:
                df[key] = result_ramp[key]* 1000
            else:
                df[key] = 0
        
        """
        Saving results
        """
        
        path = r'.\simulations\firstsim'
        if not os.path.exists(path):
            os.makedirs(path)
        name = names[ii]+'_sim'+str(jj+1)+'.pkl'
        file = os.path.join(path,name)
        df.to_pickle(file)
    
        """
        Saving results for prosumpy
        """
    
        df = df.sum(axis=1)
        # Resampling at 15 min
        df = df.to_frame()
        df = df.resample('15Min').mean()
        # Extracting ref year used in the simulation
        df.index = pd.to_datetime(df.index)
        year = df.index.year[0]
        # Remove last row if is from next year
        nye = pd.Timestamp(str(year+1)+'-01-01 00:00:00')
        df = df.drop(nye)
        # save
        name = 'prosumpy'+names[ii]+'_sim'+str(jj+1)+'.pkl'
        file = os.path.join(path,name)
        df.to_pickle(file)
        
        exectime = (time.time() - start_time)/60.
        print('Simulation: '+names[ii]+' Run: '+str(jj+1))
        print(' It took {:.1f} minutes'.format(exectime))









        