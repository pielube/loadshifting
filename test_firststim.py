
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
from sklearn.metrics import r2_score




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
    
    results = []
    occupancy = []
  
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
        
        exectime = (time.time() - start_time)/60.
        print('Simulation: '+names[ii]+' Run: '+str(jj+1))
        print(' It took {:.1f} minutes'.format(exectime))
        
        results.append(df)
        occupancy.append(result['occupancy'][n_scen])
    
    df_tot = results[0]
    df_tot = df_tot.sum(axis=1)
    df_tot = df_tot.to_frame()
    df_tot.columns=[str(1)]
    for kk in range(len(results)-1):
        df = results[kk+1]
        df = df.sum(axis=1)
        df = df.to_frame()
        df.columns=[str(kk+2)]
        df_tot = df_tot.join(df)
    
    # Calculating mean demand
    
    df_mean = df_tot.mean(axis=1)
    df_mean = df_mean.to_frame()
 
    # Calculating most representative curve
    # as the curve minimizing its R2 wrt the mean curve    
 
    bestr2 = -float('inf')
    bestr2_index = 0

    for ll in range(len(results)):
        r2 = (r2_score(df_mean[0], df_tot.iloc[:,ll]))
        print(r2)
        if r2 > bestr2:
            bestr2 = r2
            bestr2_index = ll
    
    print('For '+names[ii]+' best R2 index: '+str(ll))
        
    """
    Saving results and occupancy
    """
    
    path = r'.\simulations\firstsim'
    if not os.path.exists(path):
        os.makedirs(path)
    name = names[ii]+'.pkl' #names[ii]+'_sim'+str(jj+1)+'.pkl'
    file = os.path.join(path,name)
    with open(file, 'wb') as b:
        pickle.dump(results)
        
    name = names[ii]+'_occ.pkl'
    file = os.path.join(path,name)
    with open(file, 'wb') as b:
        pickle.dump(occupancy)
    









        