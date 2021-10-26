#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import strobe
import ramp
import json
import os


# Loading inputs for the desired case

case = 1
with open('inputs/case_'+str(case)+'.json') as f:
  inputs = json.load(f)


# Simulation function

def simulation(inputs):
    
    # Strobe
    result,textoutput = strobe.simulate_scenarios(1, inputs)
    n_scen = 0 # Working only with the first scenario
    
    # RAMP-mobility
    if inputs['EV']:
        result_ramp = ramp.EVCharging(inputs, result['occupancy'][n_scen])
    else:
        result_ramp=pd.DataFrame()
    
    # Creating dataframe with the results
    n_steps = np.size(result['StaticLoad'][n_scen,:])
    index = pd.date_range(start='2016-01-01 00:00', periods=n_steps, freq='1min')
    df = pd.DataFrame(index=index,columns=['StaticLoad','TumbleDryer','DishWasher','WashingMachine','ElectricalBoiler','HeatPumpPower','EVCharging'])
    result_ramp.loc[df.index[-1],'EVCharging'] = 0
    #df.index.union(result_ramp.index)        # too slow
    
    for key in df.columns:
        if key in result:
            df[key] = result[key][n_scen,:]
        elif key in result_ramp:
            df[key] = result_ramp[key]*1000
        else:
            df[key] = 0
    
    return(df)

# Running ten simulations for the case selected
 
nsimulations = 10
simulations = []

for i in range(nsimulations):
    # Running i-th simulation
    df = simulation(inputs)
    # Storing i-th simulation results
    newpath = r'.\simulations\case_'+str(case) 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    filename = 'run_{0}.pkl'.format(i+1)
    filename = os.path.join(newpath,filename)
    df.to_pickle(filename)
    # Appending i-th simulation results to results' list
    simulations.append(df)
    

    

        