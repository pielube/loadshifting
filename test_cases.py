#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import strobe
import ramp
import json
import os
import matplotlib.pyplot as plt


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
    # n_steps = 1000
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


def profile_average(stoch_profiles):
    nelem = np.size(stoch_profiles[0])
    Profile_avg = np.zeros(nelem)
    for pr in stoch_profiles:
        Profile_avg = Profile_avg + pr
    Profile_avg = Profile_avg/len(stoch_profiles) 
    return Profile_avg

def Profile_cloud_plot(stoch_profiles,stoch_profiles_avg):
    nelem = np.size(stoch_profiles_avg)
    plt.figure(figsize=(10,5))
    for n in stoch_profiles:
        plt.plot(np.arange(nelem),n,'#b0c4de')
        plt.xlabel('Time (hours)')
        plt.ylabel('Power (W)')
        plt.ylim(ymin=0)
        #plt.ylim(ymax=5000)
        plt.margins(x=0)
        plt.margins(y=0)
    plt.plot(np.arange(nelem),stoch_profiles_avg,'#4169e1')
    plt.xticks([0,1440*31,1440*60,1440*91,1440*121,1440*152,1440*182,1440*213,1440*244,1440*274,1440*305,1440*335],["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dic"])
    #plt.savefig('profiles.eps', format='eps', dpi=1000)
    plt.show()

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
    df = df.sum(axis=1)
    profile = df.to_numpy()
    simulations.append(profile)
    
profile_average = profile_average(simulations)
Profile_cloud_plot(simulations,profile_average)


    

    

    

        