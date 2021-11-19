#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import strobe
import ramp
import json
import os
import matplotlib.pyplot as plt
import time
import random
from strobe.Data.Households import households


def ProcebarExtractor(buildtype):
    
    """
    Given the building type, input data required by the 5R1C model 
    are obtained based on a simple elaboration of Procebar data.
    Thermal and geometric characteristics are randomly picked from 
    Procebar data according to Procebar's probability distribution
    of the given building type to have such characteristics
    
    input:
    buildtype   str defining building type (according to Procebar types('Freestanding','Semi-detached','Terraced','Apartment'))
    
    output:
    output      dict with params needed by the 5R1C model
    """

    
    # Opening building stock excel file
    # Selecting type of building wanted
    # Getting random (weighted) house thermal parameters
    # Getting corresponding reference geometry
                
    filename1 = r'inputs\Building Stock arboresence_SG_130118_EME.xls'
    sheets1 = ['Freestanding','Semi-detached','Terraced','Apartment']
    data1 = pd.read_excel (filename1,sheet_name=sheets1,header=0)
        
    df = data1[buildtype]
    df.columns = df.columns.str.rstrip()
    
    df["Occurence"].replace({np.nan: 0, -1: 0}, inplace=True)
    totprob = df["Occurence"].sum()
    df["Occurence"] = df["Occurence"]/totprob
    
    rndrow = df["Occurence"].sample(1,weights=df["Occurence"])
    rowind = rndrow.index.values[0]
    rowgeom = df.iloc[rowind]['Geometry reference']
    
    # Opening geometry excel file
    # Getting geometry parameters based on reference geometry just obtained
    
    filename2 = r'inputs\Arborescence_geometry_SG_130118.xls'
    sheets2 = [101,102,103,104,201,202,203,204,301,302,303,304,401,402,403,404]
    sheets2 = [str(i) for i in sheets2]
    data2 = pd.read_excel (filename2,sheet_name=sheets2)
    
    df2 = data2[str(rowgeom)]
    
    df3 = df2.iloc[0:7].iloc[:,0:2]
    df3 = df3.set_index(df3.iloc[:,0])
    df3 = df3.drop('General characteristics',axis=1)
    
    df4 = df2.iloc[9:16].iloc[:,0:3]
    df4.columns = df4.iloc[0]
    df4 = df4.drop(9)
    df4 = df4.reset_index(drop=True)
    df4 = df4.set_index(df4.iloc[:,0])
    df4 = df4.drop(np.nan,axis=1)
    
    df5 = df2.iloc[18:26].iloc[:,0:7]
    df5.columns = df5.iloc[0]
    df5 = df5.drop(18)
    df5 = df5.reset_index(drop=True)
    df5 = df5.set_index(df5.iloc[:,0])
    df5 = df5.drop(np.nan,axis=1)
    
    # Obtaining the parameters needed by the RC model
    # Single circuit
    
    heatedareas1 = ['Life','Night','Kitchen','Bathroom']
    heatedareas2 = ['Alife','Anight','Akitchen','Abathroom']
    
    Awindows  = df5[heatedareas1].loc['Awind'].sum() 
    Aglazed   = Awindows
    Awalls    = df5[heatedareas1].loc['Awall'].sum()
    Aroof     = df5[heatedareas1].loc['Aroof'].sum()
    Aopaque   = Awalls + Aroof
    Afloor    = df5[heatedareas1].loc['Afloor'].sum()
    Ainternal = df5[heatedareas1].loc['Aint'].sum()
    
    volume = df4['Volume [m3]'].loc[heatedareas2].sum()
    
    Atotal = Aglazed+Aopaque+Afloor+Ainternal
    
    Uwalls = df.iloc[rowind]['U_Wall']
    Uwindows = df.iloc[rowind]['U_Window']
    
    ACH_vent = 0.6 # Air changes per hour through ventilation [Air Changes Per Hour]
    ACH_infl = 0.6 # Air changes per hour through infiltration [Air Changes Per Hour]
    VentEff = 0. # The efficiency of the heat recovery system for ventilation. Set to 0 if there is no heat recovery []
    
    Ctot = df.iloc[rowind]['C_Roof'] + df.iloc[rowind]['C_Wall'] + \
            df.iloc[rowind]['C_Floor'] + df.iloc[rowind]['C_Window'] + \
            df.iloc[rowind]['C_Door']
    
    outputs = {
        'Aglazed': Awindows,
        'Aopaque': Aopaque,
        'Afloor': Afloor,
        'volume': volume,
        'Atotal': Atotal,
        'Uwalls': Uwalls,
        'Uwindows': Uwindows,
        'ACH_vent': ACH_vent,
        'ACH_infl': ACH_infl,
        'VentEff': VentEff,
        'Ctot': Ctot
        }
    
    return outputs


def HouseholdMembers(buildtype):
    
    """
    Given the building type, household members are obtained.
    The number of household per type of  building is defined according to Profils_Definition.xlsx
    Semi-detached houses not considered since not considered in Profils_Definition.xlsx
    Household members are randomly picked from StRoBe list of dwellings with 
    the specified number of inhabitants
    
    input:
    buildtype   str defining building type (according to Procebar types('Freestanding','Terraced','Apartment'))
    
    output:
    output      list of dwelling members

    """
    
    adults = ['FTE','PTE','Retired','Unemployed']

    nhouseholds = 0    

    if buildtype == 'Apartment':
        nhouseholds = 1
    elif buildtype == 'Terraced':
        nhouseholds = 2
    elif buildtype == 'Freestanding':
        nhouseholds = 4
  
    output = []
    
    # picking one random composition from strobe's list
    # and checking that there is at least one adult
    
    finished = False
    while not finished: 
        subset = {key: value for key, value in households.items() if np.size(value) == nhouseholds}
        output = random.choice(list(subset.values()))
        finished = not set(output).isdisjoint(adults)
    
    return output


def simulation(inputs):
    
    """
    Given the inputs dictionary runs the corresponding min by min
    annual simualtion, calling the functions nested in the strobe folder, i.e.:
        - StRoBe itself
        - House thermal model
        - Heat pump
        - Domestic hot water
    and the functions of the ramp folder, i.e.:
        - RAMP-mobility
        
    input:
    inputs  dict with all required inputs
    
    output:
    df      dataframe with electricity demand profiles of 'StaticLoad','TumbleDryer','DishWasher','WashingMachine','DomesticHotWater','HeatPumpPower','EVCharging'
    """
    
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
    df = pd.DataFrame(index=index,columns=['StaticLoad','TumbleDryer','DishWasher','WashingMachine','DomesticHotWater','HeatPumpPower','EVCharging'])
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
        plt.xlim([0,1440])
        plt.ylim(ymin=0)
        #plt.ylim(ymax=5000)
        plt.margins(x=0)
        plt.margins(y=0)
    plt.plot(np.arange(nelem),stoch_profiles_avg,'#4169e1')
    # plt.xticks([0,1440*31,1440*60,1440*91,1440*121,1440*152,1440*182,1440*213,1440*244,1440*274,1440*305,1440*335],["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dic"])
    #plt.savefig('profiles.eps', format='eps', dpi=1000)
    plt.show()


"""
Calling the previously defined functions to simulate nsimulations times 
the ncases defined by the json files in the inputs folder
"""

ncases = 12 # number of cases to be simulated
nsimulations = 10 # number of simulation per case


start_time = time.time()


for i in range(ncases):
        
    case = i+1+24
    
    # Reading fixed inputs from json file
    
    with open(r'inputs\case_'+str(case)+'.json') as f:
      inputs = json.load(f)
     
    simulations = []
    
    for i in range(nsimulations):
        
        sim = i+1
        message = '################### Running case {0}, simulation {1} ###################'.format(case,sim)
        print(message)

        # Adding to the inputs dictionary randomly generated inputs
        # that are changing at each simulation
        
        # People living in the dwelling
        # Taken from StRoBe list
        
        inputs['members'] = HouseholdMembers(inputs['dwelling_type'])
        
        # Thermal parameters of the dwelling
        # Taken from Procebar .xls files
        
        procebinp = ProcebarExtractor(inputs['dwelling_type'])
        inputs = {**inputs,**procebinp}
        
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
        
    profile_avg = profile_average(simulations)
    Profile_cloud_plot(simulations,profile_avg)

print("--- %s seconds ---" % (time.time() - start_time))




    

    

        