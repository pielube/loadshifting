
"""Simulate demand scenarios at building level, for specific household."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import strobe
import ramp
import json
import time
import random
from strobe.Data.Households import households


start_time = time.time()

"""
Functions used by the launcher
"""

def ProcebarExtractor(buildtype,wellinsulated):
    
    """
    Given the building type, input data required by the 5R1C model 
    are obtained based on a simple elaboration of Procebar data.
    Thermal and geometric characteristics are randomly picked from 
    Procebar data according to Procebar's probability distribution
    of the given building type to have such characteristics
    
    input:
    buildtype   str defining building type (according to Procebar types('Freestanding','Semi-detached','Terraced','Apartment'))
    wellinsulated   bool if true only well insulated houses considered (according to column fitforHP in the excel file)
    
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
    df['fitforHP'].replace({np.nan: 0, -1: 0}, inplace=True)

    if wellinsulated:
        df["Occurence"]=df["Occurence"]*df['fitforHP']
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
                + for 'Semi-detached' 3 household members considered
    
    output:
    output      list of dwelling members

    """
    
    adults = ['FTE','PTE','Retired','Unemployed']

    nhouseholds = 0    

    if buildtype == 'Apartment':
        nhouseholds = 1
    elif buildtype == 'Terraced':
        nhouseholds = 2
    elif buildtype == 'Semi-detached':
        nhouseholds = 3
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
  

"""
Loading inputs
"""

with open('inputs/example.json') as f:
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
df = pd.DataFrame(index=index,columns=['StaticLoad','TumbleDryer','DishWasher','WashingMachine','DomesticHotWater','HeatPumpPower','EVCharging'])

result_ramp.loc[df.index[-1],'EVCharging']=0
#df.index.union(result_ramp.index)        # too slow

for key in df.columns:
    if key in result:
        df[key] = result[key][n_scen,:]
    elif key in result_ramp:
        df[key] = result_ramp[key]* 1000
    else:
        df[key] = 0

"""
Plotting
"""

rng = pd.date_range(start='2015-08-09',end='2015-08-16',freq='min')
ax = df.loc[rng].plot.area(lw=0)
ax.set(ylabel = "Power [W]")
plt.legend(loc='upper left')

ax = df.loc['2015-08-06'].plot.area(figsize=(8,4),lw=0)
ax.set(xlabel = 'Time [min]')
ax.set(ylabel = "Power [W]")
plt.legend(loc='upper left')


exectime = (time.time() - start_time)/60.
print("{:.1f} minutes".format(exectime))



        