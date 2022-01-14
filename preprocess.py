
"""Functions to read input data"""

import numpy as np
import pandas as pd
import random
from strobe.Data.Households import households
import datetime




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
                
    filename1 = r'inputs/Building Stock arboresence_SG_130118_EME.xls'
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
    
    filename2 = r'inputs/Arborescence_geometry_SG_130118.xls'
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
  

def yearlyprices(scenario,timeslots,prices,stepperhour):
    
    stepperhour = int(stepperhour)

    endday = datetime.datetime.strptime('1900-01-02 00:00:00',"%Y-%m-%d %H:%M:%S")

    HSdayhours = []
    HSdaytariffs = []
    CSdayhours = []
    CSdaytariffs = []
    
    for i in timeslots['HSday']:
        starthour = datetime.datetime.strptime(i[0],'%H:%M:%S')
        HSdayhours.append(starthour)
    
    for i in range(len(HSdayhours)):
        start = HSdayhours[i]
        end = HSdayhours[i+1] if i < len(HSdayhours)-1 else endday
        delta = end - start
        for j in range(stepperhour*int(delta.seconds/3600)):
            price = prices[scenario][timeslots['HSday'][i][1]]/1000.
            HSdaytariffs.append(price)
    
    for i in timeslots['CSday']:
        starthour = datetime.datetime.strptime(i[0],'%H:%M:%S')
        CSdayhours.append(starthour)
    
    for i in range(len(CSdayhours)):
        start = CSdayhours[i]
        end = CSdayhours[i+1] if i < len(CSdayhours)-1 else endday
        delta = end - start
        for j in range(stepperhour*int(delta.seconds/3600)):
            price = prices[scenario][timeslots['CSday'][i][1]]/1000.
            CSdaytariffs.append(price)
    
    startyear = datetime.datetime.strptime('2015-01-01 00:00:00',"%Y-%m-%d %H:%M:%S")
    HSstart = datetime.datetime.strptime(timeslots['HSstart'],"%Y-%m-%d %H:%M:%S")
    CSstart = datetime.datetime.strptime(timeslots['CSstart'],"%Y-%m-%d %H:%M:%S")
    endyear = datetime.datetime.strptime('2016-01-01 00:00:00',"%Y-%m-%d %H:%M:%S")
    
    ytariffs = []
    
    deltaCS1 = HSstart - startyear
    deltaHS  = CSstart - HSstart
    deltaCS2 = endyear - CSstart
    
    for i in range(deltaCS1.days):
        ytariffs.extend(CSdaytariffs)
    for i in range(deltaHS.days):
        ytariffs.extend(HSdaytariffs)
    for i in range(deltaCS2.days):
        ytariffs.extend(CSdaytariffs)
    
    out = np.asarray(ytariffs)
            
    return out