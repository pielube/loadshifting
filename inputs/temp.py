# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 16:54:50 2021

@author: pietro
"""
import pandas as pd
import json
import os
import numpy as np
import random

import pathlib
cwd = os.getcwd()
loadshiftpath = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(loadshiftpath)
from strobe.Data.Households import households
os.chdir(cwd)

def CasesPerSheet(inputs,df,nsheet,households):  
    for index, row in df.iterrows():
        
        tempjson = inputs
    
        # Household members
        subset = {key: value for key, value in households.items() if np.size(value) == row['Ménage']}
        inputs['members'] = random.choice(list(subset.values()))
              
        # Type of building
        if row['Façades'] == 1:
            inputs['dwelling_type'] = "Terraced"
            print('WARNING: missing apartment data')
        elif row['Façades'] == 2:
            inputs['dwelling_type'] = "Terraced"
        elif row['Façades'] == 3:
            inputs['dwelling_type'] = "Semi-Detached"
        elif row['Façades'] == 4:
            inputs['dwelling_type'] = "Detached"
    
        # Appliances
        if row['Machines'] == 0:
            inputs['appliances'] = []
        elif row['Machines'] == 1:
            inputs['appliances'] = ["WashingMachine","TumbleDryer","DishWasher"]
            
        # Heat Pump
        if row['PAC'] == 0:
            inputs['HeatPump'] = False
        elif row['PAC'] == 1:
            inputs['HeatPump'] = True
            
        # Electric boiler
        if row['ECS'] == 0:
            inputs['ElectricBoiler'] = False
        elif row['ECS'] == 1:
            inputs['ElectricBoiler'] = True
        elif row['ECS'] == 2:
            inputs['ElectricBoiler'] = True
            print('WARNING: Electric boiler = 2')
    
        # Electric vehicle
        if row['VE'] == 0:
            inputs['EV'] = False
        elif row['VE'] == 1:
            inputs['EV'] = True
 
        filenumber = nsheet*12+index+1    
        filename = 'case_'+str(filenumber)+'.json'
        with open(filename, 'w',encoding='utf-8') as f:
            json.dump(tempjson, f,ensure_ascii=False, indent=4)


data = pd.read_excel (r'.\Profils_Definition.xlsx',sheet_name=['4F','2F','1F'],header=3)

with open(r'.\loadshift_inputs.json') as f:
  inputs = json.load(f)
 
count = 0
for key,value in data.items():
    prova = 'Paramètres\n/Caractéristiques'
    df = value.drop(['Pilotage','Investissement [€]','Elec [€]'], 1)
    nsheet = count
    CasesPerSheet(inputs,df,nsheet,households)
    count += 1
 

