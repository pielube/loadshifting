# -*- coding: utf-8 -*-

"""
Code to generate .json input files for the cases defined in Profils_Definition.xslx
"""

import pandas as pd
import json
import os

import pathlib
cwd = os.getcwd()
loadshiftpath = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(loadshiftpath)
from strobe.Data.Households import households
os.chdir(cwd)


def CasesPerSheet(inputs,df,nsheet,households):  
    for index, row in df.iterrows():
        
        tempjson = inputs
                
        # Type of building
        if row['Façades'] == 1:
            inputs['HP']['dwelling_type'] = "Apartment"
        elif row['Façades'] == 2:
            inputs['HP']['dwelling_type'] = "Terraced"
        elif row['Façades'] == 3:
            inputs['HP']['dwelling_type'] = "Semi-detached"
        elif row['Façades'] == 4:
            inputs['HP']['dwelling_type'] = "Freestanding"
    
        # Appliances
        if row['Machines'] == 0:
            inputs['appliances']['apps'] = []
        elif row['Machines'] == 1:
            inputs['appliances']['apps'] = ["WashingMachine","TumbleDryer","DishWasher"]
            
        # Heat Pump
        if row['PAC'] == 0:
            inputs['HP']['loadshift'] = False
        elif row['PAC'] == 1:
            inputs['HP']['loadshift'] = True
            
        # Electric boiler
        inputs = DHWinputs(row['Ménage'],row['ECS'],inputs)
    
        # Electric vehicle
        if row['VE'] == 0:
            inputs['EV']['loadshift'] = False
        elif row['VE'] == 1:
            inputs['EV']['loadshift'] = True
 
        filenumber = nsheet*12+index+1    
        filename = 'case_'+str(filenumber)+'.json'
        with open(filename, 'w',encoding='utf-8') as f:
            json.dump(tempjson, f,ensure_ascii=False, indent=4)


def DHWinputs(npeople,ECS,inputs):
    
    if ECS == 0:
        inputs['DHW']['loadshift'] = False

    elif ECS == 1:
        inputs['DHW']['loadshift'] = True
        inputs['DHW']['type'] = 1
        
        if npeople == 1:
            inputs['DHW']["PowerElMax"]= 2000.0
            inputs['DHW']["Ttarget"]= 53.0 
            inputs['DHW']["Tcw"]= 10.0     
            inputs['DHW']["Vcyl"]= 20.0  
            inputs['DHW']["Hloss"]= 0.5
        elif npeople in {2,3}:
            inputs['DHW']["PowerElMax"]= 2000.0
            inputs['DHW']["Ttarget"]= 53.0 
            inputs['DHW']["Tcw"]= 10.0     
            inputs['DHW']["Vcyl"]= 54.0  
            inputs['DHW']["Hloss"]= 0.5
        elif npeople >= 4:
            inputs['DHW']["PowerElMax"]= 2000.0
            inputs['DHW']["Ttarget"]= 53.0 
            inputs['DHW']["Tcw"]= 10.0     
            inputs['DHW']["Vcyl"]= 87.0  
            inputs['DHW']["Hloss"]= 1.0
                  
    elif ECS == 2:
        inputs['DHW']['loadshift'] = True
        inputs['DHW']['type'] = 2
        
        if npeople == 1:
            inputs['DHW']["PowerElMax"]= 350.0
            inputs['DHW']["Ttarget"]= 53.0 
            inputs['DHW']["Tcw"]= 10.0     
            inputs['DHW']["Vcyl"]= 50.0  
            inputs['DHW']["Hloss"]= 0.5
        elif npeople in {2,3}:
            inputs['DHW']["PowerElMax"]= 700.0
            inputs['DHW']["Ttarget"]= 53.0 
            inputs['DHW']["Tcw"]= 10.0     
            inputs['DHW']["Vcyl"]= 85.0  
            inputs['DHW']["Hloss"]= 1.0
        elif npeople >= 4:
            inputs['DHW']["PowerElMax"]= 1000.0
            inputs['DHW']["Ttarget"]= 53.0 
            inputs['DHW']["Tcw"]= 10.0     
            inputs['DHW']["Vcyl"]= 184.0  
            inputs['DHW']["Hloss"]= 1.5  

    return(inputs)
            

data = pd.read_excel (r'.\Profils_Definition.xlsx',sheet_name=['4F','2F','1F'],header=3)

with open(r'.\case_ref.json') as f:
  inputs = json.load(f)
 
count = 0
for key,value in data.items():
    df = value.drop(['Paramètres\n/Caractéristiques','Pilotage','Investissement [€]','Elec [€]'], 1)
    nsheet = count
    CasesPerSheet(inputs,df,nsheet,households)
    count += 1
 

