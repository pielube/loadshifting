# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:52:37 2022

@author: Ilian
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta 
from simulation import *
from demands import *
from plots import make_demand_plot

def realcases() :
    
    
    '''

    Returns
    -------
    cases_real : dict
        contains caracteristic of each consumer profile

    '''
    
    
    questionnaires = ['inputs\Data_consumer_0{}.xlsx'.format(i) for i in range(1,7)]
    
    case=    {
    	       "house": "f",
               "sheet": "F",
    		   "row": 0,
               "columns": [],
    		   "TechsShift": [],
    		   "WetAppBool": False, 
               "WetAppManBool": False,
               "WetAppAutoBool": False,
               "PVBool": False,
               "BattBool": False,
               "DHWBool": False,
               "HeatingBool": False,
               "EVBool": False
               }

    housetype = {
    	
    "name": "Apartment",
    "year": 2015,
    "ndays": 365,
    
    "members": 0,
    
    "thermal_parameters": None,
    
    "appliances": 
        {
        "loadshift": True,
        "apps": ["WashingMachine","TumbleDryer","DishWasher"],
        "prob_own":
            {
            "Hob": 0,
            "Microwave": 0,
            "PC": 0,
            "TV1": 0,
            "TumbleDryer": 0,
            "DishWasher": 0,
            "WashingMachine": 0,
            "WasherDryer": 0,
            "FridgeFreezer": 1,
            "Refrigerator": 1,
            "ChestFreezer": 0,
            "UprightFreezer": 0
            } 
        },
    
    "DHW": 
        {
        "loadshift": False,
        "type": "ElectricBoiler",
        "PowerElMax": 2000.0,
        "Ttarget": 60.0, 
        "Tcw": 10.0,     
        "Vcyl": 40.0,    
        "Hloss": 0.5
        },
    
    "HP": 
        {
        "loadshift": False,
        "model": "5R1C",
        "dwelling_type": "Apartment",
        "Tthermostatsetpoint": 20.0, 
        "ThermostatDeadband": 2.0,   
        "Temittersetpoint": 40.0,    
        "EmitterDeadband": 5.0,      
        "HeatPumpThermalPower": {}
        }, 
    
    "EV": 
        {
        "loadshift": True
        }
    
    }
    cases_real ={}
    housetypes_real ={}
    i=0
    for consumer in questionnaires :
        newcase = case.copy()
        newhousetype = housetype.copy()
        sheet_name = "Questionnaire"
        df = pd.read_excel(consumer, sheet_name)
        
        #Home feature
        newhousetype["name"] = str(df['Unnamed: 2'][45])
        newhousetype["members"] = None #df['Unnamed: 2'][110]
        newhousetype["HP"]["dwelling_type"] = str(df['Unnamed: 2'][45])
        
        #Type of dwelling
        if str(df['Unnamed: 2'][45]) == 'Appartement' :
            newcase['house'] = '1f'
            newcase ['sheet'] = '1F'
            newhousetype["HP"]["dwelling_type"] = 'Apartment'
        elif str(df['Unnamed: 2'][45]) == '2 façades' :
            newcase['house'] = '2f'
            newcase ['sheet'] = '2F'
            newhousetype["HP"]["dwelling_type"] = 'Terraced'
        elif str(df['Unnamed: 2'][45]) == '4 façades' :
            newcase['house'] = '1f'
            newcase ['sheet'] = '1F'
            newhousetype["HP"]["dwelling_type"] = 'Freestanding'
        
        #Large appliance
        newcase['row'] = i
        columns = ["StaticLoad","DishWasher","WashingMachine"]
        if str(df['Unnamed: 2'][49]) == 'oui' :
            columns.append('HeatPumpPower')
            newhousetype["HP"]["loadshift"] = True
        if str(df['Unnamed: 2'][69]) == 'oui' or str(df['Unnamed: 2'][76]) == 'oui' or str(df['Unnamed: 2'][81]) == 'oui': 
            columns.append('DomesticHotWater')
            newhousetype["DHW"]["loadshift"] = True
            if str(df['Unnamed: 2'][69]) == 'oui' :
                newhousetype["DHW"]["type"] = "ElectricBoiler"
            if str(df['Unnamed: 2'][81]) == 'oui' :
                newhousetype["DHW"]["type"] = "HeatPump"
        
        
        #Home appliance
        Appliance_list = ["PC","TV1","Hob","Microwave","WashingMachine","TumbleDryer","DishWasher","Refrigerator","ChestFreezer"]
        k=23
        for appliance in Appliance_list :
            if str(df['Unnamed: 5'][k]) != 'nan' :
                newhousetype["appliances"]["prob_own"][appliance] = 1
            k=k+1   
                
        if str(df['Unnamed: 5'][31]) != 'nan' :
            columns.append('TumbleDryer')
        
        newcase['columns'] = columns
        
    
        cases_real['case'+str(i+1)] = newcase
        housetypes_real['case'+str(i+1)] = newhousetype
     
        i=i+1
    return  (cases_real,housetypes_real)


cases_real,housetypes_real = realcases()


with open ('inputs\cases.json') as test :
    cases = json.load(test)
    




#Creation of JSON type file with the dict
filename = 'inputs\cases_real.json'
with open(filename, 'w',encoding='utf-8') as f:
    json.dump(cases_real, f,ensure_ascii=False, indent=4)

#Creation of housestype for each consumer profile
filename = 'inputs\housetypes_real.json'
with open(filename, 'w',encoding='utf-8') as f:
    json.dump(housetypes_real, f,ensure_ascii=False, indent=4)



#Configuration of the simulation
# conf = load_config('case1',cf_cases='cases_real.json',cf_pvbatt = 'pvbatt_param.json',cf_econ='econ_param.json',cf_tariff = 'tariffs.json', cf_house='housetypes_real.json')
# config,pvbatt_param,econ_param,tariffs,housetype,N = conf['config'],conf['pvbatt_param'],conf['econ_param'],conf['tariffs'],conf['housetype'],conf['N']
N=10
out = compute_demand(housetypes_real['case1'],N)
results,occupancy,input_data = out['results'],out['occupancy'],out['input_data']


#Statement comsuption 
def real_consumption ():
    statement_consumption = {}
    questionnaires = ['inputs\Data_consumer_0{}.xlsx'.format(i+1) for i in range(6)]
    i=1
    for consumer in questionnaires :
        statement_consumption_f={}
        if i == 1 :
            sheet_name = "AOU19_JUI21"
            colonne1 = 'Volume (kWh)'
            colonne2 = 'FromDate (GMT+1)'
        else :
            sheet_name = "data"
            colonne1 ='Volume en kWh'
            colonne2 = 'Date & Heure'
        df = pd.read_excel(consumer, sheet_name, header=0)
        statement_consumption_f['Date'] = df[colonne2]
        statement_consumption_f['Consumption'] = df[colonne1]*[1000]
        statement_consumption['consumer{}'.format(i)]=statement_consumption_f
        i=i+1
    return (statement_consumption)



data = results[0]
date = data.index.tolist()
comsuption = {}

columns=data.columns.tolist()

i=1
figure = {}
statement_consumption = real_consumption ()
for data in results :
    
    comsuption['case{}'.format(i)]= []
    
    for k in range (len(date)) :
        sum_kWh = data.iloc[k:k+1,:].sum(axis=1)
        comsuption['case{}'.format(i)].append(sum_kWh)
        if round(((100/N)*(k/len(date)+(i-1))),5)%1==0 :
            print('Chargement à {} %'.format(round((100/N)*(k/len(date)+(i-1)))))
        
    plt.plot(date,comsuption['case{}'.format(i)], label = '{} simulation consumption for consumer 1'.format(i))
    i=i+1
    
i=1
for k in range(1) :        
    plt.plot(statement_consumption['consumer{}'.format(i)]['Date'],statement_consumption['consumer{}'.format(i)]['Consumption'],ls =':',label='Real consumption for consumer {}'.format(i))
    i=i+1
plt.legend()
plt.show()
    

    
    
    
    
    
    
    
    
    
    

    
    
    
    
    






















# cases_real_len = len(cases_real)
# cases_len = len(cases)
# total_dict_count=cases_real_len+cases_len
# shared_dict = {}

# for i in cases_real:
#     if (i in cases) and (cases_real[i] == cases[i]):
#         shared_dict[i] = cases_real[i]

# len_shared_dict=len(shared_dict)        

# print("The items common between the dictionaries are -",shared_dict)
# print("The number of items common between the dictionaries are -", len_shared_dict)

# if (len_shared_dict==total_dict_count/2):
#     print("The dictionaries are identical")
# else:
#     print("The dictionaries are non-identical")
# for case_test in cases :
#     for case_real in cases_real :
#         if case_test == case_real :
#             print ('{} est le même cas que le {}'.format(case_test,case_real))









# with open('tariffs.json') as tarif:
# 		tariffs = json.load(tarif)

# with open('econ_param.json') as param :
#         econ_param = json.load(param)

# excel_cas = "casesmatrix.xlsx"       
# case_matrix = pd.read_excel(excel_cas, "Matrix")


# prices_scen = tariffs['prices']
# times_lots = tariffs['timeslots']

# for t in range (times_lots['HSstart'],times_lots['CSstart'],timedelta(hours=1)) :
#     for case in econ_param  :
#         scen = case['scenario']
#         price = prices_scen [scen]
        