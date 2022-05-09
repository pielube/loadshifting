# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:52:37 2022

@author: Ilian
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta 
from simulation.py import load_config



def realcases() :
    questionnaires = ['inputs\Data_consumer_0{}.xlsx'.format(i) for i in range(1,6)]
    
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


    cases_real ={}
    i=0
    for consumer in questionnaires :
        newcase = case.copy()
        sheet_name = "Questionnaire"
        df = pd.read_excel(consumer, sheet_name)
        
        if str(df['Unnamed: 2'][45]) == 'Appartement' :
            newcase['house'] = '1f'
            newcase ['sheet'] = '1F'
        elif str(df['Unnamed: 2'][45]) == '2 façades' :
            newcase['house'] = '2f'
            newcase ['sheet'] = '2F'
        elif str(df['Unnamed: 2'][45]) == '4 façades' :
            newcase['house'] = '1f'
            newcase ['sheet'] = '1F'
        
        newcase['row'] = i
        columns = ["StaticLoad","DishWasher","WashingMachine"]
        if str(df['Unnamed: 2'][49]) == 'oui' :
            columns.append('HeatPumpPower')
        if str(df['Unnamed: 2'][69]) == 'oui' : 
            columns.append('DomesticHotWater')
        if str(df['Unnamed: 5'][31]) != 'nan' :
            columns.append('TumbleDryer')
        
        newcase['columns'] = columns
        
    
        cases_real['case'+str(i+1)] = newcase
        i=i+1
    return  (cases_real)


cases_real = realcases()

with open ('inputs\cases.json') as test :
    cases = json.load(test)
    

cases_real_len = len(cases_real)
cases_len = len(cases)
total_dict_count=cases_real_len+cases_len
shared_dict = {}

for i in cases_real:
    if (i in cases) and (cases_real[i] == cases[i]):
        shared_dict[i] = cases_real[i]

len_shared_dict=len(shared_dict)        

print("The items common between the dictionaries are -",shared_dict)
print("The number of items common between the dictionaries are -", len_shared_dict)

if (len_shared_dict==total_dict_count/2):
    print("The dictionaries are identical")
else:
    print("The dictionaries are non-identical")



filename = 'inputs\cases_real.json'
with open(filename, 'w',encoding='utf-8') as f:
    json.dump(cases_real, f,ensure_ascii=False, indent=4)






out = load_config(1,cf_cases='inputs\cases_real.json',cf_pvbatt = 'pvbatt_param.json',cf_econ='econ_param.json',cf_tariff = 'tariffs.json', cf_house='housetypes.json')
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
        