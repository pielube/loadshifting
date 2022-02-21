# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 18:52:07 2022

@author: pietro
"""

import json
import pandas as pd

filename = r'casesmatrix.xlsx'
sheet = ['Matrix']
data = pd.read_excel (filename,sheet_name=sheet,header=0)
data = data['Matrix']

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

cases ={}

econparam = {
		   "WACC": 0.05,
           "net_metering": False,
           "time_horizon": 20,
           "C_grid_fixed": 0.0,
           "C_grid_kW": 0.0,
           "P_FtG": 40.0,
		   "FixedPVCost": 0.0,
           "PVCost_kW": 1500.0,
           "FixedBatteryCost":0.0,
           "BatteryCost_kWh":600,
           "PVLifetime": 20,
           "BatteryLifetime": 10,
           "OM": 0.015,
           "FixedControlCost": 0.0,
           "AnnualControlCost": 0.0,
		   "scenario": "test",
           "thresholdprice": "hollow"
           }

econparams = {}

for i in range(83):
    
    newcase = case.copy()
    
    newcase['house'] = str(data['facades'][i])+'f'
    newcase['sheet'] = str(data['facades'][i])+'F'
    newcase['row'] = i
    
    columns = []
    if data['static'][i] == 1:
        columns.append('StaticLoad')
    if data['wetapp'][i] == 1:
        columns.append('TumbleDryer')
        columns.append('DishWasher')
        columns.append('WashingMachine')
    if data['dhw'][i] == 1:
        columns.append('DomesticHotWater')
    if data['househeat'][i] == 1:
        columns.append('HeatPumpPower')
    if data['ev'][i] == 1:
        columns.append('EVCharging')
    newcase['columns'] = columns
    
    columns_shift = []
    if data['wetapp_shift'][i] == 1:
        columns_shift.append('TumbleDryer')
        columns_shift.append('DishWasher')
        columns_shift.append('WashingMachine')
        newcase['WetAppBool'] = True
    if data['wetapp_shift_manual'][i] == 1:
        newcase['WetAppManBool'] = True
    if data['wetapp_shift_auto'][i] == 1:
        newcase['WetAppAutoBool'] = True
    if data['dhw_shift'][i] == 1:
        columns_shift.append('DomesticHotWater')
        newcase['DHWBool'] = True
    if data['househeat_shift'][i] == 1:
        columns_shift.append('HeatPumpPower')
        newcase['HeatingBool'] = True
    if data['ev_shift'][i] == 1:
        columns_shift.append('EVCharging')
        newcase['EVBool'] = True
    newcase['TechsShift'] = columns_shift

    if data['pv'][i] == 1:
        newcase['PVBool'] =True
    if data['battery'][i] ==1:
        newcase['BattBool'] = True

    cases['case'+str(i+1)] = newcase
    
    neweconparam = econparam.copy()
    
    fixed = 0
    annual = 0
    if data['wetapp_shift_auto'][i] == 1:
        fixed += 50.
        neweconparam['thresholdprice'] = 'heel'

    aa = data['dhw_shift'][i] == 1
    bb = data['househeat_shift'][i] == 1
    cc = data['ev_shift'][i] == 1
    dd = data['battery'][i] == 1
    if aa or bb or cc or dd:
        fixed += 500.
        annual += 30.
    
    neweconparam['FixedControlCost'] = fixed
    neweconparam['AnnualControlCost'] = annual

    econparams['case'+str(i+1)] = neweconparam


filename = 'cases.json'
with open(filename, 'w',encoding='utf-8') as f:
    json.dump(cases, f,ensure_ascii=False, indent=4)
    

filename = 'econ_param.json'
with open(filename, 'w',encoding='utf-8') as f:
    json.dump(econparams, f,ensure_ascii=False, indent=4) 



















