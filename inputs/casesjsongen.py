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
           "WetAppManualShifting": False,
           "PresenceOfPV": False,
           "PresenceOfBattery": False,
           }

case_default = {
        "house": "4f",
        "sheet": "4F",
        "row": 0,
        "columns": [
            "StaticLoad",
            "TumbleDryer",
            "DishWasher",
            "WashingMachine",
            "DomesticHotWater",
            "HeatPumpPower",
            "EVCharging"
        ],
        "TechsShift": [
            "TumbleDryer",
            "DishWasher",
            "WashingMachine",
            "DomesticHotWater",
            "HeatPumpPower",
            "EVCharging"
        ],
        "WetAppManualShifting": True,
        "PresenceOfPV": True,
        "PresenceOfBattery": True
    }

cases ={}
cases['default'] = case_default

econparam = {
		   "WACC": 0.04,
           "elpriceincr": 0.02,
           "net_metering": False,
           "start_year": 2022,
           "time_horizon": 30,
           "C_grid_fixed": 100.0,
           "C_grid_kW": 0.0,
           "C_pros_tax": 88.81,
           "P_FtG": 40.0,
		   "FixedPVCost": 0.0,
           "PVCost_kW": 1300.0,
           "FixedBatteryCost":0.0,
           "BatteryCost_kWh":600.0,
           "FixedInverterCost": 0.0,
           "InverterCost_kW": 100.0,
           "PVLifetime": 30,
           "BatteryLifetime": 10,
           "InverterLifetime": 15,
           "OM": 0.0,
           "FixedControlCost": 0.0,
           "AnnualControlCost": 0.0,
		   "scenario": "test",
           "thresholdprice": "hollow"
           }


inp = {
       'start_year': 2023,
       'time_horizon': 30,
       'WACC': 0.04,
       'elpriceincrease': 0.02,
       'tariff': 'multi_price',
       'meter': 'smart_meter',
       't_PV': 30,
       't_battery': 10,
       't_inverter': 15,
       'C_PV_fix': 0.0,
       'C_PV_kW': 1300.0,
       'C_batt_fix': 0.0,
       'C_batt_kWh': 600.0,
       'C_invert_fix': 0.0,
       'C_invert_kW': 100.0,
       'C_control_fix': 500.0,
       'C_OM_annual': 0.0,
       'C_grid_fix_annual': 0.0,
       'C_grid_kW_annual': 0.0,
       'C_control_fix_annual': 0.0,
       'C_prosumertax': 88.81,
       "thresholdprice": 0.3,
       'tariff_ref': 'multi_price',
       'meter_ref': 'smart_meter'
       }

inps = {}
inps['default'] = inp

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
    if data['wetapp_shift_manual'][i] == 1:
        newcase['WetAppManualShifting'] = True
    if data['dhw_shift'][i] == 1:
        columns_shift.append('DomesticHotWater')
    if data['househeat_shift'][i] == 1:
        columns_shift.append('HeatPumpPower')
    if data['ev_shift'][i] == 1:
        columns_shift.append('EVCharging')
    newcase['TechsShift'] = columns_shift

    if data['pv'][i] == 1:
        newcase['PresenceOfPV'] =True
    if data['battery'][i] ==1:
        newcase['PresenceOfBattery'] = True

    cases['case'+str(i+1)] = newcase
    
    newinp = inp.copy()
    
    fixed = 0
    annual = 0
    
    if data['wetapp_shift_auto'][i] == 1:
        fixed += 50.
        newinp['thresholdprice'] = 0.2

    aa = data['dhw_shift'][i] == 1
    bb = data['househeat_shift'][i] == 1
    cc = data['ev_shift'][i] == 1
    dd = data['battery'][i] == 1
    
    if aa or bb or cc or dd:
        fixed += 500.
        annual += 30.
    
    newinp['C_control_fix'] = fixed
    newinp['C_control_fix_annual'] = annual

    inps['case'+str(i+1)] = newinp


filename = 'cases.json'
with open(filename, 'w',encoding='utf-8') as f:
    json.dump(cases, f,ensure_ascii=False, indent=4)
    

filename = 'econ_param.json'
with open(filename, 'w',encoding='utf-8') as f:
    json.dump(inps, f,ensure_ascii=False, indent=4) 



















