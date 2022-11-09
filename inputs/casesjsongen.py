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

# default, generated with: pp = pprint.PrettyPrinter(depth=4) ; pp.pprint(conf)
default = {'batt': {'capacity': 14,
          'efficiency': 0.9,
          'lifetime': 10,
          'pnom': 4,
          'yesno': True},
 'cont': {'strategy': 'self-consumption',
          'thresholdprice': 0.3,
          'wetapp': 'automated'},
 'dhw': {'hloss': 2,
         'loadshift': True,
         'pnom': 2000,
         'set_point': 60,
         'tcold': 10,
         'tfaucet': 40,
         'type': 'ElectricBoiler',
         'vol': 150,
         'yesno': True},
 'dwelling': {'dish_washer': True,
              'member1': None,
              'member2': None,
              'member3': None,
              'member4': None,
              'member5': None,
              'tumble_dryer': True,
              'type': '1f',
              'washing_machine': True},
 'econ': {'C_OM_annual': 0,
          'C_PV_fix': 10,
          'C_PV_kW': 10,
          'C_batt_fix': 10,
          'C_batt_kWh': 10,
          'C_control': 500,
          'C_control_annual': 0,
          'C_grid_fix_annual': 0,
          'C_grid_kW_annual': 0,
          'C_invert_share': 0.077,
          'C_prosumertax': 88.81,
          'C_capacitytariff': 0.4,
          'elpriceincrease': 0.02,
          'meter': 'smart_r3',
          'start_year': 2023,
          'tariff': 'multi',
          'time_horizon': 30,
          'wacc': 0.04},
 'ev': {'loadshift': True, 'yesno': True},
 'hp': {'automatic_sizing': True,
        'deadband': 2,
        'loadshift': True,
        'pnom': 5000,
        'set_point': 20,
        'yesno': True},
 'loc': {'altitude': 20,
         'latitude': 50.6,
         'longitude': 4.3,
         'name': 'Europe/Brussels',
         'timezone': 'Etc/GMT-2'},
 'ownership': {'ChestFreezer': 0.56,
               'DishWasher': 0.66,
               'FridgeFreezer': 0.64,
               'Hob': 0.751,
               'Microwave': 0.9,
               'PC': 0.8,
               'Refrigerator': 0.61,
               'TV1': 0.95,
               'TumbleDryer': 0.6,
               'UprightFreezer': 0.0,
               'WasherDryer': 0.0,
               'WashingMachine': 0.93},
 'prices': 'config_prices.csv',
 'pv': {'automatic_sizing': True,
        'azimut': 0,
        'inverter_automatic_sizing': True,
        'inverter_lifetime': 15,
        'inverter_pmax': 5,
        'lifetime': 30,
        'losses': 0.13,
        'plim_kva': 10,
        'powerfactor': 0.9,
        'ppeak': 5,
        'tilt': 35,
        'yesno': True},
 'sim': {'N': 10, 'ndays': 365, 'ts': 0.25, 'year': 2015}}

cases ={}
cases['default'] = default

for i in range(83):
    
    newcase = default.copy()
    
    newcase['dwelling']['type'] = str(data['facades'][i])+'f'
    newcase['row'] = i
    
    columns = []
    if data['static'][i] == 1:
        columns.append('StaticLoad')
    if data['wetapp'][i] == 1:
        newcase['dwelling']['washing_machine'] = True
        newcase['dwelling']['dish_washer'] = True
        newcase['dwelling']['tumble_dryer'] = True
    else:
        newcase['dwelling']['washing_machine'] = False
        newcase['dwelling']['dish_washer'] = False
        newcase['dwelling']['tumble_dryer'] = False        
    if data['dhw'][i] == 1:
        newcase['dhw']['yesno'] = True
    else:
        newcase['dhw']['yesno'] = False
    if data['househeat'][i] == 1:
        newcase['hp']['yesno'] = True
    else:
        newcase['hp']['yesno'] = False
    if data['ev'][i] == 1:
        newcase['ev']['yesno'] = True
    else:
        newcase['ev']['yesno'] = False
    
    if data['wetapp_shift'][i] == 1:
        newcase['cont']['wetapp'] = 'automated'
    else:
        newcase['cont']['wetapp'] = 'none'
    if data['wetapp_shift_manual'][i] == 1:
        newcase['cont']['wetapp'] = 'manual'
    
    if data['dhw_shift'][i] == 1:
        newcase['dhw']['loadshift'] = True
    else:
        newcase['dhw']['loadshift'] = False
    if data['househeat_shift'][i] == 1:
        newcase['hp']['loadshift'] = True
    else:
        newcase['hp']['loadshift'] = False
    if data['ev_shift'][i] == 1:
        newcase['ev']['loadshift'] = True
    else:
        newcase['ev']['loadshift'] = False

    if data['pv'][i] == 1:
        newcase['pv']['yesno'] =True
    else:
        newcase['pv']['yesno'] =False
    if data['battery'][i] ==1:
        newcase['batt']['yesno'] =True
    else:
        newcase['batt']['yesno'] =True

    # compute the cost of the control system:    
    fixed = 0
    annual = 0
    
    if newcase['cont']['wetapp']:
        fixed += 50.
        newcase['cont']['thresholdprice'] = 0.2

    aa = data['dhw_shift'][i] == 1
    bb = data['househeat_shift'][i] == 1
    cc = data['ev_shift'][i] == 1
    dd = data['battery'][i] == 1
    
    if newcase['dhw']['loadshift'] or newcase['hp']['loadshift'] or newcase['ev']['loadshift'] or newcase['batt']['yesno']:
        fixed += 500.
        annual += 30.
    
    newcase['econ']['C_control'] = fixed
    newcase['econ']['C_control_annual'] = annual

    cases['case'+str(i+1)] = newcase

filename = 'cases.json'
with open(filename, 'w',encoding='utf-8') as f:
    json.dump(cases, f,ensure_ascii=False, indent=4)
    




















