#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 09:38:12 2022

@author: sylvain
"""

# Number of stochastic demand scenarios to run:
N = 1

# Heat pump
hp_thermal_power = 7000


# Sanitary hot water:
Vol_DHW = 200
T_sp_DHW = 60

# Set points without/with occupancy
T_sp_low = 15
T_sp_occ = 20

# Increase in the set point when shifting hp load:
Tincrease = 3

# maximum number of hours allowed to consider pre-heating
t_preheat = 1  

# Heating season definition (in days)
heatseas_st = 244
heatseas_end = 151

# Heat pump sizing (TODO: define this)
fracmaxP = 0.8

#PV 
pv_power = 4.5  #kWp
pv_lim = 12  # kWp max for home installations
inv_lim = 10 # kVA max for home installations
powerfactor = 0.9 # kW/kVA to tranform inverter limit to kW

#battery
bat_cap = 14
bat_power = 4

#DHW
T_min_dhw = 45

# Probability of load shifting for a particular appliance [0,1]:
probshift = 1
# Minimum size of the admitted time window relative to the average length of the appliance duty cycle
threshold_window = 0.5
# Maximum time over which the timing of an appliance cycle can be shifted (in hours)
max_shift = 24

# Verbosity:
verbose = 0

# reference year for the weather data 
year = 2015

# translation the input variables into standard english to be used in the library:
translate = {
    'dwelling_type': {'Appartement':'1f',
                      "2 façades":'2f',
                      '3 façades':'3f',
                      '4 façades':'4f'},
    'econ_tariff': {'Facturation nette':'net-metering',
                    'Double flux:':'bi-directional'},
    'cont_strategy': {'Aucune':'none',
                      'Auto-consommation':'self-consumption',
                      'Tarif variable':'time-of-use'},
    'cont_wetapp': {'Aucun':'none',
                    'Manuel':'manual',
                    'Automatisé':'automated'},
    'dwelling_member1':{'Aucun':None,
              'Aléatoire':'Random',
              'Travailleur à temps plein':'FTE',
              'Travailleur à temps partiel':'PTE',
              'Enfant':'U12',
              'Retraité':'Retired',
              'Sans emploi':'Unemployed',
              'Etudiant':'School'},
    'dwelling_member2':{'Aucun':None,
              'Aléatoire':'Random',
              'Travailleur à temps plein':'FTE',
              'Travailleur à temps partiel':'PTE',
              'Enfant':'U12',
              'Retraité':'Retired',
              'Sans emploi':'Unemployed',
              'Etudiant':'School'},
    'dwelling_member3':{'Aucun':None,
              'Aléatoire':'Random',
              'Travailleur à temps plein':'FTE',
              'Travailleur à temps partiel':'PTE',
              'Enfant':'U12',
              'Retraité':'Retired',
              'Sans emploi':'Unemployed',
              'Etudiant':'School'},
    'dwelling_member4':{'Aucun':None,
              'Aléatoire':'Random',
              'Travailleur à temps plein':'FTE',
              'Travailleur à temps partiel':'PTE',
              'Enfant':'U12',
              'Retraité':'Retired',
              'Sans emploi':'Unemployed',
              'Etudiant':'School'},
    'dwelling_member5':{'Aucun':None,
              'Aléatoire':'Random',
              'Travailleur à temps plein':'FTE',
              'Travailleur à temps partiel':'PTE',
              'Enfant':'U12',
              'Retraité':'Retired',
              'Sans emploi':'Unemployed',
              'Etudiant':'School'},
    'dhw_type': {'Résistif':'ElectricBoiler',
                 'Thermodynamique':'HeatPump'}
    }


#Translate the loadprogen building type into the procebar building types:
convert_building = {'1f':'Apartment','2f':'Terraced','3f':'Semi-detached','4f':'Freestanding'}

# colors for plotting:

defaultcolors1 = {'StaticLoad':'#636EFA', 'WashingMachine':'#00CC96', 'TumbleDryer':'#AB63FA',
                 'DishWasher':'#FFA15A','HeatPumpPower':'#19D3F3', 'DomesticHotWater':'#FF6692',
                 'EVCharging':'#B6E880','BatteryConsumption':'#FF97FF'}

defaultcolors = {'StaticLoad':'#7fe5ca', 
                 'WashingMachine':'#ffb2c8', 
                 'TumbleDryer':'#d5b0fc',
                 'DishWasher':'#ffd0ac',
                 'HeatPumpPower':'#daf3bf', 
                 'DomesticHotWater':'#8be9f9',
                 'EVCharging':'#ffcbff',
                 'BatteryConsumption':'#b0b6fc',
                 'PVGeneration': '#ffffbe'}