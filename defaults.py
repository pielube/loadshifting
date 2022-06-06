#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 09:38:12 2022

@author: sylvain
"""


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
t_preheat = 3  

# Heating season definition (in days)
heatseas_st = 244
heatseas_end = 151

# Heat pump sizing (TODO: define this)
fracmaxP = 0.8

#PV 
pv_power = 4000
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