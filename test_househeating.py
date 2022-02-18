
"""Testing shift house thermal demand"""

import os
import strobe
import ramp
import json
import time
import random
import numpy as np
import pandas as pd
from preprocess import ProcebarExtractor,HouseholdMembers,yearlyprices
from strobe.RC_BuildingSimulator import Zone
from itertools import chain
from temp_functions import HPSizing




def ambientdata(datapath):
    temp = np.loadtxt(datapath + '/Climate/temperature.txt')
    # temp = np.append(temp,temp[-24*60:]) # add december 31 to end of year in case of leap year
    irr  = np.loadtxt(datapath + '/Climate/irradiance.txt')
    # irr  = np.append(irr,irr[-24*60:]) # add december 31 to end of year in case of leap year
    return temp,irr

def COP_Tamb(Temp):
    COP = 0.001*Temp**2 + 0.0471*Temp + 2.1259
    return COP

def househeating(inputs,QheatHP,Tset,Qintgains,Tamb,irr,nminutes,heatseas_st,heatseas_end):

    # Rough estimation of solar gains based on data from Crest
    # Could be improved
    
    typeofdwelling = inputs['HP']['dwelling_type'] 
    if typeofdwelling == 'Freestanding':
        A_s = 4.327106037
    elif typeofdwelling == 'Semi-detached':
        A_s = 4.862912117
    elif typeofdwelling == 'Terraced':
        A_s = 2.790283243
    elif typeofdwelling == 'Apartment':
        A_s = 1.5   
    Qsolgains = irr * A_s
        
    # Defining the house to be modelled with obtained HP size
    House = Zone(window_area=inputs['HP']['Aglazed'],
                walls_area=inputs['HP']['Aopaque'],
                floor_area=inputs['HP']['Afloor'],
                room_vol=inputs['HP']['volume'],
                total_internal_area=inputs['HP']['Atotal'],
                u_walls=inputs['HP']['Uwalls'],
                u_windows=inputs['HP']['Uwindows'],
                ach_vent=inputs['HP']['ACH_vent']/60,
                ach_infl=inputs['HP']['ACH_infl']/60,
                ventilation_efficiency=inputs['HP']['VentEff'],
                thermal_capacitance=inputs['HP']['Ctot'],
                t_set_heating=Tset[0], #inputs['HP']['Tthermostatsetpoint'],
                max_heating_power=QheatHP)
            
    Qheat = np.zeros(nminutes)
    Tinside = np.zeros(nminutes)

    d1 = 60*24*heatseas_end-1
    d2 = 60*24*heatseas_st-1
    concatenated = chain(range(1,d1), range(d2,nminutes))

    Tair = max(16.,Tamb[0])
    House.t_set_heating = Tset[0]    
    House.solve_energy(Qintgains[0], Qsolgains[0], Tamb[0], Tair)
    Qheat[0]   = House.heating_demand
    Tinside[0] = House.t_air    

    for i in concatenated:
        
        if i == d2:
            Tinside[i-1] = max(16.,Tamb[i-1])

        if Tset[i] != Tset[i-1]:
            House.t_set_heating = Tset[i]    
            
        House.solve_energy(Qintgains[i], Qsolgains[i], Tamb[i], Tinside[i-1])
        Qheat[i]   = House.heating_demand
        Tinside[i] = House.t_air
                       
    return Qheat, Tinside


file = r'./examples/example_res_occ.pkl'
occupancys = pd.read_pickle(file)
occ = np.zeros(len(occupancys[0][0]))
for i in range(len(occupancys[0])):
    occupancys[0][i] = [1 if a==1 else 0 for a in occupancys[0][i]]
    occ += occupancys[0][i] 
occ = [1 if a >=1 else 0 for a in occ]    
occ = occ[:-1].copy()
occupancy = np.zeros(nminutes)
for i in range(len(occ)):
    for j in range(10):
        occupancy[i*10+j] = occ[i]
occupancy[-1] = occupancy[-2]


datapath = r'./strobe/Data'
temp, irr = ambientdata(datapath)
temp = np.delete(temp,-1)
irr = np.delete(irr,-1)
Qintgains = np.zeros(len(temp))
nminutes = len(temp)
pv_long = np.zeros(nminutes)


pvpeak = 10. #kW
pvfile = r'./simulations/pv.pkl'
pvadim = pd.read_pickle(pvfile)
pv = pvadim * pvpeak # kW

for i in range(len(pv)):
    for j in range(15):
        pv_long[i*15+j]=pv[i]


Tset = [20. if a >0 else 15. for a in occupancy]
Tset = np.array(Tset)
 
fracmaxP=1.
QheatHP = HPSizing(inputs,fracmaxP)

heatseas_st = 244
heatseas_end = 151


time1 = time.time()


"""Reference thermal demand"""
# Qheat,Tin = househeating(inputs,QheatHP,Tset,Qintgains,temp,irr,nminutes,heatseas_st,heatseas_end)

""" PV shifting strategy """
# Tset[pv_long>0] = 25.
# Qheat_sh,Tin_sh = househeating(inputs,QheatHP,Tset,Qintgains,temp,irr,nminutes,heatseas_st,heatseas_end)

""" Tariff shifting strategy """
# Shifting-specific inputs
Tincrease = 3. # °C T increase wrt min T setpoint (heating off)
t_preheat = 3  # h max time allowed to consider pre-heating
t_preheat_min = t_preheat*60

offset = Tset.min()
Tset = Tset - offset

mask_z = Tset>0
idx_z = np.flatnonzero(mask_z)
idx_nz = np.flatnonzero(~mask_z)

idx_z = np.r_[idx_z,len(Tset)]

out = np.zeros(len(Tset), dtype=int)
idx = np.searchsorted(idx_z, idx_nz)
out[~mask_z] = idx_z[idx] - idx_nz

admhours = [1. if 0<a<t_preheat_min else 0. for a in out]
admhours = np.array(admhours)

# Resulting hours in which to increase setpoint
idx = np.where(admprices*admhours)

# Recalculating T setpoint array with increase
Tset += offset
Tset[idx] += Tincrease

Qheat_sh,Tin_sh = househeating(inputs,QheatHP,Tset,Qintgains,temp,irr,nminutes,heatseas_st,heatseas_end)

time2 = time.time()
print('It took {:.2f} seconds'.format(time2 - time1))








        