# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 15:28:07 2022

@author: pietro
"""

import pandas as pd
import ramp
import calendar
import time
import numpy as np
from pv import pvgis_hist
import json
import math
import matplotlib.pyplot as plt

"""
Charge profile
"""

# Get simulation data
occupancy = pd.read_pickle('./simulations/2f_occ.pkl')[0]
inputs = pd.read_pickle('./simulations/2f_inputs.pkl')[0]

# Various array sizes and timesteps used throughout the code
days = 365        
if calendar.isleap(inputs['year']):
    days = 366 
n1min = days*24*60
index1min  = pd.date_range(start='2015-01-01',end='2015-12-31 23:59:00',freq='T')
index10min = pd.date_range(start='2015-01-01',end='2015-12-31 23:50:00',freq='10T')

# Define required inputs
# TODO check this values
Pcharge = 3.7 #kW charging power 
margin = 0.5 # -
battery_eff = 0.9

# From RAMP - battery capacities
Battery_cap = {}
Battery_cap['small']  = 37 #kWh
Battery_cap['medium'] = 60 #kWh
Battery_cap['large']  = 100 #kWh

# From RAMP - Definition of battery limits to avoid degradation
SOC_max = 0.8 # Maximum SOC
SOC_min = 0.25 # Minimum SOC

# Running RAMP-mobility    
charge_home,charge_gen,SOC,MD = ramp.EVCharging(inputs, occupancy)

# Occupancy of main driver
# 1 at home (active or inactive) 0 not at home
occupancy_10min = occupancy[inputs['members'].index(MD)][:-1]
occupancy_10min = pd.Series(data=np.where(occupancy_10min<3,1,0),index=index10min)
occupancy_1min = occupancy_10min.reindex(index1min,method='pad').to_numpy()

# Selecting the non-empty SOC profile
for key, value in SOC.items():
    if value:
        SOC = value[0]
        if 'Large' in key:
            CP = Battery_cap['large']
        elif 'Medium' in key:
            CP = Battery_cap['medium']
        elif 'Small' in key:
            CP = Battery_cap['small']

# TODO why max SOC > 0.8           
SOC_max=max(np.max(SOC),SOC_max)
SOC_min=min(np.min(SOC),SOC_min)

print('Max SOC: {:.2f}'.format(np.max(SOC)))            
print('Min SOC: {:.2f}'.format(np.min(SOC)))

# Removing dummy days   
dummy = int((len(SOC)-days*24*60)/2)
SOC = SOC[dummy:-dummy] 

charge_home = charge_home.iloc[:,0].to_numpy()

"""
PV
"""

# PV and battery technology parameters
with open('inputs/pvbatt_param.json','r') as f:
    pvbatt_param = json.load(f)
config_pv = pvbatt_param['pv']
pvadim = pvgis_hist(config_pv) 
pv_15min = pvadim*10.
index1min  = pd.date_range(start='2015-01-01',end='2015-12-31 23:59:00',freq='T')
pv_1min = pv_15min.resample('T').pad().reindex(index1min,method='nearest')#.to_numpy() # kW

"""
At-home time windows
"""

# Find arrival and departure times of MD from home
# shift occupancy vector by one time step
occupancy_1min_s  = np.roll(occupancy_1min,1)

# locate all the points whit a start or a shutdown
arriving_times = (occupancy_1min>0) * (occupancy_1min_s==0)
leaving_times = (occupancy_1min_s>0) * (occupancy_1min==0)

# List the indexes of all start-ups and shutdowns
arrive = np.where(arriving_times)[0]
leave  = np.where(leaving_times)[0]

# Forcing arrays to have the same size
# Forcing first thing to be an arrival (at time 0 if already at home)
if len(arrive)>len(leave):
    leave = np.append(leave,n1min-1)
elif len(arrive)<len(leave):
    arrive = np.insert(arrive,0,0)
else:
    if leave[0]<arrive[0]:
        arrive = np.insert(arrive,0,0)
        leave = np.append(leave,n1min-1)
        
"""
Charging at-home time windows
"""

# Find starting and stopping to charge times
# Shift the app consumption vector by one time step:
charge_home_s  = np.roll(charge_home,1)

# locate all the points whit a start or a end
starting_times = (charge_home>0) * (charge_home_s==0)
stopping_times = (charge_home_s>0) * (charge_home==0)

# List the indexes of all start and end charging
starts = np.where(starting_times)[0]
ends   = np.where(stopping_times)[0]

"""
Consumptions when charging at home
"""

# # Faster, for some reason less precise and requires SOC
# consumptions = (SOC[ends]-SOC[starts-1])*CP #kWh
# # Minor fix TODO this should not be required, check what happens ()
# consumptions[consumptions<0] = 0

# slower, more precise and does not require SOC
consumptions = np.zeros(len(starts))
for i in range(len(starts)):
    consumptions[i] = np.sum(charge_home[starts[i]:ends[i]])/60

"""
Ramps
"""

chargelen = ends - starts
ramps = np.zeros(n1min) # kWh
for i in range(len(starts)):
    add = np.linspace(0,consumptions[i],num=chargelen[i]+1)
    ramps[starts[i]-1:ends[i]] += add

"""
At-home windows
"""   
 
idx_athomewindows = np.zeros(len(starts),dtype=int)
for i in range(len(starts)):
    idx = np.searchsorted(leave,[ends[i]-1],side='right')[0]
    idx_athomewindows[i] = idx

"""
Min LOC
"""    
LOC_min = ramps.copy()
for i in range(len(starts)):
    LOC_min[ends[i]:leave[idx_athomewindows[i]]] += ramps[ends[i]-1]


idx_s_e = 2
idx_a_l = np.searchsorted(leave,[ends[idx_s_e]-1],side='right')[0]

x = np.arange(arrive[idx_a_l]-1,leave[idx_a_l])
y1 = LOC_min[arrive[idx_a_l]-1:leave[idx_a_l]]

fig, ax1 = plt.subplots()
ax1.plot(x, y1)
  
"""
Define inputs for shifting function
"""

param = {}
param['BatteryCapacity'] = CP
param['MaxPower'] = np.max(charge_home)#Pcharge
param['BatteryEfficiency'] = battery_eff
param['InverterEfficiency'] = 1.
param['timestep'] = 1/60


"""
Max battery capacity based on consumptions
"""

LOC_max = np.zeros(len(consumptions))
oldidx = 0
count = 0
LOC_max_t = 0

for i in range(len(consumptions)):
    
    if idx_athomewindows[i] == oldidx:
        LOC_max_t += consumptions[i]
        count += 1
    else:
        LOC_max_t = consumptions[i]
        count = 1
        
    oldidx = idx_athomewindows[i]
    LOC_max[i+1-count:i+1] = LOC_max_t
        

def shift(pv,arrive,leave,starts,ends,LOC_min,param,idx_athomewindows,LOC_max,return_series=False):
    
    bat_size_p_adj = param['MaxPower']
    n_inv = param['InverterEfficiency']
    timestep = param['timestep']
    
    Nsteps = len(pv)
     
    pv2inv = np.zeros(Nsteps)
    inv2grid = np.zeros(Nsteps)
    inv2store = np.zeros(Nsteps)
    grid2store = np.zeros(Nsteps)
    LOC = np.zeros(Nsteps)
    
    testing = 0.
    testing2 = 0.
    
    idx_athomewindows,idxs = np.unique(idx_athomewindows,return_index=True)
    LOC_max = LOC_max[idxs]
    
    for i in range(len(idx_athomewindows)):
        
        LOC[arrive[idx_athomewindows[i]]-1] = 0
        
        for j in range(arrive[idx_athomewindows[i]],leave[idx_athomewindows[i]]):
                        
            pv2inv[j] = pv[j] # kW
            
            inv2store_t = min(pv2inv[j]*n_inv,bat_size_p_adj)            
            LOC_t = LOC[j-1] + inv2store_t*timestep
            
            if LOC_t < LOC_min[j]:
                
                inv2store[j]  = inv2store_t
                grid2store[j] = (LOC_min[j]-LOC_t)/timestep
                
                LOC[j] = LOC[j-1] + inv2store[j]*timestep + grid2store[j]*timestep
                if LOC[j]<LOC_min[j]:
                    print(LOC_min[j]-LOC[j])
            
            elif  LOC_min[j] <= LOC_t <= LOC_max[i]:
                
                inv2store[j]  = inv2store_t
                
                LOC[j] = LOC_t
                                
            elif LOC_t > LOC_max[i]:
                    
                inv2store[j] = (LOC_max[i]-LOC[j-1]) /timestep
                LOC[j] = LOC_max[i]

    
    inv2grid = pv2inv*n_inv - inv2store
    
    print(testing)
    print(testing2)
    
    out = {'pv2inv': pv2inv,
           'inv2grid': inv2grid,
           'inv2store': inv2store,
           'grid2store': grid2store,
           'LevelOfCharge': LOC
            }
    
    if return_series:
        out_pd = {}
        for k, v in out.items():  # Create dictionary of pandas series with same index as the input pv
            out_pd[k] = pd.Series(v, index=pv.index)
        out = out_pd
    return out

time1 = time.time()
out = shift(pv_1min,arrive,leave,starts,ends,LOC_min,param,idx_athomewindows,LOC_max,return_series=False)
time2 = time.time()
print('It required {:.2f} seconds to shift EV charging'.format(time2-time1))


# check: aa+bb=cc
# PASSED
aa = np.sum(out['inv2store'])/60  
bb = np.sum(out['grid2store'])/60
cc = np.sum(charge_home)/60

# check: cc = dd
# PASSED
dd = np.sum(charge_home*occupancy_1min)/60

# check: cc = ee
# PASSED
ee = np.sum(consumptions)



