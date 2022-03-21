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
import matplotlib.pyplot as plt

"""
Inputs
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

"""
RAMP-mobility
"""
# Running RAMP-mobility    
out = ramp.EVCharging(inputs, occupancy)

MD  = out['main_driver']
charge_home = out['charge_profile_home']
charge_home = charge_home.iloc[:,0].to_numpy()

"""
Occupancy of the main driver
"""

# Occupancy of main driver
# 1 at home (active or inactive) 0 not at home
occupancy_10min = occupancy[inputs['members'].index(MD)][:-1]
occupancy_10min = pd.Series(data=np.where(occupancy_10min<3,1,0),index=index10min)
occupancy_1min = occupancy_10min.reindex(index1min,method='pad').to_numpy()

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
pv_1min = pv_15min.resample('T').pad().reindex(index1min,method='nearest').to_numpy() # kW

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
Finding in which at-home time windows each charging window is
"""   
 
idx_athomewindows = np.zeros(len(starts),dtype=int)
for i in range(len(starts)):
    idx = np.searchsorted(leave,[ends[i]-1],side='right')[0]
    idx_athomewindows[i] = idx

"""
Minimum Level Of Charge
"""    
LOC_min = ramps.copy()
for i in range(len(starts)):
    LOC_min[ends[i]:leave[idx_athomewindows[i]]] += ramps[ends[i]-1]

# idx_s_e = 2
# idx_a_l = np.searchsorted(leave,[ends[idx_s_e]-1],side='right')[0]
# x = np.arange(arrive[idx_a_l]-1,leave[idx_a_l])
# y1 = LOC_min[arrive[idx_a_l]-1:leave[idx_a_l]]
# fig, ax1 = plt.subplots()
# ax1.plot(x, y1)

"""
Maximum Level Of Charge
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
        
"""
Define inputs for shifting function
"""

param = {}
param['MaxPower'] = np.max(charge_home) # Pcharge
param['InverterEfficiency'] = 1.
param['timestep'] = 1/60



def EVshift_PV(pv,arrive,leave,starts,ends,idx_athomewindows,LOC_min,LOC_max,param,return_series=False):
    
    """
    Function to shift at-home charging based on PV production
    It requires start and end indexes of at-home time windows and charging events and
    to which at-home time window each charging event belongs. 
    For each at home time window LOC_min is defined as the charge obtained from reference at-home charging events
    and LOC_max as the total consumption of charging events in that at-home time window.
    
    Parameters:
        pv (numpy array): vector Nsteps long with residual PV production, kW DC
        arrive (numpy array): vector of indexes, start at-home time windows  
        leave  (numpy array): vector of indexes, end   at-home time windows
        starts (numpy array): vector of indexes, start charging at-home time windows
        ends   (numpy array): vector of indexes, end   charging at-home time windows
        idx_athomewindows (numpy array): vector with which at-home window corresponds to each charging window
        LOC_min (numpy array): vector Nsteps long with min LOC, kWh
        LOC_max (numpy array): vector long as the number of at-home time windows with max LOC, kWh
        param (dict): dictionary with charge power [kW], inverter efficiency [-] and timestep [h]
        return_series (bool): if True then the return will be a dictionary of series. 
                              Otherwise it will be a dictionary of ndarrays.
                              It is reccommended to return ndarrays if speed is an issue (e.g. for batch runs).
                              Default is False.

    Returns:
        out (dict): dict with numpy arrays or pandas series with energy fluxes and LOC 
    """
    
    bat_size_p_adj = param['MaxPower']
    n_inv = param['InverterEfficiency']
    timestep = param['timestep']
    
    Nsteps = len(pv)
     
    pv2inv = np.zeros(Nsteps)
    inv2grid = np.zeros(Nsteps)
    inv2store = np.zeros(Nsteps)
    grid2store = np.zeros(Nsteps)
    LOC = np.zeros(Nsteps)
    
    # Not going twice through the same at-home time window    
    idx_athomewindows,idxs = np.unique(idx_athomewindows,return_index=True)
    LOC_max = LOC_max[idxs]
    
    for i in range(len(idx_athomewindows)): # iter over at-home time windows
        
        LOC[arrive[idx_athomewindows[i]]-1] = 0
        
        for j in range(arrive[idx_athomewindows[i]],leave[idx_athomewindows[i]]): # iter inside at-home time windows
                        
            pv2inv[j] = pv[j] # kW
            
            inv2store_t = min(pv2inv[j]*n_inv,bat_size_p_adj) # kW          
            LOC_t = LOC[j-1] + inv2store_t*timestep # kWh
            
            if LOC_t < LOC_min[j]:
                
                inv2store[j]  = inv2store_t # kW
                grid2store[j] = min(bat_size_p_adj-inv2store[j],(LOC_min[j]-LOC_t)/timestep) # kW
                                
                LOC[j] = LOC[j-1] + inv2store[j]*timestep + grid2store[j]*timestep # kWh
            
            elif  LOC_min[j] <= LOC_t <= LOC_max[i]:
                
                inv2store[j]  = inv2store_t # kW
                
                LOC[j] = LOC_t # kWh
                                
            elif LOC_t > LOC_max[i]:
                    
                inv2store[j] = (LOC_max[i]-LOC[j-1]) /timestep # kW
                
                LOC[j] = LOC_max[i] # kWh
   
    inv2grid = pv2inv*n_inv - inv2store # kW
        
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
out = EVshift_PV(pv_1min,arrive,leave,starts,ends,idx_athomewindows,LOC_min,LOC_max,param,return_series=False)
time2 = time.time()
print('It required {:.2f} seconds to shift EV charging'.format(time2-time1))


# check: aa+bb=cc
# PASSED
aa = np.sum(out['inv2store'])/60  
bb = np.sum(out['grid2store'])/60
cc = np.sum(charge_home)/60


