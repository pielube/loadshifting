# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 13:38:50 2022

@author: pietro
"""

import os
import json
import pandas as pd
import numpy as np
from temp_functions import yearlyprices



def dispatch_tariffs(demand, prices, thresholdprice, param, return_series=False):
    """ Tariffs-based battery dispatch algorithm.
    Battery is charged when energy price is below the threshold limit and as long as it is not fully charged.
    It is discharged as soon as the energy price is over the threshold limit and as long as it is not fully discharged.

    Arguments:
        demand (pd.Series): Vector of household consumption, kW
        prices (pd.Series): Vector of energy prices, €/kW
        param (dict): Dictionary with the simulation parameters:
                timestep (float): Simulation time step (in hours)
                BatteryCapacity: Available battery capacity (i.e. only the the available DOD), kWh
                BatteryEfficiency: Battery round-trip efficiency, -
                InverterEfficiency: Inverter efficiency, -
                MaxPower: Maximum battery charging or discharging powers (assumed to be equal), kW
        return_series(bool): if True then the return will be a dictionary of series. Otherwise it will be a dictionary of ndarrays.
                        It is reccommended to return ndarrays if speed is an issue (e.g. for batch runs).
    Returns:
        dict: Dictionary of Time series

    """

    bat_size_e_adj = param['BatteryCapacity']
    bat_size_p_adj = param['MaxPower']
    timestep = param['timestep']
    
    # We work with np.ndarrays as they are much faster than pd.Series
    Nsteps = len(demand)
    LevelOfCharge = np.zeros(Nsteps)
    grid2store = np.zeros(Nsteps)
    grid2load  = np.zeros(Nsteps)
    store2load = np.zeros(Nsteps)

    admprices = np.where(prices <= thresholdprice,1,0)
      
    demand1 = demand.to_numpy()

    LevelOfCharge[0] = bat_size_e_adj / 2.
    
    for i in range(1,Nsteps):
        
        if admprices[i] == 1: # low prices
        
            grid2load[i] = demand[i]
            
            if LevelOfCharge[i-1] >= bat_size_e_adj:  # if battery is full
                grid2store[i] = 0
            else:
                grid2store[i] = min((bat_size_e_adj - LevelOfCharge[i-1]) / timestep, bat_size_p_adj-demand[i])
                
        else: # high prices
        
            store2load[i] = min((LevelOfCharge[i-1] / timestep),demand[i])
            grid2load[i] = demand[i] - store2load[i]
            
        LevelOfCharge[i] =  LevelOfCharge[i-1]+grid2store[i]*timestep-store2load[i]*timestep

    out = {'grid2store': grid2store,
           'grid2load': grid2load,
           'store2load': store2load,
           'LevelOfCharge': LevelOfCharge}
    
    if return_series:
        out_pd = {}
        for k, v in out.items():  # Create dictionary of pandas series with same index as the input pv
            out_pd[k] = pd.Series(v, index=pv.index)
        out = out_pd
        
    return out


def dispatch_tariffs2(demand, prices, thresholdprice, param, return_series=False):
    """ Tariffs-based battery dispatch algorithm.
    Battery is charged when energy price is below the threshold limit and as long as it is not fully charged.
    It is discharged as soon as the energy price is over the threshold limit and as long as it is not fully discharged.

    Arguments:
        demand (pd.Series): Vector of household consumption, kW
        prices (np.array): Vector of energy prices, €/kW
        param (dict): Dictionary with the simulation parameters:
                timestep (float): Simulation time step (in hours)
                BatteryCapacity: Available battery capacity (i.e. only the the available DOD), kWh
                BatteryEfficiency: Battery round-trip efficiency, -
                InverterEfficiency: Inverter efficiency, -
                MaxPower: Maximum battery charging or discharging powers (assumed to be equal), kW
        return_series(bool): if True then the return will be a dictionary of series. Otherwise it will be a dictionary of ndarrays.
                        It is reccommended to return ndarrays if speed is an issue (e.g. for batch runs).
    Returns:
        dict: Dictionary of Time series

    """

    bat_size_e_adj = param['BatteryCapacity']
    bat_size_p_adj = param['MaxPower']
    timestep = param['timestep']
    
    # We work with np.ndarrays as they are much faster than pd.Series
    Nsteps = len(demand)
    LevelOfCharge = np.zeros(Nsteps)
    grid2store = np.zeros(Nsteps)
    store2load = np.zeros(Nsteps)

    admprices = np.where(prices <= thresholdprice,1,0)   
    demand1 = demand.to_numpy()

    LevelOfCharge[0] = bat_size_e_adj / 2.
    
    for i in range(1,Nsteps):
        
        if admprices[i] == 1: # low prices
            if LevelOfCharge[i-1] < bat_size_e_adj:  # if battery is full
                grid2store[i] = min((bat_size_e_adj - LevelOfCharge[i-1]) / timestep, bat_size_p_adj-demand[i])
            LevelOfCharge[i] =  LevelOfCharge[i-1]+grid2store[i]*timestep
                
        else: # high prices
            store2load[i] = min((LevelOfCharge[i-1] / timestep),demand[i],bat_size_p_adj)
            LevelOfCharge[i] =  LevelOfCharge[i-1]-store2load[i]*timestep

    grid2load = demand - store2load

    out = {'grid2store': grid2store,
           'grid2load': grid2load,
           'store2load': store2load,
           'LevelOfCharge': LevelOfCharge}
    
    if return_series:
        out_pd = {}
        for k, v in out.items():  # Create dictionary of pandas series with same index as the input demand
            out_pd[k] = pd.Series(v, index=demand.index)
        out = out_pd
        
    return out


# Demand
house = '4f'
name = house+'.pkl'
path = r'./simulations'
file = os.path.join(path,name)
demands = pd.read_pickle(file) # W
index = 0
columns = ["StaticLoad","TumbleDryer","DishWasher","WashingMachine","DomesticHotWater","HeatPumpPower"]
demand_pspy = demands[index][columns]/1000. # kW
demand_pspy = demand_pspy.resample('15Min').mean()[:-1] # kW

# Energy prices
with open(r'./inputs/tariffs.json') as f:
  econ = json.load(f)
scenario = 'test'
prices = econ['prices']
timeslots = econ['timeslots']
stepperh_15min = 4
yprices_15min = yearlyprices(scenario,timeslots,prices,stepperh_15min) # €/kWh
thresholdprice = 'hollow'
thprice = prices[scenario][thresholdprice]/1000. # €/kWh


# Demands
with open('inputs/' + house+'.json') as f:
  inputs = json.load(f)
  
Vcyl = inputs['DHW']['Vcyl'] # litres
Ttarget = inputs['DHW']['Ttarget'] # °C
PowerDHWMax = inputs['DHW']['PowerElMax']/1000. # kW

Tmin = 45. # °C
Ccyl = Vcyl * 1000. /1000. * 4200. # J/K
capacity = Ccyl*(Ttarget-Tmin)/3600./1000. # kWh
  
param = {'BatteryCapacity': capacity,
          'MaxPower': PowerDHWMax,
          'timestep': 0.25}


prices = yearlyprices(scenario,timeslots,prices,stepperh_15min) # €/kWh

time1 = time.time()

res1 = dispatch_tariffs(demand_pspy['DomesticHotWater'], yprices_15min, thprice, param, return_series=False)

time2 = time.time()

res2 = dispatch_tariffs2(demand_pspy['DomesticHotWater'], yprices_15min, thprice, param, return_series=False)

time3 = time.time()

print('It took {:.2f} seconds'.format(time2 - time1))
print('It took {:.2f} seconds'.format(time3 - time2))




# Graphs

df = demand_pspy['DomesticHotWater'].copy()
df = df.to_frame()
df['grid2store'] = res2['grid2store']
df['grid2load'] = res2['grid2load']
df['store2load'] = res2['store2load']

day = '2015-01-07'
rng = pd.date_range(start = day,end=day+' 23:45:00',freq='15T')


import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default='browser'


traces = []

marker = dict(color='goldenrod')
trace = go.Scatter(x=df.loc[rng].index,
                    y=df.loc[rng]['DomesticHotWater'],
                    name='DHW',
                    marker=marker,
                    fill='tonexty',
                    fillcolor='rgba(218, 165, 32, 0.15)',
                    yaxis='y2')
traces.append(trace)

for col in ['grid2store','grid2load','store2load']:
    
    trace = go.Bar(x=df.loc[rng].index,
                    y=df.loc[rng][col],
                    name=col)
    traces.append(trace)
    

      
layout = go.Layout(yaxis2=dict(overlaying='y'),
                    barmode='stack')

fig = go.Figure(data=traces,
                layout=layout)

fig.update_yaxes(range = [0,3])

fig.add_vrect(x0=day, x1=day+' 10:00:00', 
              fillcolor="green", 
              opacity=0.15, 
              line_width=0)
fig.add_vrect(x0=day+' 23:00:00', x1=day+' 23:59:00', 
              fillcolor="green", 
              opacity=0.15, 
              line_width=0)

fig.show()

