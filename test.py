#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simulate demand scenarios at building level, for specific household."""

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import strobe
import json


# Loading inputs:
with open('inputs/loadshift_inputs.json') as f:
  inputs = json.load(f)

"""
Actual simulations
"""

# Ambient data
temp, irr = strobe.ambientdata()

# Strobe
#res_el,res_DHW,Tbath,res_Qgains = simulate_scenarios(1, year, nday) # members=['FTE', 'PTE', 'School']
res_el,res_DHW,Tbath,res_Qgains,textoutput = strobe.simulate_scenarios(1, inputs)

# House heating model
ressize = np.size(res_el['pstatic'])
phi_c = res_Qgains['Qrad']+res_Qgains['Qcon']
timersetting = strobe.HeatingTimer(inputs)
phi_h_space,Tem_test = strobe.HouseThermalModel(inputs,ressize,temp,irr,phi_c,timersetting)
thermal_load = int(sum(phi_h_space)/1000./60.)
print(' - Thermal demand for space heating is ',thermal_load,' kWh')

# Heat pump electric load
phi_hp = strobe.ElLoadHP(temp,phi_h_space)

# Electric boiler and hot water tank
phi_a = strobe.HotWaterTankModel(inputs,res_DHW['mDHW'],Tbath)

# Creating dataframe with the results
df = pd.DataFrame(data=res_el)
df['elboiler'] = phi_a
df['heatpump'] = phi_hp
time = list(range(0,np.size(res_el['pstatic'])))
time = [datetime.datetime(2020,1,1) + datetime.timedelta(minutes=each) for each in time]
df.index = time


# Plotting

rng = pd.date_range(start='2020-01-02',end='2020-01-09',freq='min')
ax = df.loc[rng].plot.area(lw=0)
ax.set(ylabel = "Power [W]")
plt.legend(loc='upper left')

ax = df.loc['2020-01-06'].plot.area(lw=0)
ax.set(ylabel = "Power [W]")
plt.legend(loc='upper left')
