#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simulate demand scenarios at building level, for specific household."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import strobe
import json


# Loading inputs:
with open('inputs/loadshift_inputs.json') as f:
  inputs = json.load(f)

"""
Actual simulations
"""

# Strobe
result,textoutput = strobe.simulate_scenarios(1, inputs)

# Creating dataframe with the results (only for the first scenarioi)
n_scen = 0
n_steps = np.size(result['StaticLoad'][n_scen,:])
index = pd.date_range(start='2020-01-01 00:00', periods=n_steps, freq='1min')
df = pd.DataFrame(index=index,columns=['StaticLoad','TumbleDryer','DishWasher','WashingMachine','ElectricalBoiler','HeatPumpPower','EVCharging'])

for key in df.columns:
    if key in result:
        df[key] = result[key][n_scen,:]
    else:
        df[key] = 0

# Plotting

rng = pd.date_range(start='2020-01-02',end='2020-01-09',freq='min')
ax = df.loc[rng].plot.area(lw=0)
ax.set(ylabel = "Power [W]")
plt.legend(loc='upper left')

ax = df.loc['2020-01-06'].plot.area(lw=0)
ax.set(ylabel = "Power [W]")
plt.legend(loc='upper left')

        