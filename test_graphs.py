
import os
import numpy as np
import pandas as pd
import json
import time
from temp_functions import yearlyprices,mostrapcurve, run,strategy1,writetoexcel
from prosumpy import dispatch_max_sc_withsd,print_analysis



    
"""
Choosing most representative curve according to house and technologies considered
"""
house = '4f'
# Demands
name = house+'.pkl'
path = r'./simulations'
file = os.path.join(path,name)
demands = pd.read_pickle(file)

# Occupancy
name = house+'_occ.pkl'
file = os.path.join(path,name)
occupancys = pd.read_pickle(file)

# Technology parameters required by prosumpy
param_tech_mrc = {'BatteryCapacity': 0.,
                  'BatteryEfficiency': 0.9,
                  'MaxPower': 0.,
                  'InverterEfficiency': 1.,
                  'timestep': 0.25}

# Technology parameters required by economic analysis (updated inside run function)
inputs_mrc = {'CapacityPV': 0.,
          'CapacityBattery': 0.}

# Technology costs required by prosumpy and economic analysis
Inv = {'FixedPVCost':0,
        'PVCost_kW':1500,
        'FixedBatteryCost':0,
        'BatteryCost_kWh':600,
        'PVLifetime':20,
        'BatteryLifetime':10,
        'OM':0.015, # eur/year/eur of capex (both for PV and battery)
        'FixedControlCost': 0,
        'AnnualControlCost': 0} 

# Economic parameteres required by yearly prices function and economic analysis
with open(r'./inputs/economics.json') as g:
  econ = json.load(g)

timeslots = econ['timeslots']
prices = econ['prices']
scenario = 'test'

EconomicVar = {'WACC': 0.05, # weighted average cost of capital
               'net_metering': False, # type of tarification scheme
               'time_horizon': 20,
               'C_grid_fixed':prices[scenario]['fixed'], # € annual fixed grid costs
               'C_grid_kW': prices[scenario]['capacity'], # €/kW annual grid cost per kW 
               'P_FtG': 40.}     # €/MWh electricity price to sell to the grid

# Timestep used by prosumpy and economic analysis
timestep = 0.25 # hrs
stepperhour = 4

columns = list(demands[0].columns)


# Choosing most rapresentative curve
ElPrices = yearlyprices(scenario,timeslots,prices,stepperhour)

index = mostrapcurve(demands,param_tech_mrc,inputs_mrc,EconomicVar,Inv,ElPrices,timestep,columns)




import pandas as pd
import os
import matplotlib.pyplot as plt
import calendar
  
# Plotting one day of the year

rng = pd.date_range(start='2015-08-06',end='2015-08-07',freq='min')
dem1 = demands[0].sum(axis=1)
ax = dem1.loc[rng].plot(figsize=(8,4),color='#b0c4de',legend=False)

for dem in demands:
    dem = dem.sum(axis=1)
    dem.loc[rng].plot(ax=ax,color='#b0c4de',legend=False)

demred = demands[index].sum(axis=1)    
demred.loc[rng].plot(ax=ax,color ='red',legend=False)


ax.xaxis.set_label_text('Time [min]')
ax.yaxis.set_label_text('Power [W]')


# Saving figure

# fig = ax.get_figure()

# newpath = r'.\simulations\plots' 
# if not os.path.exists(newpath):
#     os.makedirs(newpath)
# figname = 'case_{0}.png'.format(ncase)

# figpath = os.path.join(newpath,figname)
# fig.savefig(figpath,format='png',bbox_inches='tight')

# plt.close(fig)
