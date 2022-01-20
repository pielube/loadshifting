
import os
import numpy as np
import pandas as pd
import json
import time
import pickle
from temp_functions import yearlyprices,mostrapcurve, run,strategy1
from prosumpy import dispatch_max_sc_withsd,print_analysis



"""
Inputs:
    pv profile,
    pv capacity, 
    battery capacity, 
    price scenario, 
    technology costs, 
    economic variables,
    timestep,
    step per hour,
    demand

Functions:
    choose most rapresentative curve,
    models cases from excel
        prosumpy functions
        economic analysis functions
        load shifting functions
    
"""

inputs = {'CapacityPV': 10.,
          'CapacityBattery': 0.}

pvpeak = 10. #kW 
# pv.values[:] = 0

pvfile = r'./simulations/pv.pkl'
pvadim = pd.read_pickle(pvfile)
pv = pvadim * pvpeak # kW
pv = pv.iloc[:,0]


name = '1f.pkl'
path = r'./simulations'
file = os.path.join(path,name)
demands = pd.read_pickle(file)
columns = ['StaticLoad','TumbleDryer','DishWasher','WashingMachine','DomesticHotWater','HeatPumpPower','EVCharging']
appshift = ['WashingMachine','TumbleDryer','DishWasher']

name = '1f_occ.pkl'
file = os.path.join(path,name)
occupancys = pd.read_pickle(file)

param_tech = {'BatteryCapacity':  0.,
              'BatteryEfficiency': 0.9,
              'MaxPower': 7.,
              'InverterEfficiency': 1.,
              'timestep': .25}

# Technology costs
Inv = {'FixedPVCost':0,
        'PVCost_kW':1500,
        'FixedBatteryCost':100,
        'BatteryCost_kWh':200,
        'PVLifetime':20,
        'BatteryLifetime':10,
        'OM':0.015} # eur/year/eur of capex (both for PV and battery)

with open(r'./inputs/economics.json') as g:
  econ = json.load(g)

timeslots = econ['timeslots']
prices = econ['prices']
scenario = 'test'

# Economic parameteres
EconomicVar = {'WACC': 0.05, # weighted average cost of capital
               'net_metering': False, # type of tarification scheme
               'time_horizon': 20,
               'C_grid_fixed':prices[scenario]['fixed'], # € annual fixed grid costs
               'C_grid_kW': prices[scenario]['capacity'], # €/kW annual grid cost per kW 
               'P_FtG': 40.}     # €/MWh electricity price to sell to the grid

timestep = 0.25 # hrs
stepperhour = 4

ElPrices = yearlyprices(scenario,timeslots,prices,stepperhour)
index = mostrapcurve(pv,demands,param_tech,inputs,EconomicVar,Inv,ElPrices,timestep,columns)

test = run(pv,demands[index],param_tech,inputs,EconomicVar,Inv,ElPrices,timestep,columns,prices,scenario)


"""
Load shifting for the appliances
Strategy 1
"""

stepperhourshift=60
yprices = yearlyprices(scenario,timeslots,prices,stepperhourshift)
admprices = np.where(yprices <= prices[scenario]['hollow']/1000,1.,0.)
admprices = np.append(admprices,yprices[-1])

# Custom admissible windows
admcustom = np.ones(len(admprices))
for i in range(len(admprices)-60):
    if admprices[i]-admprices[i+60] == 1.:
        admcustom[i] = 0
               
# Adimissibile time windows according to occupancy

occ = np.zeros(len(occupancys[index][0]))
for i in range(len(occupancys[index])):
    occupancys[index][i] = [1 if a==1 else 0 for a in occupancys[index][i]]
    occ += occupancys[index][i]
    
occ = [1 if a >=1 else 0 for a in occ]    
occ = occ[:-1].copy()
occupancy = np.zeros(len(demands[index]['StaticLoad']))
for i in range(len(occ)):
    for j in range(10):
        occupancy[i*10+j] = occ[i]
occupancy[-1] = occupancy[-2]

# Resulting admissibile time windows
admtimewin = admprices*admcustom*occupancy

# Probability of load being shifted
probshift = 1.

startshift = time.time()

for app in appshift:
    print("---"+str(app)+"---")
    app_n,ncyc,ncycshift,maxshift,avgshift,cycnotshift = strategy1(demands[index][app],admtimewin,probshift)

    demands[index].insert(len(demands[index].columns),app+'Shift', app_n,True)
    
    conspre  = sum(demands[index][app])/60./1000.
    conspost = sum(demands[index][app+'Shift'])/60./1000.
    print("Original consumption: {:.2f}".format(conspre))
    print("Number of cycles: {:}".format(ncyc))
    print("Number of cycles shifted: {:}".format(ncycshift))
    print("Consumption after shifting (check): {:.2f}".format(conspost))
    print("Max shift: {:.2f} hours".format(maxshift))
    print("Avg shift: {:.2f} hours".format(avgshift))
    print("Unable to shift {:} cycles".format(cycnotshift))

execshift = (time.time() - startshift)
print("Time to shift the appliances: {:.1f} seconds".format(execshift))

result = demands[index]

path = r'./simulations/results'
if not os.path.exists(path):
    os.makedirs(path)
    
name = 'test.pkl'
file = os.path.join(path,name)
with open(file, 'wb') as b:
    pickle.dump(result,b)

"""
Load shifting for DHW whit PV panels
Strategy 2
"""

# demand2 = demands[index]

# columnsnotshift = ['StaticLoad','TumbleDryer','DishWasher','WashingMachine','HeatPumpPower','EVCharging']
# demand2_notshift = demand2[columns]
# demand2_notshift = demand2_notshift.sum(axis=1)
# demand2_notshift = demand2_notshift/1000. # W to kW
# demand2_notshift = demand2_notshift.to_frame()
# demand2_notshift = demand2_notshift.resample('15Min').mean() # resampling at 15 min
# demand2_notshift.index = pd.to_datetime(demand2_notshift.index)
# year = demand2_notshift.index.year[0] # extracting ref year used in the simulation
# nye = pd.Timestamp(str(year+1)+'-01-01 00:00:00') # remove last row if is from next year
# demand2_notshift = demand2_notshift.drop(nye)
# demand2_notshift = demand2_notshift.iloc[:,0]

# demand2_dhw = demand2['DomesticHotWater']
# # demand2_dhw = demand2_dhw.sum(axis=1)
# demand2_dhw = demand2_dhw/1000. # W to kW
# demand2_dhw = demand2_dhw.to_frame()
# demand2_dhw = demand2_dhw.resample('15Min').mean() # resampling at 15 min
# demand2_dhw.index = pd.to_datetime(demand2_dhw.index)
# year = demand2_dhw.index.year[0] # extracting ref year used in the simulation
# nye = pd.Timestamp(str(year+1)+'-01-01 00:00:00') # remove last row if is from next year
# demand2_dhw = demand2_dhw.drop(nye)
# demand2_dhw = demand2_dhw.iloc[:,0]


# pv2 = []
# for i in range(len(pv)):
#     pvres = max(pv[i]-demand2_notshift[i],0)
#     pv2.append(pvres)
    
# pv2 = pd.Series(pv2)
# pv2.index = pv.index    

# with open('inputs/example.json') as f:
#   inputs2 = json.load(f)

# Ccyl = inputs2['DHW']['Vcyl'] * 1000. /1000. * 4200. # J/K
# capacity = Ccyl*(inputs2['DHW']['Ttarget']-inputs2['DHW']['Tcw'])/3600./1000. # kWh

# # Troom = 15. #°C
# # selfdis = inputs2['DHW']['Hloss']/1000.*timestep*(inputs2['DHW']['Ttarget']-Troom) # kW
  
# param_tech2 = {'BatteryCapacity':  capacity,
#               'BatteryEfficiency': 1.,
#               'MaxPower': inputs2['DHW']['PowerElMax'],
#               'InverterEfficiency': 1.,
#               'timestep': .25,
#               'SelfDisLin': 0.,
#               'SelfDisFix':0.}

# outs = dispatch_max_sc_withsd(pv2,demand2_dhw,param_tech2,return_series=False)
# print_analysis(pv2, demand2_dhw, param_tech2, outs)

