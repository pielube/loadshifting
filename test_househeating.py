
import time

import numpy as np
import pandas as pd

from simulation import load_config
from functions import ProcebarExtractor, HPSizing, yearlyprices, HouseHeating, load_climate_data, COP_Tamb
from demands import compute_demand
import defaults

index1min  = pd.date_range(start='2015-01-01',end='2015-12-31 23:59:00',freq='T')
index10min = pd.date_range(start='2015-01-01',end='2015-12-31 23:59:00',freq='10T')
index15min = pd.date_range(start='2015-01-01',end='2015-12-31 23:59:00',freq='15T')
n1min = len(index1min)
stepperh_1min = 60
stepperh_15min = 4

"""
Demand of always the same house
"""

temp, irr = load_climate_data()
conf = load_config('case16')
config,pvbatt_param,econ_param,tariffs,housetype,N = conf['config'],conf['pvbatt_param'],conf['econ_param'],conf['tariffs'],conf['housetype'],conf['N']
# buildtype='Freestanding'
# wellinsulated = True
# procebinp = ProcebarExtractor(buildtype,wellinsulated)
procebinp={ 'ACH_infl': 0.6,        # forcing thermal parameters
  'ACH_vent': 0.6,
  'Afloor': 42.56,
  'Aglazed': 16.43,
  'Aopaque': 105.07360152315982,
  'Atotal': 262.7636015231598,
  'Ctot': 300000.,#157148.60871502483,
  'Uwalls': 0.32,
  'Uwindows': 2,
  'VentEff': 0.0,
  'volume': 224.99799999999996}
members = ['FTE','FTE','U12']       # forcing members
out = compute_demand(housetype,N,members= members,thermal_parameters=procebinp)
occ = out['occupancy'][0]
occupancy_10min = (occ==1).sum(axis=1)                     # when occupancy==1, the person is in the house and not sleeping
occupancy_10min = (occupancy_10min>0)                       # if there is at least one person awake in the house
occupancy_1min = occupancy_10min.reindex(index1min,method='nearest')
occupancy_15min = occupancy_10min.reindex(index15min,method='nearest')

"""
Recompute thermal demand
"""


procebinp={ 'ACH_infl': 0.6,
  'ACH_vent': 0.6,
  'Afloor': 42.56,
  'Aglazed': 16.43,
  'Aopaque': 105.07360152315982,
  'Atotal': 262.7636015231598,
  'Ctot': 300000.,#157148.60871502483,
  'Uwalls': 0.1,#0.32,
  'Uwindows': 0.5,#2,
  'VentEff': 0.0,
  'volume': 224.99799999999996}
housetype['HP'] = {**housetype['HP'],**procebinp}


Tset_ref = np.full(n1min,defaults.T_sp_low) + np.full(n1min,defaults.T_sp_occ-defaults.T_sp_low) * occupancy_1min

fracmaxP = defaults.fracmaxP 
QheatHP = HPSizing(housetype,fracmaxP)

Qintgains = out['results'][0]['InternalGains']    
Qheat,Tin_heat = HouseHeating(housetype,QheatHP,Tset_ref,Qintgains,temp,irr,n1min,defaults.heatseas_st,defaults.heatseas_end)

Eheat = np.zeros(n1min+1)
for i in range(n1min):
    COP = COP_Tamb(temp[i])
    Eheat[i] = Qheat[i]/COP # W

out['results'][0]['HeatPumpPower'] = Eheat

  
"""
Shifting
"""

heatseas_st = defaults.heatseas_st
heatseas_end = defaults.heatseas_end

scenario = econ_param['scenario']
timeslots = tariffs['timeslots']
enprices = tariffs['prices']
gridfees = tariffs['gridfees']
thresholdprice = econ_param['thresholdprice']

yenprices_1min = yearlyprices(scenario,timeslots,enprices,stepperh_1min) # €/kWh
ygridfees_1min = yearlyprices(scenario,timeslots,gridfees,stepperh_1min) # €/kWh
yprices_1min = yenprices_1min + ygridfees_1min  # €/kWh
admprice = (enprices[scenario][thresholdprice] + gridfees[scenario][thresholdprice])/1000
admprices = np.where(yprices_1min <= admprice+0.01,1.,0.)


time1 = time.time()

""" PV shifting strategy """
# Tset[pv_long>0] = 25.
# Qheat_sh,Tin_sh = househeating(inputs,QheatHP,Tset,Qintgains,temp,irr,nminutes,heatseas_st,heatseas_end)

""" Tariff shifting strategy """
# Shifting-specific inputs
Tincrease = 3. # °C T increase wrt min T setpoint (heating off)
t_preheat = 3  # h max time allowed to consider pre-heating
t_preheat_min = t_preheat*60

Tset_ref = np.full(n1min,defaults.T_sp_low) + np.full(n1min,defaults.T_sp_occ-defaults.T_sp_low) * occupancy_1min

offset = Tset_ref.min()
Tset = Tset_ref - offset

mask_z = Tset>0
idx_z = np.flatnonzero(mask_z)
idx_nz = np.flatnonzero(~mask_z)

idx_z = np.r_[idx_z,len(Tset)]

out_r = np.zeros(len(Tset), dtype=int)
idx_r = np.searchsorted(idx_z, idx_nz)
out_r[~mask_z] = idx_z[idx_r] - idx_nz

admhours = [1. if 0<a<t_preheat_min else 0. for a in out_r]
admhours = np.array(admhours)

# Resulting hours in which to increase setpoint
idx = np.where(admprices*admhours)[0]

# Recalculating T setpoint array with increase
Tset += offset
Tset[idx] += Tincrease

Qheat_sh,Tin_sh = HouseHeating(housetype,QheatHP,Tset,Qintgains,temp,irr,n1min,heatseas_st,heatseas_end)

time2 = time.time()
print('It took {:.2f} seconds'.format(time2 - time1))


"""
Postprocess
"""

# T analysis
Twhenon    = Tin_sh*occupancy_1min.values # °C
Twhenon_hs = Twhenon[np.r_[0:60*24*defaults.heatseas_end,60*24*defaults.heatseas_st:-1]] # °C
whenon     = np.nonzero(Twhenon_hs)
Twhenon_hs_mean = np.mean(Twhenon_hs[whenon]) # °C
Twhenon_hs_min  = np.min(Twhenon_hs[whenon]) # °C
Twhenon_hs_max  = np.max(Twhenon_hs[whenon]) # °C

# Electricity consumption
Eshift = np.zeros(n1min) 
for i in range(n1min):
    COP = COP_Tamb(temp[i])
    Eshift[i] = Qheat_sh[i]/COP # W

# Updating demand dataframe
demand_HP_shift = pd.Series(data=Eshift,index=index1min)/1000.
demand_HP_shift = demand_HP_shift.resample('15Min').mean() # kW

# Check results
demand_HP = pd.Series(data=out['results'][0]['HeatPumpPower'][:-1]/1000.,index=index1min)
demand_HP = demand_HP.resample('15Min').mean() # kW


HPcons_pre  = np.sum(demand_HP)/4. # kWh
HPcons_post = np.sum(demand_HP_shift)/4. # kWh
HPconsincr = (HPcons_post-HPcons_pre)/HPcons_pre*100 # %
print("Original consumption: {:.2f} kWh".format(HPcons_pre))
print("Consumption after shifting: {:.2f} kWh".format(HPcons_post))
print("Consumption increase: {:.2f}%".format(HPconsincr))

Tin_pre = pd.Series(data=Tin_heat,index=index1min)
Tin_pre = Tin_pre.reindex(index15min,method='nearest')
Tin_post = pd.Series(data=Tin_sh,index=index1min)
Tin_post = Tin_post.reindex(index15min,method='nearest')

yenprices_15min = yearlyprices(scenario,timeslots,enprices,stepperh_15min) # €/kWh
ygridfees_15min = yearlyprices(scenario,timeslots,gridfees,stepperh_15min) # €/kWh
yprices_15min = yenprices_15min + ygridfees_15min  # €/kWh

xx1 = sum(demand_HP*yprices_15min)
xx2 = sum(demand_HP_shift*yprices_15min)

"""
Plots
"""

day ='2015-01-02'
ax = demand_HP.loc[day].plot(legend=True)
ax1 = ax.twinx()

demand_HP_shift.loc[day].plot(ax=ax,legend=True)

occgraph = occupancy_15min*3
occgraph.loc[day].plot.area(ax=ax,alpha=0.3)

# Tset_ref.loc[day].plot(ax=ax1,c='tab:green',legend=True)
# Tset.loc[day].plot(ax=ax1,c='tab:red',legend=True)
Tin_pre.loc[day].plot(ax=ax1,c='tab:green',ls='dashed',legend=True)
Tin_post.loc[day].plot(ax=ax1,c='tab:red',ls='dashed',legend=True)

ax.set_ylabel('Power demand [kW]')
ax1.set_ylabel('T setpoint [°C]')
ax.set_ylim(0,2)
ax1.set_ylim(0,25)
ax.legend(labels=['HP pre shift','HP after shift'],loc='upper left',bbox_to_anchor=(0,-0.12))
ax1.legend(labels=['T set pre shift','T set post shift','T in pre shift', 'T in post shift'],loc = 'upper right',bbox_to_anchor=(1,-0.12))





        