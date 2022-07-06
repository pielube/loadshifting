
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
index60min = pd.date_range(start='2015-01-01',end='2015-12-31 23:59:00',freq='60T')
n1min  = len(index1min)
n15min = len(index15min)
n60min = len(index60min)
stepperh_1min  = 60
stepperh_15min = 4
ts1min  = 1/60
ts15min = 1/4
ts60min = 1

heatseas_st = defaults.heatseas_st
heatseas_end = defaults.heatseas_end

"""
Demand of always the same house
"""

temp, irr = load_climate_data()
temp = pd.Series(data=temp[:-1],index=index1min)
temp15min = temp.resample('15Min').mean()
irr = pd.Series(data=irr[:-1],index=index1min)
irr15min = irr.resample('15Min').mean()

conf = load_config('case16')
config,pvbatt_param,econ_param,tariffs,housetype,N = conf['config'],conf['pvbatt_param'],conf['econ_param'],conf['tariffs'],conf['housetype'],conf['N']

# housetype['HP']['HeatPumpThermalPower'] = 6000.

# buildtype='Freestanding'
# wellinsulated = True
# procebinp = ProcebarExtractor(buildtype,wellinsulated)

procebinp={ 'ACH_infl': 0.5,        #forcing thermal parameters
  'ACH_vent': 0.0,
  'Afloor': 131.7,
  'Aglazed': 46.5,
  'Aopaque': 92.2,
  'Atotal': 270.2,
  'Ctot': 16833573.,
  'Uwalls': 0.5,
  'Uwindows': 2.5,
  'VentEff': 0.0,
  'volume': 375.0}

members = ['FTE','FTE','U12']       # forcing members
out = compute_demand(housetype,N,members= members,thermal_parameters=procebinp)
occ = out['occupancy'][0]
occupancy_10min = (occ==1).sum(axis=1)                     # when occupancy==1, the person is in the house and not sleeping
occupancy_10min = (occupancy_10min>0)                       # if there is at least one person awake in the house
occupancy_1min = occupancy_10min.reindex(index1min,method='nearest')
occupancy_15min = occupancy_10min.reindex(index15min,method='nearest')
occupancy_60min = occupancy_10min.reindex(index60min,method='nearest')

"""
Recompute thermal demand
"""
procebinp={ 'ACH_infl': 0.5,        #forcing thermal parameters
  'ACH_vent': 0.0,
  'Afloor': 131.7,
  'Aglazed': 46.5,
  'Aopaque': 92.2,
  'Atotal': 270.2,
  'Ctot': 16833573.,
  'Uwalls': 0.5,
  'Uwindows': 2.5,
  'VentEff': 0.0,
  'volume': 375.0}

housetype['HP'] = {**housetype['HP'],**procebinp}

Tset_ref = np.full(n15min,defaults.T_sp_low) + np.full(n15min,defaults.T_sp_occ-defaults.T_sp_low) * occupancy_15min

housetype['HP']['HeatPumpThermalPower'] = None
fracmaxP = 0.8 #defaults.fracmaxP 
QheatHP = HPSizing(housetype,fracmaxP)

# QheatHP = 8000.
# housetype['HP']['HeatPumpThermalPower'] = QheatHP

Qintgains = out['results'][0]['InternalGains']
Qintgains = Qintgains.resample('15Min').mean() 
res = HouseHeating(housetype,QheatHP,Tset_ref,Qintgains,temp15min,irr15min,n15min,defaults.heatseas_st,defaults.heatseas_end,ts15min)
Qheat = res['Qheat']
Tin_heat = res['Tinside']
Tm = res['Tm']


Eheat = np.zeros(n15min+1)
for i in range(n15min):
    COP = COP_Tamb(temp[i])
    Eheat[i] = Qheat[i]/COP # W

Eheat = pd.Series(data=Eheat[:-1],index=index15min)
Eheat = Eheat.resample('1Min').ffill()
Eheat = Eheat.reindex(index1min).fillna(method='ffill')
Eheat = Eheat.to_numpy()
Eheat = np.append(Eheat,Eheat[-1])
out['results'][0]['HeatPumpPower'] = Eheat

  
"""
Shifting
"""

scenario = econ_param['scenario']
timeslots = tariffs['timeslots']
enprices = tariffs['prices']
gridfees = tariffs['gridfees']
thresholdprice = econ_param['thresholdprice']

yenprices_15min = yearlyprices(scenario,timeslots,enprices,stepperh_15min) # €/kWh
ygridfees_15min = yearlyprices(scenario,timeslots,gridfees,stepperh_15min) # €/kWh
yprices_15min = yenprices_15min + ygridfees_15min  # €/kWh
admprice = (enprices[scenario][thresholdprice] + gridfees[scenario][thresholdprice])/1000
admprices = np.where(yprices_15min <= admprice+0.01,1.,0.)


time1 = time.time()

""" PV shifting strategy """
# Tset[pv_long>0] = 25.
# res_sh = HouseHeating(housetype,QheatHP,Tset,Qintgains,temp,irr,n1min,defaults.heatseas_st,defaults.heatseas_end,ts)
# Qheat_sh = res_sh['Qheat']
# Tin_sh = res_sh['Tinside']
# Tm_sh = res_sh['Tm']

""" Tariff shifting strategy """
# Shifting-specific inputs
Tincrease = 3. # °C T increase wrt min T setpoint (heating off)
t_preheat = 1  # h max time allowed to consider pre-heating
t_preheat_15min = t_preheat*stepperh_15min

Tset_ref = np.full(n15min,defaults.T_sp_low) + np.full(n15min,defaults.T_sp_occ-defaults.T_sp_low) * occupancy_15min

offset = Tset_ref.min()
Tset = Tset_ref - offset

mask_z = Tset>0
idx_z = np.flatnonzero(mask_z)
idx_nz = np.flatnonzero(~mask_z)

idx_z = np.r_[idx_z,len(Tset)]

out_r = np.zeros(len(Tset), dtype=int)
idx_r = np.searchsorted(idx_z, idx_nz)
out_r[~mask_z] = idx_z[idx_r] - idx_nz

admhours = [1. if 0<a<t_preheat_15min else 0. for a in out_r]
admhours = np.array(admhours)

# Resulting hours in which to increase setpoint
idx = np.where(admprices*admhours)[0]

# Recalculating T setpoint array with increase
Tset += offset
Tset[idx] += Tincrease

# Recomputing heating after shifting
res_sh = HouseHeating(housetype,QheatHP,Tset,Qintgains,temp15min,irr15min,n15min,heatseas_st,heatseas_end,ts15min)
Qheat_sh = res_sh['Qheat']
Tin_sh = res_sh['Tinside']
Tm_sh = res_sh['Tm']

time2 = time.time()
print('It took {:.2f} seconds'.format(time2 - time1))


"""
Postprocess
"""

# T analysis
Twhenon    = Tin_sh*occupancy_15min.values # °C
Twhenon_hs = Twhenon[np.r_[0:4*24*defaults.heatseas_end,4*24*defaults.heatseas_st:-1]] # °C
whenon     = np.nonzero(Twhenon_hs)
Twhenon_hs_mean = np.mean(Twhenon_hs[whenon]) # °C
Twhenon_hs_min  = np.min(Twhenon_hs[whenon]) # °C
Twhenon_hs_max  = np.max(Twhenon_hs[whenon]) # °C

# Electricity consumption
Eshift = np.zeros(n15min) 
for i in range(n15min):
    COP = COP_Tamb(temp[i])
    Eshift[i] = Qheat_sh[i]/COP # W

# Updating demand dataframe
demand_HP_shift = pd.Series(data=Eshift,index=index15min)/1000.

# Check results
demand_HP = pd.Series(data=out['results'][0]['HeatPumpPower'][:-1]/1000.,index=index1min)
demand_HP = demand_HP.resample('15Min').mean() # kW

HPcons_pre  = np.sum(demand_HP)/4. # kWh
HPcons_post = np.sum(demand_HP_shift)/4. # kWh
HPconsincr = (HPcons_post-HPcons_pre)/HPcons_pre*100 # %
print("Original consumption: {:.2f} kWh".format(HPcons_pre))
print("Consumption after shifting: {:.2f} kWh".format(HPcons_post))
print("Consumption increase: {:.2f}%".format(HPconsincr))


Tin_pre = pd.Series(data=Tin_heat,index=index15min)
Tin_post = pd.Series(data=Tin_sh,index=index15min)

Tm_pre = pd.Series(data=Tm,index=index15min)
Tm_post = pd.Series(data=Tm_sh,index=index15min)

xx1 = sum(demand_HP/4*yprices_15min)
xx2 = sum(demand_HP_shift/4*yprices_15min)
print("Annual expense pre: {:.2f} kWh".format(xx1))
print("Annual expense post: {:.2f} kWh".format(xx2))

xxx = demand_HP_shift - demand_HP
yy1 = abs(sum(xxx[np.where(xxx>0)[0]]))
yy2 = abs(sum(xxx[np.where(xxx<0)[0]]))
print("Preheating: {:.2f} kWh".format(yy1))
print("Heating avoided: {:.2f} kWh".format(yy2))


"""
Plots
"""

day ='2015-01-03'
ax = demand_HP.loc[day].plot(legend=True)
ax1 = ax.twinx()

demand_HP_shift.loc[day].plot(ax=ax,legend=True)

df = occupancy_15min.loc[day] > 0
fst = df.index[df & ~ df.shift(1).fillna(False)]
lst = df.index[df & ~ df.shift(-1).fillna(False)]
pr = [(i, j) for i, j in zip(fst, lst)]
for i, j in pr:
    ax.axvspan(i,j, color='tab:green', alpha=0.3)

Tin_pre.loc[day].plot(ax=ax1,c='tab:blue',ls='dashed',legend=True)
Tin_post.loc[day].plot(ax=ax1,c='tab:orange',ls='dashed',legend=True)
Tm_pre.loc[day].plot(ax=ax1,c='tab:blue',ls='dotted',legend=True)
Tm_post.loc[day].plot(ax=ax1,c='tab:orange',ls='dotted',legend=True)

ax1.axhline(y=20, color='tab:red', ls='-',alpha=0.4)
ax1.axhline(y=15, color='tab:purple', ls='-',alpha=0.4)

ax.set_ylabel('P$_{el,HP}$ [kW]')
ax1.set_ylabel('T [°C]')
ax.set_ylim(0,10)
ax1.set_ylim(0,25)

ax.legend(labels=['Pre','Post'],loc = 'upper left',bbox_to_anchor=(0,-0.12),frameon=False)

from matplotlib.lines import Line2D
line1 = Line2D([0,1],[0,1],linestyle='-', color='grey')
line2 = Line2D([0,1],[0,1],linestyle='dashed', color='grey')
line3 = Line2D([0,1],[0,1],linestyle='dotted', color='grey')
ax1.legend([line1,line2,line3],['Power','T inside','T wall'],loc='upper left',bbox_to_anchor=(0.2,-0.12),frameon=False)


"""
Check on changing timestep
"""

# # 1 min timestep
# ts1min=1/60
# housetype['HP']['Ctot']=16833573.#*60
# res1min = HouseHeating(housetype,QheatHP,Tset_ref,Qintgains,temp,irr,n1min,heatseas_st,heatseas_end,ts1min)
# Qheat1min = res1min['Qheat']
# Tin1min = res1min['Tinside']
# Tm1min = res1min['Tm']
# totcons1min = np.sum(Qheat1min/60/1000) #kWh
# print("Consumption 1-min timestp: {:.2f} kWh".format(totcons1min))


# # 15 min timestep
# Tset15min = Tset_ref.resample('15Min').mean()
# Qintgains15min = Qintgains.resample('15Min').mean() # kW
# temp = pd.Series(data=temp[:-1],index=index1min)
# temp15min = temp.resample('15Min').mean()
# irr = pd.Series(data=irr[:-1],index=index1min)
# irr15min = irr.resample('15Min').mean()
# n15min = 35040
# ts15min = 0.25
# housetype['HP']['Ctot']=16833573.#*4
# res15min = HouseHeating(housetype,QheatHP,Tset15min,Qintgains15min,temp15min,irr15min,n15min,heatseas_st,heatseas_end,ts15min)
# Qheat15min = res15min['Qheat']
# Tin15min = res15min['Tinside']
# Tm15min = res15min['Tm']
# totcons15min = np.sum(Qheat15min/4/1000) #kWh
# print("Consumption 15-min timestep: {:.2f} kWh".format(totcons15min))


# # 1 h timestep
# Tset1h = Tset_ref.resample('60Min').mean()
# Qintgains1h = Qintgains.resample('60Min').mean() # kW
# temp = pd.Series(data=temp[:-1],index=index1min)
# temp1h = temp.resample('60Min').mean()
# # temp1h = temp.reindex(index60min,method='nearest')
# irr = pd.Series(data=irr[:-1],index=index1min)
# irr1h = irr.resample('60Min').mean()
# # irr1h = irr.reindex(index60min,method='nearest')
# n60min = 8760
# ts1h = 1
# housetype['HP']['Ctot']=16833573.
# res1hour = HouseHeating(housetype,QheatHP,Tset1h,Qintgains1h,temp1h,irr1h,n60min,heatseas_st,heatseas_end,ts1h)
# Qheat1hour = res1hour['Qheat']
# Tin1hour = res1hour['Tinside']
# Tm1hour = res1hour['Tm']
# totcons1hour = np.sum(Qheat1hour/1000) #kWh
# print("Consumption 1-hour timestep: {:.2f} kWh".format(totcons1hour))


# day ='2015-01-03'

# # heating
# Qheat1min = pd.Series(data=Qheat1min,index=index1min)
# Qheat15min = pd.Series(data=Qheat15min,index=index15min)
# Qheat1hour = pd.Series(data=Qheat1hour,index=index60min)
# ax = Qheat1min.loc[day].plot(label='1-min',legend=True)
# Qheat15min.loc[day].plot(ax=ax,label= '15-min',legend=True)
# Qheat1hour.loc[day].plot(ax=ax,label= '1-hour',legend=True)

# ax1 = ax.twinx()

# # # occupancy
# # occupancy_1min=occupancy_1min.astype(int)
# # occupancy_60min=occupancy_60min.astype(int)

# # air temperature
# Tin1min = pd.Series(data=Tin1min,index=index1min)
# Tin15min = pd.Series(data=Tin15min,index=index15min)
# Tin1hour = pd.Series(data=Tin1hour,index=index60min)
# Tin1min.loc[day].plot(ax=ax1,c='tab:blue',ls='dashed')
# Tin15min.loc[day].plot(ax=ax1,c='tab:orange',ls='dashed')
# Tin1hour.loc[day].plot(ax=ax1,c='tab:green',ls='dashed')

# # wall temperature
# Tm1min = pd.Series(data=Tm1min,index=index1min)
# Tm15min = pd.Series(data=Tm15min,index=index15min)
# Tm1hour = pd.Series(data=Tm1hour,index=index60min)
# Tm1min.loc[day].plot(ax=ax1,c='tab:blue',ls='dotted')
# Tm15min.loc[day].plot(ax=ax1,c='tab:orange',ls='dotted')
# Tm1hour.loc[day].plot(ax=ax1,c='tab:green',ls='dotted')

# ax.set_ylim(0,10000)
# ax1.set_ylim(0,21)







