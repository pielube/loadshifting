
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


"""
Loading inputs
"""

with open('inputs/example.json') as f:
  inputs = json.load(f)
  
# People living in the dwelling
# Taken from StRoBe list
cond1 = 'members' not in inputs
cond2 = 'members' in inputs and inputs['members'] == None
if cond1 or cond2:
    inputs['members'] = HouseholdMembers(inputs['HP']['dwelling_type'])

# Thermal parameters of the dwelling
# Taken from Procebar .xls files

procebinp = ProcebarExtractor(inputs['HP']['dwelling_type'],True)
inputs['HP'] = {**inputs['HP'],**procebinp}  
  
start_time = time.time()

"""
Running the models
"""

# Strobe
# House thermal model + HP
# DHW
result,textoutput = strobe.simulate_scenarios(1,inputs)
n_scen = 0 # Working only with the first scenario

# RAMP-mobility
if inputs['EV']['loadshift']:
    result_ramp = ramp.EVCharging(inputs, result['occupancy'][n_scen])
else:
    result_ramp=pd.DataFrame()

exectime = (time.time() - start_time)/60.
print("Time to run the models: {:.1f} minutes".format(exectime))



pvpeak = 10. #kW
pvfile = r'./simulations/pv.pkl'
pvadim = pd.read_pickle(pvfile)
pv = pvadim * pvpeak # kW
pv = pv.iloc[:,0]



def ambientdata(datapath):
    temp = np.loadtxt(datapath + '/Climate/temperature.txt')
    # temp = np.append(temp,temp[-24*60:]) # add december 31 to end of year in case of leap year
    irr  = np.loadtxt(datapath + '/Climate/irradiance.txt')
    # irr  = np.append(irr,irr[-24*60:]) # add december 31 to end of year in case of leap year
    return temp,irr

def COP_Tamb(Temp):
    COP = 0.001*Temp**2 + 0.0471*Temp + 2.1259
    return COP


def test_noshift(inputs,nminutes,Tamb,irr,Qintgains,Tset,fracmaxP):

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

    if inputs['HP']['HeatPumpThermalPower'] == None:
        # Heat pump sizing
        # External T = -10°C, internal T = 21°C
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
                     t_set_heating=21.,
                     max_heating_power=float('inf'))
        Tair = 21.
        House.solve_energy(0.,0.,-10.,Tair)
        QheatHP = House.heating_demand*fracmaxP
        # Ttemp = House.t_m_next this should be the T that would be reached with no heating
        # Tair = House.t_air # T actually reached in the house (in this case should be = to initial Tair)
        
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
                    t_set_heating=Tset[0],
                    max_heating_power=QheatHP)
        
    else:
        # Heat pump size given as an input
        # directly defining the house to be modelled
        QheatHP = inputs['HP']['HeatPumpThermalPower']
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
                    t_set_heating=Tset[0],
                    max_heating_power=inputs['HP']['HeatPumpThermalPower'])  
            

    Tair = max(16.,Tamb[0])  + random.random()*2. #°C
    Qheat = np.zeros(nminutes)
    Tinside = np.zeros(nminutes)
    Tsetold = Tset[0]

    for i in range(nminutes):
        
        Tset_ts = Tset[i]
        if Tset_ts != Tsetold:
            
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
                        t_set_heating=Tset_ts,
                        max_heating_power=QheatHP)
            
            Tsetold = Tset_ts
        
        if 60*24*151 < i <= 60*24*244: # heating season
            Qheat[i] = 0
            if i == 60*24*244:
                Tair = Tamb[i]
        else:
            House.solve_energy(Qintgains[i], Qsolgains[i], Tamb[i], Tair)
            Tair      = House.t_air
            Qheat[i] = House.heating_demand
            Tinside[i] = Tair
        
    return Qheat, QheatHP, Tinside



def test_shift(inputs,nminutes,Tamb,irr,Qintgains,QheatHP,pv,Tset):

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
            
    Tair = max(16.,Tamb[0])  + random.random()*2. #°C
    Qheat = np.zeros(nminutes)
    Tinside = np.zeros(nminutes)
    Tsetold = Tset[0] #inputs['HP']['Tthermostatsetpoint']

    for i in range(nminutes):
        
        if 60*24*151 < i <= 60*24*244: # heating season
            Qheat[i] = 0
            if i == 60*24*244:
                Tair = Tamb[i]
        else:
            
            if pv[i] > 0.:
                Tset_ts = 25.
            else:
                Tset_ts = Tset[i] #inputs['HP']['Tthermostatsetpoint']
                
            if Tset_ts != Tsetold:
                
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
                            t_set_heating=Tset_ts,
                            max_heating_power=QheatHP)
                
            House.solve_energy(Qintgains[i], Qsolgains[i], Tamb[i], Tair)
            Tair      = House.t_air
            Qheat[i] = House.heating_demand
            Tinside[i] = Tair
            
            Tsetold = Tset_ts
           
    return Qheat, Tinside




datapath = r'./strobe/Data'
temp, irr = ambientdata(datapath)
temp = np.delete(temp,-1)
irr = np.delete(irr,-1)
Qintgains = np.zeros(len(temp))
nminutes = len(temp)
pv_long = np.zeros(nminutes)

for i in range(len(pv)):
    for j in range(15):
        pv_long[i*15+j]=pv[i]


occ = np.zeros(len(result['occupancy'][0][0]))
for i in range(len(result['occupancy'][0])):
    result['occupancy'][0][i] = [1 if a==1 else 0 for a in result['occupancy'][0][i]]
    occ += result['occupancy'][0][i] 
occ = [1 if a >=1 else 0 for a in occ]    
occ = occ[:-1].copy()
occupancy = np.zeros(nminutes)
for i in range(len(occ)):
    for j in range(10):
        occupancy[i*10+j] = occ[i]
occupancy[-1] = occupancy[-2]

Tset = np.zeros(nminutes)
for i in range(nminutes):
    if occupancy[i] != 0:
        Tset[i] = 20.
    else:
        Tset[i] = 15.
 
fracmaxP=1.

Qnoshift,QheatHP,Tin_noshift = test_noshift(inputs,nminutes,temp,irr,Qintgains,Tset,fracmaxP)
Qshift,Tin_shift = test_shift(inputs,nminutes,temp,irr,Qintgains,QheatHP,pv_long,Tset)

# Tin_noshift_hs = Tin_noshift[np.r_[0:60*24*151,60*24*244:-1]]
# Tin_shift_hs   = Tin_shift[np.r_[0:60*24*151,60*24*244:-1]]

Twhenon_noshift = Tin_noshift*occupancy
Twhenon_noshift_hs = Twhenon_noshift[np.r_[0:60*24*151,60*24*244:-1]]
nonzero_noshift = np.nonzero(Twhenon_noshift_hs)
Twhenon_noshift_hs_mean = np.mean(Twhenon_noshift_hs[nonzero_noshift])
Twhenon_noshift_hs_min = np.min(Twhenon_noshift_hs[nonzero_noshift])
Twhenon_noshift_hs_max = np.max(Twhenon_noshift_hs[nonzero_noshift])


Twhenon_shift = Tin_shift*occupancy
Twhenon_shift_hs = Twhenon_shift[np.r_[0:60*24*151,60*24*244:-1]]
nonzero_shift = np.nonzero(Twhenon_shift_hs)
Twhenon_shift_hs_mean = np.mean(Twhenon_shift_hs[nonzero_shift])
Twhenon_shift_hs_min = np.min(Twhenon_shift_hs[nonzero_shift])
Twhenon_shift_hs_max = np.max(Twhenon_shift_hs[nonzero_shift])

actpv = [1 if a>0 else 0 for a in pv_long]

Qnoshift_pv = Qnoshift * actpv
Qshift_pv = Qshift *actpv

# Qnoshift_pv_hs = Qnoshift_pv[np.r_[0:60*24*151,60*24*244:len(Qnoshift_pv)]]
# Qshift_pv_hs = Qshift_pv[np.r_[0:60*24*151,60*24*244:len(Qnoshift_pv)]]

Enoshift = np.zeros(nminutes)
Eshift = np.zeros(nminutes)

for i in range(len(Qnoshift)):
    COP = COP_Tamb(temp[i])
    Enoshift[i] = Qnoshift[i]/COP
    Eshift[i] = Qshift[i]/COP
    
totcons_noshift = np.sum(Enoshift)/60./1000.
totcons_shift = np.sum(Eshift)/60./1000.
roundtripeff = totcons_noshift/totcons_shift


# heating si attiva quando PV sopra certa soglia
# a seconda delle priorità si potrebbe usare PV residuo
# se attivo perchè ho PV ma energia elettrica non viene da PV non ha senso
# calcolo basato su admtimewin?
# ricordati di riaggiungere gli internal heat gains


# model predictive control
# and then see battery equivalent








        