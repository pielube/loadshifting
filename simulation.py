#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 01:31:36 2022

@author: sylvain
"""

import os
import numpy as np
import pandas as pd
import json
import time
from prosumpy import dispatch_max_sc,print_analysis
from functions import yearlyprices,HPSizing,COP_deltaT
from functions import MostRepCurve,DHWShiftTariffs,HouseHeating,EVshift_PV,EVshift_tariffs,ResultsAnalysis,WriteResToExcel,load_climate_data
from functions import shift_appliance,scale_timeseries
from pv import pvgis_hist
from demands import compute_demand
import defaults

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

from joblib import Memory
memory = Memory(__location__ + '/cache/', verbose=defaults.verbose)

def load_config(namecase,cf_cases='cases.json',cf_pvbatt = 'pvbatt_param.json',cf_econ='econ_param.json',cf_tariff = 'tariffs.json', cf_house='housetypes.json'):
    '''
    Load the config files for a specific simulation
    
    Parameters
    ----------
    namecase : str
        case to be exctracted from the configuration (e.g. 'default')
        
    Returns
    -------
    dictionary with the main inputs   

    '''
    inputhpath = __location__ + '/inputs/'
    out = {}
    
    out['namecase'] = namecase
    out['N'] = 1 # Number of stochastic simulations to be run for the demand curves. # to be changed
    
    # Case description
    with open(inputhpath + cf_cases,'r') as f:
        cases = json.load(f)
    
    # PV and battery technology parameters
    with open(inputhpath + cf_pvbatt,'r') as f:
        out['pvbatt_param'] = json.load(f)
    
    # Economic parameters
    with open(inputhpath + cf_econ,'r') as f:
        econ_cases = json.load(f)
    
    # Time of use tariffs
    with open(inputhpath + cf_tariff,'r') as f:
        out['tariffs'] = json.load(f)
    
    # Parameters for the dwelling
    with open(inputhpath + cf_house,'r') as f:
        housetypes = json.load(f)    
    
    if defaults.verbose > 0:
        print('###########################')
        print('  Loading config: '+ namecase )
        print('###########################')
    
    out['config'] = cases[namecase]
    out['econ_param'] = econ_cases[namecase]
    out['housetype'] = housetypes[cases[namecase]['house']]
        
    return out

def load_cases(cf_cases='cases.json'):
    '''
    Load the list of cases from the corresponding json file
        
    Returns
    -------
    list with the cases  

    '''
    inputpath = __location__ + '/inputs/'
    
    # Case description
    with open(inputpath + cf_cases,'r') as f:
        cases = json.load(f)
    
    out = list(cases.keys())
    
    # Add user-defined cases:
    inputpath = __location__ + '/scenarios/'
    listfiles = [x.strip('.json') for x in os.listdir(inputpath) if x[-5:]=='.json']
    
    out = out + listfiles
    
    if 'default' in out:
        out.remove('default')
        
    return out


@memory.cache
def shift_load(config,pvbatt_param,econ_param,tariffs,inputs,N):
    '''
    
    Parameters
    ----------
    config : dict
        Pre-defined simulation case to be simulated.
    pvbatt_param : dict
        PV and battery parameters.
    econ_param : dict
        Economic parameters.
    tariffs : dict
        Time-of-use electricity tariffs.
    inputs : dict
        House parameters.
    N : int
        Number of stochastic scenarios to be simulated.

    Returns
    -------
    outs : dict
        Main simulation results.

    '''

    house          = config['house']
    columns        = config['columns'] 
    TechsShift     = config['TechsShift']
    WetAppShift    = [x for x in TechsShift if x in ['TumbleDryer','DishWasher','WashingMachine']]
    TechsNoShift   = [x for x in columns if x not in TechsShift]
    WetAppBool     = len(WetAppShift)>0
    WetAppManBool  = config['WetAppManualShifting']
    PVBool         = config['PresenceOfPV']
    BattBool       = config['PresenceOfBattery']
    DHWBool        = "DomesticHotWater" in TechsShift
    HeatingBool    = "HeatPumpPower" in TechsShift
    EVBool         = "EVCharging" in TechsShift
    
    FixedControl   = econ_param['FixedControlCost']
    AnnualControl  = econ_param['AnnualControlCost']
    thresholdprice = econ_param['thresholdprice']
    
    demands = compute_demand(inputs,N,inputs['members'],inputs['thermal_parameters'])
    
    config_pv  = pvbatt_param['pv']
    config_inv = pvbatt_param['inv']
    config_bat = pvbatt_param['battery']
    
    pvadim = pvgis_hist(config_pv)  
    
    # Various array sizes and timesteps used throughout the code
    index1min  = pd.date_range(start='2015-01-01',end='2015-12-31 23:59:00',freq='T')
    index10min = pd.date_range(start='2015-01-01',end='2015-12-31 23:50:00',freq='10T')
    index15min = pd.date_range(start='2015-01-01',end='2015-12-31 23:45:00',freq='15T')
    n1min  = len(index1min)
    n10min = int(n1min/10)
    n15min = int(n1min/15)
    stepperh_1min = 60 # 1/h
    stepperh_15min = 4 # 1/h
    ts_15min = 0.25 # h
    
    #%%
    # Electricity prices array - 15 min timestep
    scenario = econ_param['scenario']
    timeslots = tariffs['timeslots']
    enprices = tariffs['prices']
    gridfees = tariffs['gridfees']
    yenprices_15min = yearlyprices(scenario,timeslots,enprices,stepperh_15min) # €/kWh
    ygridfees_15min = yearlyprices(scenario,timeslots,gridfees,stepperh_15min) # €/kWh
    yprices_15min = yenprices_15min + ygridfees_15min  # €/kWh
    
    #%%
    
    """
    3) Most representative curve
    """
    
    idx = MostRepCurve(demands['results'],columns,yenprices_15min,ygridfees_15min,ts_15min,econ_param)
    
    # Inputs relative to the most representative curve:
    inputs = demands['input_data'][idx]
    if defaults.verbose > 0:
        print('Most representative curve index: {:}'.format(idx))
    
    """
    4) Demand
    """
    
    # Demand prosumpy-compatible
    # Meaning 15 min timestep and in kW
    # Selecting only techs (columns) of the case of interest
    
    demand_15min = demands['results'][idx][columns]/1000. # kW
    demand_15min = demand_15min.resample('15Min').mean()[:-1] # kW
    
    # define the shifted demand dataframe. To be updated in the code
    demand_shifted = demand_15min.copy()
    
    # Define a dataframe with the main power flows (in kW)
    pflows = pd.DataFrame(index=demand_15min.index)
    
    # Reference demand
    # Aggregated demand pre-shifting
    pflows['demand_noshift'] = demand_15min.sum(axis=1).to_numpy() # kW
    ydemand = np.sum(pflows['demand_noshift'])/4

    
    """
    5) Occupancy
    """
    
    # Occupancy array, built checking if at least one active person at home
    # 1-yes 0-no
    # 1 min timestep
    
    occ = demands['occupancy'][idx]
    occupancy_10min = (occ==1).sum(axis=1)                         # when occupancy==1, the person is in the house and not sleeping
    occupancy_10min = (occupancy_10min>0)                       # if there is at least one person awake in the house
    occupancy_1min = occupancy_10min.reindex(index1min,method='nearest')
    occupancy_15min = occupancy_10min.reindex(index15min,method='nearest')
  
    
    """
    6) PV production
    """
    
    if PVBool:
    
        if config_pv['AutomaticSizing']:
            yield_pv = pvadim.sum()/4
            pvpeak = min(ydemand/yield_pv,config_pv['Plim']) # kWp
        else:
            pvpeak = config_pv['Ppeak']
            
        if config_inv['AutomaticSizing']:
            inv_lim = config_inv['Plim_kVA'] * config_inv['powerfactor'] # kW max inv power
            invpeak = min(pvpeak/1.2,inv_lim)
        else:
            invpeak = config_inv['Ppeak']
            
        # 15 min timestep series, with upper limit given by inverter   
        pflows['pv'] = np.clip(pvadim*pvpeak,None,invpeak) # kW
        # 1 min timestep array
        pv_1min = scale_timeseries(pflows.pv,index1min)   # kW
                
        # Residual PV
        # 15 min timestep series
        pv_15min_res = np.maximum(0,pflows.pv - demand_15min[TechsNoShift].sum(axis=1)) # kW
        # 1 min timestep array
        demnoshift = demands['results'][idx][TechsNoShift].sum(axis=1)[:-1].to_numpy()/1000. # kW
        pv_1min_res = np.maximum(0,pv_1min-demnoshift) # kW 
      
    else:
        pflows['pv'] = pd.Series(0,index=index15min) # kW
        pvpeak = 0. # kWp
        invpeak = 0. # kWi
    
    # Update PV capacity
    pvbatt_param['pv']['Ppeak'] = pvpeak # kWp
    config_pv['Ppeak'] = pvpeak
    
    # Update inverter capacity
    pvbatt_param['inv']['Ppeak'] = invpeak # kWi
    config_inv['Ppeak'] = invpeak
    
    """
    7) Battery size
    """
    
    if not BattBool:
        pvbatt_param['battery']['BatteryCapacity'] = 0. # kWh
        config_bat['BatteryCapacity'] = 0.
    	   
    
    """
    8) Shifting
    """
    
    """
    8A) Load shifting - Wet appliances
        NB for the shifting of the appliances we work with 1-min timestep and power in W   
    """
    
    if WetAppBool:
        
        if defaults.verbose > 0:
            print('--- Shifting wet appliances ---')
        
        # Wet app demands to be shifted, 1 min timestep
        demshift = demands['results'][idx][WetAppShift][:-1] # W
    
        """
        Admissible time windows
        """
        
        # Admissible time window based on electricity prices
        yenprices_1min = yearlyprices(scenario,timeslots,enprices,stepperh_1min) # €/kWh
        ygridfees_1min = yearlyprices(scenario,timeslots,gridfees,stepperh_1min) # €/kWh
        yprices_1min = yenprices_1min + ygridfees_1min  # €/kWh
        admprice = (enprices[scenario][thresholdprice] + gridfees[scenario][thresholdprice])/1000
        admprices = np.where(yprices_1min <= admprice+0.001,1.,0.) #TODO fix this
    
        # Reducing prices adm time window as not to be on the final hedge of useful windows
        admcustom = np.ones(n1min)
        for i in range(n1min-60):
            if admprices[i]-admprices[i+60] == 1.:
                admcustom[i] = 0
    
        if WetAppManBool:
            admtimewin = admprices*admcustom*occupancy_1min
        else:
            admtimewin = admprices*admcustom
            
        # Admissible time window based on pv generation and residual load
        if PVBool:
            admtimewin_pv = (pv_1min_res > 0)
    
        """
        Shifting wet appliances
        """
        for app in WetAppShift:
        
            # Admissible time windows according to PV production
            # Adm starting times are when:
            # residual PV covers at least 90% of cycle consumption of the average cycle 
            
            threshold_window = defaults.threshold_window
            if PVBool:          # if there is PV, priority is given to self-consumption
                admtimewin = admtimewin_pv
                threshold_window = 0.9
        
            # Calling function to shift the app
            if defaults.verbose > 0:
                print("---"+str(app)+"---")
            app_n,ncyc,ncycshift,enshift = shift_appliance(demands['results'][idx][app][:-1],admtimewin,defaults.probshift,max_shift=defaults.max_shift*60,threshold_window=threshold_window,verbose=True)
            
            # Resizing shifted array
            app_n_15min = pd.Series(data=app_n,index=index1min).resample('15Min').mean().to_numpy()/1000. # kW
            
            # updating demand dataframe
            demand_shifted[app] = app_n_15min     # kW
            
            # Updating residual PV considering the consumption of the app just shifted
            if PVBool:  
                pv_1min_res = pv_1min_res - app_n/1000. # kW
                pv_15min_res = pd.Series(data=pv_1min_res,index=index1min).resample('15Min').mean() # kW
      
    """
    8B) Load shifting - DHW
    """
    
    
    if DHWBool:
        
        if defaults.verbose > 0:
            print('--- Shifting domestic hot water ---')
    
        # demand of domestic hot water (to be used with battery equivalent approach)
        demand_dhw = demand_15min['DomesticHotWater'] # kW
    
        # equivalent battery
        # TODO check these entries
        Vcyl = inputs['DHW']['Vcyl'] # litres
        Ttarget = inputs['DHW']['Ttarget'] # °C
        PowerDHWMax = inputs['DHW']['PowerElMax']/1000. # kW
    
        Tmin = defaults.T_min_dhw # °C
        Ccyl = Vcyl * 1000. /1000. * 4200. # J/K
        capacity = Ccyl*(Ttarget-Tmin)/3600./1000. # kWh
          
        param_tech_dhw = {'BatteryCapacity': capacity,
                          'BatteryEfficiency': 1.,
                          'MaxPower': PowerDHWMax,
                          'InverterEfficiency': 1.,
                          'timestep': .25,
                          'SelfDisLin': 0., # not used right now
                          'SelfDisFix':0.}  # not used right now
            
        if PVBool: # strategy based on enhancing self-consumption
    
            # prosumpy
            outs = dispatch_max_sc(pv_15min_res,demand_dhw,param_tech_dhw,return_series=False)
            demand_dhw_shift = outs['pv2store']+outs['pv2inv']-outs['inv2grid']+outs['grid2load'] # kW
            demand_dhw_shift = demand_dhw_shift.astype('float64') # kW
            
            # updating residual pv
            pv_15min_res = np.maximum(0,pv_15min_res-demand_dhw_shift) # kW
            pv_1min_res  = scale_timeseries(pv_15min_res,index1min) # kW 
            
        else: # strategy based on tariffs
            
            # prosumpy inspired tariffs based function
            admprice = (enprices[scenario][thresholdprice] + gridfees[scenario][thresholdprice])/1000
            outs = DHWShiftTariffs(demand_dhw, yprices_15min, admprice, param_tech_dhw, return_series=False)
            demand_dhw_shift = outs['grid2load']+outs['grid2store'] # kW
            demand_dhw_shift = demand_dhw_shift.astype('float64')   # kW
            
        # updating demand dataframe
        demand_shifted['DomesticHotWater'] = demand_dhw_shift     # kW
        
        # check on shifting
        conspre  = np.sum(demand_15min['DomesticHotWater'])/4. # kWh
        conspost = np.sum(demand_shifted['DomesticHotWater'])/4. # kWh
        
        if defaults.verbose > 0:
            print("Original consumption: {:.2f} kWh".format(conspre))
            print("Consumption after shifting (check): {:.2f} kWh".format(conspost))
            
    """
    8C) Load shifting - House heating
    """  
    
    # TODO
    # - save T inside house if needed to check how shift affects it
    # - T setpoint could be saved as done for dem, occ, etc., here recalculated
    # - harmonize ambient data used in all simulations
    #   use TMY obtained from PVGIS everywhere
    # - add fraction of max power when sizing HP
    # - revise heating season
    # - check how much a 15 min ts would affect the results
    # - to decide how to handle T increase
    
    if HeatingBool:
        
        if defaults.verbose > 0:
            print('--- Shifting house heating ---')
        
        temp,irr = load_climate_data()
        temp = np.delete(temp,-1) # °C
        irr = np.delete(irr,-1) # W/m2
        ts = 1/60               # Temporary, to be changed
        
        # internal gains
        Qintgains = demands['results'][idx]['InternalGains'][:-1].to_numpy() # W
    
        # T setpoint based on occupancy
        Tset = np.full(n1min,defaults.T_sp_low) + np.full(n1min,defaults.T_sp_occ-defaults.T_sp_low) * occupancy_1min
        
        # Heat pump sizing
        if inputs['HP']['HeatPumpThermalPower'] is not None:
            QheatHP = inputs['HP']['HeatPumpThermalPower']
        else:
            QheatHP = HPSizing(inputs,defaults.fracmaxP) # W
        
        if PVBool: # strategy based on enhancing self-consumption
        
            # Strategy here:
            # increasing setpoint T of Tincrease when:
            # residual PV > 0.
        
            Tset[pv_1min_res>0] += defaults.Tincrease
       
        else: # strategy based on tariffs
        
            # Strategy here:
            # increasing setpoint T of Tincrease when:
            # in the 3 hour time window before heating on
            # AND
            # tariffs are low
            
            # Hours with admissible prices
            yenprices_1min = yearlyprices(scenario,timeslots,enprices,stepperh_1min) # €/kWh
            ygridfees_1min = yearlyprices(scenario,timeslots,gridfees,stepperh_1min) # €/kWh
            yprices_1min = yenprices_1min + ygridfees_1min  # €/kWh
            admprice = (enprices[scenario][thresholdprice] + gridfees[scenario][thresholdprice])/1000
            admprices = np.where(yprices_1min <= admprice+0.01,1.,0.)
            
            # Hours close enough to when heating will be required
            offset = Tset.min()
            Tset = Tset - offset
            
            mask_z = Tset>0
            idx_z = np.flatnonzero(mask_z)
            idx_nz = np.flatnonzero(~mask_z)
            
            idx_z = np.r_[idx_z,len(Tset)]
            
            timeleftarr = np.zeros(len(Tset), dtype=int)
            idx_r = np.searchsorted(idx_z, idx_nz)
            timeleftarr[~mask_z] = idx_z[idx_r] - idx_nz
            
            admtimes = (timeleftarr<defaults.t_preheat*60)
            
            # Resulting hours in which to increase setpoint
            idx_tincrease = np.where(admprices*admtimes)[0]
            
            # Recalculating T setpoint array with increase
            Tset += offset
            Tset[idx_tincrease] += defaults.Tincrease
            
        
        res_househeat = HouseHeating(inputs,QheatHP,Tset,Qintgains,temp,irr,n1min,defaults.heatseas_st,defaults.heatseas_end,ts)
        Qshift = res_househeat['Qheat']
        Tin_shift = res_househeat['Tinside']
        
        # T analysis
        Twhenon    = Tin_shift*occupancy_1min.values # °C
        Twhenon_hs = Twhenon[np.r_[0:60*24*defaults.heatseas_end,60*24*defaults.heatseas_st:-1]] # °C
        whenon     = np.nonzero(Twhenon_hs)
        Twhenon_hs_mean = np.mean(Twhenon_hs[whenon]) # °C
        Twhenon_hs_min  = np.min(Twhenon_hs[whenon]) # °C
        Twhenon_hs_max  = np.max(Twhenon_hs[whenon]) # °C
        
        # Electricity consumption
        Eshift = np.zeros(n1min) 
        for i in range(n1min):
            COP = COP_deltaT(temp[i])
            Eshift[i] = Qshift[i]/COP # W
        
        # Updating demand dataframe
        demand_HP_shift = pd.Series(data=Eshift,index=index1min)
        demand_shifted['HeatPumpPower'] = demand_HP_shift.resample('15Min').mean().to_numpy()/1000. # kW
        
        # Check results
        HPcons_pre  = np.sum(demand_15min['HeatPumpPower'])/4. # kWh
        HPcons_post = np.sum(demand_shifted['HeatPumpPower'])/4. # kWh
        HPconsincr = (HPcons_post-HPcons_pre)/HPcons_pre*100 # %
        if defaults.verbose > 0:
            print("Original consumption: {:.2f} kWh".format(HPcons_pre))
            print("Consumption after shifting: {:.2f} kWh".format(HPcons_post))
            print("Consumption increase: {:.2f}%".format(HPconsincr))
        
        if PVBool:        
            # Updating residual PV
            pv_15min_res = np.maximum(0,pv_15min_res-demand_HP_shift) # kW
            pv_1min_res  = scale_timeseries(pv_15min_res,index1min) # kW 
        
            
    
    """
    8D) Load shifting - EV
    """
    
    #demand_shifted['EVCharging'] = demand_15min['EVCharging']  # temporary value to run the rest of the code        
    
    if EVBool:
        
        # Main driver - household member using the car
        MD = inputs['EV']['MainDriver']
        
        # Home charging profile
        charge_home = demands['results'][idx]['EVCharging']/1000. # kW
        charge_home = charge_home.to_numpy()
        
        # Occupancy of the main driver profile
        # 1 at home (active or inactive) 0 not at home
        occ_10min_MD = demands['occupancy'][idx].iloc[:,inputs['members'].index(MD)][:-1]
        occ_10min_MD = pd.Series(data=np.where(occ_10min_MD<3,1,0),index=index10min)
        occ_1min_MD  = occ_10min_MD.reindex(index1min,method='pad').to_numpy()
        
        # At-home time windows
        # Find arrival and departure times of MD from home
        
        # shift occupancy vector by one time step
        occ_1min_MD_s  = np.roll(occ_1min_MD,1)
        
        # locate all the points whit a start or a shutdown
        arriving_times_MD = (occ_1min_MD>0) * (occ_1min_MD_s==0)
        leaving_times_MD  = (occ_1min_MD_s>0) * (occ_1min_MD==0)
        
        # List the indexes of all start-ups and shutdowns
        arrive = np.where(arriving_times_MD)[0]
        leave  = np.where(leaving_times_MD)[0]
        
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
                
        #  Charging at-home time window: find starting and stopping charge times
        # Shift the app consumption vector by one time step:
        charge_home_s  = np.roll(charge_home,1)
        
        # locate all the points whit a start or a end
        starting_times_chhome = (charge_home>0) * (charge_home_s==0)
        stopping_times_chhome = (charge_home_s>0) * (charge_home==0)
        
        # List the indexes of all start and end charging
        starts_chhome = np.where(starting_times_chhome)[0]
        ends_chhome   = np.where(stopping_times_chhome)[0]
        
        # Consumptions when charging at home
        consumptions = np.zeros(len(starts_chhome))
        for i in range(len(starts_chhome)):
            consumptions[i] = np.sum(charge_home[starts_chhome[i]:ends_chhome[i]])/60
        
        # Finding in which at-home time windows each charging window is
        idx_athomewindows = np.zeros(len(starts_chhome),dtype=int)
        for i in range(len(starts_chhome)):
            idx_i = np.searchsorted(leave,[ends_chhome[i]-1],side='left')[0]
            idx_athomewindows[i] = idx_i
        
        # Minimum level of charge
        # LOC ramps when charging
        chargelen = ends_chhome - starts_chhome
        ramps = np.zeros(n1min) # kWh
        for i in range(len(starts_chhome)):
            add = np.linspace(0,consumptions[i],num=chargelen[i]+1)
            ramps[starts_chhome[i]:ends_chhome[i]] += add[1:]
        # LOC_min
        LOC_min_EV = ramps.copy()
        for i in range(len(starts_chhome)):
            LOC_min_EV[ends_chhome[i]:leave[idx_athomewindows[i]]] += ramps[ends_chhome[i]-1]
        
        # LOC_max
        LOC_max_EV = np.zeros(len(consumptions))
        oldidx = 0
        count = 0
        LOC_max_EV_t = 0
        
        for i in range(len(consumptions)): 
            if idx_athomewindows[i] == oldidx:
                LOC_max_EV_t += consumptions[i]
                count += 1
            else:
                LOC_max_EV_t = consumptions[i]
                count = 1
            oldidx = idx_athomewindows[i]
            LOC_max_EV[i+1-count:i+1] = LOC_max_EV_t
            
        
        # Define inputs for shifting function
        paramEVshift = {}
        paramEVshift['MaxPower'] = np.max(charge_home) # Pcharge
        paramEVshift['InverterEfficiency'] = 1.
        paramEVshift['timestep'] = 1/60
        
        if PVBool:
            out_EV = EVshift_PV(pv_1min_res,
                                arrive,leave,
                                starts_chhome,ends_chhome,
                                idx_athomewindows,
                                LOC_min_EV,LOC_max_EV,
                                paramEVshift,return_series=False)
            
            demand_EV_shift = out_EV['inv2store']+out_EV['grid2store']
            
        else:
            
            yenprices_1min = yearlyprices(scenario,timeslots,enprices,stepperh_1min) # €/kWh
            ygridfees_1min = yearlyprices(scenario,timeslots,gridfees,stepperh_1min) # €/kWh
            yprices_1min = yenprices_1min + ygridfees_1min  # €/kWh            
            yprices_1min = pd.Series(data=yprices_1min,index=index1min)
            pricelim = (enprices[scenario][thresholdprice] + gridfees[scenario][thresholdprice])/1000
            
            out_EV = EVshift_tariffs(yprices_1min,pricelim,
                                     arrive,leave,
                                     starts_chhome,ends_chhome,
                                     idx_athomewindows,
                                     LOC_min_EV,LOC_max_EV,
                                     paramEVshift,return_series=False)
            
            demand_EV_shift = out_EV['grid2store']

        demand_EV_shift = pd.Series(data=demand_EV_shift,index=index1min)
        demand_shifted['EVCharging'] = demand_EV_shift.resample('15Min').mean().to_numpy() # kW
        
        if PVBool:        
            # Updating residual PV
            pv_15min_res = np.maximum(0,pv_15min_res-demand_EV_shift) # kW
            pv_1min_res  = scale_timeseries(pv_15min_res,index1min) # kW 
            
  
    """
    8E) Final aggregated demand before battery shifting
    """

    # Saving demand profile obtained thanks to shifting techs
    # Not yet considering battery
    # Equal to pflows['demand_noshift'] if no shifting
    pflows['demand_shifted_nobatt'] = demand_shifted[TechsNoShift+TechsShift].sum(axis=1) # kW
    #demand_pspy.insert(len(demand_pspy.columns),'TotalShiftPreBattery',demand_prebatt.to_numpy(),True) # kW
    
       
    """
    8F) Load shifting - Standard battery
    """
    
    
    if BattBool:
        
        if defaults.verbose > 0:
            print('--- Shifting resulting demand with battery ---')
         
        # Battery applied to demand profile shifted by all other shifting techs
        param_tech_batt_pspy = pvbatt_param['battery']
        param_tech_batt_pspy['timestep']=.25
        
        dispatch_bat = dispatch_max_sc(pflows.pv,pflows.demand_shifted_nobatt,param_tech_batt_pspy,return_series=False)
        
        if defaults.verbose > 0:
            print_analysis(pflows.pv,pflows.demand_shifted_nobatt,param_tech_batt_pspy, dispatch_bat)
        
        # The charging of the battery is considered as an additional load. It is the difference between the original PV genration and the generation from prosumpy
        demand_shifted['BatteryConsumption'] = np.maximum(0,pflows.pv - dispatch_bat['inv2load'] - dispatch_bat['inv2grid'])
        
        # The discharge of the battery is considered as a negative load:
        demand_shifted['BatteryGeneration'] = np.minimum(0,pflows.pv - dispatch_bat['inv2load'] - dispatch_bat['inv2grid'])
        
        # Saving demand profile considering also battery shifting
        pflows['fromgrid'] = pd.Series(data=dispatch_bat['grid2load'],index=index15min) # kW
        pflows['togrid'] = pd.Series(data=dispatch_bat['inv2grid'],index=index15min) # kW
        
        pflows['demand_shifted'] = demand_shifted.sum(axis=1)
        
    else:
        
         pflows['demand_shifted'] = pflows['demand_shifted_nobatt']
  
    
    """
    9) Final analisys of the results (including economic analysis)
    """
    
    # TODO
    # - add how much energy has been shifted by each technology
    # - add to outs:
    #   - time horizon
    #   - energy prices
    #   - fixed and capacity-related tariffs
    
    outs = ResultsAnalysis(config_pv['Ppeak'],config_bat['BatteryCapacity'],config_inv['Ppeak'],pflows,yenprices_15min,ygridfees_15min,enprices,gridfees,scenario,econ_param)

    return outs,demand_15min,demand_shifted,pflows


if __name__ == '__main__':
    
    conf = load_config('default')
    config,pvbatt_param,econ_param,tariffs,housetype,N = conf['config'],conf['pvbatt_param'],conf['econ_param'],conf['tariffs'],conf['housetype'],conf['N']
    
    results,demand_15min,demand_shifted,pflows = shift_load(config,pvbatt_param,econ_param,tariffs,housetype,N)
    
    print(json.dumps(results, indent=4))
    