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
from prosumpy import dispatch_max_sc,print_analysis
from functions import HPSizing,COP_deltaT
from functions import scale_vector,MostRepCurve,DHWShiftTariffs,HouseHeating,EVshift_PV,EVshift_tariffs,ResultsAnalysis,load_climate_data
from readinputs import read_config
from functions import shift_appliance,scale_timeseries
from pv import pvgis_hist
from demands import compute_demand
import defaults

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

from joblib import Memory
memory = Memory(__location__ + '/cache/', verbose=defaults.verbose)

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

def list_appliances(conf):
    '''
    Small function that generates a list of appliances present in the household.
    If they are shiftable, a second list is returned with the flexible appliances
    '''
    techs = ['StaticLoad']
    techshift = []
    if conf['dwelling']['washing_machine']:
        techs.append('WashingMachine')
        if conf['cont']['wetapp'] in ['automated','manual']:
            techshift.append('WashingMachine')
    if conf['dwelling']['tumble_dryer']:
        techs.append('TumbleDryer')
        if conf['cont']['wetapp'] in ['automated','manual']:
            techshift.append('TumbleDryer')
    if conf['dwelling']['dish_washer']:
        techs.append('DishWasher')
        if conf['cont']['wetapp'] in ['automated','manual']:
            techshift.append('DishWasher')
    if conf['hp']['yesno']:
        techs.append('HeatPumpPower')
        if conf['hp']['loadshift']:
            techshift.append('HeatPumpPower')
    if conf['dhw']['yesno']:
        techs.append('DomesticHotWater')
        if conf['dhw']['loadshift']:
            techshift.append('DomesticHotWater')
    if conf['ev']['yesno']:
        techs.append('EVCharging')
        if conf['ev']['loadshift']:
            techshift.append('EVCharging')
    return techs,techshift
    

@memory.cache
def shift_load(conf,prices):
    '''
    
    Parameters
    ----------
    conf : dict
        Pre-defined simulation case to be simulated.

    Returns
    -------
    outs : dict
        Main simulation results.

    '''

    columns,TechsShift = list_appliances(conf)
    WetAppShift    = [x for x in TechsShift if x in ['TumbleDryer','DishWasher','WashingMachine']]
    TechsNoShift   = [x for x in columns if x not in TechsShift]
    WetAppBool     = len(WetAppShift)>0
    WetAppManBool  = conf['cont']['wetapp'] == 'manual'
    PVBool         = conf['pv']['yesno']
    BattBool       = conf['batt']['yesno']
    DHWBool        = "DomesticHotWater" in TechsShift
    HeatingBool    = "HeatPumpPower" in TechsShift
    EVBool         = "EVCharging" in TechsShift
    
    thresholdprice = conf['cont']['thresholdprice']
    
    #demands = compute_demand(conf,N,inputs['members'],inputs['thermal_parameters'])
    demands = compute_demand(conf)
    
    pvadim = pvgis_hist(conf['pv'],conf['loc'])  
    
    # Various array sizes and timesteps used throughout the code
    index1min  = pd.date_range(start='2015-01-01',end='2015-12-31 23:59:00',freq='T')
    index10min = pd.date_range(start='2015-01-01',end='2015-12-31 23:50:00',freq='10T')
    index15min = pd.date_range(start='2015-01-01',end='2015-12-31 23:45:00',freq='15T')
    n1min  = len(index1min)
    n15min = int(n1min/15)
    ts_15min = 0.25 # h
           
    yenprices_15min = scale_vector(prices['energy'].to_numpy(),len(index15min),silent=True) # €/kWh
    ygridfees_15min = scale_vector(prices['grid'].to_numpy(),len(index15min),silent=True)  # €/kWh
    yprices_15min = yenprices_15min + ygridfees_15min  # €/kWh
    
    
    """
    3) Most representative curve
    """
    
    idx = MostRepCurve(conf,prices,demands['results'],columns,ts_15min)
    
    # Inputs relative to the most representative curve:
    conf = demands['input_data'][idx]              # this overwrites the conf dictionnary passed as argument!!
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
    
        if conf['pv']['automatic_sizing']:
            yield_pv = pvadim.sum()/4
            pvpeak = min(ydemand/yield_pv,conf['pv']['plim_kva']) # kWp  TODO: check thi!
        else:
            pvpeak = conf['pv']['ppeak']
            
        if conf['pv']['automatic_sizing']:
            inv_lim = conf['pv']['plim_kva'] * conf['pv']['powerfactor'] # kW max inv power
            invpeak = min(pvpeak/1.2,inv_lim)
        else:
            invpeak = conf['pv']['inverter_pmax']
            
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
    conf['pv']['ppeak'] = pvpeak # kWp
    
    # Update inverter capacity
    conf['pv']['inverter_pmax'] = invpeak # kWi
    
    """
    7) Battery size
    """
    
    if not BattBool:
        print('No battery!!')
        conf['batt']['capacity'] = 0. # kWh
    	   
    
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
    
        """
        Admissible time windows
        """
        
        # Admissible time window based on electricity prices
        yprices_1min   = pd.Series(index=index15min,data=yprices_15min).reindex(index1min,method='ffill').to_numpy() # €/kWh
        admprices = np.where(yprices_1min <= thresholdprice+0.001,1.,0.) #TODO fix this
    
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
        Vcyl = conf['dhw']['vol'] # litres
        Ttarget = conf['dhw']['set_point'] # °C
        PowerDHWMax = conf['dhw']['pnom']/1000. # kW
    
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
            outs = DHWShiftTariffs(demand_dhw, yprices_15min, thresholdprice, param_tech_dhw, return_series=False)
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
    # - harmonize ambient data used in all simulations
    #   use TMY obtained from PVGIS everywhere
    # - revise heating season
    # - to decide how to handle T increase
    
    if HeatingBool:
        
        if defaults.verbose > 0:
            print('--- Shifting house heating ---')
        
        temp,irr = load_climate_data()
        temp = pd.Series(data=temp[:-1],index=index1min)
        temp15min = temp.resample('15Min').mean()
        irr = pd.Series(data=irr[:-1],index=index1min)
        irr15min = irr.resample('15Min').mean()
        
        ts = 1/4
        
        # internal gains
        Qintgains = demands['results'][idx]['InternalGains'][:-1].to_numpy() # W
    
        # T setpoint based on occupancy
        Tset = np.full(n15min,defaults.T_sp_low) + np.full(n15min,defaults.T_sp_occ-defaults.T_sp_low) * occupancy_15min
        
        # Heat pump sizing
        if conf['hp']['pnom'] is not None:
            QheatHP = conf['hp']['pnom']
        else:
            QheatHP = HPSizing(conf['BuildingEnvelope'],defaults.fracmaxP) # W   conf['BuildingEnvelope'] should was defined in the function compute_demand
        
        if PVBool: # strategy based on enhancing self-consumption
        
            # Strategy here:
            # increasing setpoint T of Tincrease when:
            # residual PV > 0.
        
            Tset[pv_15min_res>0] += defaults.Tincrease
       
        else: # strategy based on tariffs
        
            # Strategy here:
            # increasing setpoint T of Tincrease when:
            # in the 3 hour time window before heating on
            # AND
            # tariffs are low
            
            # Hours with admissible prices
            admprices = np.where(yprices_15min <= thresholdprice+0.01,1.,0.)
            
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
            
            # admtimes = (timeleftarr<defaults.t_preheat*int(1/ts))
            admtimes = [1. if 0<a<defaults.t_preheat*int(1/ts) else 0. for a in timeleftarr]
            admtimes = np.array(admtimes)
            
            # Resulting hours in which to increase setpoint
            idx_tincrease = np.where(admprices*admtimes)[0]
            
            # Recalculating T setpoint array with increase
            Tset += offset
            Tset[idx_tincrease] += defaults.Tincrease
        
        res_househeat = HouseHeating(conf['BuildingEnvelope'],QheatHP,Tset,Qintgains,temp15min,irr15min,n15min,defaults.heatseas_st,defaults.heatseas_end,ts)
        Qshift = res_househeat['Qheat']
        
        # Electricity consumption
        Eshift = np.zeros(n15min) 
        for i in range(n15min):
            COP = COP_deltaT(temp15min[i])
            Eshift[i] = Qshift[i]/COP # W
        
        # Updating demand dataframe
        demand_shifted['HeatPumpPower'] = Eshift/1000. # kW
        
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
            pv_15min_res = np.maximum(0,pv_15min_res- demand_shifted['HeatPumpPower']) # kW
            pv_1min_res  = scale_timeseries(pv_15min_res,index1min) # kW 
        
            
    
    """
    8D) Load shifting - EV
    """
        
    if EVBool:
        
        # Main driver - household member using the car
        MD = conf['ev']['MainDriver']       # this was defined in the function compute_demand
        
        # Home charging profile
        charge_home = demands['results'][idx]['EVCharging']/1000. # kW
        charge_home = charge_home.to_numpy()
        
        # Occupancy of the main driver profile
        # 1 at home (active or inactive) 0 not at home
        occ_10min_MD = demands['occupancy'][idx].iloc[:,conf['members'].index(MD)][:-1]
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
                
        # Charging at-home time window: find starting and stopping charge times
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
            
            yprices_1min   = pd.Series(index=index15min,data=yprices_15min).reindex(index1min,method='ffill').to_numpy() # €/kWh
            
            out_EV = EVshift_tariffs(yprices_1min,thresholdprice,
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
    
    
    """
    8F) Load shifting - Standard battery
    """
    
    
    if BattBool:
        
        if defaults.verbose > 0:
            print('--- Shifting resulting demand with battery ---')
         
        # Battery applied to demand profile shifted by all other shifting techs
        param_tech_batt_pspy = {'BatteryCapacity':conf['batt']['capacity'],
                                'BatteryEfficiency':conf['batt']['efficiency'],
                                'InverterEfficiency':1,
                                'timestep':conf['sim']['ts'],
                                'MaxPower':conf['batt']['pnom']
            
            }
        
        dispatch_bat = dispatch_max_sc(pflows.pv,pflows.demand_shifted_nobatt,param_tech_batt_pspy,return_series=False)
        
        if defaults.verbose > 0:
            print_analysis(pflows.pv,pflows.demand_shifted_nobatt,param_tech_batt_pspy, dispatch_bat)
        
        # The charging of the battery is considered as an additional load. It is the difference between the original PV genration and the generation from prosumpy
        demand_shifted['BatteryConsumption'] = np.maximum(0,pflows.pv - dispatch_bat['inv2load'] - dispatch_bat['inv2grid'])
        
        # The discharge of the battery is considered as a negative load:
        demand_shifted['BatteryGeneration'] = np.minimum(0,pflows.pv - dispatch_bat['inv2load'] - dispatch_bat['inv2grid'])
        
        pflows['demand_shifted'] = demand_shifted.sum(axis=1)
        # Saving demand profile considering also battery shifting
        pflows['fromgrid'] = pd.Series(data=dispatch_bat['grid2load'],index=index15min) # kW
        pflows['togrid'] = pd.Series(data=dispatch_bat['inv2grid'],index=index15min) # kW
        pflows['BatteryConsumption'] = demand_shifted['BatteryConsumption']
        pflows['BatteryGeneration'] = demand_shifted['BatteryGeneration']
        
    else:
        pflows['demand_shifted'] = pflows['demand_shifted_nobatt']
        pflows['fromgrid'] = np.maximum(0,pflows['demand_shifted_nobatt'] - pflows['pv'])
        pflows['togrid'] = np.maximum(0,-pflows['demand_shifted_nobatt'] + pflows['pv'])
        pflows['BatteryConsumption'] = pd.Series(0,index=index15min)
        pflows['BatteryGeneration'] = pd.Series(0,index=index15min)    

  
    
    """
    9) Final analysis of the results (including economic analysis)
    """
    
    # TODO
    # - add how much energy has been shifted by each technology
    
    results,econ = ResultsAnalysis(conf,prices,pflows)

    return results,demand_15min,demand_shifted,pflows


if __name__ == '__main__':
    
    conf,prices,config_full = read_config(__location__ + '/inputs/config.xlsx')
    
    # delete unnecessary entries:
    results,demand_15min,demand_shifted,pflows = shift_load(conf,prices)
    
    print(results)
    
    # plotting the results
    #from plots import make_demand_plot, make_pflow_plot
    # fig = make_demand_plot(demand_15min.index,demand_shifted,PV = pflows.pv,title='Consumption and generation')
    # fig.show()
    #fig2 = make_pflow_plot(demand_15min.index,pflows)
    #fig2.show()
 
    
    
    