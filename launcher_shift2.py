
import os
import numpy as np
import pandas as pd
import json
import time
from prosumpy import dispatch_max_sc,print_analysis
from temp_functions import yearlyprices,HPSizing,COP_Tamb
from launcher_shift_functions import MostRepCurve,DHWShiftTariffs,HouseHeating,ResultsAnalysis,WriteResToExcel,load_climate_data
from temp_functions import shift_appliance,scale_timeseries
from pv import pvgis_hist
from demands import compute_demand
import defaults


start_time = time.time()


#%% Main simulation parameters

N = 1 # Number of stochastic simulations to be run for the demand curves

idx_casestobesim = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 22, 24, 26, 28, 30, 32,
        34, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 60, 62, 64, 66,
        68, 70, 72, 74, 76, 77, 78, 79, 80, 81, 82]
idx_casestobesim = [0]

#%% Loading inputs

# Case description
with open('inputs/cases.json','r') as f:
    cases = json.load(f)

# PV and battery technology parameters
with open('inputs/pvbatt_param.json','r') as f:
    pvbatt_param = json.load(f)

# Economic parameters
with open('inputs/econ_param.json','r') as f:
    econ_param = json.load(f)

# Time of use tariffs
with open('inputs/tariffs.json','r') as f:
    tariffs = json.load(f)

# Parameters for the dwelling
with open('inputs/housetypes.json','r') as f:
    housetypes = json.load(f)
        

for jjj in idx_casestobesim:
    namecase = 'case'+str(jjj+1)
    namecase = 'default'
    
    print('###########################')
    print('   Simulating: '+ namecase )
    print('###########################')
    
    house          = cases[namecase]['house']
    sheet          = cases[namecase]['sheet']
    row            = cases[namecase]['row']
    columns        = cases[namecase]['columns'] 
    TechsShift     = cases[namecase]['TechsShift']
    WetAppShift    = [x for x in TechsShift if x in ['TumbleDryer','DishWasher','WashingMachine']]
    TechsNoShift   = [x for x in columns if x not in TechsShift]
    WetAppBool     = cases[namecase]['WetAppBool']
    WetAppManBool  = cases[namecase]['WetAppManBool']
    WetAppAutoBool = cases[namecase]['WetAppAutoBool']
    PVBool         = cases[namecase]['PVBool']
    BattBool       = cases[namecase]['BattBool']
    DHWBool        = cases[namecase]['DHWBool']
    HeatingBool    = cases[namecase]['HeatingBool']
    EVBool         = cases[namecase]['EVBool']
    
    FixedControl   = econ_param[namecase]['FixedControlCost']
    AnnualControl  = econ_param[namecase]['AnnualControlCost']
    thresholdprice = econ_param[namecase]['thresholdprice']
    

    inputs = housetypes[house]
    demands = compute_demand(inputs,N,inputs['members'],inputs['thermal_parameters'])
    
    config_pv = pvbatt_param['pv']
    
    config_bat = pvbatt_param['battery']
    
    pvadim = pvgis_hist(config_pv)  
    
    # Various array sizes and timesteps used throughout the code
    index1min  = pd.date_range(start='2015-01-01',end='2015-12-31 23:59:00',freq='T')
    index15min = pd.date_range(start='2015-01-01',end='2015-12-31 23:45:00',freq='15T')
    n1min  = len(index1min)
    n10min = int(n1min/10)
    n15min = int(n1min/15)
    stepperh_1min = 60 # 1/h
    stepperh_15min = 4 # 1/h
    ts_15min = 0.25 # h
    
    #%%
    # Electricity prices array - 15 min timestep
    scenario = econ_param[namecase]['scenario']
    timeslots = tariffs['timeslots']
    prices = tariffs['prices']
    yprices_15min = yearlyprices(scenario,timeslots,prices,stepperh_15min) # €/kWh
    
    #%%
    
    """
    3) Most representative curve
    """
    
    idx = MostRepCurve(demands['results'],columns,yprices_15min,ts_15min,econ_param[namecase])
    
    # Inputs relative to the most representative curve:
    inputs = demands['input_data'][idx]
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
    demand_shifted = demand_15min
    
    # Reference demand
    # Aggregated demand pre-shifting
    demand_ref = demand_15min.sum(axis=1).to_numpy() # kW
    ydemand = np.sum(demand_ref)/4
    #demand_15min.insert(len(demand_pspy.columns),'TotalReference',demand_ref,True)
    
    
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
            # Sizing
            pvpeak = ydemand/yield_pv  # kWp
        else:
            pvpeak = config_pv['Ppeak']
        # 15 min timestep series
        pv_15min = pvadim * pvpeak # kW
        # 1 min timestep array
        pv_1min = scale_timeseries(pv_15min,index1min)   # kW
                
        # Residual PV
        # 15 min timestep series
        dem_15min_noshift = demand_15min[TechsNoShift].sum(axis=1) # kW
        pv_15min_res = np.maximum(0,pv_15min - dem_15min_noshift) # kW
        # 1 min timestep array
        demnoshift = demands['results'][idx][TechsNoShift].sum(axis=1)[:-1].to_numpy()/1000. # kW
        pv_1min_res = np.maximum(0,pv_1min-demnoshift) # kW 
      
    else:
        pv_15min = pd.Series(0,index=index15min) # kW
        pvpeak = 0. # kWp
    
    # Update PV capacity
    pvbatt_param['pv']['Ppeak'] = pvpeak # kWp
    
    """
    7) Battery size
    """
    
    if not BattBool:
        pvbatt_param['battery']['BatteryCapacity'] = 0. # kWh
    	   
    
    """
    8) Shifting
    """
    
    """
    8A) Load shifting - Wet appliances
        NB for the shifting of the appliances we work with 1-min timestep and power in W   
    """
    
    if WetAppBool:
        
        print('--- Shifting wet appliances ---')
        
        # Wet app demands to be shifted, 1 min timestep
        demshift = demands['results'][idx][WetAppShift][:-1] # W
    
        """
        Admissible time windows
        """
        
        # Admissible time window based on electricity prices
        yprices_1min = yearlyprices(scenario,timeslots,prices,stepperh_1min) # €/kWh
        admprices = np.where(yprices_1min <= prices[scenario][thresholdprice]/1000,1.,0.)
    
        # Reducing prices adm time window as not to be on the final hedge of useful windows
        admcustom = np.ones(n1min)
        for i in range(n1min-60):
            if admprices[i]-admprices[i+60] == 1.:
                admcustom[i] = 0
    
        if WetAppManBool:
            admtimewin = admprices*admcustom*occupancy_1min
        
        if WetAppAutoBool:
            admtimewin = admprices*admcustom
            
        # Admissible time window based on pv generation and residual load
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
            outs = DHWShiftTariffs(demand_dhw, yprices_15min, prices[scenario][thresholdprice], param_tech_dhw, return_series=False)
            demand_dhw_shift = outs['grid2load']+outs['grid2store'] # kW
            demand_dhw_shift = demand_dhw_shift.astype('float64')   # kW
            
        # updating demand dataframe
        demand_shifted['DomesticHotWater'] = demand_dhw_shift     # kW
        
        # check on shifting
        conspre  = np.sum(demand_15min['DomesticHotWater'])/4. # kWh
        conspost = np.sum(demand_shifted['DomesticHotWater'])/4. # kWh
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
        
        print('--- Shifting house heating ---')
        
        temp,irr = load_climate_data()
        temp = np.delete(temp,-1) # °C
        irr = np.delete(irr,-1) # W/m2
        
        # internal gains
        Qintgains = demands['results'][idx]['InternalGains'][:-1].to_numpy() # W
    
        # T setpoint based on occupancy
        Tset = np.full(n1min,defaults.T_sp_low) + np.full(n1min,defaults.T_sp_occ-defaults.T_sp_low) * occupancy_1min
        
        # Heat pump sizing
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
            yprices_1min = yearlyprices(scenario,timeslots,prices,stepperh_1min) # €/kWh
            admprices = np.where(yprices_1min <= prices[scenario][thresholdprice]/1000,1.,0.)
            
            # Hours close enough to when heating will be required
            offset = Tset.min()
            Tset = Tset - offset
            
            mask_z = Tset>0
            idx_z = np.flatnonzero(mask_z)
            idx_nz = np.flatnonzero(~mask_z)
            
            idx_z = np.r_[idx_z,len(Tset)]
            
            timeleftarr = np.zeros(len(Tset), dtype=int)
            idx = np.searchsorted(idx_z, idx_nz)
            timeleftarr[~mask_z] = idx_z[idx] - idx_nz
            
            admtimes = (timeleftarr<defaults.t_preheat*60)
            
            # Resulting hours in which to increase setpoint
            idx_tincrease = np.where(admprices*admtimes)[0]
            
            # Recalculating T setpoint array with increase
            Tset += offset
            Tset[idx_tincrease] += defaults.Tincrease
            
        
        Qshift,Tin_shift = HouseHeating(inputs,QheatHP,Tset,Qintgains,temp,irr,n1min,defaults.heatseas_st,defaults.heatseas_end)
        
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
            COP = COP_Tamb(temp[i])
            Eshift[i] = Qshift[i]/COP # W
        
        # Updating demand dataframe
        demand_HP_shift = pd.Series(data=Eshift,index=index1min)
        demand_shifted['HeatPumpPower'] = demand_HP_shift.resample('15Min').mean().to_numpy()/1000. # kW
        
        # Check results
        HPcons_pre  = np.sum(demand_15min['HeatPumpPower'])/4. # kWh
        HPcons_post = np.sum(demand_shifted['HeatPumpPower'])/4. # kWh
        HPconsincr = (HPcons_post-HPcons_pre)/HPcons_pre*100 # %
        print("Original consumption: {:.2f} kWh".format(HPcons_pre))
        print("Consumption after shifting: {:.2f} kWh".format(HPcons_post))
        print("Consumption increase: {:.2f}%".format(HPconsincr))
        
        if PVBool:        
            # Updating residual PV
            pv_15min_res = np.maximum(0,pv_15min_res-demand_HP_shift) # kW
            #pv_1min_res  = scale_timeseries(pv_15min_res,index1min) # kW 
        
            
    
    """
    8D) Load shifting - EV
    """
    
    #TODO
    #demand_shifted['EVCharging'] = demand_15min['EVCharging']              # temporary value to run the rest of the code
    
    """
    8E) Final aggregated demand before battery shifting
    """

    # Saving demand profile obtained thanks to shifting techs
    # Not yet considering battery
    # Equal to demand_ref if no shifting
    demand_prebatt = demand_shifted[TechsNoShift+TechsShift].sum(axis=1) # kW
    #demand_pspy.insert(len(demand_pspy.columns),'TotalShiftPreBattery',demand_prebatt.to_numpy(),True) # kW
    
       
    """
    8F) Load shifting - Standard battery
    """
    
    
    if BattBool:
        
        print('--- Shifting resulting demand with battery ---')
         
        # Battery applied to demand profile shifted by all other shifting techs
        param_tech_batt_pspy = pvbatt_param['battery']
        param_tech_batt_pspy['timestep']=.25
        
        dispatch_bat = dispatch_max_sc(pv_15min,demand_prebatt,param_tech_batt_pspy,return_series=False)
        print_analysis(pv_15min,demand_prebatt,param_tech_batt_pspy, dispatch_bat)
        
        # The charging of the battery is considered as an additional load. It is the difference between the original PV genration and the generation from prosumpy
        demand_shifted['BatteryConsumption'] = np.maximum(0,pv_15min - dispatch_bat['inv2load'] - dispatch_bat['inv2grid'])
        
        # The discharge of the battery is considered as a negative load:
        demand_shifted['BatteryGeneration'] = np.minimum(0,pv_15min - dispatch_bat['inv2load'] - dispatch_bat['inv2grid'])
        
        # Saving demand profile considering also battery shifting
        demand_fromgrid = pd.Series(data=dispatch_bat['grid2load'],index=index15min) # kW
        #demand_pspy.insert(len(demand_pspy.columns),'TotalShiftPostBattery',demand_postbatt,True) # kW
            
    
    """
    9) Final analisys of the results (including economic analysis)
    """
    
    # TODO
    # - add how much energy has been shifted by each technology
    # - add to outs:
    #   - time horizon
    #   - energy prices
    #   - fixed and capacity-related tariffs
    
    outs = ResultsAnalysis(pvbatt_param['pv']['Ppeak'],pvbatt_param['battery']['BatteryCapacity'],pv_15min,demand_ref,demand_fromgrid,yprices_15min,prices,scenario,econ_param[namecase])
    
    
    """
    10) Saving results to Excel
    """
    # TODO
    #   - add column with time horizion EconomicVar['time_horizon']
    #   - add columns with el prices
    #   - add columns with capacity-related prices
    #   - add in previous passages overall electricity shifted (right here set to 0)
    
    ###### TEMP ########
    outs['el_shifted'] = 0. 
    
    # Saving results to excel
    file = 'simulations/test'+house+'.xlsx'
    WriteResToExcel(file,sheet,outs,econ_param[namecase],prices[scenario],row)


exectime = (time.time() - start_time)
print('It all took {:.1f} seconds'.format(exectime))




















