
import os
import numpy as np
import pandas as pd
import json
import time
from prosumpy import dispatch_max_sc,print_analysis
from temp_functions import yearlyprices,HPSizing,COP_Tamb
from launcher_shift_functions import MostRepCurve,DHWShiftTariffs,HouseHeating,ResultsAnalysis,WriteResToExcel
from temp_functions import shift_appliance
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
    
    print('###########################')
    print('        Case'+str(jjj+1))
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
    
    
    # Demands
    # with open('inputs/' + house+'.json') as f:
    #     inputs = json.load(f)
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
    
    demand_pspy = demands['results'][idx][columns]/1000. # kW
    demand_pspy = demand_pspy.resample('15Min').mean()[:-1] # kW
    
    # Reference demand
    # Aggregated demand pre-shifting
    
    demand_ref = demand_pspy.sum(axis=1).to_numpy() # kW
    ydemand = np.sum(demand_ref)/4
    demand_pspy.insert(len(demand_pspy.columns),'TotalReference',demand_ref,True)
    
    
    """
    5) Occupancy
    """
    
    # Occupancy array, built checking if at least one active person at home
    # 1-yes 0-no
    # 1 min timestep
    
    occ = np.zeros(n10min)
    for i in range(len(demands['occupancy'][idx])):
        singlehouseholdocc = [1 if a==1 else 0 for a in demands['occupancy'][idx][i][:-1]]
        occ += singlehouseholdocc
    occ = [1 if a >=1 else 0 for a in occ]    
    occupancy = np.zeros(n1min)
    for i in range(n10min):
        for j in range(10):
            occupancy[i*10+j] = occ[i]
    
    
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
        pv_1min = pv_15min.resample('T').pad().reindex(index1min,method='nearest').to_numpy() # kW
                
        # Residual PV
        # 15 min timestep series
        dem_15min_noshift = demand_pspy[TechsNoShift].sum(axis=1) # kW
        pv_15min_res = [a if a>0 else 0 for a in (pv_15min.to_numpy() - dem_15min_noshift.to_numpy())] # kW
        pv_15min_res = pd.Series(data=pv_15min_res,index=index15min) # kW
        # 1 min timestep array
        demnoshift = demands['results'][idx][TechsNoShift].sum(axis=1)[:-1].to_numpy()/1000. # kW
        pv_1min_res = [a if a>0. else 0. for a in pv_1min-demnoshift] # kW 
        pv_1min_res = np.array(pv_1min_res) # kW
      
    else:
        pv_15min = np.zeros(n15min) # kW
        pv_15min = pd.Series(data=pv_15min,index=index15min) # kW
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
            admtimewin = admprices*admcustom*occupancy
        
        if WetAppAutoBool:
            admtimewin = admprices*admcustom
    
        """
        Shifting wet appliances
        """
        for app in WetAppShift:
        
            # Admissible time windows according to PV production
            # Adm starting times are when:
            # residual PV covers at least 90% of cycle consumption of the longest cycle of the year
            
            if PVBool:
                
                admpv = np.zeros(len(demnoshift))
                
                apparr = demshift[app].to_numpy() # W
                app_s  = np.roll(apparr,1)
                starts = np.where(apparr-app_s>1)[0]
                ends   = np.where(apparr-app_s<-1)[0]
                lengths = ends-starts
                maxcyclen = np.max(lengths)            
                cons = np.ones(maxcyclen)*apparr[starts[0]+1] # W
                sumcons = np.sum(cons)*ts_15min # Wh
                
                for i in range(len(demshift)-maxcyclen*60):
                    
                    prod = pv_1min_res[i:i+maxcyclen]*1000. # W
                    diff = np.array(cons-prod) # W
                    diff = np.where(diff>=0,diff,0)
                    notcov = np.sum(diff)*ts_15min # Wh
                    share = 1-notcov/sumcons
                    if share >= 0.9:
                        admpv[i]=1
                
                # Updating admissibile time windows
                admtimewin = admtimewin + admpv*occupancy
        
            # Calling function to shift the app
            print("---"+str(app)+"---")
            #app_n,enshift = AdmTimeWinShift(demands['results'][idx][app][:-1],admtimewin,probshift) # W, Wh
            app_n,ncyc,ncycshift,enshift = shift_appliance(demands['results'][idx][app][:-1],admtimewin,defaults.probshift,max_shift=24*60)
            
            # Resizing shifted array
            app_n_series = pd.Series(data=app_n,index=index1min) # W
            app_n_15min = app_n_series.resample('15Min').mean().to_numpy()/1000. # kW
            
            # updating demand dataframe
            demand_pspy.insert(len(demand_pspy.columns),app+'Shift',app_n_15min,True) # kW
            
            # Updating residual PV considering the consumption of the app just shifted
            if PVBool:  
                pv_1min_res = pv_1min_res - app_n/1000. # kW
                pv_1min_res_series = pd.Series(data=pv_1min_res,index=index1min) # kW
                pv_15min_res = pv_1min_res_series.resample('15Min').mean() # kW
      
    """
    8B) Load shifting - DHW
    """
    
    
    if DHWBool:
    
        print('--- Shifting domestic hot water ---')
    
        # demand of domestic hot water (to be used with battery equivalent approach)
        demand_dhw = demand_pspy['DomesticHotWater'] # kW
    
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
            demand_dhw_shift = outs['pv2store']+outs['grid2load'] # kW
            demand_dhw_shift = demand_dhw_shift.astype('float64') # kW
            
            # updating residual pv
            pv_15min_res = [a if a>0 else 0 for a in (pv_15min_res.to_numpy()-demand_dhw_shift)] # kW
            pv_15min_res = pd.Series(data=pv_15min_res,index=index15min) # kW
            pv_1min_res  = pv_15min_res.resample('T').pad().reindex(index1min,method='nearest').to_numpy() # kW 
            
        else: # strategy based on tariffs
            
            # prosumpy inspired tariffs based function
            outs = DHWShiftTariffs(demand_dhw, yprices_15min, prices[scenario][thresholdprice], param_tech_dhw, return_series=False)
            demand_dhw_shift = outs['grid2load']+outs['grid2store'] # kW
            demand_dhw_shift = demand_dhw_shift.astype('float64')   # kW
            
        # updating demand dataframe
        demand_pspy.insert(len(demand_pspy.columns),'DomesticHotWaterShift',demand_dhw_shift,True) # kW
        
        # check on shifting
        conspre  = np.sum(demand_pspy['DomesticHotWater'])/4. # kWh
        conspost = np.sum(demand_pspy['DomesticHotWaterShift'])/4. # kWh
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
        
        # ambient data
        datapath = r'./strobe/Data'
        temp = np.loadtxt(datapath + '/Climate/temperature.txt')
        irr  = np.loadtxt(datapath + '/Climate/irradiance.txt')   
        temp = np.delete(temp,-1) # °C
        irr = np.delete(irr,-1) # W/m2
        
        # internal gains
        Qintgains = demands['results'][idx]['InternalGains'][:-1].to_numpy() # W
    
        # T setpoint based on occupancy
        Tset = [20. if a == 1 else 15. for a in occupancy] # °C
        Tset = np.array(Tset)
        
        # Heat pump sizing
        fracmaxP = 0.8
        QheatHP = HPSizing(inputs,fracmaxP) # W
        
        # Heating season
        heatseas_st = 244
        heatseas_end = 151
        
        if PVBool: # strategy based on enhancing self-consumption
        
            # Strategy here:
            # increasing setpoint T of Tincrease when:
            # residual PV > 0.
        
            Tincrease = 3.    
            Tset[pv_1min_res>0] += Tincrease
       
        else: # strategy based on tariffs
        
            # Strategy here:
            # increasing setpoint T of Tincrease when:
            # in the 3 hour time window before heating on
            # AND
            # tariffs are low
        
            # Shifting based on tariffs-specific inputs
            Tincrease = 3. # °C T increase wrt min T setpoint (heating off)
            t_preheat = 3  # h max time allowed to consider pre-heating
            t_preheat_min = t_preheat*60
            
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
            
            admhours = [1. if 0<a<t_preheat_min else 0. for a in timeleftarr]
            admhours = np.array(admhours)
            
            # Resulting hours in which to increase setpoint
            idx = np.where(admprices*admhours)
            
            # Recalculating T setpoint array with increase
            Tset += offset
            Tset[idx] += Tincrease
            
        
        Qshift,Tin_shift = HouseHeating(inputs,QheatHP,Tset,Qintgains,temp,irr,n1min,heatseas_st,heatseas_end)
        
        # T analysis
        Twhenon    = Tin_shift*occupancy # °C
        Twhenon_hs = Twhenon[np.r_[0:60*24*heatseas_end,60*24*heatseas_st:-1]] # °C
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
        demand_HP_shift = pd.Series(data=Eshift,index=pd.date_range(start='2015-01-01',end='2015-12-31 23:59:00',freq='min'))
        demand_HP_shift = demand_HP_shift.resample('15Min').mean().to_numpy()/1000. # kW
        demand_pspy.insert(len(demand_pspy.columns),'HeatPumpPowerShift',demand_HP_shift,True) #kW
        
        # Check results
        HPcons_pre  = np.sum(demand_pspy['HeatPumpPower'])/4. # kWh
        HPcons_post = np.sum(demand_pspy['HeatPumpPowerShift'])/4. # kWh
        HPconsincr = (HPcons_post-HPcons_pre)/HPcons_pre*100 # %
        print("Original consumption: {:.2f} kWh".format(HPcons_pre))
        print("Consumption after shifting: {:.2f} kWh".format(HPcons_post))
        print("Consumption increase: {:.2f}%".format(HPconsincr))
        
        if PVBool:        
            # Updating residual PV
            pv_15min_res = [a if a>0 else 0 for a in (pv_15min_res.to_numpy()-demand_HP_shift)] # kW
            pv_15min_res = pd.Series(data=pv_15min_res,index=index15min) # kW
            pv_1min_res  = pv_15min_res.resample('T').pad().reindex(index1min,method='nearest').to_numpy() # kW 
        
            
    
    """
    8D) Load shifting - EV
    """
    
    #TODO
    
    """
    8E) Final aggregated demand before battery shifting
    """
    
    # Columns to be considered for the final demand profile
    finalcols = TechsNoShift
    for app in TechsShift:
        finalcols.append(app+'Shift')
    
    # Saving demand profile obtained thanks to shifting techs
    # Not yet considering battery
    # Equal to demand_ref if no shifting
    demand_prebatt = demand_pspy[finalcols].sum(axis=1) # kW
    demand_pspy.insert(len(demand_pspy.columns),'TotalShiftPreBattery',demand_prebatt.to_numpy(),True) # kW
    
    # If no battery this demand profile is the definitive one
    demand_final = demand_prebatt # kW
       
    """
    8F) Load shifting - Standard battery
    """
    
    
    if BattBool:
        
        print('--- Shifting resulting demand with battery ---')
         
        # Battery applied to demand profile shifted by all other shifting techs
        param_tech_batt_pspy = {'BatteryCapacity': pvbatt_param['BatteryCapacity'],
                                'BatteryEfficiency': pvbatt_param['BatteryEfficiency'],
                                'MaxPower': pvbatt_param['MaxPower'],
                                'InverterEfficiency': pvbatt_param['InverterEfficiency'],
                                'timestep': .25} 
        outs = dispatch_max_sc(pv_15min,demand_prebatt,param_tech_batt_pspy,return_series=False)
        print_analysis(pv_15min,demand_prebatt,param_tech_batt_pspy, outs)
        
        # Saving demand profile considering also battery shifting
        demand_postbatt = pd.Series(data=outs['grid2load'],index=index15min) # kW
        demand_pspy.insert(len(demand_pspy.columns),'TotalShiftPostBattery',demand_postbatt,True) # kW
        
        # Updating final demand profile
        demand_final = demand_postbatt # kW
    
    
    """
    9) Final analisys of the results (including economic analysis)
    """
    
    # TODO
    # - add how much energy has been shifted by each technology
    # - add to outs:
    #   - time horizon
    #   - energy prices
    #   - fixed and capacity-related tariffs
    
    demand_final = pd.Series(data=demand_final,index=index15min)
    outs = ResultsAnalysis(pvbatt_param['pv']['Ppeak'],pvbatt_param['battery']['BatteryCapacity'],pv_15min,demand_ref,demand_final,yprices_15min,prices,scenario,econ_param[namecase])
    
    
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


exectime = (time.time() - start_time)/60.
print('It all took {:.1f} minutes'.format(exectime))




















