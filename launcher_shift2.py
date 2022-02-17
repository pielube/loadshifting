
import os
import numpy as np
import pandas as pd
import json
import time
from prosumpy import dispatch_max_sc,print_analysis
from temp_functions import yearlyprices,HPSizing,COP_Tamb
from launcher_shift_functions import MostRepCurve,DHWShiftTariffs,HouseHeatingShiftSC,ResultsAnalysis,WriteResToExcel
from temp_functions import shift_appliance
from pv import pvgis_hist
from demands import compute_demand


start_time = time.time()


#%%

N = 2 # Number of stochastic simulations to be run for the demand curves
jjj = 12  
namecase = 'case'+str(jjj)

print('###########################')
print('        Case'+str(jjj))
print('###########################')


#%%
# Case description
with open('inputs/cases.json','r') as f:
    cases = json.load(f)

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

#%%
# Adimensional PV curve
with open('inputs/pv.json','r') as f:
    config_pv = json.load(f)

pvadim = pvgis_hist(config_pv)  

#%%
# Demands
with open('inputs/' + house+'.json') as f:
  inputs = json.load(f)

demands = compute_demand(inputs,N,inputs['members'],inputs['thermal_parameters'])

#%%
# Economic parameters
with open('inputs/econ_param.json','r') as f:
    econ_param = json.load(f)

FixedControl   = econ_param[namecase]['FixedControlCost']
AnnualControl  = econ_param[namecase]['AnnualControlCost']
thresholdprice = econ_param[namecase]['thresholdprice']

#%%
# PV and battery technology parameters
with open('inputs/pvbatt_param.json','r') as f:
    pvbatt_param = json.load(f)

#%%
# Time of use tariffs
with open('inputs/tariffs.json','r') as f:
    tariffs = json.load(f)

#%%
# Various array sizes and timesteps used throughout the code

n1min  = np.size(demands['results'][0]['StaticLoad'])-1
n10min = int(n1min/10)
n15min = int(n1min/15)
stepperh_1min = 60 # 1/h
stepperh_15min = 4 # 1/h
ts_15min = 0.25 # h
index1min  = pd.date_range(start='2015-01-01',end='2015-12-31 23:59:00',freq='T')
index15min = pd.date_range(start='2015-01-01',end='2015-12-31 23:45:00',freq='15T')

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

    # Sizing
    pvpeak = ydemand/950. if ydemand/950. < 10. else 10. # kWp
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

# Update PV capacity
pvbatt_param['PVCapacity'] = pvpeak # kWp


"""
7) Battery size
"""

if BattBool:
    pvbatt_param['BatteryCapacity'] = 10. # kWh
else:
    pvbatt_param['BatteryCapacity'] = 0. # kWh


"""
8) Shifting
"""

"""
8A) Load shifting - Wet appliances
    NB for the shifting of the appliances we work with 1-min timestep and power in W   
"""

if WetAppBool:
    
    print('--- Shifting wet appliances ---')

    # Probability of load being shifted
    # TODO add to cases inputs?
    probshift = 1. 
    
    # Wet app demands to be shifted, 1 min timestep
    demshift = demands['results'][idx][WetAppShift][:-1] # W

    """
    Admissible time windows
    """
    
    # Admissible time window based on electricity prices
    yprices_1min = yearlyprices(scenario,timeslots,prices,stepperh_1min) # €/kWh
    admprices = np.where(yprices_1min <= prices[scenario][thresholdprice]/1000,1.,0.)
    admprices = np.append(admprices,yprices_1min[-1])

    # Reducing prices adm time window as not to be on the final hedge of useful windows
    admcustom = np.ones(len(admprices))
    for i in range(len(admprices)-60):
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
        #app_n,enshift = AdmTimeWinShift(demands['results'][idx][app],admtimewin,probshift) # W, Wh
        app_n,ncyc,ncycshift,enshift = shift_appliance(demands[idx][app],admtimewin,probshift,max_shift=24*60)
        
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

    Tmin = 45. # °C
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
        demand_dhw_shift = outs['inv2load']+outs['grid2load'] # kW
        demand_dhw_shift = demand_dhw_shift.astype('float64') # kW
        
        # updating residual pv
        pv_15min_res = [a if a>0 else 0 for a in (pv_15min_res.to_numpy()-demand_dhw_shift)] # kW
        pv_15min_res = pd.Series(data=pv_15min_res,index=index15min) # kW
        pv_1min_res  = pv_15min_res.resample('T').pad().reindex(index1min,method='nearest').to_numpy() # kW 
        
    else: # strategy based on tariffs
        
        # prosumpy inspired tariffs based function
        outs = DHWShiftTariffs(demand_dhw, yprices_15min, thresholdprice, param_tech_dhw, return_series=False)
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
    
    # Heat pump sizing
    fracmaxP = 0.8
    QheatHP = HPSizing(inputs,fracmaxP) # W
    
    if PVBool: # strategy based on enhancing self-consumption
        
        # Heat shifted
        Qshift,Tin_shift = HouseHeatingShiftSC(inputs,n1min,temp,irr,Qintgains,QheatHP,pv_1min,Tset) # W, °C
        
        # T analysis
        Twhenon    = Tin_shift*occupancy # °C
        Twhenon_hs = Twhenon[np.r_[0:60*24*151,60*24*244:-1]] # °C
        whenon     = np.nonzero(Twhenon_hs)
        Twhenon_hs_mean = np.mean(Twhenon_hs[whenon]) # °C
        Twhenon_hs_min  = np.min(Twhenon_hs[whenon]) # °C
        Twhenon_hs_max  = np.max(Twhenon_hs[whenon]) # °C
        
        # Electricity consumption
        Eshift = np.zeros(n1min) 
        for i in range(n1min):
            COP = COP_Tamb(temp[i])
            Eshift[i] = Qshift[i]/COP # W
        
        # updating demand dataframe
        demand_HP_shift = pd.Series(data=Eshift,index=pd.date_range(start='2015-01-01',end='2015-12-31 23:59:00',freq='min'))
        demand_HP_shift = demand_HP_shift.resample('15Min').mean().to_numpy()/1000. # kW
        demand_pspy.insert(len(demand_pspy.columns),'HeatPumpPowerShift',demand_HP_shift,True) #kW
        
        # checking
        HPcons_pre  = np.sum(demand_pspy['HeatPumpPower'])/4. # kWh
        HPcons_post = np.sum(demand_pspy['HeatPumpPowerShift'])/4. # kWh
        HPconsincr = (HPcons_post-HPcons_pre)/HPcons_pre*100 # %
        print("Original consumption: {:.2f} kWh".format(HPcons_pre))
        print("Consumption after shifting: {:.2f} kWh".format(HPcons_post))
        print("Consumption increase: {:.2f}%".format(HPconsincr))

        
        # updating residual PV
        pv_15min_res = [a if a>0 else 0 for a in (pv_15min_res.to_numpy()-demand_HP_shift)] # kW
        pv_15min_res = pd.Series(data=pv_15min_res,index=index15min) # kW
        pv_1min_res  = pv_15min_res.resample('T').pad().reindex(index1min,method='nearest').to_numpy() # kW   
    
    #else: # strategy based on tariffs

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
    outs = dispatch_max_sc(pv_15min,demand_prebatt,param_tech,return_series=False)
    print_analysis(pv_15min,demand_prebatt,param_tech, outputs)
    
    # Saving demand profile considering also battery shifting
    demand_postbatt = pd.Series(data=outs['grid2load'],index=index15min) # kW
    demand_pspy.insert(len(demand_pspy.columns),'TotalShiftPostBattery',demand_afterbattshift,True) # kW
    
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
outs = ResultsAnalysis(pvbatt_param['PVCapacity'],pvbatt_param['BatteryCapacity'],pv_15min,demand_ref,demand_final,yprices_15min,prices,scenario,econ_param[namecase])


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
WriteResToExcel(file,sheet,outs,row)


exectime = (time.time() - start_time)/60.
print('It all took {:.1f} minutes'.format(exectime))




















