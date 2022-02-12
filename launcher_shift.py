
import os
import numpy as np
import pandas as pd
import json
import time
from prosumpy import dispatch_max_sc
from temp_functions import yearlyprices,HPSizing,COP_Tamb
from launcher_shift_functions import MostRapCurve,AdmTimeWinShift,HouseHeatingShiftSC,ResultsAnalysis,WriteResToExcel
from preprocess import ProcebarExtractor,HouseholdMembers


start_time = time.time()

# Reading cases to be simulated
path_cases = r'./inputs'
name_cases = 'cases.json'
file_cases = os.path.join(path_cases,name_cases)
with open(file_cases) as f:
  casesjson = json.load(f)

# Selecting 
casesarr = [12]

"""
1) Reading inputs that define the case to be simulated
"""

jjj = 12
    
namecase = 'case'+str(jjj)

print('###########################')
print('        Case'+str(jjj))
print('###########################')

house          = casesjson[namecase]['house']
sheet          = casesjson[namecase]['sheet']
columns        = casesjson[namecase]['columns'] 
TechsShift     = casesjson[namecase]['TechsShift']
WetAppShift    = [x for x in TechsShift if x in ['TumbleDryer','DishWasher','WashingMachine']]
TechsNoShift   = [x for x in columns if x not in TechsShift]

WetAppBool     = casesjson[namecase]['WetAppBool']
WetAppManBool  = casesjson[namecase]['WetAppManBool']
WetAppAutoBool = casesjson[namecase]['WetAppAutoBool']
PVBool         = casesjson[namecase]['PVBool']
BattBool       = casesjson[namecase]['BattBool']
DHWBool        = casesjson[namecase]['DHWBool']
HeatingBool    = casesjson[namecase]['HeatingBool']
EVBool         = casesjson[namecase]['EVBool']


row            = casesjson[namecase]['row']
FixedControl   = casesjson[namecase]['FixedControl']
AnnualControl  = casesjson[namecase]['AnnualControl']

thresholdprice = casesjson[namecase]['thresholdprice']


"""
2) Putting all inputs required together
"""

# Adimensional PV curve

pvfile = r'./simulations/pv.pkl'
pvadim = pd.read_pickle(pvfile)
#TODO fix pvadim where generated
pvadim = [a if a > 0.01 else 0 for a in pvadim.to_numpy()]
pvadim = pd.Series(data=pvadim,index=pd.date_range(start='2015-01-01',end='2015-12-31 23:45:00',freq='15T'))

# Demands

name = house+'.pkl'
path = r'./simulations'
file = os.path.join(path,name)
demands = pd.read_pickle(file)

# Occupancy

name = house+'_occ.pkl'
file = os.path.join(path,name)
occupancys = pd.read_pickle(file)

# Various array sizes and timesteps used throughout the code

n1min  = np.size(demands[0]['StaticLoad'])-1
n10min = int(n1min/10)
n15min = int(n1min/15)
stepperh_1min = 60
stepperh_15min = 4
ts_15min = 0.25
index1min  = pd.date_range(start='2015-01-01',end='2015-12-31 23:59:00',freq='T')
index15min = pd.date_range(start='2015-01-01',end='2015-12-31 23:45:00',freq='15T')

# PV and battery capacities and parameters initialization

pvpeak = 0.
battcapacity = 0.

capacities = {'CapacityPV': pvpeak,
              'CapacityBattery': battcapacity}

param_tech = {'BatteryCapacity': battcapacity,
              'BatteryEfficiency': 0.9,
              'MaxPower': 7.,
              'InverterEfficiency': 1.,
              'timestep': 0.25}

# Inputs used to generate demands

path = r'./inputs'
name = casesjson[namecase]['house']+'.json'
file = os.path.join(path,name)
with open(file) as f:
  inputs2 = json.load(f)    

# Economic parameteres

with open(r'./inputs/economics.json') as f:
  econ = json.load(f)

scenario = 'test'
prices = econ['prices']
timeslots = econ['timeslots']

EconomicVar = {'WACC': 0.05, # weighted average cost of capital
               'net_metering': False, # type of tarification scheme
               'time_horizon': 20,
               'C_grid_fixed':prices[scenario]['fixed'], # € annual fixed grid costs
               'C_grid_kW': prices[scenario]['capacity'], # €/kW annual grid cost per kW 
               'P_FtG': 40.}     # €/MWh electricity price to sell to the grid

Inv = {'FixedPVCost':0,
        'PVCost_kW':1500,
        'FixedBatteryCost':0,
        'BatteryCost_kWh':600,
        'PVLifetime':20,
        'BatteryLifetime':10,
        'OM':0.015, # eur/year/eur of capex (both for PV and battery)
        'FixedControlCost': FixedControl,
        'AnnualControlCost': AnnualControl} 

# Electricity prices array - 15 min timestep
yprices_15min = yearlyprices(scenario,timeslots,prices,stepperh_15min)


"""
3) Most rapresentative curve
"""

print('--- Selecting most representative curve ---')

index = MostRapCurve(demands,columns,yprices_15min,ts_15min,EconomicVar)

print('Curve index: {:}'.format(index))


"""
4) Demand
"""

# Demand prosumpy-compatible
# Meaning 15 min timestep and in kW
# Selecting only techs (columns) of the case of interest

demand_pspy = demands[index][columns]
demand_pspy = demand_pspy/1000. # W to kW
demand_pspy = demand_pspy.resample('15Min').mean()
year = demand_pspy.index.year[0]
nye = pd.Timestamp(str(year+1)+'-01-01 00:00:00')
demand_pspy = demand_pspy.drop(nye)

# Reference demand
# Aggregated demand pre-shifting

demand_ref = demand_pspy.sum(axis=1)
demand_ref = demand_ref.to_numpy()
ydemand = np.sum(demand_ref)/4
demand_pspy.insert(len(demand_pspy.columns),'TotalReference',demand_ref,True)


"""
5) Occupancy
"""

# Occupancy array, built checking if t least one active person at home
# 1-yes 0-no
# 1 min timestep

occ = np.zeros(n10min)
for i in range(len(occupancys[index])):
    singlehouseholdocc = [1 if a==1 else 0 for a in occupancys[index][i][:-1]]
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
    pvpeak = ydemand/950. if ydemand/950. < 10. else 10. #kW
    # 15 min timestep series
    pv_15min = pvadim * pvpeak # kW
    # 1 min timestep array
    pv_1min = pv_15min.resample('T').pad().reindex(index1min,method='nearest').to_numpy()
            
    # Residual PV
    # 15 min timestep series
    dem_15min_noshift = demand_pspy[TechsNoShift].sum(axis=1)
    pv_15min_res = [a if a>0 else 0 for a in (pv_15min.to_numpy() - dem_15min_noshift.to_numpy())]
    pv_15min_res = pd.Series(data=pv_15min_res,index=index15min)
    # 1 min timestep array
    demnoshift = demands[index][TechsNoShift].sum(axis=1)[:-1].to_numpy()
    pv_1min_res = [a if a>0 else 0 for a in pv_1min-demnoshift]    
    pv_1min_res = np.array(pv_1min_res)
    
else:
    pv_15min = np.zeros(n15min)
    pv_15min = pd.Series(data=pv_15min,index=index15min)

# Update PV capacity
capacities['CapacityPV'] = pvpeak


"""
7) Battery size
"""

if BattBool:
    battcapacity = 10.
else:
    battcapacity = 0.

# Update battery capacity
param_tech['BatteryCapacity'] = battcapacity
capacities['CapacityBattery'] = battcapacity


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
    demshift = demands[index][WetAppShift][:-1]

    """
    Admissible time windows
    """
    
    # Admissible time window based on electricity prices
    yprices_1min = yearlyprices(scenario,timeslots,prices,stepperh_1min)
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
            
            apparr = demshift[app].to_numpy()
            app_s  = np.roll(apparr,1)
            starts = np.where(apparr-app_s>1)[0]
            ends   = np.where(apparr-app_s<-1)[0]
            lengths = ends-starts
            maxcyclen = np.max(lengths)            
            cons = np.ones(maxcyclen)*apparr[starts[0]+1]
            sumcons = np.sum(cons)
            
            for i in range(len(demshift)-maxcyclen*60):
                
                prod = pv_1min_res[i:i+maxcyclen]
                diff = np.array(cons-prod)
                diff = np.where(diff>=0,diff,0)
                notcov = np.sum(diff)*0.25
                share = 1-notcov/sumcons
                if share >= 0.9:
                    admpv[i]=1
            
            # Updating admissibile time windows
            admtimewin = admtimewin + admpv*occupancy
    
        # Calling function to shift the app
        print("---"+str(app)+"---")
        app_n,enshift = AdmTimeWinShift(demands[index][app],admtimewin,probshift)
        # TODO uniform way of computing total energy shifted
        # totenshift += enshift/60./1000. # kWh
        
        # Resizing shifted array
        app_n_series = pd.Series(data=app_n,index=index1min)
        app_n_15min = app_n_series.resample('15Min').mean().to_numpy()
        
        # updating demand dataframe
        demand_pspy.insert(len(demand_pspy.columns),app+'Shift',app_n_15min,True)
        
        # Updating residual PV considering the consumption of the app just shifted
        if PVBool:  
            pv_1min_res = pv_1min_res - app_n
            pv_1min_res_series = pd.Series(data=pv_1min_res,index=index1min)
            pv_15min_res = pv_1min_res_series.resample('15Min').mean()
  
"""
8B) Load shifting - DHW
"""


if DHWBool:

    print('--- Shifting domestic hot water ---')

    
    if PVBool: # strategy based on enhancing self-consumption
       
        # demand of domestic hot water (to be used with battery equivalent approach)
        demand_dhw = demand_pspy['DomesticHotWater']
    
        # equivalent battery
        # TODO check these entries
        Vcyl = inputs2['DHW']['Vcyl'] # litres
        Ttarget = inputs2['DHW']['Ttarget'] # °C
        PowerDHWMax = inputs2['DHW']['PowerElMax']/1000. # kW

        Tmin = 45. # °C
        Ccyl = Vcyl * 1000. /1000. * 4200. # J/K
        capacity = Ccyl*(Ttarget-Tmin)/3600./1000. # kWh
          
        param_tech_dhw = {'BatteryCapacity': capacity,
                          'BatteryEfficiency': 1.,
                          'MaxPower': PowerDHWMax,
                          'InverterEfficiency': 1.,
                          'timestep': .25,
                          'SelfDisLin': 0.,
                          'SelfDisFix':0.}
    
        # prosumpy
        outs = dispatch_max_sc(pv_15min_res,demand_dhw,param_tech_dhw,return_series=False)
        demand_dhw_shift = outs['inv2load']+outs['grid2load']
        demand_dhw_shift = demand_dhw_shift.astype('float64')
    
        # updating demand dataframe
        demand_pspy.insert(len(demand_pspy.columns),'DomesticHotWaterShift',demand_dhw_shift,True)
        
        # check on shifting
        conspre  = np.sum(demand_pspy['DomesticHotWater'])/4. # kWh
        conspost = np.sum(demand_pspy['DomesticHotWaterShift'])/4. # kWh
        print("Original consumption: {:.2f} kWh".format(conspre))
        print("Consumption after shifting (check): {:.2f} kWh".format(conspost))
        
        # updating residual pv
        pv_15min_res = [a if a>0 else 0 for a in (pv_15min_res.to_numpy()-demand_dhw_shift)]
        pv_15min_res = pd.Series(data=pv_15min_res,index=index15min)
        pv_1min_res  = pv_15min_res.resample('T').pad().reindex(index1min,method='nearest').to_numpy()
        
    #else: # strategy based on tariffs
        # TODO (function as in prosumpy but based on el price)

"""
8C) Load shifting - House heating
""" 
   

# TODO
# - save Q internal gains
# - save info on building, as extracted from Procebar
# - save T inside house if needed to check how shift affects it
# - revise temperature setpoint in the demand simulation
#   here T set could recalculated (as now) or saved as dem, occ, etc.
# - harmonize ambient data used in all simulations
#   use TMY obtained from PVGIS everywhere
# - add fraction of max power when sizing HP
# - revise heating season

if HeatingBool:
    
    print('--- Shifting house heating ---')

    # Thermal parameters of the dwelling
    # Taken from Procebar xls file
    # WRONG: data should be saved when running 10 cases and loaded here
    procebinp = ProcebarExtractor(inputs2['HP']['dwelling_type'],True)
    inputs2['HP'] = {**inputs2['HP'],**procebinp}  
    
    # ambient data
    datapath = r'./strobe/Data'
    temp = np.loadtxt(datapath + '/Climate/temperature.txt')
    irr  = np.loadtxt(datapath + '/Climate/irradiance.txt')   
    temp = np.delete(temp,-1)
    irr = np.delete(irr,-1)
    
    # internal gains
    # WRONG: data should be saved when running 10 cases and loaded here
    Qintgains = np.zeros(len(temp))

    # T setpoint based on occupancy
    Tset = [20. if a == 1 else 20. for a in occupancy]
    
    # Heat pump sizing
    fracmaxP = 1.
    QheatHP = HPSizing(inputs2,fracmaxP)
    
    if PVBool: # strategy based on enhancing self-consumption
        
        # Heat shifted
        Qshift,Tin_shift = HouseHeatingShiftSC(inputs2,n1min,temp,irr,Qintgains,QheatHP,pv_1min,Tset)
        
        # T analysis
        Twhenon    = Tin_shift*occupancy
        Twhenon_hs = Twhenon[np.r_[0:60*24*151,60*24*244:-1]]
        whenon     = np.nonzero(Twhenon_hs)
        Twhenon_hs_mean = np.mean(Twhenon_hs[whenon])
        Twhenon_hs_min  = np.min(Twhenon_hs[whenon])
        Twhenon_hs_max  = np.max(Twhenon_hs[whenon])
        
        # Electricity consumption
        Eshift = np.zeros(n1min) 
        for i in range(n1min):
            COP = COP_Tamb(temp[i])
            Eshift[i] = Qshift[i]/COP
        
        # updating demand dataframe
        demand_HP_shift = pd.Series(data=Eshift,index=pd.date_range(start='2015-01-01',end='2015-12-31 23:59:00',freq='min'))
        demand_HP_shift = demand_HP_shift.resample('15Min').mean().to_numpy()/1000.
        demand_pspy.insert(len(demand_pspy.columns),'HeatPumpPowerShift',demand_HP_shift,True)
        
        # checking
        HPcons_pre  = np.sum(demand_pspy['HeatPumpPower'])/4.
        HPcons_post = np.sum(demand_pspy['HeatPumpPowerShift'])/4.
        HPconsincr = (HPcons_post-HPcons_pre)/HPcons_pre*100
        print("Original consumption: {:.2f} kWh".format(HPcons_pre))
        print("Consumption after shifting: {:.2f} kWh".format(HPcons_post))
        print("Consumption increase: {:.2f}%".format(HPconsincr))
        
        # updating residual PV
        pv_15min_res = [a if a>0 else 0 for a in (pv_15min_res.to_numpy()-demand_HP_shift)]
        pv_15min_res = pd.Series(data=pv_15min_res,index=index15min)
        pv_1min_res  = pv_15min_res.resample('T').pad().reindex(index1min,method='nearest').to_numpy()   
    
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
demand_prebatt = demand_pspy[finalcols].sum(axis=1)
demand_pspy.insert(len(demand_pspy.columns),'TotalShiftPreBattery',demand_prebatt.to_numpy(),True)

# If no battery this demand profile is the definitive one
demand_final = demand_prebatt
   
"""
8F) Load shifting - Standard battery
"""


if BattBool:
    
    print('--- Shifting resulting demand with battery ---')
     
    # Battery applied to demand profile shifted by all other shifting techs          
    outs = dispatch_max_sc(pv_15min,demand_prebatt,param_tech,return_series=False)
    print_analysis(pv_15min,demand_prebatt,param_tech, outputs)
    
    # Saving demand profile considering also battery shifting
    demand_postbatt = pd.Series(data=outs['grid2load'],index=index15min)
    demand_pspy.insert(len(demand_pspy.columns),'TotalShiftPostBattery',demand_afterbattshift,True)
    
    # Updating final demand profile
    demand_final = demand_postbatt


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
outs = ResultsAnalysis(capacities,pv_15min,demand_ref,demand_final,yprices_15min,prices,scenario,EconomicVar,Inv)


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
file = 'test'+house+'.xlsx'
WriteResToExcel(file,sheet,outs,row)




exectime = (time.time() - start_time)/60.
print('It all took {:.1f} minutes'.format(exectime))




















