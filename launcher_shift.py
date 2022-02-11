
import os
import numpy as np
import pandas as pd
import json
import time
from temp_functions import yearlyprices,mostrapcurve, run,strategy1,writetoexcel,shift_appliance
from prosumpy import dispatch_max_sc_withsd,print_analysis


start_time = time.time()

path_cases = r'./inputs'
name_cases = 'cases.json'
file_cases = os.path.join(path_cases,name_cases)
with open(file_cases) as f:
  casesjson = json.load(f)

casesarr = [1,2,3,4,5,6,7,8,9,
            11,13,15,17,19,
            21,23,25,27,29,
            31,33,35,37,39,
            40,41,42,43,44,45,46,47,49,
            51,53,55,57,59,
            61,63,65,67,69,
            71,73,75,77,78,79,
            80,82]

index = 0 

for jjj in casesarr:
    
    namecase = 'case'+str(jjj)
    
    print('###########################')
    print('        Case'+str(jjj))
    print('###########################')
    
    house = casesjson[namecase]['house']
    sheet = casesjson[namecase]['sheet']
    columns = casesjson[namecase]['columns']
    IndexBool = casesjson[namecase]['IndexBool']
    row=casesjson[namecase]['row']
    appshift =casesjson[namecase]['appshift']
    appnoshift = [x for x in columns if x not in appshift]
    PVBool = casesjson[namecase]['PVBool']
    BattBool = casesjson[namecase]['BattBool']
    FixedControl = casesjson[namecase]['FixedControl']
    AnnualControl = casesjson[namecase]['AnnualControl']
    ShiftBool = casesjson[namecase]['ShiftBool']
    thresholdprice = casesjson[namecase]['thresholdprice']
    OccBool = casesjson[namecase]['OccBool']
    ShiftPVBool = casesjson[namecase]['ShiftPVBool']
    
    """
    Choosing most representative curve according to house and technologies considered
    """
    
    # Demands
    name = house+'.pkl'
    path = r'./simulations'
    file = os.path.join(path,name)
    demands = pd.read_pickle(file)
    
    # Occupancy
    name = house+'_occ.pkl'
    file = os.path.join(path,name)
    occupancys = pd.read_pickle(file)
    
    # Technology parameters required by prosumpy
    param_tech_mrc = {'BatteryCapacity': 0.,
                      'BatteryEfficiency': 0.9,
                      'MaxPower': 0.,
                      'InverterEfficiency': 1.,
                      'timestep': 0.25}
    
    # Technology parameters required by economic analysis (updated inside run function)
    inputs_mrc = {'CapacityPV': 0.,
              'CapacityBattery': 0.}
    
    # Technology costs required by prosumpy and economic analysis
    Inv = {'FixedPVCost':0,
            'PVCost_kW':1500,
            'FixedBatteryCost':0,
            'BatteryCost_kWh':600,
            'PVLifetime':20,
            'BatteryLifetime':10,
            'OM':0.015, # eur/year/eur of capex (both for PV and battery)
            'FixedControlCost': 0,
            'AnnualControlCost': 0} 
    
    # Economic parameteres required by yearly prices function and economic analysis
    with open(r'./inputs/economics.json') as g:
      econ = json.load(g)
    
    timeslots = econ['timeslots']
    prices = econ['prices']
    scenario = 'test'
    
    EconomicVar = {'WACC': 0.05, # weighted average cost of capital
                   'net_metering': False, # type of tarification scheme
                   'time_horizon': 20,
                   'C_grid_fixed':prices[scenario]['fixed'], # € annual fixed grid costs
                   'C_grid_kW': prices[scenario]['capacity'], # €/kW annual grid cost per kW 
                   'P_FtG': 40.}     # €/MWh electricity price to sell to the grid
    
    # Timestep used by prosumpy and economic analysis
    timestep = 0.25 # hrs
    stepperhour = 4
    
    # Choosing most rapresentative curve
    ElPrices = yearlyprices(scenario,timeslots,prices,stepperhour)
    
    if IndexBool:
        index = mostrapcurve(demands,param_tech_mrc,inputs_mrc,EconomicVar,Inv,ElPrices,timestep,columns)
    
    
    """
    Modelling all cases that have the most rapresentative curve in common
    """
    
    # Total demand
    demand_ref = demands[index][columns]
    demand_ref = demand_ref.sum(axis=1)
    demand_ref = demand_ref/1000. # W to kW
    demand_ref = demand_ref.to_frame()
    demand_ref = demand_ref.resample('15Min').mean() # resampling at 15 min
    demand_ref.index = pd.to_datetime(demand_ref.index)
    year = demand_ref.index.year[0] # extracting ref year used in the simulation
    nye = pd.Timestamp(str(year+1)+'-01-01 00:00:00') # remove last row if is from next year
    demand_ref = demand_ref.drop(nye)
    demand_ref = demand_ref.iloc[:,0]
    demand_ref = demand_ref.to_numpy()
    ydemand = np.sum(demand_ref)/4
    
    # PV production
    pvpeak = ydemand/950. if ydemand/950. < 10. else 10. #kW
    pvfile = r'./simulations/pv.pkl'
    pvadim = pd.read_pickle(pvfile)
    pv = pvadim * pvpeak # kW
    pv = pv.iloc[:,0]
    if not PVBool:
        pvpeak = 0
        pv.values[:] = 0 
    
    # Battery data
    if BattBool:
        battcapacity = 10.
    else:
        battcapacity = 0.
        
    battmaxpower = 7.
    param_tech = {'BatteryCapacity': battcapacity,
                  'BatteryEfficiency': 0.9,
                  'MaxPower': battmaxpower,
                  'InverterEfficiency': 1.,
                  'timestep': 0.25}
    
    inputs = {'CapacityPV': pvpeak,
              'CapacityBattery': battcapacity}
    
    Inv['FixedControlCost']= FixedControl
    Inv['AnnualControlCost'] = AnnualControl
    
    
    results = []
    
    if not ShiftBool:
        results = run(pv,demands[index],param_tech,inputs,EconomicVar,Inv,ElPrices,timestep,columns,prices,scenario,demand_ref)
    
    
    """
    Load shifting for wet appliances
    Manual starts, scheduled starts without and with PV
    """
    
    if ShiftBool:
        # Admissible prices time windows
        stepperhourshift=60
        yprices = yearlyprices(scenario,timeslots,prices,stepperhourshift)
        admprices = np.where(yprices <= prices[scenario][thresholdprice]/1000,1.,0.)
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
        if OccBool:
            admtimewin = admprices*admcustom*occupancy
        else:
            admtimewin = admprices*admcustom
        
        # Probability of load being shifted
        probshift = 1.
        
        demnoshift = []
        demshift = []
        pv2 = []
        totenshift = 0.
        
        # Admissible time windows according to PV production - General part
        if ShiftPVBool:
            demand2 = demands[index]
            demnoshift = demand2[appnoshift]
            demnoshift = demnoshift.sum(axis=1)
            demnoshift = demnoshift.to_frame()
            demnoshift.index = pd.to_datetime(demnoshift.index)
            year = demnoshift.index.year[0] # extracting ref year used in the simulation
            nye = pd.Timestamp(str(year+1)+'-01-01 00:00:00') # remove last row if is from next year
            demnoshift = demnoshift.drop(nye)
            demnoshift = demnoshift.iloc[:,0]   
            
            demshift = demand2[appshift]
            demshift.index = pd.to_datetime(demshift.index)
            year = demshift.index.year[0] # extracting ref year used in the simulation
            nye = pd.Timestamp(str(year+1)+'-01-01 00:00:00') # remove last row if is from next year
            demshift = demshift.drop(nye)
            
            pv2 = []
            for i in range(len(pv)):
                pvres = max(pv[i]-demnoshift[i]/1000.,0)
                for j in range(15):
                    pv2.append(pvres)
            pv2 = np.array(pv2)*1000 
        
        for app in appshift:
        
            # Admissible time windows according to PV production - Technology-specific part    
            if ShiftPVBool:  
                adm = np.zeros(len(demand2))
                
                apparr = demshift[app].to_numpy()
                app_s  = np.roll(apparr,1)
                starts   = np.where(apparr-app_s>1)[0]
                ends   = np.where(apparr-app_s<-1)[0]
                lengths = ends-starts
                maxcyclen = np.max(lengths)
                meancyclen = int(np.mean(lengths))
                
                cons = np.ones(maxcyclen)*apparr[starts[0]+1]
                sumcons = np.sum(cons)
                
                for i in range(len(demshift)-maxcyclen*60):
                    
                    prod = pv2[i:i+maxcyclen]
                    diff = np.array(cons-prod)
                    diff = np.where(diff>=0,diff,0)
                    notcov = np.sum(diff)*0.25
                    share = 1-notcov/sumcons
                    if share >= 0.9:
                        adm[i]=1
                
                # Updating admissibile time windows considering PV
                admtimewin = admtimewin + adm*occupancy
            
            print("---"+str(app)+"---")
            app_n,ncyc,ncycshift,enshift = shift_appliance(demands[index][app],admtimewin,probshift,max_shift=24*60)
            #app_n,ncyc,ncycshift,maxshift,avgshift,cycnotshift,enshift = strategy1(demands[index][app],admtimewin,probshift)
        
            demands[index].insert(len(demands[index].columns),app+'Shift', app_n,True)
            
            conspre  = sum(demands[index][app])/60./1000.
            conspost = sum(demands[index][app+'Shift'])/60./1000.
            print("Original consumption: {:.2f}".format(conspre))
            print("Number of cycles: {:}".format(ncyc))
            print("Number of cycles shifted: {:}".format(ncycshift))
            #print("Consumption after shifting (check): {:.2f}".format(conspost))
            #print("Max shift: {:.2f} hours".format(maxshift))
            #print("Avg shift: {:.2f} hours".format(avgshift))
            #print("Unable to shift {:} cycles".format(cycnotshift))
            
            totenshift += enshift/60./1000.
            
            if ShiftPVBool:  
                # Updating PV residual
                tempdem = demands[index][app+'Shift'].drop(nye)
                pv2 = pv2 - tempdem.to_numpy()
            
        newdemands = demands[index]
        
        # Run!
        columnsshift = ['StaticLoad','TumbleDryerShift','DishWasherShift','WashingMachineShift']
        results = run(pv,newdemands,param_tech,inputs,EconomicVar,Inv,ElPrices,timestep,columnsshift,prices,scenario,demand_ref)
        results['el_shifted'] = totenshift
    
    else:
        results['el_shifted'] = 0.
        results['varperc_cons_heel'] = 0.
        results['varperc_cons_hollow'] =0.
        results['varperc_cons_full'] = 0.
        results['varperc_cons_peak'] = 0.
    
    
    # Saving results to excel
    file = 'test'+house+'.xlsx'
    writetoexcel(file,sheet,results,row)

exectime = (time.time() - start_time)/60.
print(' It took {:.1f} minutes'.format(exectime))

# """
# Load shifting for DHW with PV panels
# """

# demand2 = demands[index]

# columnsnotshift = ['StaticLoad','TumbleDryer','DishWasher','WashingMachine','HeatPumpPower','EVCharging']
# demand2_notshift = demand2[columnsnotshift]
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

# Tmin = 45.
# Ccyl = inputs2['DHW']['Vcyl'] * 1000. /1000. * 4200. # J/K
# capacity = Ccyl*(inputs2['DHW']['Ttarget']-Tmin)/3600./1000. # kWh
  
# param_tech2 = {'BatteryCapacity':  capacity,
#               'BatteryEfficiency': 1.,
#               'MaxPower': inputs2['DHW']['PowerElMax'],
#               'InverterEfficiency': 1.,
#               'timestep': .25,
#               'SelfDisLin': 0.,
#               'SelfDisFix':0.}

# outs = dispatch_max_sc_withsd(pv2,demand2_dhw,param_tech2,return_series=False)
# print_analysis(pv2, demand2_dhw, param_tech2, outs)

