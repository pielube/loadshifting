
import random
import numpy as np
import pandas as pd
from statistics import mean

from prosumpy import dispatch_max_sc
from strobe.RC_BuildingSimulator import Zone

from temp_functions import EconomicAnalysis

# List of functions this .py file should contain

# - EconomicAnalysis
# - MostRapCurve
# - AdmTimeWinShift
# - DHWShiftTariffs
# - DHWShiftSC
# - HouseHeatingShiftTariffs
# - HouseHeatingShiftSC
# - EVShiftTariffs
# - EVShiftSC
# - ResultsAnalysis
# - WriteToExcel

# TODO
# - add description of all functions
# - comment all functions
# - change name of this file



    
def MostRapCurve(demands,columns,ElPrices,timestep,EconomicVar):
    
    """
    Choosing most representative curve among a list of demand curves
    based on electricity bill buying all electricity from grid
    hence wiithout PV or batteries
    """

    # Technology parameters required by prosumpy
    # Battery forced to be 0 
    param_tech = {'BatteryCapacity': 0.,
                  'BatteryEfficiency': 1.0,
                  'MaxPower': 0.,
                  'InverterEfficiency': 1.,
                  'timestep': 0.25}
    
    # Technology parameters required by economic analysis
    # PV and battery forced to be 0
    inputs = {'CapacityPV': 0.,
              'CapacityBattery': 0.}
    
    # Technology costs required economic analysis
    # Not relevant
    Inv = {'FixedPVCost':0.,
           'PVCost_kW':0.,
           'FixedBatteryCost':0.,
           'BatteryCost_kWh':0.,
           'PVLifetime':20.,
           'BatteryLifetime':0.,
           'OM': 0.,
           'FixedControlCost': 0.,
           'AnnualControlCost': 0.} 
    
    results = []
    
    pv = np.zeros(int((len(demands[0])-1)/15))
    date = pd.date_range(start='2015-01-01 00:00:00',end='2015-12-31 23:45:00',freq='15Min')
    pv = pd.DataFrame(data=pv,index=date)
    pv = pv.iloc[:,0]
    
    for ii in range(len(demands)):
        demand = demands[ii][columns]
        demand = demand.sum(axis=1)
        demand = demand/1000. # W to kW
        demand = demand.to_frame()
        demand = demand.resample('15Min').mean() # resampling at 15 min
        demand.index = pd.to_datetime(demand.index)
        year = demand.index.year[0] # extracting ref year used in the simulation
        nye = pd.Timestamp(str(year+1)+'-01-01 00:00:00') # remove last row if is from next year
        demand = demand.drop(nye)
        demand = demand.iloc[:,0]
        
        outputs = dispatch_max_sc(pv,demand,param_tech,return_series=False)
        # print_analysis(pv, demand, param_tech, outputs)
        
        inputs['ACGeneration'] = pv.to_numpy() # should be equal to outputs['inv2grid']+outputs['inv2load']
        inputs['Load'] = demand.to_numpy()
        inputs['ToGrid'] = outputs['inv2grid']
        inputs['FromGrid'] = outputs['grid2load']
        inputs['SC'] = outputs['inv2load']
        inputs['FromBattery'] = outputs['store2inv']
        
        out = EconomicAnalysis(inputs,EconomicVar,Inv,ElPrices,timestep,inputs['Load'])
        results.append(out['ElBill'])
    
    meanelbill = mean(results)
    var = results-meanelbill
    index = min(range(len(var)), key=var.__getitem__)
        
    return index



def AdmTimeWinShift(app,admtimewin,probshift):
   
    ncycshift = 0
    ncycnotshift = 0
    maxshift = 0
    totshift = 0
    enshift = 0.
    
    app_s  = np.roll(app,1)
    starts   = np.where(app-app_s>1)[0]
    ends   = np.where(app-app_s<-1)[0]
    
    app_n = np.zeros(len(app))
    
    for i in range(len(starts)):
        
        if admtimewin[starts[i]] == 1:
            app_n[starts[i]:ends[i]] += app[starts[i]:ends[i]]
        
    for i in range(len(starts)):
        
        if admtimewin[starts[i]] == 0:
            
            if random.random() > probshift:
                app_n[starts[i]:ends[i]] += app[starts[i]:ends[i]]
            else:
                
                ncycshift += 1
                
                non_zeros = np.nonzero(admtimewin)[0] # array of indexes of non 0 elements
                distances = np.abs(non_zeros-starts[i]) # array of distances btw non 0 elem and ref           
                closest_idx = np.where(distances == np.min(distances))[0]
                newstart = non_zeros[closest_idx][0]
                cyclen = ends[i]-starts[i]
                newend = newstart + cyclen
                
                while any(app_n[newstart:newend]):
                    non_zeros = np.delete(non_zeros,closest_idx)
                    if np.size(non_zeros)==0:
                        newstart = starts[i]
                        newend = ends[i]
                        ncycnotshift += 1
                        break
                    distances = np.abs(non_zeros-starts[i])
                    closest_idx = np.where(distances == np.min(distances))[0]
                    newstart = non_zeros[closest_idx][0]
                    cyclen = ends[i]-starts[i]
                    newend = newstart + cyclen
                           
                if newend > len(app)-1:
                    newend = len(app)-1
                    cyclen = newend-newstart
                    app_n[newstart:newend] += app[starts[i]:starts[i]+cyclen]
                else:
                    app_n[newstart:newend] += app[starts[i]:ends[i]]
            
            enshift += np.sum(app_n[newstart:newend])/60.
            maxshift = max(maxshift,abs(newstart-starts[i])/60.)
            totshift += abs(newstart-starts[i])
    
    avgshift = totshift/len(starts)/60.
    app_n=np.where(app_n==0,1,app_n)
    ncyc = len(starts)
    ncycshift = ncycshift - ncycnotshift
    
    if ncycnotshift > 0:
        val = np.sort(np.unique(app_n))
        if np.size(val) > 2:
            indexes = np.where(app_n==val[-1])[0]
            app_n[indexes]=val[-2]
    
    conspre  = sum(app)/60./1000.
    conspost = sum(app_n)/60./1000.
    print("Original consumption: {:.2f}".format(conspre))
    print("Number of cycles: {:}".format(ncyc))
    print("Number of cycles shifted: {:}".format(ncycshift))
    print("Consumption after shifting (check): {:.2f}".format(conspost))
    print("Max shift: {:.2f} hours".format(maxshift))
    print("Avg shift: {:.2f} hours".format(avgshift))
    print("Unable to shift {:} cycles".format(ncycnotshift))
                
    return app_n,enshift



def HouseHeatingShiftSC(inputs,nminutes,Tamb,irr,Qintgains,QheatHP,pv,Tset):

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



def ResultsAnalysis(capacities,pv,demand_ref,demand,ElPrices,prices,scenario,EconomicVar,Inv):
    
    # Running prosumpy to get SC and SSR
    # All shifting must have already been modelled, including battery
    # param_tech is hence defined here and battery forced to be 0
    
    param_tech = {'BatteryCapacity': 0.,
                  'BatteryEfficiency': 1.,
                  'MaxPower': 0.,
                  'InverterEfficiency': 1.,
                  'timestep': 0.25}
    
    res_pspy = dispatch_max_sc(pv,demand,param_tech,return_series=False)
    #print_analysis(pv, demand, param_tech, outputs)

    # Economic analysis

    Epspy = {}
    
    Epspy['CapacityPV']      = capacities['CapacityPV']
    Epspy['CapacityBattery'] = capacities['CapacityBattery']
    Epspy['ACGeneration'] = pv.to_numpy()
    Epspy['Load']         = demand.to_numpy()
    Epspy['ToGrid']       = res_pspy['inv2grid']
    Epspy['FromGrid']     = res_pspy['grid2load']
    Epspy['SC']           = res_pspy['inv2load']
    # Not used by economic analysis 
    # and would be all 0 considerng how prosumpy has been used
    #Epspy['FromBattery'] = outputs['store2inv']
    
    timestep = 0.25
    
    res_EA = EconomicAnalysis(Epspy,EconomicVar,Inv,ElPrices,timestep,demand_ref)
    
    # Preparing function outputs
    
    out = {}

    heel   = np.where(ElPrices == prices[scenario]['heel']/1000,1.,0.)
    hollow = np.where(ElPrices == prices[scenario]['hollow']/1000,1.,0.)
    full   = np.where(ElPrices == prices[scenario]['full']/1000,1.,0.)
    peak   = np.where(ElPrices == prices[scenario]['peak']/1000,1.,0.)
   
    consrefheel   = np.sum(demand_ref*heel)*timestep
    consrefhollow = np.sum(demand_ref*hollow)*timestep
    consreffull   = np.sum(demand_ref*full)*timestep
    consrefpeak   = np.sum(demand_ref*peak)*timestep

    out['CapacityPV']      = res_EA['CapacityPV'] 
    out['CapacityBattery'] = res_EA['CapacityBattery'] 
    
    out['CostPV']      = res_EA['CostPV']
    out['CostBattery'] = res_EA['CostBattery']

    out['peakdem'] = np.max(demand)
    
    out['cons_total'] = np.sum(demand)*timestep

    out['cons_heel']   = np.sum(res_pspy['grid2load']*heel)*timestep
    out['cons_hollow'] = np.sum(res_pspy['grid2load']*hollow)*timestep
    out['cons_full']   = np.sum(res_pspy['grid2load']*full)*timestep
    out['cons_peak']   = np.sum(res_pspy['grid2load']*peak)*timestep
        
    out['varperc_cons_heel']   = (out['cons_heel']-consrefheel)/out['cons_total']*100
    out['varperc_cons_hollow'] = (out['cons_hollow']-consrefhollow)/out['cons_total']*100
    out['varperc_cons_full']   = (out['cons_full']-consreffull)/out['cons_total']*100
    out['varperc_cons_peak']   = (out['cons_peak']-consrefpeak)/out['cons_total']*100
    
    out['el_prod']           = np.sum(pv)*timestep
    out['el_selfcons']       = np.sum(res_pspy['inv2load'])*timestep
    out['el_soldtogrid']     = np.sum(res_pspy['inv2grid'])*timestep
    out['el_boughtfromgrid'] = np.sum(res_pspy['grid2load'])*timestep
    
    out['selfsuffrate'] = out['el_selfcons']/out['cons_total']
    
    if out['el_prod'] == 0:
        out['selfconsrate'] = 0
    else:
        out['selfconsrate'] = out['el_selfcons']/out['el_prod']
    
    out['el_soldtogrid_rev']      = res_EA['RevSelling']
    out['el_boughtfromgrid_cost'] = res_EA['CostBuying']
    out['annualgridcosts']        = res_EA['AnnualGridCosts']
    out['el_netexpend']           = res_EA['ElBill']
    out['el_costperkwh']          = res_EA['costpermwh']/1000.
    
    out['PBP'] = res_EA['PBP']
    out['NPV'] = res_EA['NPV']
    out['PI']  = res_EA['PI']
    
    return out



def WriteResToExcel(file,sheet,results,row):
    
    df = pd.read_excel(file,sheet_name=sheet,header=0,index_col=0)
    
    df.at[row,'PV [kWp]'] = results['CapacityPV']
    df.at[row,'Battery [kWh]'] = results['CapacityBattery']
    
    df.at[row,'Investissement PV [€]'] = results['CostPV'] 
    df.at[row,'Investissement Battery [€]'] = results['CostBattery']
    
    df.at[row,"Capacité d'accès kW"]  = results['peakdem']
    
    df.at[row,'Total  [kWh]']  = results['cons_total']

    df.at[row,'Electricté Produite [kWh] ']  = results['el_prod']
    df.at[row,'Electricté autoconsommée [kWh] ']  = results['el_selfcons']
    df.at[row,'Electricté injectée [kWh] '] = results['el_soldtogrid']
    df.at[row,'Electricté achetée [kWh] '] = results['el_boughtfromgrid']
    
    df.at[row,'Heure Talon [1h-7h] [kWh]'] = results['cons_heel']
    df.at[row,'Heure creuse [23h-1h et 7h-10h] [kWh]']  = results['cons_hollow']
    df.at[row,'Heure pleine [10h-18h et 21h-23h]  [kWh]']  = results['cons_full']
    df.at[row,'Heure pointe [18h-21h]  [kWh]']  =  results['cons_peak']
    
    df.at[row,'Variation Heure Talon [1h-7h]'] = results['varperc_cons_heel']
    df.at[row,'Variation Heure creuse [23h-1h et 7h-10h]'] = results['varperc_cons_hollow']
    df.at[row,'Variation Heure pleine [10h-18h et 21h-23h]'] =  results['varperc_cons_full']
    df.at[row,'Variation Heure pointe [18h-21h]'] = results['varperc_cons_peak']
    
    df.at[row,'Taux autosuff [%]']  = results['selfsuffrate']
    df.at[row,'Taux autocons [%]']  = results['selfconsrate']
    
    df.at[row,'Electricitédéplacée [kWh]'] = results['el_shifted']
    
    df.at[row,'Electricté vendue [€] '] = results['el_soldtogrid_rev']
    df.at[row,'Electricté achetée [€] '] = results['el_boughtfromgrid_cost']
    df.at[row,'Electricté couts du réseau [€]'] = results['annualgridcosts']

    df.at[row,'Elec dépenses [€] = Vend-Ach-CoutsRes'] = results['el_netexpend']
    df.at[row,"Coût de l'électricité [€/kWh]"] = results['el_costperkwh']
        
    df.at[row,'PBP [years]'] = results['PBP']
    df.at[row,'NPV [€]'] = results['NPV']
    df.at[row,'PI [-]'] = results['PI'] 
    
    df.to_excel(file,sheet_name=sheet)













