
import numpy as np
import pandas as pd
from statistics import mean
from prosumpy import dispatch_max_sc,print_analysis
import datetime
import numpy_financial as npf
import random
from strobe.RC_BuildingSimulator import Zone
import os
import pickle
import time
import json
from typing import Dict, Any      # for the dictionnary hashing funciton
import hashlib  # for the dictionnary hashing funciton



def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()
    
def cache_func(func):
    '''
    Decorator function that saves the output of a function into a pickle file in 
    the /cache directory and reloads it at the next function calls.
    For the results to be reloaded, the parameters of the function should be 
    exactly the same. 
    To remain computationally efficient, the function inputs should not contain
    heavy data such as numpy arrays of pandas dataframes.
    
    @author: Sylvain Quoilin
    '''
    def timed(*args, **kwargs):
        # generating hash for the current config:   
        dhash = dict_hash({'list':args,'dict':kwargs})
        folder = os.path.join('cache',func.__name__)
        filename = os.path.join(folder, dhash)
        # create cache directory if not present
        if not os.path.isdir('cache'):
            os.mkdir('cache')
        if not os.path.isdir(folder):
            os.mkdir(folder)
        if os.path.isfile(filename):
            #results = pd.read_pickle('data.p')           # temporary file to speed up things
            print('Function "' + func.__name__ + '": Reading previous (cached) simulation data with hash ' + dhash)
            return pickle.load(open(filename,'rb'))
        else:
            print('Function "' + func.__name__ + '": No previous (cached) simulation found with hash ' + dhash)     
            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            print('    Function run in {:.2f} seconds'.format(te - ts))
            pickle.dump(result,open(filename,"wb"))   
            return result
    return timed    
    
    

def EconomicAnalysis(E,EconomicVar,Inv,ElPrices,timestep,demand_ref):
    '''
    Calculation of the profits linked to the PV/battery installation, user perspective
       
    :param E: Output of the "EnergyFlows" function: dictionary of the computed yearly qunatities relative to the PV battery installation 
    :param EconomicVar: Dictionary with the financial variables of the considered country   
    :param Inv: Investment data. Defined as a dictionary with the fields 'FixedPVCost','PVCost_kW','FixedBatteryCost','BatteryCost_kWh','PVLifetime','BatteryLifetime','OM'
    :array demand: Energy consumption in the reference case
    :return: List comprising the Profitability Ratio and the system LCOE
    '''
        
    # Defining output dictionnary
    out = {}
    
    # Updating the fixed costs if PV or batteries capacities = 0
    if E['CapacityPV'] == 0:
        FixedPVCost = 0
    else:
        FixedPVCost = Inv['FixedPVCost']
        
    if E['CapacityBattery'] == 0:
        FixedBatteryCost = 0
    else:
        FixedBatteryCost = Inv['FixedBatteryCost']
        
    out['CapacityPV'] = E['CapacityPV']
    out['CapacityBattery'] = E['CapacityBattery']
     
    # Load economic data:

    # General
    interest = EconomicVar['WACC'] # Discount rate, -
    net_metering = EconomicVar['net_metering']  # Boolean variable for the net metering scheme 
    years = EconomicVar['time_horizon'] # time horizon for the investment

    # Grid connection
    C_grid_fixed = EconomicVar['C_grid_fixed']  # Fixed grid tariff per year, €
    C_grid_kW    = EconomicVar['C_grid_kW']     # Fixed cost per installed grid capacity, €/kW 

    # Sell to the grid
    P_FtG      = EconomicVar['P_FtG']       # Purchase price of electricity fed to the grid, €/MWh  (price of energy sold to the grid)
    C_grid_FtG = 0.  # Grid fees for electricity fed to the grid, €/MWh      (cost to sell electricity to the grid)  
    C_TL_FtG   = 0.    # Tax and levies for electricity fed to the grid, €/MWh (cost to sell electricity to the grid)

    # Buy from the grid
    P_retail = ElPrices # array, it was EconomicVar['P_retail']

    # PV and batteries supports
    supportPV_INV  = 0.  # Investment support, % of investment
    supportPV_kW   = 0.  # Investment support proportional to the size, €/kW
    supportBat_INV = 0.  # to be added
    supportBat_kW  = 0.  # to be added

    # Self consumption
    P_support_SC = 0.    # Support to self-consumption, €/MWh                    (incentive to self consumption)  
    C_grid_SC    = 0.    # Grid fees for self-consumed electricity, €/MWh        (cost to do self consumption)
    C_TL_SC      = 0.    # Tax and levies for self-consumed electricity, €/MWh   (cost to do self consumption)
    
    
    # Initialize cash flows array 
    CashFlows = np.zeros(int(years)+1)    

    # PV investment, no replacements:
    PVInvestment = FixedPVCost + Inv['PVCost_kW'] * E['CapacityPV']
    
    # Battery investment with one replacement after the battery lifetime (10 years)
    BatteryInvestment  = (FixedBatteryCost + Inv['BatteryCost_kWh'] * E['CapacityBattery'])
    
    out['CostPV'] = PVInvestment
    out['CostBattery'] = BatteryInvestment
    
    # Inverter
    # Inverter should be considered as well, with a lifetime of 10 years
    
    # Control strategy initial investment
    ControlInvestment = Inv['FixedControlCost']   

    # Initial investment
    InitialInvestment =  PVInvestment + BatteryInvestment + ControlInvestment

    # Adding investment costs to cash flows array
    CashFlows[0]  = - InitialInvestment
    CashFlows[10] = - BatteryInvestment
    
    # O&M
    CashFlows[1:21] = CashFlows[1:21] - Inv['OM'] * (BatteryInvestment + PVInvestment)
    
    # Annual costs for controller
    CashFlows[1:21] = CashFlows[1:21] - Inv['AnnualControlCost']
    
    # Annual costs for grid connection
    Capacity = np.max(E['Load'])
    AnnualCostGrid = C_grid_fixed + C_grid_kW * Capacity
    
    # Energy expenditure and revenues
    # Both in case of net metering or not
    
    if net_metering:
        # Revenues selling to the grid
        # Fixed selling price and cost
        Income_FtG = np.maximum(0,sum(E['ACGeneration']-E['Load'])*timestep) * (P_FtG - C_grid_FtG - C_TL_FtG)/1000
        Income_SC = 0
        """
        Old equations:
        Income_FtG = np.maximum(0,E['ACGeneration']-E['Load']) * (P_FtG - C_grid_FtG - C_TL_FtG)/1000
        Income_SC = (P_support_SC + P_retail - C_grid_SC - C_TL_SC) * np.minimum(E['ACGeneration'],E['Load'])/1000  # the retail price on the self-consumed part is included here since it can be considered as a support to SC    
        """
        # Expenditures buying from the grid
        Cost_BtG = np.maximum(sum(P_retail*(E['Load']-E['ACGeneration'])*timestep),0)      
    else:
        # Revenues selling to the grid
        # Fixed selling price and cost
        Income_FtG = sum(E['ToGrid']*timestep) * (P_FtG - C_grid_FtG - C_TL_FtG)/1000
        Income_SC = 0
        """
        Old equations:
        Income_FtG = E['ToGrid'] * (P_FtG - C_grid_FtG - C_TL_FtG)/1000
        Income_SC = (P_support_SC + P_retail - C_grid_SC - C_TL_SC) * E['SC']/1000  # the retail price on the self-consumed part is included here since it can be considered as a support to SC
        """
        # Expenditures buying from the grid
        Cost_BtG = sum(P_retail * E['FromGrid']*timestep)
    
    
    # Reference case energy expenditure
    RefEnExpend = sum(demand_ref*ElPrices*timestep)
    
    CashFlows[1:21] = CashFlows[1:21] + Income_FtG + Income_SC - Cost_BtG - AnnualCostGrid + RefEnExpend
    CashFlowsAct = np.zeros(len(CashFlows))
    NPVcurve = np.zeros(len(CashFlows))

    for i in range(len(CashFlows)):
        CashFlowsAct[i] = CashFlows[i]/(1+interest)**(i)

    NPVcurve[0] = CashFlowsAct[0]

    for i in range(len(CashFlows)-1):
        NPVcurve[i+1] = NPVcurve[i]+CashFlowsAct[i+1]
        
    NPV = npf.npv(interest,CashFlows)
    out['NPV'] = NPV
       
    # plt.plot(NPVcurve)
       
    zerocross = np.where(np.diff(np.sign(NPVcurve)))[0]
    if bool(zerocross):
        x1 = zerocross[0]
        x2 = zerocross[0]+1
        xs = [x1,x2]
        y1 = NPVcurve[zerocross[0]]
        y2 = NPVcurve[zerocross[0]+1]
        ys = [y1,y2]
        PBP = np.interp(0,ys,xs)
    else:
        PBP = None #9999.
    
    out['PBP'] = PBP
    
    IRR = npf.irr(CashFlows)
    out['IRR'] = IRR
    
    if InitialInvestment == 0:
        PI = None
    else:
        PI = NPV/InitialInvestment
    out['PI'] = PI

    # Annual electricity bill
    out['RevSelling'] = Income_FtG
    out['CostBuying'] = Cost_BtG
    out['AnnualGridCosts'] = AnnualCostGrid
    out['ElBill'] = Income_FtG - Cost_BtG - AnnualCostGrid # eur/y
       
    # LCOE equivalent, as if the grid was a generator
    NPV_Battery_reinvestment = (FixedBatteryCost + Inv['BatteryCost_kWh'] * E['CapacityBattery']) / (1+interest)**Inv['BatteryLifetime']
    BatteryInvestment += NPV_Battery_reinvestment
    CRF = interest * (1+interest)**Inv['PVLifetime']/((1+interest)**Inv['PVLifetime']-1)
    NetSystemCost = PVInvestment * (1 - supportPV_INV) - supportPV_kW * E['CapacityPV']  \
                    + BatteryInvestment * (1 - supportBat_INV) - supportBat_kW * E['CapacityBattery']
    AnnualInvestment = NetSystemCost * CRF + Inv['OM'] * (BatteryInvestment + PVInvestment)
    out['costpermwh'] = ((AnnualInvestment + AnnualCostGrid - Income_FtG - (P_support_SC - C_grid_SC - C_TL_SC)*sum(E['SC']*timestep)/1000 + Cost_BtG) / sum(E['Load']*timestep))*1000. #eur/MWh
    out['cost_grid'] = AnnualCostGrid/sum(E['Load']*timestep)*1000
    
    return out


def yearlyprices(scenario,timeslots,prices,stepperhour):

    stepperhour = int(stepperhour)    

    endday = datetime.datetime.strptime('1900-01-02 00:00:00',"%Y-%m-%d %H:%M:%S")

    HSdayhours = []
    HSdaytariffs = []
    CSdayhours = []
    CSdaytariffs = []
    
    for i in timeslots['HSday']:
        starthour = datetime.datetime.strptime(i[0],'%H:%M:%S')
        HSdayhours.append(starthour)
    
    for i in range(len(HSdayhours)):
        start = HSdayhours[i]
        end = HSdayhours[i+1] if i < len(HSdayhours)-1 else endday
        delta = end - start
        for j in range(stepperhour*int(delta.seconds/3600)):
            price = prices[scenario][timeslots['HSday'][i][1]]/1000.
            HSdaytariffs.append(price)
    
    for i in timeslots['CSday']:
        starthour = datetime.datetime.strptime(i[0],'%H:%M:%S')
        CSdayhours.append(starthour)
    
    for i in range(len(CSdayhours)):
        start = CSdayhours[i]
        end = CSdayhours[i+1] if i < len(CSdayhours)-1 else endday
        delta = end - start
        for j in range(stepperhour*int(delta.seconds/3600)):
            price = prices[scenario][timeslots['CSday'][i][1]]/1000.
            CSdaytariffs.append(price)
    
    startyear = datetime.datetime.strptime('2015-01-01 00:00:00',"%Y-%m-%d %H:%M:%S")
    HSstart = datetime.datetime.strptime(timeslots['HSstart'],"%Y-%m-%d %H:%M:%S")
    CSstart = datetime.datetime.strptime(timeslots['CSstart'],"%Y-%m-%d %H:%M:%S")
    endyear = datetime.datetime.strptime('2016-01-01 00:00:00',"%Y-%m-%d %H:%M:%S")
    
    ytariffs = []
    
    deltaCS1 = HSstart - startyear
    deltaHS  = CSstart - HSstart
    deltaCS2 = endyear - CSstart
    
    for i in range(deltaCS1.days):
        ytariffs.extend(CSdaytariffs)
    for i in range(deltaHS.days):
        ytariffs.extend(HSdaytariffs)
    for i in range(deltaCS2.days):
        ytariffs.extend(CSdaytariffs)
    
    out = np.asarray(ytariffs)
            
    return out

def mostrapcurve(demands,param_tech,inputs,EconomicVar,Inv,ElPrices,timestep,columns):
    
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
        print_analysis(pv, demand, param_tech, outputs)
        
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
    mostrapcurveindex = min(range(len(var)), key=var.__getitem__)
    
    return mostrapcurveindex


def run(pv,demand,param_tech,inputs,EconomicVar,Inv,ElPrices,timestep,columns,prices,scenario,demand_ref):
    
    heel = np.where(ElPrices == prices[scenario]['heel']/1000,1.,0.)
    hollow = np.where(ElPrices == prices[scenario]['hollow']/1000,1.,0.)
    full =  np.where(ElPrices == prices[scenario]['full']/1000,1.,0.)
    peak =  np.where(ElPrices == prices[scenario]['peak']/1000,1.,0.)
    
    demand = demand[columns]
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
    print_analysis(pv, demand, param_tech, outputs)
    
    inputs['ACGeneration'] = pv.to_numpy() # should be equal to outputs['inv2grid']+outputs['inv2load']
    inputs['Load'] = demand.to_numpy()
    inputs['ToGrid'] = outputs['inv2grid']
    inputs['FromGrid'] = outputs['grid2load']
    inputs['SC'] = outputs['inv2load']
    inputs['FromBattery'] = outputs['store2inv']
    
    res = EconomicAnalysis(inputs,EconomicVar,Inv,ElPrices,timestep,demand_ref)
    
    out = {}

    consrefheel = np.sum(demand_ref*heel)*timestep
    consrefhollow = np.sum(demand_ref*hollow)*timestep
    consreffull = np.sum(demand_ref*full)*timestep
    consrefpeak = np.sum(demand_ref*peak)*timestep

    out['CapacityPV'] = res['CapacityPV'] 
    out['CapacityBattery'] = res['CapacityBattery'] 
    
    out['CostPV'] = res['CostPV']
    out['CostBattery'] = res['CostBattery']

    out['peakdem'] = np.max(demand)
    
    out['cons_total'] = np.sum(demand)*timestep

    out['cons_heel'] = np.sum(outputs['grid2load']*heel)*timestep
    out['cons_hollow'] = np.sum(outputs['grid2load']*hollow)*timestep
    out['cons_full'] = np.sum(outputs['grid2load']*full)*timestep
    out['cons_peak'] = np.sum(outputs['grid2load']*peak)*timestep
        
    out['varperc_cons_heel'] = (consrefheel-out['cons_heel'])/out['cons_heel']*100
    out['varperc_cons_hollow'] = (consrefhollow-out['cons_hollow'])/out['cons_hollow']*100
    out['varperc_cons_full'] = (consreffull-out['cons_full'])/out['cons_full']*100
    out['varperc_cons_peak'] = (consrefpeak-out['cons_peak'])/out['cons_peak']*100
    
    out['el_prod'] = np.sum(pv)*timestep
    out['el_selfcons'] = np.sum(outputs['inv2load'])*timestep
    out['el_soldtogrid'] = np.sum(outputs['inv2grid'])*timestep
    out['el_boughtfromgrid'] = np.sum(outputs['grid2load'])*timestep
    
    out['selfsuffrate'] = out['el_selfcons']/out['cons_total']
    if out['el_prod'] == 0:
        out['selfconsrate'] = 0
    else:
        out['selfconsrate'] = out['el_selfcons']/out['el_prod']
    
    out['el_soldtogrid_rev'] = res['RevSelling']
    out['el_boughtfromgrid_cost'] = res['CostBuying']
    out['annualgridcosts'] = res['AnnualGridCosts']
    out['el_netexpend'] = res['ElBill']
    out['el_costperkwh'] = res['costpermwh']/1000.
    
    out['PBP'] = res['PBP']
    out['NPV'] = res['NPV']
    out['PI'] = res['PI']
    
    return out
    

def strategy1(app,admtimewin,probshift):
   
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
            
            enshift += np.sum(app_n[newstart:newend])
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
                
    return app_n,ncyc,ncycshift,maxshift,avgshift,ncycnotshift,enshift



def shift_appliance(app,admtimewin,probshift,max_shift=None,verbose=False):
    '''
    This function shifts the duty duty cycles of a particular appliances according
    to a vector of admitted time windows.

    Parameters
    ----------
    app : numpy.array
        Original appliance consumption vector to be shifted
    admtimewin : numpy.array
        Vector of admitted time windows, where load should be shifted.
    probshift : float
        Probability (between 0 and 1) of a given cycle to be shifted
    max_shift : int
        Maximum number of time steps over which a duty cycle can be shifted
    verbose : bool
        Print messages or not. The default is False.

    Returns
    -------
    tuple with the shifted appliance load, the total number of duty cycles and 
    the number of shifted cycles

    '''
    ncycshift = 0                   # initialize the counter of shifted duty cycles
    if max_shift is None:
        max_shift = 24*60                    # maximmum time over which load can be shifted
    
    #remove offset from consumption vector:
    offset = app.min()
    app = app - offset
    
    # Define the shifted consumption vector for the appliance:
    app_n = np.full(len(app),offset)
    
    # Shift the app consumption vector by one time step:
    app_s  = np.roll(app,1)
    
    # locate all the points whit a start or a shutdown
    starting_times = (app-app_s>1) * (app_s==0)
    stopping_times = (app-app_s<-1) * (app==0)
    
    # List the indexes of all start-ups and shutdowns
    starts   = np.where(starting_times)[0]
    ends   = np.where(stopping_times)[0]
    means = (( starts + ends)/2 ).astype('int')
    
    # Define the indexes of each admitted time window
    admtimewin_s = np.roll(admtimewin,1)
    adm_starts   = np.where(admtimewin-admtimewin_s>0.1)[0]
    adm_ends   = np.where(admtimewin-admtimewin_s<-0.1)[0]
    adm_means = (( adm_starts + adm_ends)/2 ).astype('int')
    admtimewin_j = np.zeros(len(app),dtype='int')
    
    for j in range(len(adm_starts)):            # create a time vector with the index number of the current time window
        admtimewin_j[adm_starts[j]:adm_ends[j]] = j
    
    #Change the time window array to boolean
    if admtimewin.dtype != 'bool':
        admtimewin = (admtimewin == 1)    
    
    # For all activations events:
    for i in range(len(starts)):
        length = ends[i]- starts[i]
        
        if admtimewin[starts[i]] and admtimewin[ends[i]]:           # if the whole activation length is within the admitted time windows
            app_n[starts[i]:ends[i]] += app[starts[i]:ends[i]]
            j = admtimewin_j[starts[i]]
            admtimewin[adm_starts[j]:adm_ends[j]] = False       # make the whole time window unavailable for further shifting
            adm_means[j] = -max_shift -999999
            
        else:     # if the activation length is outside admitted windows
            if random.random() > probshift:
                app_n[starts[i]:ends[i]] += app[starts[i]:ends[i]]
            else:
                j_min = np.argmin(np.abs(adm_means-means[i]))          # find the closest admissible time window
                if np.abs(adm_means[j_min]-means[i]) > max_shift:     # The closest time window is too far away, no shifting possible
                    app_n[starts[i]:ends[i]] += app[starts[i]:ends[i]]
                else:
                    ncycshift += 1                                      # increment the counter of shifted cycles
                    delta = (adm_ends[j_min] - adm_starts[j_min]) - length
                    if delta < 0:                                        # if the admissible window is smaller than the activation length
                        t_start = int(adm_starts[j_min] - length/2)
                        app_n[t_start:t_start+length] += app[starts[i]:ends[i]] 
                    else:
                        delay = random.randrange(1+int(delta/2))             # dandomize the activation time within the allowed window
                        app_n[adm_starts[j_min]+delay:adm_starts[j_min]+delay+length] += app[starts[i]:ends[i]]
                    admtimewin[adm_starts[j_min]:adm_ends[j_min]] = False       # make the whole time window unavailable for further shifting
                    adm_means[j_min] = -max_shift -999999  
    
    app = app + offset
    enshift = np.abs(app_n - app).sum()/2
    
    if verbose: 
        if np.abs(app_n.sum() - app.sum())/app.sum() > 0.01:    # check that the total consumption is unchanged
            print('WARNING: the total shifted consumption is ' + str(app_n.sum()) + ' while the original consumption is ' + str(app.sum()))
        print(str(len(starts)) + ' duty cycles detected. ' + str(ncycshift) + ' cycles shifted in time')
        print('Total shifted energy : {:.2f}% of the total load'.format(enshift/app.sum()*100))

    return app_n,len(starts),ncycshift,enshift



def writetoexcel(file,sheet,results,row):
    
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
            
            enshift += np.sum(app_n[newstart:newend])
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



def HPSizing(inputs,fracmaxP):

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

        
    else:
        # Heat pump size given as an input
        QheatHP = inputs['HP']['HeatPumpThermalPower']
    
    return QheatHP

def COP_Tamb(Temp):
    COP = 0.001*Temp**2 + 0.0471*Temp + 2.1259
    return COP


    
































