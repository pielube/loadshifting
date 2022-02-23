
import numpy as np
import pandas as pd
import datetime
import numpy_financial as npf
import random
from strobe.RC_BuildingSimulator import Zone


from joblib import Memory
memory = Memory('./cache/', verbose=1)



def scale_timeseries(data,index):
    ''' 
    Function that takes a pandas Dataframe as input and interpolates it to the proper datetime index
    '''
    if isinstance(data,pd.Series):
        data_df = pd.DataFrame(data)
    elif isinstance(data,pd.DataFrame):
        data_df = data
    else:
        raise Exception("The input must be a pandas series or dataframe")
    dd = data_df.reindex(data_df.index.union(index)).interpolate(method='time').reindex(index)
    if isinstance(data,pd.Series):
        return dd.iloc[:,0]
    else:
        return dd
    
    

def EconomicAnalysis(E,econ_param,ElPrices,timestep,demand_ref):
    '''
    Calculation of the profits linked to the PV/battery installation, user perspective
       
    :param E: Output of the "EnergyFlows" function: dictionary of the computed yearly qunatities relative to the PV battery installation 
    :param econ_param: Dictionary with the financial variables of the considered country   
    :param econ_param: Investment data. Defined as a dictionary with the fields 'FixedPVCost','PVCost_kW','FixedBatteryCost','BatteryCost_kWh','PVLifetime','BatteryLifetime','OM'
    :array demand: Energy consumption in the reference case
    :return: List comprising the Profitability Ratio and the system LCOE
    '''
        
    # Defining output dictionnary
    out = {}
    
    # Updating the fixed costs if PV or batteries capacities = 0
    if E['PVCapacity'] == 0:
        FixedPVCost = 0
    else:
        FixedPVCost = econ_param['FixedPVCost']
        
    if E['BatteryCapacity'] == 0:
        FixedBatteryCost = 0
    else:
        FixedBatteryCost = econ_param['FixedBatteryCost']
        
    out['PVCapacity'] = E['PVCapacity']
    out['BatteryCapacity'] = E['BatteryCapacity']
     
    # Load economic data:

    # General
    interest = econ_param['WACC'] # Discount rate, -
    net_metering = econ_param['net_metering']  # Boolean variable for the net metering scheme 
    years = econ_param['time_horizon'] # time horizon for the investment

    # Grid connection
    C_grid_fixed = econ_param['C_grid_fixed']  # Fixed grid tariff per year, €
    C_grid_kW    = econ_param['C_grid_kW']     # Fixed cost per installed grid capacity, €/kW 

    # Sell to the grid
    P_FtG      = econ_param['P_FtG']       # Purchase price of electricity fed to the grid, €/MWh  (price of energy sold to the grid)
    C_grid_FtG = 0.  # Grid fees for electricity fed to the grid, €/MWh      (cost to sell electricity to the grid)  
    C_TL_FtG   = 0.    # Tax and levies for electricity fed to the grid, €/MWh (cost to sell electricity to the grid)

    # Buy from the grid
    P_retail = ElPrices # array, it was econ_param['P_retail']

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
    PVInvestment = FixedPVCost + econ_param['PVCost_kW'] * E['PVCapacity']
    
    # Battery investment with one replacement after the battery lifetime (10 years)
    BatteryInvestment  = (FixedBatteryCost + econ_param['BatteryCost_kWh'] * E['BatteryCapacity'])
    
    out['CostPV'] = PVInvestment
    out['CostBattery'] = BatteryInvestment
    
    # Inverter
    # Inverter should be considered as well, with a lifetime of 10 years
    
    # Control strategy initial investment
    ControlInvestment = econ_param['FixedControlCost']   

    # Initial investment
    InitialInvestment =  PVInvestment + BatteryInvestment + ControlInvestment

    # Adding investment costs to cash flows array
    CashFlows[0]  = - InitialInvestment
    CashFlows[10] = - BatteryInvestment
    
    # O&M
    CashFlows[1:21] = CashFlows[1:21] - econ_param['OM'] * (BatteryInvestment + PVInvestment)
    
    # Annual costs for controller
    CashFlows[1:21] = CashFlows[1:21] - econ_param['AnnualControlCost']
    
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
    
    zerocross = np.where(np.diff(np.sign(NPVcurve)))[0]
    if len(zerocross) > 0: 
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
    NPV_Battery_reinvestment = (FixedBatteryCost + econ_param['BatteryCost_kWh'] * E['BatteryCapacity']) / (1+interest)**econ_param['BatteryLifetime']
    BatteryInvestment += NPV_Battery_reinvestment
    CRF = interest * (1+interest)**econ_param['PVLifetime']/((1+interest)**econ_param['PVLifetime']-1)
    NetSystemCost = PVInvestment * (1 - supportPV_INV) - supportPV_kW * E['PVCapacity']  \
                    + BatteryInvestment * (1 - supportBat_INV) - supportBat_kW * E['BatteryCapacity']
    AnnualInvestment = NetSystemCost * CRF + econ_param['OM'] * (BatteryInvestment + PVInvestment)
    out['costpermwh'] = ((AnnualInvestment + AnnualCostGrid - Income_FtG - (P_support_SC - C_grid_SC - C_TL_SC)*sum(E['SC']*timestep)/1000 + Cost_BtG) / sum(E['Load']*timestep))*1000. #eur/MWh
    out['cost_grid'] = AnnualCostGrid/sum(E['Load']*timestep)*1000
    
    return out

def EconomicAnalysisRefPV(E,econ_param,ElPrices,timestep,E_ref):
    '''
    Calculation of the profits linked to the PV/battery installation, user perspective
       
    :param E: Output of the "EnergyFlows" function: dictionary of the computed yearly qunatities relative to the PV battery installation 
    :param econ_param: Dictionary with the financial variables of the considered country   
    :param econ_param: Investment data. Defined as a dictionary with the fields 'FixedPVCost','PVCost_kW','FixedBatteryCost','BatteryCost_kWh','PVLifetime','BatteryLifetime','OM'
    :array demand: Energy consumption in the reference case
    :return: List comprising the Profitability Ratio and the system LCOE
    '''
        
    # Defining output dictionnary
    out = {}
    
    # Fixed PV costs always 0 as PV also in the reference case

    # Updating the fixed costs if PV or batteries capacities = 0
    if E['PVCapacity'] == 0:
        FixedPVCost = 0
    else:
        FixedPVCost = econ_param['FixedPVCost']
        
    if E['BatteryCapacity'] == 0:
        FixedBatteryCost = 0
    else:
        FixedBatteryCost = econ_param['FixedBatteryCost']
        
    out['PVCapacity'] = E['PVCapacity']
    out['BatteryCapacity'] = E['BatteryCapacity']
     
    # Load economic data:

    # General
    interest = econ_param['WACC'] # Discount rate, -
    net_metering = econ_param['net_metering']  # Boolean variable for the net metering scheme 
    years = econ_param['time_horizon'] # time horizon for the investment

    # Grid connection
    C_grid_fixed = econ_param['C_grid_fixed']  # Fixed grid tariff per year, €
    C_grid_kW    = econ_param['C_grid_kW']     # Fixed cost per installed grid capacity, €/kW 

    # Sell to the grid
    P_FtG      = econ_param['P_FtG']       # Purchase price of electricity fed to the grid, €/MWh  (price of energy sold to the grid)
    C_grid_FtG = 0.  # Grid fees for electricity fed to the grid, €/MWh      (cost to sell electricity to the grid)  
    C_TL_FtG   = 0.    # Tax and levies for electricity fed to the grid, €/MWh (cost to sell electricity to the grid)

    # Buy from the grid
    P_retail = ElPrices # array, it was econ_param['P_retail']

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
    PVInvestment = 0.
    
    # Battery investment with one replacement after the battery lifetime (10 years)
    BatteryInvestment  = (FixedBatteryCost + econ_param['BatteryCost_kWh'] * E['BatteryCapacity'])
    
    out['CostPV'] = PVInvestment
    out['CostBattery'] = BatteryInvestment
    
    # Inverter
    # Inverter should be considered as well, with a lifetime of 10 years
    
    # Control strategy initial investment
    ControlInvestment = econ_param['FixedControlCost']   

    # Initial investment
    InitialInvestment =  PVInvestment + BatteryInvestment + ControlInvestment

    # Adding investment costs to cash flows array
    CashFlows[0]  = - InitialInvestment
    CashFlows[10] = - BatteryInvestment
    
    # O&M
    CashFlows[1:21] = CashFlows[1:21] - econ_param['OM'] * (BatteryInvestment + PVInvestment)
    
    # Annual costs for controller
    CashFlows[1:21] = CashFlows[1:21] - econ_param['AnnualControlCost']
    
    # Annual costs for grid connection
    Capacity = np.max(E['Load'])
    AnnualCostGrid = C_grid_fixed + C_grid_kW * Capacity
    
    # Energy expenditure and revenues
    # Both in case of net metering or not
    
    if net_metering:
        
        # Case analyzed
        # Revenues selling to the grid
        Income_FtG = np.maximum(0,sum(E['ACGeneration']-E['Load'])*timestep) * (P_FtG - C_grid_FtG - C_TL_FtG)/1000
        Income_SC = 0
        # Expenditures buying from the grid
        Cost_BtG = np.maximum(sum(P_retail*(E['Load']-E['ACGeneration'])*timestep),0)
        
        # Reference case
        # Revenues selling to the grid
        Income_FtG_ref = np.maximum(0,sum(E_ref['ACGeneration']-E_ref['Load'])*timestep) * (P_FtG - C_grid_FtG - C_TL_FtG)/1000
        Income_SC_ref = 0
        # Expenditures buying from the grid
        Cost_BtG_ref = np.maximum(sum(P_retail*(E_ref['Load']-E_ref['ACGeneration'])*timestep),0) 
        
    else:
        
        # Case analyzed
        # Revenues selling to the grid
        Income_FtG = sum(E['ToGrid']*timestep) * (P_FtG - C_grid_FtG - C_TL_FtG)/1000
        Income_SC = 0
        # Expenditures buying from the grid
        Cost_BtG = sum(P_retail * E['FromGrid']*timestep)
        
        # Reference case
        # Revenues selling to the grid
        Income_FtG_ref = sum(E_ref['ToGrid']*timestep) * (P_FtG - C_grid_FtG - C_TL_FtG)/1000
        Income_SC_ref = 0
        # Expenditures buying from the grid
        Cost_BtG_ref = sum(P_retail * E_ref['FromGrid']*timestep)
    
    CashFlows[1:21] = CashFlows[1:21] + Income_FtG + Income_SC -Cost_BtG -(Income_FtG_ref + Income_SC_ref - Cost_BtG_ref - AnnualCostGrid)- AnnualCostGrid
    CashFlowsAct = np.zeros(len(CashFlows))
    NPVcurve = np.zeros(len(CashFlows))

    for i in range(len(CashFlows)):
        CashFlowsAct[i] = CashFlows[i]/(1+interest)**(i)

    NPVcurve[0] = CashFlowsAct[0]

    for i in range(len(CashFlows)-1):
        NPVcurve[i+1] = NPVcurve[i]+CashFlowsAct[i+1]
        
    NPV = npf.npv(interest,CashFlows)
    out['NPV'] = NPV
         
    zerocross = np.where(np.diff(np.sign(NPVcurve)))[0]
    if len(zerocross) > 0:
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
    out['ElBill'] = Income_FtG ++ Income_SC - Cost_BtG - AnnualCostGrid # eur/y
    out['ElBill_ref'] = Income_FtG_ref + Income_SC_ref - Cost_BtG_ref - AnnualCostGrid
       
    # LCOE equivalent, as if the grid was a generator
    NPV_Battery_reinvestment = (FixedBatteryCost + econ_param['BatteryCost_kWh'] * E['BatteryCapacity']) / (1+interest)**econ_param['BatteryLifetime']
    BatteryInvestment += NPV_Battery_reinvestment
    CRF = interest * (1+interest)**econ_param['PVLifetime']/((1+interest)**econ_param['PVLifetime']-1)
    NetSystemCost = PVInvestment * (1 - supportPV_INV) - supportPV_kW * E['PVCapacity']  \
                    + BatteryInvestment * (1 - supportBat_INV) - supportBat_kW * E['BatteryCapacity']
    AnnualInvestment = NetSystemCost * CRF + econ_param['OM'] * (BatteryInvestment + PVInvestment)
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


@memory.cache
def shift_appliance(app,admtimewin,probshift,max_shift=None,threshold_window=0,verbose=False):
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
    threshold_window: float [0,1]
        Share of the average cycle length below which an admissible time window is considered as unsuitable and discarded
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
    
    # check if admtimewin is boolean:
    if not admtimewin.dtype=='bool':
        if (admtimewin>1).any() or (admtimewin<0).any():
            print('WARNING: Some values of the admitted time windows are higher than 1 or lower than 0')
        admtimewin = (admtimewin>0)
    
    # Define the shifted consumption vector for the appliance:
    app_n = np.full(len(app),offset)
    
    # Shift the app consumption vector by one time step:
    app_s  = np.roll(app,1)
    
    # locate all the points whit a start or a shutdown
    starting_times = (app>0) * (app_s==0)
    stopping_times = (app_s>0) * (app==0)
    
    # List the indexes of all start-ups and shutdowns
    starts   = np.where(starting_times)[0]
    ends   = np.where(stopping_times)[0]
    means = (( starts + ends)/2 ).astype('int')
    lengths = ends - starts
    
    # Define the indexes of each admitted time window
    admtimewin_s = np.roll(admtimewin,1)
    admtimewin_s[0] = False; admtimewin[-1] = False
    adm_starts   = np.where(admtimewin * np.logical_not(admtimewin_s))[0]
    adm_ends   = np.where(admtimewin_s * np.logical_not(admtimewin))[0]
    adm_lengths = adm_ends - adm_starts
    adm_means = (( adm_starts + adm_ends)/2 ).astype('int')
    admtimewin_j = np.zeros(len(app),dtype='int')
    
    # remove all windows that are shorter than the average cycle length:
    tooshort = adm_lengths < lengths.mean() * threshold_window
    adm_means[tooshort] = -max_shift -999999            # setting the mean to a low value makes this window unavailable
    
    for j in range(len(adm_starts)):            # create a time vector with the index number of the current time window
        admtimewin_j[adm_starts[j]:adm_ends[j]] = j
    
    # For all activations events:
    for i in range(len(starts)):
        length = lengths[i]
        
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
                    delta = adm_lengths[j_min] - length
                    if delta < 0:                                        # if the admissible window is smaller than the activation length
                        t_start = int(adm_starts[j_min] - length/2)
                        t_start = np.minimum(t_start,len(app)-length)    # ensure that there is sufficient space for the whole activation at the end of the vector
                        app_n[t_start:t_start+length] += app[starts[i]:ends[i]] 
                        admtimewin[adm_starts[j_min]:adm_ends[j_min]] = False       # make the whole time window unavailable for further shifting
                        adm_means[j_min] = -max_shift -999999  
                    elif delta < length:                                    # This an arbitrary value
                        delay = random.randrange(1+delta)             # randomize the activation time within the allowed window
                        app_n[adm_starts[j_min]+delay:adm_starts[j_min]+delay+length] += app[starts[i]:ends[i]]
                        admtimewin[adm_starts[j_min]:adm_ends[j_min]] = False       # make the whole time window unavailable for further shifting
                        adm_means[j_min] = -max_shift -999999  
                    else:                                                    # the time window is longer than two times the activation. We split it and keep the first part
                        delay = random.randrange(1+length)                # randomize the activation time within the allowed window
                        app_n[adm_starts[j_min]+delay:adm_starts[j_min]+delay+length] += app[starts[i]:ends[i]]
                        admtimewin[adm_starts[j_min]:adm_starts[j_min]+2*length] = False       # make the first part of the time window unavailable for further shifting
                        adm_starts[j_min] = adm_starts[j_min]+2*length+1                   # Update the size of this time window
                        adm_means[j_min] = (( adm_starts[j_min] + adm_ends[j_min])/2 ).astype('int')
                        adm_lengths[j_min] = adm_ends[j_min] - adm_starts[j_min]
    app = app + offset
    enshift = np.abs(app_n - app).sum()/2
    
    if verbose: 
        if np.abs(app_n.sum() - app.sum())/app.sum() > 0.01:    # check that the total consumption is unchanged
            print('WARNING: the total shifted consumption is ' + str(app_n.sum()) + ' while the original consumption is ' + str(app.sum()))
        print(str(len(starts)) + ' duty cycles detected. ' + str(ncycshift) + ' cycles shifted in time')
        print(str(tooshort.sum()) + ' admissible time windows were discarded because they were too short')
        print('Total shifted energy : {:.2f}% of the total load'.format(enshift/app.sum()*100))

    return app_n,len(starts),ncycshift,enshift



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


    
































