
import numpy as np
import pandas as pd
import datetime
import numpy_financial as npf
import random
from strobe.RC_BuildingSimulator import Zone

import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

from joblib import Memory
memory = Memory(__location__ + '/cache/', verbose=1)



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
    
    

def EconomicAnalysis(E,econ_param,yenprices,ygridfees,timestep,demand_ref):
    '''
    Economic analysis
    NPV analysis and mean electricity price
       
    E: dictionary of yearly arrays of energy flows and PV and battery capacity 
    econ_param: Dictionary with economic param required for the analysis   
    ElPrices: array of yearly energy prices
    timestep: timestep of the arrays
    demand_ref: power demand curve in the reference scenario
    
    out: dictionary with results of the economic analysis
    '''
        
    # Defining output dictionnary
    out = {}
    
    # Adding PV and battery capacities to outputs
    out['PVCapacity'] = E['PVCapacity']
    out['BatteryCapacity'] = E['BatteryCapacity']
    
    # Updating fixed costs if PV or battery capacities = 0
    
    if E['PVCapacity'] == 0:
        FixedPVCost = 0
        FixedInverterCost = 0
    else:
        FixedPVCost = econ_param['FixedPVCost']
        FixedInverterCost = econ_param['FixedInverterCost']
        
    if E['BatteryCapacity'] == 0:
        FixedBatteryCost = 0
    else:
        FixedBatteryCost = econ_param['FixedBatteryCost']
     
    # Load economic data

    # General
    interest = econ_param['WACC']              # Discount rate, -
    elpriceincr = econ_param['elpriceincr']    # Annual electricity price increase, -
    net_metering = econ_param['net_metering']  # Boolean variable for the net metering scheme 
    years = econ_param['time_horizon']         # Time horizon of the investment
    start_year = econ_param['start_year']      # year in which the analysis start

    # Buying from the grid
    P_retail = yenprices + ygridfees # Array of prices
    C_grid_fixed = econ_param['C_grid_fixed'] # Fixed grid tariff per year, €
    C_grid_kW    = econ_param['C_grid_kW']    # Fixed cost per installed grid capacity, €/kW 

    # Selling to the grid
    P_FtG      = econ_param['P_FtG'] # Purchase price of electricity fed to the grid, €/MWh  (price of energy sold to the grid)
    C_grid_FtG = 0.                  # Grid fees for electricity fed to the grid, €/MWh      (cost to sell electricity to the grid)  
    C_TL_FtG   = 0.                  # Tax and levies for electricity fed to the grid, €/MWh (cost to sell electricity to the grid)
    C_pros_tax = econ_param['C_pros_tax'] # Prosumer tax, €/kW  fixed cost per min power installed between PV and inverter

    # PV and batteries supports
    supportPV_INV   = 0.  # Investment support, % of investment
    supportPV_kW    = 0.  # Investment support proportional to the size, €/kW
    supportBat_INV  = 0.  # Investment support, % of investment
    supportBat_kWh  = 0.  # Investment support proportional to the size, €/kW

    # Self consumption
    P_support_SC = 0. # Support to self-consumption, €/MWh                  (incentive to self consumption)  
    C_grid_SC    = 0. # Grid fees for self-consumed electricity, €/MWh      (cost to do self consumption)
    C_TL_SC      = 0. # Tax and levies for self-consumed electricity, €/MWh (cost to do self consumption)
 
    # Investment and replacement costs  

    # PV investment cost
    PVInvestment = FixedPVCost + econ_param['PVCost_kW'] * E['PVCapacity']
    out['CostPV'] = PVInvestment

    # Battery investment cost
    BatteryInvestment  = FixedBatteryCost + econ_param['BatteryCost_kWh'] * E['BatteryCapacity']
    out['CostBattery'] = BatteryInvestment
    
    # Inverter investment cost
    InverterInvestment = FixedInverterCost + econ_param['InverterCost_kW'] * E['PVCapacity']
    out['CostInverter'] = InverterInvestment
    
    # Control strategy initial investment
    ControlInvestment = econ_param['FixedControlCost']   

    # Initial investment
    InitialInvestment =  PVInvestment + BatteryInvestment + InverterInvestment + ControlInvestment

    # Initialize cashflows array 
    CashFlows = np.zeros(int(years)+1)   

    # Adding initial investment costs to cashflows array
    CashFlows[0]  = - InitialInvestment
    
    # Adding replacement costs to cashflows array
    NBattRep = int((years-1)/econ_param['BatteryLifetime'])
    for i in range(NBattRep):
        iyear = (i+1)*econ_param['BatteryLifetime']
        CashFlows[iyear] = - BatteryInvestment
    
    NInvRep = int((years-1)/econ_param['InverterLifetime'])
    for i in range(NInvRep):
        iyear = (i+1)*econ_param['InverterLifetime']
        CashFlows[iyear] = - InverterInvestment
        
    # Other costs
    
    # O&Ms
    CashFlows[1:years+1] = CashFlows[1:years+1] - econ_param['OM'] * (PVInvestment + BatteryInvestment)
    
    # Annual costs for controller
    CashFlows[1:years+1] = CashFlows[1:years+1] - econ_param['AnnualControlCost']
   
    # Energy expenditure and revenues
    # Both in case of net metering or not
    
    if net_metering:
        print('WARNING: net metering to be revised')
        # # Revenues selling to the grid
        # # Fixed selling price and cost
        # Income_FtG = np.maximum(0,sum(E['ACGeneration']-E['Load'])*timestep) * (P_FtG - C_grid_FtG - C_TL_FtG)/1000        
        # Income_SC = (P_support_SC - C_grid_SC - C_TL_SC)*sum(E['SC']*timestep)/1000
        # # Energy expenditure buying from the grid
        # Cost_BtG_energy = np.maximum(sum(yenprices*(E['Load']-E['ACGeneration'])*timestep),0)
        # Cost_BtG_grid   = np.maximum(sum(ygridfees*(E['Load']-E['ACGeneration'])*timestep),0)
        # Cost_BtG = Cost_BtG_energy + Cost_BtG_grid

    else:
        
        # Selling to the grid
        Income_FtG = sum(E['ToGrid']*timestep) * P_FtG/1000
        Cost_FtG = sum(E['ToGrid']*timestep) * (C_grid_FtG + C_TL_FtG)/1000
        Total_FtG = Income_FtG - Cost_FtG 
        # Buying from the grid
        Cost_BtG_energy = sum(yenprices * E['FromGrid']*timestep)
        Cost_BtG_grid   = sum(ygridfees * E['FromGrid']*timestep)
        Total_BtG = Cost_BtG_energy + Cost_BtG_grid
        # Self-consumption support schemes
        Income_SC = (P_support_SC - C_grid_SC - C_TL_SC)*sum(E['SC']*timestep)/1000

        # Pre 2030 revenues for selling to the grid if PV installed before 2024
        if (E['PVCapacity'] > 0 and start_year < 2024):
            
            # Selling to the grid
            Income_FtG_pre2030 = sum(E['ToGrid']*yenprices*timestep)
            Cost_FtG_pre2030 = np.maximum(sum(E['ToGrid']*ygridfees*timestep),C_pros_tax*E['PVCapacity'])
            Total_FtG_pre2030 = Income_FtG_pre2030 - Cost_FtG_pre2030


    # Annual costs for grid connection
    Capacity = np.max(E['Load'])
    AnnualCostGrid = C_grid_fixed + C_grid_kW * Capacity 
       
    # Adding revenues and expenditures from buying and selling  
    if (E['PVCapacity'] > 0 and start_year < 2024):
        end2030 = 2030-start_year
        for i in range(1,end2030+1):
            CashFlows[i] = CashFlows[i] + Income_SC - AnnualCostGrid + (Total_FtG_pre2030 - Total_BtG)*(1+elpriceincr)**(i-1)
        for i in range(end2030+2,years+1):
            CashFlows[i] = CashFlows[i] + Income_SC - AnnualCostGrid + (Total_FtG         - Total_BtG)*(1+elpriceincr)**(i-1)
    else:
        for i in range(1,years+1):
            CashFlows[i] = CashFlows[i] + Income_SC - AnnualCostGrid + (Total_FtG         - Total_BtG)*(1+elpriceincr)**(i-1)
    
    # Reference case energy expenditure
    # Buying all energy from the grid
    RefEnExpend = sum(demand_ref*P_retail*timestep)
    
    # Adding energy expenditure from reference case as savings
    for i in range(1,years+1):
        CashFlows[i] = CashFlows[i] + RefEnExpend*(1+elpriceincr)**(i-1)
     
    # Actualized cashflows
    CashFlowsAct = np.zeros(len(CashFlows))
    for i in range(len(CashFlows)):
        CashFlowsAct[i] = CashFlows[i]/(1+interest)**(i)

    # NPV curve        
    NPVcurve = np.zeros(len(CashFlows))
    NPVcurve[0] = CashFlowsAct[0]
    for i in range(len(CashFlows)-1):
        NPVcurve[i+1] = NPVcurve[i]+CashFlowsAct[i+1]

    # Final NPV        
    NPV = npf.npv(interest,CashFlows)
    out['NPV'] = NPV

    # Pay Back Period    
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
        PBP = None  
    out['PBP'] = PBP
    
    # Internal Rate of Return
    IRR = npf.irr(CashFlows)
    out['IRR'] = IRR

    # Profit Index    
    if InitialInvestment == 0:
        PI = None
    else:
        PI = NPV/InitialInvestment
    out['PI'] = PI

    # Annual electricity bill
    
    out["EnSold"]     = Income_FtG
    out["CostToSell"] = Cost_FtG
    out["TotalSell"]  = Total_FtG
    
    out["EnBought"]  = Cost_BtG_energy
    out["CostToBuy"] = Cost_BtG_grid
    out["TotalBuy"]  = Total_BtG
    
    out['ElBill'] = Total_FtG - Total_BtG - AnnualCostGrid + Income_SC
       
    # LCOE equivalent, as if the grid was a generator
    
    # Total actualized battery investment, accounting for replacements
    TotActBatteryInvestment = BatteryInvestment   
    for i in range(NBattRep):
        iyear = (i+1)*econ_param['BatteryLifetime']
        NPV_Battery_reinvestment = (BatteryInvestment) / (1+interest)**iyear
        TotActBatteryInvestment += NPV_Battery_reinvestment

    # Total actualized inverter investment, accounting for replacements    
    TotActInverterInvestment = InverterInvestment
    for i in range(NInvRep):
        iyear = (i+1)*econ_param['InverterLifetime']
        NPV_Inverter_reinvestment = (InverterInvestment) / (1+interest)**iyear
        TotActInverterInvestment += NPV_Inverter_reinvestment
    
    # Net system costs
    NetSystemCost = PVInvestment * (1 - supportPV_INV) - supportPV_kW * E['PVCapacity']  \
                  + TotActBatteryInvestment * (1 - supportBat_INV) - supportBat_kWh * E['BatteryCapacity'] \
                  + TotActInverterInvestment
                  
    # Capital Recovery Factor
    CRF = interest * (1+interest)**years/((1+interest)**years-1)
    
    # Annual investment costs             
    AnnualInvestment = NetSystemCost * CRF + econ_param['OM'] * (PVInvestment + BatteryInvestment)
    
    # Electricity price per MWh
    ECost = AnnualInvestment - (Total_FtG - Total_BtG - AnnualCostGrid + Income_SC)
    
    # If change of tarification after 2030, Ecost averaged pre-post 2030
    if (E['PVCapacity'] > 0 and start_year < 2024):
        yearspre2030 = 2030-start_year
        ECost_pre2030 = AnnualInvestment - (Total_FtG_pre2030 - Total_BtG - AnnualCostGrid + Income_SC)
        ECost = (ECost_pre2030*yearspre2030+ECost*(years-yearspre2030)) / years
 
    # LCOE equivalent, electricity price per MWhh
    out['costpermwh'] = (ECost / sum(E['Load']*timestep))*1000. #eur/MWh
    
    # Grid cost component of the final energy price
    out['cost_grid'] = AnnualCostGrid/sum(E['Load']*timestep)*1000
    
    return out

def EconomicAnalysisRefPV(E,econ_param,yenprices,ygridfees,timestep,E_ref):
    '''
    Economic analysis with PV panels being reference case
    NPV analysis and mean electricity price
    
       
    E: dictionary of yearly arrays of energy flows and PV and battery capacity 
    econ_param: Dictionary with economic param required for the analysis   
    ElPrices: array of yearly energy prices
    timestep: timestep of the arrays
    E_ref:  dictionary of yearly arrays of energy flows in ref scenario (only PV)
    
    out: dictionary with results of the economic analysis
    '''
        
    # Defining output dictionnary
    out = {}
    
    # Adding PV and battery capacities to outputs
    out['PVCapacity'] = E['PVCapacity']
    out['BatteryCapacity'] = E['BatteryCapacity']
    
    # If PV capacity is zero, no need to calcuate anything
    if E['PVCapacity'] == 0:
        out['PBP'] = 0
        out['IRR'] = 0
        out['PI']  = 0
        
    
    # Updating fixed costs if battery caapcity = 0
    if E['BatteryCapacity'] == 0:
        FixedBatteryCost = 0
    else:
        FixedBatteryCost = econ_param['FixedBatteryCost']
     
    # Load economic data

    # General
    interest = econ_param['WACC']              # Discount rate, -
    elpriceincr = econ_param['elpriceincr']    # Annual electricity price increase, -
    net_metering = econ_param['net_metering']  # Boolean variable for the net metering scheme 
    years = econ_param['time_horizon']         # Time horizon of the investment
    start_year = econ_param['start_year']      # year in which the analysis start

    # Buying from the grid
    P_retail = yenprices + ygridfees # Array of prices
    C_grid_fixed = econ_param['C_grid_fixed'] # Fixed grid tariff per year, €
    C_grid_kW    = econ_param['C_grid_kW']    # Fixed cost per installed grid capacity, €/kW 

    # Selling to the grid
    P_FtG      = econ_param['P_FtG'] # Purchase price of electricity fed to the grid, €/MWh  (price of energy sold to the grid)
    C_grid_FtG = 0.                  # Grid fees for electricity fed to the grid, €/MWh      (cost to sell electricity to the grid)  
    C_TL_FtG   = 0.                  # Tax and levies for electricity fed to the grid, €/MWh (cost to sell electricity to the grid)
    C_pros_tax = econ_param['C_pros_tax'] # Prosumer tax, €/kW  fixed cost per min power installed between PV and inverter

    # Self consumption
    P_support_SC = 0. # Support to self-consumption, €/MWh                  (incentive to self consumption)  
    C_grid_SC    = 0. # Grid fees for self-consumed electricity, €/MWh      (cost to do self consumption)
    C_TL_SC      = 0. # Tax and levies for self-consumed electricity, €/MWh (cost to do self consumption)
 
    # Investment and replacement costs
    # We do not evaluate PV and inverter investments as they are part of the ref case

    # Battery investment cost
    BatteryInvestment  = FixedBatteryCost + econ_param['BatteryCost_kWh'] * E['BatteryCapacity']
    out['CostBattery'] = BatteryInvestment

    # Control strategy initial investment
    ControlInvestment = econ_param['FixedControlCost']   

    # Initial investment
    InitialInvestment =  BatteryInvestment + ControlInvestment

    # Initialize cashflows array 
    CashFlows = np.zeros(int(years)+1)   

    # Adding initial investment costs to cashflows array
    CashFlows[0]  = - InitialInvestment
    
    # Adding replacement costs to cashflows array
    NBattRep = int((years-1)/econ_param['BatteryLifetime'])
    for i in range(NBattRep):
        iyear = (i+1)*econ_param['BatteryLifetime']
        CashFlows[iyear] = - BatteryInvestment
        
    # Other costs
    
    # O&Ms
    CashFlows[1:years+1] = CashFlows[1:years+1] - econ_param['OM'] * BatteryInvestment
    
    # Annual costs for controller
    CashFlows[1:years+1] = CashFlows[1:years+1] - econ_param['AnnualControlCost']
   
    # Energy expenditure and revenues
    # Both in case of net metering or not
    
    if net_metering:
        print('WARNING: net metering to be revised')
        # # Revenues selling to the grid
        # # Fixed selling price and cost
        # Income_FtG = np.maximum(0,sum(E['ACGeneration']-E['Load'])*timestep) * (P_FtG - C_grid_FtG - C_TL_FtG)/1000        
        # Income_SC = (P_support_SC - C_grid_SC - C_TL_SC)*sum(E['SC']*timestep)/1000
        # # Energy expenditure buying from the grid
        # Cost_BtG_energy = np.maximum(sum(yenprices*(E['Load']-E['ACGeneration'])*timestep),0)
        # Cost_BtG_grid   = np.maximum(sum(ygridfees*(E['Load']-E['ACGeneration'])*timestep),0)
        # Cost_BtG = Cost_BtG_energy + Cost_BtG_grid

    else:
        
        ### Considered case ###
        
        # Selling to the grid
        Income_FtG = sum(E['ToGrid']*timestep) * P_FtG/1000
        Cost_FtG = sum(E['ToGrid']*timestep) * (C_grid_FtG + C_TL_FtG)/1000
        Total_FtG = Income_FtG - Cost_FtG 
        # Buying from the grid
        Cost_BtG_energy = sum(yenprices * E['FromGrid']*timestep)
        Cost_BtG_grid   = sum(ygridfees * E['FromGrid']*timestep)
        Total_BtG = Cost_BtG_energy + Cost_BtG_grid
        # Self-consumption support schemes
        Income_SC = (P_support_SC - C_grid_SC - C_TL_SC)*sum(E['SC']*timestep)/1000

        # Pre 2030 revenues for selling to the grid if PV installed before 2024
        if (E['PVCapacity'] > 0 and start_year < 2024):
            
            # Selling to the grid
            Income_FtG_pre2030 = sum(E['ToGrid']*yenprices*timestep)
            Cost_FtG_pre2030 = np.maximum(sum(E['ToGrid']*ygridfees*timestep),C_pros_tax*E['PVCapacity'])
            Total_FtG_pre2030 = Income_FtG_pre2030 - Cost_FtG_pre2030

        ### Reference case ###
            
        # Selling to the grid
        Income_FtG_ref = sum(E_ref['ToGrid']*timestep) * P_FtG/1000
        Cost_FtG_ref = sum(E_ref['ToGrid']*timestep) * (C_grid_FtG + C_TL_FtG)/1000
        Total_FtG_ref = Income_FtG_ref - Cost_FtG_ref 
        # Buying from the grid
        Cost_BtG_energy_ref = sum(yenprices * E_ref['FromGrid']*timestep)
        Cost_BtG_grid_ref   = sum(ygridfees * E_ref['FromGrid']*timestep)
        Total_BtG_ref = Cost_BtG_energy_ref + Cost_BtG_grid_ref
        # Self-consumption support schemes
        Income_SC_ref = (P_support_SC - C_grid_SC - C_TL_SC)*sum(E_ref['SC']*timestep)/1000

        # Pre 2030 revenues for selling to the grid if PV installed before 2024
        if (E['PVCapacity'] > 0 and start_year < 2024):
            
            # Selling to the grid
            Income_FtG_pre2030_ref = sum(E_ref['ToGrid']*yenprices*timestep)
            Cost_FtG_pre2030_ref = np.maximum(sum(E_ref['ToGrid']*ygridfees*timestep),C_pros_tax*E_ref['PVCapacity'])
            Total_FtG_pre2030_ref = Income_FtG_pre2030_ref - Cost_FtG_pre2030_ref


    # Annual costs for grid connection
    Capacity = np.max(E['Load'])
    AnnualCostGrid = C_grid_fixed + C_grid_kW * Capacity 
       
    # Adding revenues and expenditures from buying and selling  
    if (E['PVCapacity'] > 0 and start_year < 2024):
        end2030 = 2030-start_year
        for i in range(1,end2030+1):
            CashFlows[i] = CashFlows[i] + Income_SC - AnnualCostGrid + (Total_FtG_pre2030 - Total_BtG)*(1+elpriceincr)**(i-1)
        for i in range(end2030+2,years+1):
            CashFlows[i] = CashFlows[i] + Income_SC - AnnualCostGrid + (Total_FtG         - Total_BtG)*(1+elpriceincr)**(i-1)
    else:
        for i in range(1,years+1):
            CashFlows[i] = CashFlows[i] + Income_SC - AnnualCostGrid + (Total_FtG         - Total_BtG)*(1+elpriceincr)**(i-1)
    
    # Reference case energy expenditure
    # PV panels with no load shifting or batteries
    # Inverted sign as considered as savings
    
    if (E['PVCapacity'] > 0 and start_year < 2024):
        end2030 = 2030-start_year
        for i in range(1,end2030+1):
            CashFlows[i] = CashFlows[i] - (Income_SC_ref - AnnualCostGrid + (Total_FtG_pre2030_ref - Total_BtG_ref)*(1+elpriceincr)**(i-1))
        for i in range(end2030+2,years+1):
            CashFlows[i] = CashFlows[i] - (Income_SC_ref - AnnualCostGrid + (Total_FtG_ref         - Total_BtG_ref)*(1+elpriceincr)**(i-1))
    else:
        for i in range(1,years+1):
            CashFlows[i] = CashFlows[i] - (Income_SC_ref - AnnualCostGrid + (Total_FtG_ref         - Total_BtG_ref)*(1+elpriceincr)**(i-1))    
    
    # Actualized cashflows
    CashFlowsAct = np.zeros(len(CashFlows))
    for i in range(len(CashFlows)):
        CashFlowsAct[i] = CashFlows[i]/(1+interest)**(i)

    # NPV curve        
    NPVcurve = np.zeros(len(CashFlows))
    NPVcurve[0] = CashFlowsAct[0]
    for i in range(len(CashFlows)-1):
        NPVcurve[i+1] = NPVcurve[i]+CashFlowsAct[i+1]

    # Final NPV        
    NPV = npf.npv(interest,CashFlows)
    out['NPV'] = NPV

    # Pay Back Period    
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
        PBP = None  
    out['PBP'] = PBP
    
    # Internal Rate of Return
    IRR = npf.irr(CashFlows)
    out['IRR'] = IRR

    # Profit Index    
    if InitialInvestment == 0:
        PI = None
    else:
        PI = NPV/InitialInvestment
    out['PI'] = PI
    
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
    
    # Imposing the extreme values
    app_s[0] = 0; app[-1] = 0
    
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


    
































