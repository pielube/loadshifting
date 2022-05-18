
import numpy as np
import numpy_financial as npf

import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

from joblib import Memory
memory = Memory(__location__ + '/cache/', verbose=1)

    
    

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
    out['InvCapacity'] = E['InvCapacity']
    
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
    InverterInvestment = FixedInverterCost + econ_param['InverterCost_kW'] * E['InvCapacity']
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
