# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:14:08 2022

@author: pietro
"""

import numpy as np
import numpy_financial as npf
import sys
import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
from joblib import Memory
memory = Memory(__location__ + '/cache/', verbose=1)



def EconomicAnalysis(conf, E, E_ref):
    
    # Initialize cashflows array 
    CashFlows = np.zeros(int(conf['econ']['time_horizon'])+1) 
    
    # Investment costs
    
    if conf['pv']['yesno']: # PV and inverter costs - Analyzed case
        Inv_PV = conf['econ']['C_PV_fix'] + conf['econ']['C_PV_kW'] * conf['pv']['ppeak']
        Inv_Invert = conf['econ']['C_invert_fix'] + conf['econ']['C_invert_kW'] * conf['pv']['inverter_pmax']
    else:
        Inv_PV = 0.
        Inv_Invert = 0
    
    if conf['econ']['PV_ref']:  # PV and inverter costs - Reference case
        Inv_PV_ref = Inv_PV
        Inv_Invert_ref = Inv_Invert
    else:
        Inv_PV_ref = 0.
        Inv_Invert_ref = 0.
        
    if conf['batt']['yesno']: # Battery costs - Analyzed case
        Inv_Batt = conf['econ']['C_batt_fix'] + conf['econ']['C_batt_kWh'] * conf['batt']['capacity']
    else:
        Inv_Batt = 0.        
    
    Inv_Control = conf['econ']['C_control'] # Control system investment cost

    InitialInvestment =  Inv_PV - Inv_PV_ref + Inv_Invert - Inv_Invert_ref + Inv_Batt + Inv_Control # Initial investment
    CashFlows[0]  = - InitialInvestment # Adding initial investment costs to cashflows array

    NBattRep = int((conf['econ']['time_horizon']-1)/conf['batt']['lifetime']) # Replacement costs - Battery
    for i in range(NBattRep):
        iyear = (i+1)*conf['batt']['lifetime']
        CashFlows[iyear] = - Inv_Batt
        
    NInvRep = int((conf['econ']['time_horizon']-1)/conf['pv']['inverter_lifetime']) # Replacement costs - Inverter
    for i in range(NInvRep):
        iyear = (i+1)*conf['pv']['inverter_lifetime']
        CashFlows[iyear] = - Inv_Invert
        
    # Annual costs
    
    CashFlows[1:conf['econ']['time_horizon']+1] += - conf['econ']['C_grid_fix_annual'] - conf['econ']['C_grid_kW_annual'] * (max(E['FromGrid'])-max(E['LoadNoshift'])) # grid fees
    CashFlows[1:conf['econ']['time_horizon']+1] += - conf['econ']['C_OM_annual'] * (Inv_PV - Inv_PV_ref + Inv_Batt)  # O&Ms   
    CashFlows[1:conf['econ']['time_horizon']+1] += - conf['econ']['C_control_annual'] # Controller
    
    # Contributions of buying and selling energy to cash flows - Analyzed case
    
    enpricekWh = conf['energyprice']
    gridfeekWh = conf['gridprice']
    enpricekWh_sell = conf['sellprice']
    
    start = 305 # November 1st 
    end   =  90 # March 31st
    a = E['FromGrid'][np.r_[0:end*24*4-1,(start-1)*24*4-1:365*24*4-1]]-10
    capterm = np.sum(np.trunc(a[a>0])+1)*conf['econ']['C_capacitytariff'] # Capacty tariff
    
    prostax = conf['econ']['C_prosumertax']*min(conf['pv']['ppeak'],conf['pv']['inverter_pmax'])
    
    res = EnergyBuyAndSell(conf,enpricekWh, gridfeekWh, enpricekWh_sell, E, prostax, capterm)
    
    end2030 = 2030 - conf['econ']['start_year']
    for i in range(1,end2030+1):
        CashFlows[i] += (res['pre2030']['IncomeStG'] - res['pre2030']['CostBfG_energy'] - res['pre2030']['CostBfG_grid'])*(1+conf['econ']['elpriceincrease'])**(i-1)
    for i in range(end2030+2,conf['econ']['time_horizon']+1):
        CashFlows[i] += (res['post2030']['IncomeStG'] - res['post2030']['CostBfG_energy'] - res['post2030']['CostBfG_grid'])*(1+conf['econ']['elpriceincrease'])**(i-1)
    
    # Contributions of buying and selling energy to cash flows - Reference case
    
    enpricekWh_ref = conf['energyprice_ref']
    gridfeekWh_ref = conf['gridprice_ref']
    enpricekWh_sell_ref = conf['sellprice_ref']
    
    start = 305 # November 1st 
    end   =  90 # March 31st
    a = E['FromGrid'][np.r_[0:end*24*4-1,(start-1)*24*4-1:365*24*4-1]]-10
    capterm_ref = np.sum(np.trunc(a[a>0])+1)*conf['econ']['C_capacitytariff'] # Capacity tariff
    
    prostax_ref = conf['econ']['C_prosumertax']*min(conf['pv']['ppeak'],conf['pv']['inverter_pmax'])
    
    res_ref = EnergyBuyAndSell(conf,enpricekWh_ref, gridfeekWh_ref, enpricekWh_sell_ref, E_ref, prostax_ref, capterm_ref)
    
    end2030 = 2030 - conf['econ']['start_year']
    for i in range(1,end2030+1):
        CashFlows[i] += (res_ref['pre2030']['IncomeStG'] - res_ref['pre2030']['CostBfG_energy'] - res_ref['pre2030']['CostBfG_grid'])*(1+conf['econ']['elpriceincrease'])**(i-1)
    for i in range(end2030+2,conf['econ']['time_horizon']+1):
        CashFlows[i] += (res_ref['post2030']['IncomeStG'] - res_ref['post2030']['CostBfG_energy'] - res_ref['post2030']['CostBfG_grid'])*(1+conf['econ']['elpriceincrease'])**(i-1)

    # NPV analysis
    
    CashFlowsAct = np.zeros(len(CashFlows)) # Actualized cashflows
    for i in range(len(CashFlows)):
        CashFlowsAct[i] = CashFlows[i]/(1+conf['econ']['wacc'])**(i)
   
    NPVcurve = np.zeros(len(CashFlows))  # NPV curve
    NPVcurve[0] = CashFlowsAct[0]
    for i in range(len(CashFlows)-1):
        NPVcurve[i+1] = NPVcurve[i]+CashFlowsAct[i+1]

    NPV = npf.npv(conf['econ']['wacc'],CashFlows) # NPV at the end of time horizon
    NPV = 0 if abs(NPV)<0.01 else NPV

    idx1 = np.where(NPVcurve[:-1] * NPVcurve[1:] < 0 )[0] +1 # Pay Back Period
    if len(idx1) > 0:
        idx1 = idx1[0]
        fractional = (0-NPVcurve[idx1-1])/CashFlowsAct[idx1]
        PBP = idx1+fractional
    else:
        PBP = None
    
    IRR = npf.irr(CashFlows) # Internal Rate of Return

    if InitialInvestment == 0:  # Profit Index
        PI = None
    else:
        PI = NPV/InitialInvestment
    
    
    # Average electricity price
    # LCOE equivalent, as if the grid was a generator
    
    Inv_Batt_act_total = Inv_Batt # Total actualized battery investment, accounting for replacements
    for i in range(NBattRep):
        iyear = (i+1)*conf['batt']['lifetime']
        NPV_Battery_reinvestment = Inv_Batt / (1+conf['econ']['wacc'])**iyear
        Inv_Batt_act_total += NPV_Battery_reinvestment

    Inv_invert_act_total = Inv_Invert # Total actualized inverter investment, accounting for replacements
    for i in range(NInvRep):
        iyear = (i+1)*conf['pv']['inverter_lifetime']
        NPV_Inverter_reinvestment = Inv_Invert / (1+conf['econ']['wacc'])**iyear
        Inv_invert_act_total += NPV_Inverter_reinvestment
    
    NetSystemCost = Inv_PV + Inv_Batt_act_total + Inv_invert_act_total # Net system costs
    
    CRF = conf['econ']['wacc'] * (1+conf['econ']['wacc'])**conf['econ']['time_horizon']/((1+conf['econ']['wacc'])**conf['econ']['time_horizon']-1) # Capital Recovery Factor
    
    AnnualInvestment = NetSystemCost * CRF + \
                       conf['econ']['C_grid_fix_annual'] + conf['econ']['C_grid_kW_annual'] * max(E['FromGrid']) + \
                       conf['econ']['C_OM_annual'] * (Inv_PV + Inv_Batt) + \
                       conf['econ']['C_control_annual'] # Annual investment costs  
    
    ECost_pre2030  = AnnualInvestment - (res['pre2030' ]['IncomeStG'] - (res['pre2030' ]['CostBfG_energy'] + res['pre2030' ]['CostBfG_grid'])) # eur Annual electricity cost pre 2030
    ECost_post2030 = AnnualInvestment - (res['post2030']['IncomeStG'] - (res['post2030']['CostBfG_energy'] + res['post2030']['CostBfG_grid'])) # eur Annual electricity cost post 2030

    ElPriceAvg_pre2030  = ECost_pre2030/sum(E['Load']*conf['sim']['ts'])*1000. # eur/MWh
    ElPriceAvg_post2030 = ECost_post2030/sum(E['Load']*conf['sim']['ts'])*1000. # eur/MWh
    
    yearspre2030 = 2030 - conf['econ']['start_year']
    ECost = (ECost_pre2030*yearspre2030 + ECost_post2030*(conf['econ']['time_horizon']-yearspre2030)) / conf['econ']['time_horizon']
    ElPriceAvg = (ElPriceAvg_pre2030*yearspre2030 + ElPriceAvg_post2030*(conf['econ']['time_horizon']-yearspre2030)) / conf['econ']['time_horizon']
    
    # TODO   
    # cost components:
    # energy
    # grid
    #   - proportional kWh
    #   - prostax
    #   - grid fee
    #   - annual grid fee kW
    # other (O&M, annual control costs)

    # Outputs

    out = {}
    out['pre2030']  = {}
    out['post2030'] = {}
    
    # Outputs - Investments 
    
    out['PVInv']       = Inv_PV - Inv_PV_ref
    out['BatteryInv']  = Inv_Batt
    out['InverterInv'] = Inv_Invert - Inv_Invert_ref

    # Outputs - NPV analysis    

    out['NPV'] = NPV
    out['PBP'] = PBP
    out['IRR'] = IRR
    out['PI']  = PI
    
    # Outputs - Average electricity price  
    
    out['costpermwh'] = ElPriceAvg
     
    return out


def scale_vector(vec_in,N,silent=False):
    ''' 
    Function that scales a numpy vector or Pandas Series to the desired length
    
    :param vec_in: Input vector
    :param N: Length of the output vector
    :param silent: Set to True to avoid verbosity
    '''    
    N_in = len(vec_in)
    if type(N) != int:
        N = int(N) 
        if not silent:
            print('Converting Argument N to int: ' + str(N))
    if N > N_in:
        if np.mod(N,N_in)==0:
            if not silent:
                print('Target size is a multiple of input vector size. Repeating values')
            vec_out = np.repeat(vec_in,N/N_in)
        else:
            if not silent:
                print('Target size is larger but not a multiple of input vector size. Interpolating')
            vec_out = np.interp(np.linspace(start=0,stop=N_in,num=N),range(N_in),vec_in)
    elif N == N_in:
        print('Target size is iqual to input vector size. Not doing anything')
        vec_out = vec_in
    else:
        if np.mod(N_in,N)==0:
            if not silent:
                print('Target size is entire divisor of the input vector size. Averaging')
            vec_out = np.zeros(N)
            for i in range(N):
                vec_out[i] = np.mean(vec_in[i*N_in/N:(i+1)*N_in/N])
        else:
            if not silent:
                print('Target size is lower but not a divisor of the input vector size. Interpolating')
            vec_out = np.interp(np.linspace(start=0,stop=N_in,num=N),range(N_in),vec_in)
    return vec_out  


def EnergyBuyAndSell(conf,enpricekWh, gridfeekWh, enpricekWh_sell, E, prostax, capterm):
    """
    Function to calculate annual energy expenditure, proportional grid fees and income from selling energy
    based on: type of meter (analogue, smart_r1, smart_r1), PV (yes/no), tariff (mono, bi, multi) and
    installation year (pre/post 2024).
    
    Cases are defined based on Energie Commune's requests.
    
    Parameters
    ----------
    conf : dict
        Dictionary with all variables describing the case studied.
    enpricekWh : numpy array
        Array with energy prices for the whole year.
    gridfeekWh : numpy array
        Array with proportiona grid fees for the whole year.
    enpricekWh_sell : numpy array
        Array with energy selling price for the whole year.
    E : dict
        Dictionary with energy balances
    prostax : float
        Prosumer tax.
    capterm : float
        Capacity term.

    Returns
    -------
    out : dict
        Dictionary with annual energy expenditure, grid fees for pre and post 2030.

    """
    
    out = {}
    
    out['pre2030'] = {}
    out['post2030'] = {}
    
    
    ts = conf['sim']['ts']
    N = len(E['FromGrid'])
    
    if ts!= 1:
        enpricekWh = scale_vector(enpricekWh,N,silent=False)
        gridfeekWh = scale_vector(gridfeekWh,N,silent=False)
        enpricekWh_sell = scale_vector(enpricekWh_sell,N,silent=False)
    
    if conf['econ_meter'] in ['analogue']: # Analogue meter
        
        if conf['pv_yesno']: # Yes PV
            
            if conf['econ_start_year'] >= 2024:
                
                print('Error: No analogue meters with PV installastions after 2024')
                sys.exit('Error: No analogue meters with PV installastions after 2024')
            
            if conf['tariff'] in ['mono','bi']:                

                # Net metering and prosumer tax
                out['pre2030']['CostBfG_energy'] = sum((E['FromGrid']-E['ToGrid'])*enpricekWh)*ts
                out['pre2030']['CostBfG_grid']   = sum((E['FromGrid']-E['ToGrid'])*gridfeekWh)*ts + prostax
                out['pre2030']['IncomeStG']      = 0.
                
                # Gross metering
                out['post2030']['CostBfG_energy'] = sum(E['FromGrid']*enpricekWh)*ts
                out['post2030']['CostBfG_grid']   = min(sum(E['FromGrid']*gridfeekWh)*ts,out['pre2030']['CostBfG_grid'])
                out['post2030']['IncomeStG']      = 0.
                

            elif conf['tariff'] in ['multi']:
                
                print('Error: No multi tariff with analogue meters')
                sys.exit('Error: No multi tariff with analogue meters')
                
            else:
                
                print('Error: Wrong tariff name')
                sys.exit('Error: Wrong tariff name')
                
        else: # No PV
            
            if conf['tariff'] in ['mono','bi']:
                
                # Gross metering (Net = Gross since no PV)
                out['pre2030']['CostBfG_energy'] = sum(E['FromGrid']*enpricekWh)*ts
                out['pre2030']['CostBfG_grid']   = sum(E['FromGrid']*gridfeekWh)*ts
                out['pre2030']['IncomeStG']      = 0.
                
                # No changes after 2030
                out['post2030']['CostBfG_energy'] = out['pre2030']['CostBfG_energy']
                out['post2030']['CostBfG_grid']   = out['pre2030']['CostBfG_grid']
                out['post2030']['IncomeStG']      = out['pre2030']['IncomeStG']
                
            elif conf['tariff'] in ['multi']:
                
                print('Error: No multi tariff with analogue meters')
                sys.exit('Error: No multi tariff with analogue meters')
                
            else:
                
                print('Error: Wrong tariff name')
                sys.exit('Error: Wrong tariff name')
                
    if conf['econ_meter'] in ['smart_r1']:
        
        if conf['pv_yesno']: # Yes PV
            
            if conf['tariff'] in ['mono','bi','multi']:
                
                if conf['econ_start_year'] < 2024:
                    
                    # Net metering and prosumer tax
                    out['pre2030']['CostBfG_energy'] = sum((E['FromGrid']-E['ToGrid'])*enpricekWh)*ts
                    out['pre2030']['CostBfG_grid']   = sum((E['FromGrid']-E['ToGrid'])*gridfeekWh)*ts + prostax
                    out['pre2030']['IncomeStG']      = 0.
                    
                    # Gross metering
                    out['post2030']['CostBfG_energy'] = sum(E['FromGrid']*enpricekWh)*ts
                    out['post2030']['CostBfG_grid']   = min(sum(E['FromGrid']*gridfeekWh)*ts,out['pre2030']['CostBfG_grid'])
                    out['post2030']['IncomeStG']      = 0.
                    
                else:
                    
                    # Gross metering 
                    out['pre2030']['CostBfG_energy'] = sum(E['FromGrid']*enpricekWh)*ts
                    out['pre2030']['CostBfG_grid']   = min(sum(E['FromGrid']*gridfeekWh)*ts,out['pre2030']['CostBfG_grid'])
                    out['pre2030']['IncomeStG']      = 0.
                    
                    # No changes after 2030
                    out['post2030']['CostBfG_energy'] = out['pre2030']['CostBfG_energy']
                    out['post2030']['CostBfG_grid']   = out['pre2030']['CostBfG_grid']
                    out['post2030']['IncomeStG']      = out['pre2030']['IncomeStG']
                
            else:
                
                print('Error: Wrong tariff name')
                sys.exit('Error: Wrong tariff name')
                
        else: # No PV

            if conf['tariff'] in ['mono','bi','multi']:
                
                # Gross metering
                out['pre2030']['CostBfG_energy'] = sum(E['FromGrid']*enpricekWh)*ts
                out['pre2030']['CostBfG_grid']   = sum(E['FromGrid']*gridfeekWh)*ts
                out['pre2030']['IncomeStG']      = 0.
                
                # No changes after 2030
                out['post2030']['CostBfG_energy'] = out['pre2030']['CostBfG_energy']
                out['post2030']['CostBfG_grid']   = out['pre2030']['CostBfG_grid']
                out['post2030']['IncomeStG']      = out['pre2030']['IncomeStG']
            
            else:
                
                print('Error: Wrong tariff name')
                sys.exit('Error: Wrong tariff name')
                
    if conf['econ_meter'] in ['smart_r3']:
        
        # No need to distinguish between with or without PV
        # If no PV E['ToGrid'] is all 0 and IncomeStG will be 0 accordingly
            
        if conf['tariff'] in ['mono','bi']:
            
            print('Error: No mono or bi tariffs with R3')
            sys.exit('Error: No mono or bi tariffs with R3')

        elif conf['tariff'] == 'multi':
            
            # Gross metering and capacity term
            out['pre2030']['CostBfG_energy'] = sum(E['FromGrid']*enpricekWh)*ts
            out['pre2030']['CostBfG_grid']   = sum(E['FromGrid']*gridfeekWh)*ts + capterm
            out['pre2030']['IncomeStG']      = sum(E['ToGrid']*enpricekWh_sell)*ts

            # No changes after 2030
            out['post2030']['CostBfG_energy'] = out['pre2030']['CostBfG_energy']
            out['post2030']['CostBfG_grid']   = out['pre2030']['CostBfG_grid']
            out['post2030']['IncomeStG']      = out['pre2030']['IncomeStG']              

        else:
            
            print('Error: Wrong tariff name')
            sys.exit('Error: Wrong tariff name')
    
    return out