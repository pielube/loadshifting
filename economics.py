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



def EconomicAnalysis(conf,prices,E,new_PV = False):
    
    if new_PV:
        # PV investment cost - Analyzed case
        Inv_PV = conf['econ']['C_PV_fix'] + conf['econ']['C_PV_kW'] * conf['pv']['ppeak']
        # Inverter investment cost - Analyzed case
        Inv_Invert = conf['econ']['C_invert_fix'] + conf['econ']['C_invert_kW'] * conf['pv']['inverter_pmax']
    else:
        Inv_PV = 0
        Inv_Invert = 0

    # Battery investment cost - Analyzed case
    Inv_Batt = conf['econ']['C_batt_fix'] + conf['econ']['C_batt_kWh'] * conf['batt']['capacity']
    
    # Control system investment cost
    Inv_Control = conf['econ']['C_control']

    # Initial investment
    InitialInvestment =  Inv_PV + Inv_Batt + Inv_Invert + Inv_Control

    # Initialize cashflows array 
    CashFlows = np.zeros(int(conf['econ']['time_horizon'])+1)   

    # Adding initial investment costs to cashflows array
    CashFlows[0]  = - InitialInvestment
    
    # Adding replacement costs to cashflows array - Battery
    NBattRep = int((conf['econ']['time_horizon']-1)/conf['batt']['lifetime'])
    for i in range(NBattRep):
        iyear = (i+1)*conf['batt']['lifetime']
        CashFlows[iyear] = - Inv_Batt
        
    # Adding replacement costs to cashflows array - Inverter
    NInvRep = int((conf['econ']['time_horizon']-1)/conf['pv']['inverter_lifetime'])
    for i in range(NInvRep):
        iyear = (i+1)*conf['pv']['inverter_lifetime']
        CashFlows[iyear] = - Inv_Invert
    # Annual costs: grid fees
    if new_PV:
        CashFlows[1:conf['econ']['time_horizon']+1] += - conf['econ']['C_grid_kW_annual'] * (max(E['FromGrid']) -  max(E['Load'])) 
    else:
        CashFlows[1:conf['econ']['time_horizon']+1] += - conf['econ']['C_grid_kW_annual'] * (max(E['FromGrid']) -  max(E['Load']-E['ACGeneration'])) 
    
    # Annual costs: O&Ms
    CashFlows[1:conf['econ']['time_horizon']+1] += - conf['econ']['C_OM_annual'] * (Inv_PV + Inv_Batt) 
    
    # Annual costs: controller
    CashFlows[1:conf['econ']['time_horizon']+1] += - conf['econ']['C_control_annual'] 
        
    # Contributions of buying and selling energy to cash flows - Analyzed case
    
    enpricekWh = prices['energy']
    gridfeekWh = prices['grid']
    enpricekWh_sell = prices['sell']
    prostax = conf['econ']['C_prosumertax']*min(conf['pv']['ppeak'],conf['pv']['inverter_pmax'])
    res = EnergyBuyAndSell(conf,enpricekWh, gridfeekWh, enpricekWh_sell, E, prostax)
        
    # Adding revenues and expenditures from buying and selling energy
    if conf['pv']['ppeak'] > 0 and conf['econ']['start_year'] < 2024 and not conf['econ']['smart_meter']:
        
        end2030 = 2030-conf['econ']['start_year']
        
        for i in range(1,end2030+1): # up to 2030
            CashFlows[i] += (res['IncomeStG_pre2030'] - res['CostStG_pre2030'] - res['CostBfG_energy'] - res['CostBfG_grid']) *(1+conf['econ']['elpriceincrease'])**(i-1)
        for i in range(end2030+2,conf['econ']['time_horizon']+1): # after 2030
            CashFlows[i] += (res['IncomeStG'] - res['CostStG'] - res['CostBfG_energy'] - res['CostBfG_grid']) *(1+conf['econ']['elpriceincrease'])**(i-1)
    else:
        
        for i in range(1,conf['econ']['time_horizon']+1): # whole time horizon, no distinction in 2030
            CashFlows[i] += (res['IncomeStG'] - res['CostStG'] - res['CostBfG_energy'] - res['CostBfG_grid']) *(1+conf['econ']['elpriceincrease'])**(i-1)


    # Actualized cashflows
    CashFlowsAct = np.zeros(len(CashFlows))
    for i in range(len(CashFlows)):
        CashFlowsAct[i] = CashFlows[i]/(1+conf['econ']['wacc'])**(i)

    # NPV curve
    NPVcurve = np.zeros(len(CashFlows))
    NPVcurve[0] = CashFlowsAct[0]
    for i in range(len(CashFlows)-1):
        NPVcurve[i+1] = NPVcurve[i]+CashFlowsAct[i+1]

    # Final NPV
    NPV = npf.npv(conf['econ']['wacc'],CashFlows)
    NPV = 0 if abs(NPV)<0.01 else NPV

    # Pay Back Period
    idx1 = np.where(NPVcurve[:-1] * NPVcurve[1:] < 0 )[0] +1
    if len(idx1) > 0:
        idx1 = idx1[0]
        fractional = (0-NPVcurve[idx1-1])/CashFlowsAct[idx1]
        PBP = idx1+fractional
    else:
        PBP = None
    
    # Internal Rate of Return
    IRR = npf.irr(CashFlows)

    # Profit Index
    if InitialInvestment == 0:
        PI = None
    else:
        PI = NPV/InitialInvestment
    
    # Total actualized battery investment, accounting for replacements
    Inv_Batt_act_total = Inv_Batt   
    for i in range(NBattRep):
        iyear = (i+1)*conf['batt']['lifetime']
        NPV_Battery_reinvestment = Inv_Batt / (1+conf['econ']['wacc'])**iyear
        Inv_Batt_act_total += NPV_Battery_reinvestment

    # Total actualized inverter investment, accounting for replacements
    Inv_invert_act_total = Inv_Invert
    for i in range(NInvRep):
        iyear = (i+1)*conf['pv']['inverter_lifetime']
        NPV_Inverter_reinvestment = Inv_Invert / (1+conf['econ']['wacc'])**iyear
        Inv_invert_act_total += NPV_Inverter_reinvestment
    
    # Net system costs
    NetSystemCost = Inv_PV + Inv_Batt_act_total + Inv_invert_act_total
                  
    # Capital Recovery Factor
    
    CRF = conf['econ']['wacc'] * (1+conf['econ']['wacc'])**conf['econ']['time_horizon']/((1+conf['econ']['wacc'])**conf['econ']['time_horizon']-1)
    
    # Annual investment costs  
           
    AnnualInvestment = NetSystemCost * CRF + \
                       conf['econ']['C_grid_fix_annual'] + conf['econ']['C_grid_kW_annual'] * max(E['FromGrid']) + \
                       conf['econ']['C_OM_annual'] * (Inv_PV + Inv_Batt) + \
                       conf['econ']['C_control_annual']
    
    # Annual electricity cost
    ECost = AnnualInvestment - (res['IncomeStG'] - res['CostStG'] - (res['CostBfG_energy'] + res['CostBfG_grid']))
    
    # Annual electricity cost updated if change of tarification after 2030
    # Ecost averaged pre-post 2030
    if conf['pv']['ppeak'] > 0 and conf['econ']['start_year'] < 2024 and not conf['econ']['smart_meter']:
        yearspre2030 = 2030 - conf['econ']['start_year']
        ECost_pre2030 = AnnualInvestment - (res['IncomeStG_pre2030'] - res['CostStG_pre2030'] - (res['CostBfG_energy'] + res['CostBfG_grid']))
        ECost = (ECost_pre2030*yearspre2030 + ECost*(conf['econ']['time_horizon']-yearspre2030)) / conf['econ']['time_horizon']
 
    # LCOE equivalent, electricity price per MWhh
    ElPriceAvg = (ECost / sum(E['Load']*conf['sim']['ts']))*1000. # eur/MWh
    
    # Grid cost component of the final energy price
    # TODO average pre post 2030
    # revise what this actually is
    ElPriceAvg_grid = 0./sum(E['Load']*conf['sim']['ts'])*1000 # eur/MWh

    out = {}

    # TODO revise
    # TODO average pre and post 2030 or separate pre and post 2030
    
    out['PVInv']       = Inv_PV 
    out['BatteryInv']  = Inv_Batt
    out['InverterInv'] = Inv_Invert 

    out["EnSold"]     = res['IncomeStG']
    out["CostToSell"] = res['CostStG']
    out["TotalSell"]  = res['IncomeStG'] - res['CostStG']
    
    out["EnBought"]  = res['CostBfG_energy']
    out["CostToBuy"] = res['CostBfG_grid']
    out["TotalBuy"]  = res['CostBfG_energy'] + res['CostBfG_grid']
    
    out['ElBill'] = res['IncomeStG'] - res['CostStG'] - (res['CostBfG_energy'] + res['CostBfG_grid']) - \
                    conf['econ']['C_grid_fix_annual'] + conf['econ']['C_grid_kW_annual'] * max(E['FromGrid']) - \
                    conf['econ']['C_OM_annual'] * (Inv_PV + Inv_Batt) - \
                    conf['econ']['C_control_annual']

    out['NPV'] = NPV
    out['PBP'] = PBP
    out['IRR'] = IRR
    out['PI']  = PI
    
    out['costpermwh'] = ElPriceAvg
    out['cost_grid']  = ElPriceAvg_grid
     
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


def EnergyBuyAndSell(conf,enpricekWh, gridfeekWh, enpricekWh_sell, E, prostax):
    
    ts = conf['sim']['ts']
    N = len(E['FromGrid'])
    
    if ts!= 1:
        enpricekWh = scale_vector(enpricekWh,N,silent=True)
        gridfeekWh = scale_vector(gridfeekWh,N,silent=True)
        enpricekWh_sell = scale_vector(enpricekWh_sell,N,silent=True)
    
    CostBfG_energy = sum(E['FromGrid']*enpricekWh)*ts
    CostBfG_grid   = sum(E['FromGrid']*gridfeekWh)*ts
    
    # Selling
    
    if conf['pv']['ppeak'] > 0: # prosumers
        
        if conf['econ']['start_year'] < 2024: # PV installed before 2024
            # distinction to be made before and after 2030 when selling
            
            if not conf['econ']['smart_meter']:
                # Selling
                # pre 2030
                # cash flow depends on tariff type
                
                if conf['econ']['tariff'] == 'net-metering':
                    
                    IncomeStG_pre2030 = sum(E['ToGrid']*(enpricekWh + gridfeekWh))*ts
                    CostStG_pre2030   = prostax
                    
                elif conf['econ']['tariff'] == 'bi-directional':
                    
                    print('Error: multi price tariff requires smart meter')
                    sys.exit('Error: multi price tariff requires smart meter')
                    
                else:
                    
                    print('Error: tariff type specified does not exist')
                    sys.exit('Error: tariff type specified does not exist')
                 
                # Selling  
                # post 2030
                # prosumers forced to install smart meter
                IncomeStG = sum(E['ToGrid']*enpricekWh_sell)*ts
                CostStG   = 0 # min(sum(E['ToGrid']*gridfeekWh)*ts, prostax)

                
            elif conf['econ']['tariff']:
                
                # Selling
                # no distinction between pre and post 2030
                IncomeStG = sum(E['ToGrid']*enpricekWh_sell)*ts
                CostStG   = 0 # min(sum(E['ToGrid']*gridfeekWh)*ts, prostax)
                IncomeStG_pre2030 = None
                CostStG_pre2030 = None
                
            
            else:
                print('Error: meter type specified does not exist')
                sys.exit('Error: meter type specified does not exist')
                
        else: # PV installed after 2024 => # no distinctions to be made before and after 2030
            IncomeStG = sum(E['ToGrid']*enpricekWh_sell)*ts
            CostStG   = 0 # min(sum(E['ToGrid']*gridfeekWh)*ts, prostax)
            IncomeStG_pre2030 = None
            CostStG_pre2030 = None
    
    else: # consumers (no PV installed)
    
        IncomeStG = 0
        CostStG   = 0
        IncomeStG_pre2030 = None
        CostStG_pre2030 = None
               
    out = {}

    out['CostBfG_energy']    = CostBfG_energy 
    out['CostBfG_grid']      = CostBfG_grid     
    out['IncomeStG']         = IncomeStG 
    out['CostStG']           = CostStG 
    out['IncomeStG_pre2030'] = IncomeStG_pre2030 
    out['CostStG_pre2030']   = CostStG_pre2030
    
    return out


