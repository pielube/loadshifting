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



def EconomicAnalysis(inp, tariffs, E, E_ref):
    
    # PV investment cost - Analyzed case
    if inp['PV'] == 0:
        Inv_PV = 0.
    else:
        Inv_PV = inp['C_PV_fix'] + inp['C_PV_kW'] * inp['PV']
    
    # PV investment cost - Reference case
    if inp['PV_ref'] == 0:
        Inv_PV_ref = 0.
    else:
        Inv_PV_ref = inp['C_PV_fix'] + inp['C_PV_kW'] * inp['PV_ref'] 

    # Battery investment cost - Analyzed case
    
    if inp['battery'] == 0:
        Inv_Batt = 0.
    else:
        Inv_Batt = inp['C_batt_fix'] + inp['C_batt_kWh'] * inp['battery']
    
    # Inverter investment cost - Analyzed case
    
    if inp['inverter'] == 0:
        Inv_Invert = 0.
    else:
        Inv_Invert = inp['C_invert_fix'] + inp['C_invert_kW'] * inp['inverter']
        
    # Inverter investment cost - Reference case
    
    if inp['inverter_ref'] == 0:
        Inv_Invert_ref = 0.
    else:
        Inv_Invert_ref = inp['C_invert_fix'] + inp['C_invert_kW'] * inp['inverter_ref']
    
    # Control system investment cost
    
    Inv_Control = inp['C_control_fix']

    # Initial investment
    
    InitialInvestment =  Inv_PV - Inv_PV_ref + Inv_Batt + Inv_Invert - Inv_Invert_ref + Inv_Control

    # Initialize cashflows array 
    
    CashFlows = np.zeros(int(inp['time_horizon'])+1)   

    # Adding initial investment costs to cashflows array
    
    CashFlows[0]  = - InitialInvestment
    
    # Adding replacement costs to cashflows array - Battery
    
    NBattRep = int((inp['time_horizon']-1)/inp['t_battery'])
    for i in range(NBattRep):
        iyear = (i+1)*inp['t_battery']
        CashFlows[iyear] = - Inv_Batt
        
    # Adding replacement costs to cashflows array - Inverter
    
    NInvRep = int((inp['time_horizon']-1)/inp['t_inverter'])
    for i in range(NInvRep):
        iyear = (i+1)*inp['t_inverter']
        CashFlows[iyear] = - Inv_Invert + Inv_Invert_ref 
            
    # Annual costs: grid
    
    CashFlows[1:inp['time_horizon']+1] += - (inp['C_grid_fix_annual'] + inp['C_grid_kW_annual'] * max(E['FromGrid'])) \
                                          +  inp['C_grid_fix_annual'] + inp['C_grid_kW_annual'] * max(E_ref['FromGrid'])
    
    # Annual costs: O&Ms
    
    CashFlows[1:inp['time_horizon']+1] += - inp['C_OM_annual'] * (Inv_PV + Inv_Batt) \
                                          + inp['C_OM_annual'] *  Inv_PV_ref
    
    # Annual costs: controller
    
    CashFlows[1:inp['time_horizon']+1] += - inp['C_control_fix_annual'] 
        
    # Contributions of buying and selling energy to cash flows - Analyzed case
    
    enpricekWh = tariffs[inp['tariff']]['energy']
    gridfeekWh = tariffs[inp['tariff']]['grid']
    enpricekWh_sell = tariffs[inp['tariff']]['sell']
    prostax = inp['C_prosumertax']*min(inp['PV'],inp['inverter'])
    res = EnergyBuyAndSell(inp,enpricekWh, gridfeekWh, enpricekWh_sell, E, inp['ts'], prostax)
        
    # Adding revenues and expenditures from buying and selling energy - Analyzed case
    
    if inp['PV'] > 0 and inp['start_year'] < 2024 and not(inp['meter']=='smart_meter'):
        
        end2030 = 2030-inp['start_year']
        
        for i in range(1,end2030+1): # up to 2030
            CashFlows[i] += (res['IncomeStG_pre2030'] - res['CostStG_pre2030'] - res['CostBfG_energy'] - res['CostBfG_grid']) *(1+inp['elpriceincrease'])**(i-1)
        for i in range(end2030+2,inp['time_horizon']+1): # after 2030
            CashFlows[i] += (res['IncomeStG'] - res['CostStG'] - res['CostBfG_energy'] - res['CostBfG_grid']) *(1+inp['elpriceincrease'])**(i-1)
    else:
        
        for i in range(1,inp['time_horizon']+1): # whole time horizon, no distinction in 2030
            CashFlows[i] += (res['IncomeStG'] - res['CostStG'] - res['CostBfG_energy'] - res['CostBfG_grid']) *(1+inp['elpriceincrease'])**(i-1)

    # Contributions of buying and selling energy to cash flows - Refence case
 
    enpricekWh_ref = tariffs[inp['tariff_ref']]['energy']
    gridfeekWh_ref = tariffs[inp['tariff_ref']]['grid']
    enpricekWh_sell_ref = tariffs[inp['tariff_ref']]['sell']
    prostax_ref = inp['C_prosumertax']*min(inp['PV_ref'],inp['inverter_ref'])
    res_ref = EnergyBuyAndSell(inp,enpricekWh_ref, gridfeekWh_ref, enpricekWh_sell_ref, E_ref, inp['ts'], prostax_ref)
    
    # Adding revenues and expenditures from buying and selling energy - Reference case

    if inp['PV_ref'] > 0 and inp['start_year'] < 2024 and not(inp['meter']=='smart_meter'):
        
        end2030 = 2030-inp['start_year']
        
        for i in range(1,end2030+1): # up to 2030
            CashFlows[i] += - (res_ref['IncomeStG_pre2030'] - res_ref['CostStG_pre2030'] - res_ref['CostBfG_energy'] - res_ref['CostBfG_grid']) *(1+inp['elpriceincrease'])**(i-1)
        for i in range(end2030+2,inp['time_horizon']+1): # after 2030
            CashFlows[i] += - (res_ref['IncomeStG'] - res_ref['CostStG'] - res_ref['CostBfG_energy'] - res_ref['CostBfG_grid']) *(1+inp['elpriceincrease'])**(i-1)
    else:
        
        for i in range(1,inp['time_horizon']+1): # whole time horizon, no distinction in 2030
            CashFlows[i] += - (res_ref['IncomeStG'] - res_ref['CostStG'] - res_ref['CostBfG_energy'] - res_ref['CostBfG_grid']) *(1+inp['elpriceincrease'])**(i-1)
    
    # Actualized cashflows
    
    CashFlowsAct = np.zeros(len(CashFlows))
    for i in range(len(CashFlows)):
        CashFlowsAct[i] = CashFlows[i]/(1+inp['WACC'])**(i)

    # NPV curve
      
    NPVcurve = np.zeros(len(CashFlows))
    NPVcurve[0] = CashFlowsAct[0]
    for i in range(len(CashFlows)-1):
        NPVcurve[i+1] = NPVcurve[i]+CashFlowsAct[i+1]
        
    # import matplotlib.pyplot as plt
    # plt.plot(NPVcurve)

    # Final NPV
     
    NPV = npf.npv(inp['WACC'],CashFlows)
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
        
    # LCOE equivalent, as if the grid was a generator
    
    # Total actualized battery investment, accounting for replacements
    
    Inv_Batt_act_total = Inv_Batt   
    for i in range(NBattRep):
        iyear = (i+1)*inp['t_battery']
        NPV_Battery_reinvestment = Inv_Batt / (1+inp['WACC'])**iyear
        Inv_Batt_act_total += NPV_Battery_reinvestment

    # Total actualized inverter investment, accounting for replacements
    
    Inv_invert_act_total = Inv_Invert
    for i in range(NInvRep):
        iyear = (i+1)*inp['t_inverter']
        NPV_Inverter_reinvestment = Inv_Invert / (1+inp['WACC'])**iyear
        Inv_invert_act_total += NPV_Inverter_reinvestment
    
    # Net system costs
    
    NetSystemCost = Inv_PV + Inv_Batt_act_total + Inv_invert_act_total
                  
    # Capital Recovery Factor
    
    CRF = inp['WACC'] * (1+inp['WACC'])**inp['time_horizon']/((1+inp['WACC'])**inp['time_horizon']-1)
    
    # Annual investment costs  
           
    AnnualInvestment = NetSystemCost * CRF + \
                       inp['C_grid_fix_annual'] + inp['C_grid_kW_annual'] * max(E['FromGrid']) + \
                       inp['C_OM_annual'] * (Inv_PV + Inv_Batt) + \
                       inp['C_control_fix_annual']
    
    # Annual electricity cost
    
    ECost = AnnualInvestment - (res['IncomeStG'] - res['CostStG'] - (res['CostBfG_energy'] + res['CostBfG_grid']))
    
    # Annual electricity cost updated if change of tarification after 2030
    # Ecost averaged pre-post 2030
    
    if inp['PV'] > 0 and inp['start_year'] < 2024 and not(inp['meter']=='smart_meter'):
        yearspre2030 = 2030 - inp['start_year']
        ECost_pre2030 = AnnualInvestment - (res['IncomeStG_pre2030'] - res['CostStG_pre2030'] - (res['CostBfG_energy'] + res['CostBfG_grid']))
        ECost = (ECost_pre2030*yearspre2030 + ECost*(inp['time_horizon']-yearspre2030)) / inp['time_horizon']
 
    # LCOE equivalent, electricity price per MWhh
    
    ElPriceAvg = (ECost / sum(E['Load']*inp['ts']))*1000. # eur/MWh
    
    # Grid cost component of the final energy price
    # TODO average pre post 2030
    # revise what this actually is
    
    ElPriceAvg_grid = 0./sum(E['Load']*inp['ts'])*1000 # eur/MWh

    out = {}

    # TODO revise
    # TODO average pre and post 2030 or separate pre and post 2030
    
    out['PVInv']       = Inv_PV - Inv_PV_ref
    out['BatteryInv']  = Inv_Batt
    out['InverterInv'] = Inv_Invert - Inv_Invert_ref

    out["EnSold"]     = res['IncomeStG']
    out["CostToSell"] = res['CostStG']
    out["TotalSell"]  = res['IncomeStG'] - res['CostStG']
    
    out["EnBought"]  = res['CostBfG_energy']
    out["CostToBuy"] = res['CostBfG_grid']
    out["TotalBuy"]  = res['CostBfG_energy'] + res['CostBfG_grid']
    
    out['ElBill'] = res['IncomeStG'] - res['CostStG'] - (res['CostBfG_energy'] + res['CostBfG_grid']) - \
                    inp['C_grid_fix_annual'] + inp['C_grid_kW_annual'] * max(E['FromGrid']) - \
                    inp['C_OM_annual'] * (Inv_PV + Inv_Batt) - \
                    inp['C_control_fix_annual']

    out['NPV'] = NPV
    out['PBP'] = PBP
    out['IRR'] = IRR
    out['PI']  = PI
    
    out['costpermwh'] = ElPriceAvg
    out['cost_grid']  = ElPriceAvg_grid
     
    return out


def EnergyBuyAndSell(inp,enpricekWh, gridfeekWh, enpricekWh_sell, E, ts, prostax):
    
    CostBfG_energy = sum(E['FromGrid']*enpricekWh)*ts
    CostBfG_grid   = sum(E['FromGrid']*gridfeekWh)*ts
    
    # Selling
    
    if inp['PV'] > 0: # prosumers
        
        if inp['start_year'] < 2024:
            
            # PV installed before 2024
            # distinction to be made before and after 2030 when selling
            
            if inp['meter'] == 'disc_meter':
                
                # Selling
                # pre 2030
                # cash flow depends on tariff type
                
                if inp['tariff'] == 'single_price':
                    
                    IncomeStG_pre2030 = sum(E['ToGrid']*(enpricekWh + gridfeekWh))*ts
                    CostStG_pre2030   = prostax
                    
                elif inp['tariff'] == 'double_price':
                    
                    IncomeStG_pre2030 = sum(E['ToGrid']*(enpricekWh + gridfeekWh))*ts
                    CostStG_pre2030   = prostax
                    
                elif inp['tariff'] == 'multi_price':
                    
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

                
            elif inp['meter'] == 'smart_meter':
                
                # Selling
                # no distinction between pre and post 2030
                IncomeStG = sum(E['ToGrid']*enpricekWh_sell)*ts
                CostStG   = 0 # min(sum(E['ToGrid']*gridfeekWh)*ts, prostax)
                IncomeStG_pre2030 = None
                CostStG_pre2030 = None
                
            
            else:
                print('Error: meter type specified does not exist')
                sys.exit('Error: meter type specified does not exist')
                
        else:
            
            # PV installed after 2024
            # no distinctions to be made before and after 2030
            IncomeStG = sum(E['ToGrid']*enpricekWh_sell)*ts
            CostStG   = 0 # min(sum(E['ToGrid']*gridfeekWh)*ts, prostax)
            IncomeStG_pre2030 = None
            CostStG_pre2030 = None
    
    else: # consumers
    
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


if __name__ == "__main__":
    
    inp = {
           'start_year': 2023,
           'time_horizon': 30,
           'WACC': 0.04,
           'elpriceincrease': 0.02,
           'tariff': 'multi_price',
           'meter': 'smart_meter',
           'PV': 10.0,
           'battery': 10.0,
           'inverter': 8.0,
           't_PV': 30,
           't_battery': 10,
           't_inverter': 15,
           'C_PV_fix': 0.0,
           'C_PV_kW': 1300.0,
           'C_batt_fix': 0.0,
           'C_batt_kWh': 600.0,
           'C_invert_fix': 0.0,
           'C_invert_kW': 100.0,
           'C_control_fix': 500.0,
           'C_OM_annual': 0.0,
           'C_grid_fix_annual': 0.0,
           'C_grid_kW_annual': 0.0,
           'C_control_fix_annual': 0.0,
           'C_prosumertax': 88.81,
           'tariff_ref': 'multi_price',
           'meter_ref': 'smart_meter',
           'PV_ref': 0.0,
           'inverter_ref': 0.0,
           'ts': 0.25
           }