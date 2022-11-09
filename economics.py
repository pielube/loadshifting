# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:14:08 2022

@author: pietro
"""

import numpy as np
import numpy_financial as npf
import pandas as pd
import sys
import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
from joblib import Memory
memory = Memory(__location__ + '/cache/', verbose=1)

def CashFlows(conf,prices,fromgrid,togrid):
    '''
    Function that calculates the cash flows linked to the household electricity consumption/generation

    Parameters
    ----------
    conf : dict
         Household configuration.
    prices : pandas.DataFrame
        Prices (buy/sell) for grid electricity.
    fromgrid : pandas.Series
        Electricity bought from the grid.
    togrid : pandas.Series
        Electricity injected in the grid.

    Returns
    -------
    CF : pandas.DataFrame
        Cash flows, disaggregated by expense/revenu type.

    '''
    
    Nyears = conf['econ']['time_horizon']+1
    CF = pd.DataFrame(index=range(Nyears))
    
    # PV investment cost:
    if conf['pv']['ppeak'] == 0:
        CF.loc[0,'Inv_PV'] = 0
    else:
        CF.loc[0,'Inv_PV'] = - (conf['econ']['C_PV_fix'] + conf['econ']['C_PV_kW'] * conf['pv']['ppeak'])
        
    # Inverter investment cost
    CF.loc[0,'Inv_Invert'] =  conf['econ']['C_invert_share']*CF.loc[0,'Inv_PV']

    # Battery investment cost 
    if conf['batt']['capacity'] == 0:
        CF.loc[0,'Inv_Batt'] = 0
    else:
        CF.loc[0,'Inv_Batt'] = - (conf['econ']['C_batt_fix'] + conf['econ']['C_batt_kWh'] * conf['batt']['capacity'])
    
    # Control system investment cost. A specific control is needed is appliances are automatically shifted, or if the EV, HP or DWH are shiftable
    control_needed = (conf['cont']['wetapp'] == 'automated') or conf['ev']['loadshift'] or conf['hp']['loadshift'] or conf['dhw']['loadshift']
    if control_needed:
        CF.loc[0,'Inv_Control'] = -conf['econ']['C_control']
    else:
        CF.loc[0,'Inv_Control'] = 0

    # Adding replacement costs to cashflows array - Battery
    NBattRep = int((conf['econ']['time_horizon']-1)/conf['batt']['lifetime'])
    for i in range(NBattRep):
        iyear = (i+1)*conf['batt']['lifetime']
        CF.loc[iyear,'Inv_Batt'] = CF.loc[0,'Inv_Batt']
        
    # Adding replacement costs to cashflows array - Inverter
    NInvRep = int((conf['econ']['time_horizon']-1)/conf['pv']['inverter_lifetime'])
    for i in range(NInvRep):
        iyear = (i+1)*conf['pv']['inverter_lifetime']
        CF.loc[iyear,'Inv_Invert'] = CF.loc[0,'Inv_Invert']
        
    # Annual costs: grid fees, proportional to the maximum load
    CF.loc[1:conf['econ']['time_horizon']+1,'C_grid'] = - conf['econ']['C_grid_kW_annual'] * max(fromgrid)

    # Annual costs: O&Ms
    CF.loc[1:conf['econ']['time_horizon']+1,'C_OM'] = - conf['econ']['C_OM_annual'] * (CF.loc[0,'Inv_PV'] + CF.loc[0,'Inv_Batt']) 
    
    # Annual costs: controller
    CF.loc[1:conf['econ']['time_horizon']+1,'C_cont'] = - conf['econ']['C_control_annual'] 
        
    # Contributions of buying and selling energy to cash flows - Analyzed case
    enpricekWh = prices['energy'].values
    gridfeekWh = prices['grid'].values
    enpricekWh_sell = prices['sell'].values
    
    # Prosumer tax
    prostax = conf['econ']['C_prosumertax']*min(conf['pv']['ppeak'],conf['pv']['inverter_pmax'])
    
    # Capacity tariff
    start = 305 # November 1st 
    end   =  90 # March 31st
    a = fromgrid[np.r_[0:end*24*4-1,(start-1)*24*4-1:365*24*4-1]]-10
    capterm = np.sum(np.trunc(a[a>0])+1)*conf['econ']['C_capacitytariff']
    
    res = EnergyBuyAndSell(conf,enpricekWh, gridfeekWh, enpricekWh_sell, fromgrid, togrid, prostax, capterm)
    
    # Adding revenues and expenditures from buying and selling energy
    end2030 = 2030 - conf['econ']['start_year']
    for i in range(1,end2030+1):
        CF.loc[i,'IncomeStG'] = res['pre2030']['IncomeStG']*(1+conf['econ']['elpriceincrease'])**(i-1)
        CF.loc[i,'CostBfG'] = (- res['pre2030']['CostBfG_energy'] - res['pre2030']['CostBfG_grid'])*(1+conf['econ']['elpriceincrease'])**(i-1)
    for i in range(end2030+2,conf['econ']['time_horizon']+1):
        CF.loc[i,'IncomeStG'] = res['post2030']['IncomeStG']*(1+conf['econ']['elpriceincrease'])**(i-1)
        CF.loc[i,'CostBfG'] = (- res['post2030']['CostBfG_energy'] - res['post2030']['CostBfG_grid'])*(1+conf['econ']['elpriceincrease'])**(i-1)

    CF.fillna(0,inplace=True)
    CF['CashFlows'] = CF.sum(axis=1)
    
    return CF

def FinancialMetrics(wacc,CF):
    '''
    Function that computes the profitability indicators from a vector of annual cash flows
    Year 0 corresponds to the investment year

    Parameters
    ----------
    wacc : float
         Weighted average cost of capital
    CF : pandas.Series
        yearly cash flows.

    '''
    N = len(CF)
    
    # Actualized cashflows
    CashFlowsAct = np.zeros(N)
    for i in range(N):
        CashFlowsAct[i] = CF[i]/(1+wacc)**(i)

    # NPV curve
    NPVcurve = np.zeros(N)
    NPVcurve[0] = CashFlowsAct[0]
    for i in range(N-1):
        NPVcurve[i+1] = NPVcurve[i]+CashFlowsAct[i+1]

    # Final NPV
    NPV = npf.npv(wacc,CF)
    # NPV = 0 if abs(NPV)<0.01 else NPV       # what is that?

    # Pay Back Period
    idx1 = np.where(NPVcurve[:-1] * NPVcurve[1:] < 0 )[0] +1
    if len(idx1) > 0:
        idx1 = idx1[0]
        fractional = (0-NPVcurve[idx1-1])/CashFlowsAct[idx1]
        PBP = idx1+fractional
    else:
        PBP = None
    
    # Internal Rate of Return
    IRR = npf.irr(CF)

    # Profit Index
    if CF[0] == 0:
        PI = None
    else:
        PI = -NPV/CF[0]
        
    return {'NPV':NPV,'IRR':IRR,'PBP':PBP,'PI':PI}


def scale_vector(vec_in,N,silent=False):
    ''' 
    Function that scales a numpy vector to the desired length
    
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



def EnergyBuyAndSell(conf,enpricekWh, gridfeekWh, enpricekWh_sell, fromgrid, togrid, prostax, capterm):
    '''
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
    '''
    
    out = {}
    
    out['pre2030'] = {}
    out['post2030'] = {}
    
    ts = conf['sim']['ts']
    N = len(fromgrid)
    
    if isinstance(enpricekWh,pd.Series):
        enpricekWh = enpricekWh.values
    if isinstance(gridfeekWh,pd.Series):
        enpricekWh = gridfeekWh.values
    if isinstance(enpricekWh_sell,pd.Series):
        enpricekWh = enpricekWh_sell.values
    
    if ts!= 1:
        enpricekWh = scale_vector(enpricekWh,N,silent=True)
        gridfeekWh = scale_vector(gridfeekWh,N,silent=True)
        enpricekWh_sell = scale_vector(enpricekWh_sell,N,silent=True)
    
    
    if conf['econ']['meter'] in ['analogue']: # Analogue meter
        
        if conf['pv']['yesno']: # Yes PV
            
            if conf['econ']['start_year'] >= 2024:
                
                print('Error: No analogue meters with PV installastions after 2024')
                sys.exit('Error: No analogue meters with PV installastions after 2024')
            
            if conf['econ']['tariff'] in ['mono','bi']:                

                # Net metering and prosumer tax
                out['pre2030']['CostBfG_energy'] = sum((fromgrid-togrid)*enpricekWh)*ts
                out['pre2030']['CostBfG_grid']   = sum((fromgrid-togrid)*gridfeekWh)*ts + prostax
                out['pre2030']['IncomeStG']      = 0.
                
                # Gross metering
                out['post2030']['CostBfG_energy'] = sum(fromgrid*enpricekWh)*ts
                out['post2030']['CostBfG_grid']   = min(sum(fromgrid*gridfeekWh)*ts,out['pre2030']['CostBfG_grid'])
                out['post2030']['IncomeStG']      = 0.
                

            elif conf['econ']['tariff'] in ['multi']:
                
                print('Error: No multi tariff with analogue meters')
                sys.exit('Error: No multi tariff with analogue meters')
                
            else:
                
                print('Error: Wrong tariff name')
                sys.exit('Error: Wrong tariff name')
                
        else: # No PV
            
            if conf['econ']['tariff'] in ['mono','bi']:
                
                # Gross metering (Net = Gross since no PV)
                out['pre2030']['CostBfG_energy'] = sum(fromgrid*enpricekWh)*ts
                out['pre2030']['CostBfG_grid']   = sum(fromgrid*gridfeekWh)*ts
                out['pre2030']['IncomeStG']      = 0.
                
                # No changes after 2030
                out['post2030']['CostBfG_energy'] = out['pre2030']['CostBfG_energy']
                out['post2030']['CostBfG_grid']   = out['pre2030']['CostBfG_grid']
                out['post2030']['IncomeStG']      = out['pre2030']['IncomeStG']
                
            elif conf['econ']['tariff'] in ['multi']:
                
                print('Error: No multi tariff with analogue meters')
                sys.exit('Error: No multi tariff with analogue meters')
                
            else:
                
                print('Error: Wrong tariff name')
                sys.exit('Error: Wrong tariff name')
                
    if conf['econ']['meter'] in ['smart_r1']:
        
        if conf['pv']['yesno']: # Yes PV
            
            if conf['econ']['tariff'] in ['mono','bi','multi']:
                
                if conf['econ']['start_year'] < 2024:
                    
                    # Net metering and prosumer tax
                    out['pre2030']['CostBfG_energy'] = sum((fromgrid-togrid)*enpricekWh)*ts
                    out['pre2030']['CostBfG_grid']   = sum((fromgrid-togrid)*gridfeekWh)*ts + prostax
                    out['pre2030']['IncomeStG']      = 0.
                    
                    # Gross metering
                    out['post2030']['CostBfG_energy'] = sum(fromgrid*enpricekWh)*ts
                    out['post2030']['CostBfG_grid']   = min(sum(fromgrid*gridfeekWh)*ts,out['pre2030']['CostBfG_grid'])
                    out['post2030']['IncomeStG']      = 0.
                    
                else:
                    
                    # Gross metering 
                    out['pre2030']['CostBfG_energy'] = sum(fromgrid*enpricekWh)*ts
                    out['pre2030']['CostBfG_grid']   = min(sum(fromgrid*gridfeekWh)*ts,out['pre2030']['CostBfG_grid'])
                    out['pre2030']['IncomeStG']      = 0.
                    
                    # No changes after 2030
                    out['post2030']['CostBfG_energy'] = out['pre2030']['CostBfG_energy']
                    out['post2030']['CostBfG_grid']   = out['pre2030']['CostBfG_grid']
                    out['post2030']['IncomeStG']      = out['pre2030']['IncomeStG']
                
            else:
                
                print('Error: Wrong tariff name')
                sys.exit('Error: Wrong tariff name')
                
        else: # No PV

            if conf['econ']['tariff'] in ['mono','bi','multi']:
                
                # Gross metering
                out['pre2030']['CostBfG_energy'] = sum(fromgrid*enpricekWh)*ts
                out['pre2030']['CostBfG_grid']   = sum(fromgrid*gridfeekWh)*ts
                out['pre2030']['IncomeStG']      = 0.
                
                # No changes after 2030
                out['post2030']['CostBfG_energy'] = out['pre2030']['CostBfG_energy']
                out['post2030']['CostBfG_grid']   = out['pre2030']['CostBfG_grid']
                out['post2030']['IncomeStG']      = out['pre2030']['IncomeStG']
            
            else:
                
                print('Error: Wrong tariff name')
                sys.exit('Error: Wrong tariff name')
                
    if conf['econ']['meter'] in ['smart_r3']:
        
        # No need to distinguish between with or without PV
        # If no PV togrid is all 0 and IncomeStG will be 0 accordingly
            
        if conf['econ']['tariff'] in ['mono','bi']:
            
            print('Error: No mono or bi tariffs with R3')
            sys.exit('Error: No mono or bi tariffs with R3')

        elif conf['econ']['tariff'] == 'multi':
            
            # Gross metering and capacity term
            out['pre2030']['CostBfG_energy'] = sum(fromgrid*enpricekWh)*ts
            out['pre2030']['CostBfG_grid']   = sum(fromgrid*gridfeekWh)*ts + capterm
            out['pre2030']['IncomeStG']      = sum(togrid*enpricekWh_sell)*ts

            # No changes after 2030
            out['post2030']['CostBfG_energy'] = out['pre2030']['CostBfG_energy']
            out['post2030']['CostBfG_grid']   = out['pre2030']['CostBfG_grid']
            out['post2030']['IncomeStG']      = out['pre2030']['IncomeStG']              

        else:
            
            print('Error: Wrong tariff name')
            sys.exit('Error: Wrong tariff name')
    
    return out