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
    CF.loc[0,'Inv_PV'] = - (conf['econ']['C_PV_fix'] + conf['econ']['C_PV_kW'] * conf['pv']['ppeak'])
        
    # Inverter investment cost 
    CF.loc[0,'Inv_Invert'] =  conf['econ']['C_invert_share']*CF.loc[0,'Inv_PV']

    # Battery investment cost 
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
    prostax = conf['econ']['C_prosumertax']*min(conf['pv']['ppeak'],conf['pv']['inverter_pmax'])
    res = EnergyBuyAndSell(conf,enpricekWh, gridfeekWh, enpricekWh_sell, fromgrid, togrid, prostax)
        
    # Adding revenues and expenditures from buying and selling energy
    if conf['pv']['ppeak'] > 0 and conf['econ']['start_year'] < 2024 and not conf['econ']['smart_meter']:
        
        end2030 = 2030-conf['econ']['start_year']
        
        for i in range(1,end2030+1): # up to 2030
            CF.loc[i,'IncomeStG'] = res['IncomeStG_pre2030']  *(1+conf['econ']['elpriceincrease'])**(i-1)
            CF.loc[i,'CostStG'] = - res['CostStG_pre2030']  *(1+conf['econ']['elpriceincrease'])**(i-1)
            CF.loc[i,'CostBfG'] = (- res['CostBfG_energy'] - res['CostBfG_grid']) *(1+conf['econ']['elpriceincrease'])**(i-1)
        for i in range(end2030+2,conf['econ']['time_horizon']+1): # after 2030
            CF.loc[i,'IncomeStG'] = res['IncomeStG']  *(1+conf['econ']['elpriceincrease'])**(i-1)
            CF.loc[i,'CostStG'] = - res['CostStG']  *(1+conf['econ']['elpriceincrease'])**(i-1)
            CF.loc[i,'CostBfG'] = (- res['CostBfG_energy'] - res['CostBfG_grid']) *(1+conf['econ']['elpriceincrease'])**(i-1)    
    else:
        for i in range(1,conf['econ']['time_horizon']+1): # whole time horizon, no distinction in 2030
            CF.loc[i,'IncomeStG'] = res['IncomeStG']  *(1+conf['econ']['elpriceincrease'])**(i-1)
            CF.loc[i,'CostStG'] = - res['CostStG']  *(1+conf['econ']['elpriceincrease'])**(i-1)
            CF.loc[i,'CostBfG'] = (- res['CostBfG_energy'] - res['CostBfG_grid']) *(1+conf['econ']['elpriceincrease'])**(i-1)  

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



def EnergyBuyAndSell(conf,enpricekWh, gridfeekWh, enpricekWh_sell, fromgrid,togrid, prostax):
    '''
    Function that provides a dictionary with the tarifs pre and post 2030 depending
    on the selected tarification scheme
    '''
    
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
    
    CostBfG_energy = sum(fromgrid*enpricekWh)*ts
    CostBfG_grid   = sum(fromgrid*gridfeekWh)*ts
    
    # Selling
    
    if conf['pv']['ppeak'] > 0: # prosumers
        
        if conf['econ']['start_year'] < 2024: # PV installed before 2024
            # distinction to be made before and after 2030 when selling
            
            if not conf['econ']['smart_meter']:
                # Selling
                # pre 2030
                # cash flow depends on tariff type
                
                if conf['econ']['tariff'] == 'net-metering':
                    
                    IncomeStG_pre2030 = sum(togrid*(enpricekWh + gridfeekWh))*ts
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
                IncomeStG = sum(togrid*enpricekWh_sell)*ts
                CostStG   = 0 # min(sum(E['ToGrid']*gridfeekWh)*ts, prostax)

                
            elif conf['econ']['tariff']:
                
                # Selling
                # no distinction between pre and post 2030
                IncomeStG = sum(togrid*enpricekWh_sell)*ts
                CostStG   = 0 # min(sum(E['ToGrid']*gridfeekWh)*ts, prostax)
                IncomeStG_pre2030 = None
                CostStG_pre2030 = None
                
            
            else:
                print('Error: meter type specified does not exist')
                sys.exit('Error: meter type specified does not exist')
                
        else: # PV installed after 2024 => # no distinctions to be made before and after 2030
            IncomeStG = sum(togrid*enpricekWh_sell)*ts
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