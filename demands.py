

import numpy as np
import pandas as pd
import strobe
import ramp
import json
from functions import ProcebarExtractor,HouseholdMembers,load_climate_data,COP_deltaT,HPSizing,HouseHeating
import os
from joblib import Memory
import defaults

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

memory = Memory(__location__ + '/cache/', verbose=defaults.verbose)

@memory.cache
def compute_demand(inputs,N,members= None,thermal_parameters=None):
    '''
    Function that generates the stochastic time series for
    - The occupancy profiles
    - The household electrical demand
    - The heat pump demand
    - The DHW demand
    - The Electric vehicle charging profile

    Parameters
    ----------
    inputs : dict
        Dictionary with the simulation inputs.
    N : int
        Number of stochastic simulations to be run.
    members: None, int, list
        Optional. If None inputs['members'] is used, if int or list follows HousholdMemebers func logic
    thermal_parameters: None, dict
        Optional. If None thermal parameters taken from Procebar

    Returns
    -------
    out : dict
        Contains, for each simulation, the demand curves, the occupancy profile and the input data.

    '''
    
    out = {'results':[],'occupancy':[],'input_data':[]}

    for jj in range(N):          # run the simulation N times and append the results to the list
    
        # People living in the dwelling, taken as input or from Strobe's list
        if members is not None:
            inputs['members'] = HouseholdMembers(members)
        else:
            inputs['members'] = HouseholdMembers(inputs['members'])
               
        # Thermal parameters of the dwelling
        # Taken from Procebar .xls files
        if thermal_parameters is not None:
            inputs['HP'] = {**inputs['HP'],**thermal_parameters}
        else:
            procebinp = ProcebarExtractor(inputs['HP']['dwelling_type'],True)
            inputs['HP'] = {**inputs['HP'],**procebinp}
        
        """
        Running the models
        """
        
        temp, irr = load_climate_data()
        index1min  = pd.date_range(start='2015-01-01',end='2016-01-01 00:00:00',freq='T')
        index10min = pd.date_range(start='2015-01-01',end='2016-01-01 00:00:00',freq='10T')
        index15min = pd.date_range(start='2015-01-01',end='2016-01-01 00:00:00',freq='15T')
        
        temp1min = pd.Series(data=temp,index=index1min)
        temp15min = temp1min.resample('15Min').mean()
        irr1min = pd.Series(data=irr,index=index1min)
        irr15min = irr1min.resample('15Min').mean()
        
        ### Strobe ###
        
        # Occupancy, appliances, water withdrawals, heat gains
        # DHW
        result,textoutput = strobe.simulate_scenarios(1,inputs)
        n_scen = 0 # Working only with the first scenario
        
        occ = np.array(result['occupancy'][n_scen])
        occupancy_10min = (occ==1).sum(axis=0) # when occupancy==1, the person is in the house and not sleeping
        occupancy_10min = (occupancy_10min>0)  # if there is at least one person awake in the house
        occupancy_10min = pd.Series(data=occupancy_10min, index = index10min)
        occupancy_15min = occupancy_10min.reindex(index15min,method='nearest')
        Qintgains = result['InternalGains'][n_scen]
        Qintgains = pd.Series(data=Qintgains,index=index1min)
        Qintgains = Qintgains.resample('15Min').mean() 
        n1min  = len(index1min)
        n15min = len(index15min)
        
        ### House heating ###
        
        Tset = np.full(n15min,defaults.T_sp_low) + np.full(n15min,defaults.T_sp_occ-defaults.T_sp_low) * occupancy_15min
        ts15min = 0.25
        
        # Heat pump sizing
        if inputs['HP']['HeatPumpThermalPower'] is not None:
            QheatHP = inputs['HP']['HeatPumpThermalPower']
        else:
            QheatHP = HPSizing(inputs,defaults.fracmaxP) # W
                        
        res_househeat = HouseHeating(inputs,QheatHP,Tset,Qintgains,temp15min,irr15min,n15min,defaults.heatseas_st,defaults.heatseas_end,ts15min)
        Qheat = res_househeat['Qheat']
        
        Eheat = np.zeros(n15min)
        Eheat_final = np.zeros((1,n1min))
        
        for i in range(n15min):
            COP = COP_deltaT(temp15min[i])
            Eheat[i] = Qheat[i]/COP # W
        
        Eheat = pd.Series(data=Eheat,index=index15min)
        Eheat = Eheat.resample('1Min').ffill()
        Eheat = Eheat.reindex(index1min).fillna(method='ffill')
        Eheat = Eheat.to_numpy()
        Eheat_final[0,:] = Eheat
        
        result['HeatPumpPower'] = Eheat_final
        
        ### RAMP-mobility ###
        
        if inputs['EV']['loadshift']:
            result_ramp = ramp.EVCharging(inputs, result['occupancy'][n_scen])
            res_ramp_charge_home = result_ramp['charge_profile_home']
            inputs['EV']['MainDriver'] = result_ramp['main_driver']
        else:
            res_ramp_charge_home = pd.DataFrame()
    
        """
        Creating dataframe with the results
        """
        
        n_steps = np.size(result['StaticLoad'][n_scen,:])
        index = pd.date_range(start='2015-01-01 00:00', periods=n_steps, freq='1min')
        
        index_10min = pd.date_range(start='2015-01-01 00:00', periods=len(result['occupancy'][n_scen][0]), freq='10min')
        
        # Dataframe of demands
        df = pd.DataFrame(index=index,columns=['StaticLoad','TumbleDryer','DishWasher','WashingMachine','DomesticHotWater','HeatPumpPower','HeatPumpPower2','EVCharging','InternalGains'],dtype=object)
        
        res_ramp_charge_home.loc[df.index[-1],'EVCharging']=0
        
        for key in df.columns:
            if key in result:
                df[key] = result[key][n_scen,:]
            elif key in res_ramp_charge_home:
                df[key] = res_ramp_charge_home[key]* 1000
            else:
                df[key] = 0
                
        # Dataframe with the occupancy data
        occupancy = pd.DataFrame(index=index_10min)
        for i,m in enumerate(result['members'][n_scen]):
            membername = str(i) + '-' + m 
            occupancy[membername] = result['occupancy'][n_scen][i]
        
        out['results'].append(df)
        out['occupancy'].append(occupancy)
        out['input_data'].append(inputs)    

    return out


if __name__ == "__main__":
    """
    Testing the main function and saving the results
    """
    path = './inputs'
    names = ['4f']# ['1f','2f','4f']
    
    for name in names:            # For each house type
        
        """
        Loading inputs
        """
        filename = name+'.json'
        file = os.path.join(path,filename)
        with open(file) as f:
          inputs = json.load(f)
        N = 1
      
        out = compute_demand(inputs,N)   









        