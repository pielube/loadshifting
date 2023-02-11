

import numpy as np
import pandas as pd
import strobe
import ramp
from functions import ProcebarExtractor,HouseholdMembers,load_climate_data,COP_deltaT,HPSizing,HouseHeating
import os,sys
from joblib import Memory
import defaults
from readinputs import read_config

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

memory = Memory(__location__ + '/cache/', verbose=defaults.verbose)

#@memory.cache
def compute_demand(conf):
    '''
    Function that generates the stochastic time series for
    - The occupancy profiles
    - The household electrical demand
    - The heat pump demand
    - The DHW demand
    - The Electric vehicle charging profile

    Parameters
    ----------
    conf : dict
        Dictionary with the simulation inputs. Requires the keys 'dwelling', 'hp', 'ev'

    Returns
    -------
    out : dict
        Contains, for each simulation, the demand curves, the occupancy profile and the input data.

    '''

    
    if 'load_demands' not in conf:

        N=conf['sim']['N']
        out = {'results':[],'occupancy':[],'input_data':[]}
    
        for jj in range(N):          # run the simulation N times and append the results to the list
        
            # People living in the dwelling, taken as input or from Strobe's list
            conf['members'] = HouseholdMembers(conf)
                   
            # Thermal parameters of the dwelling
            # Taken from Procebar .xls files
            if 'BuildingEnvelope' not in conf:
                print('randomly selecting house typology')
                conf['BuildingEnvelope'] = ProcebarExtractor(conf['dwelling']['type'],True)
                
            # Rough estimation of solar gains based on data from Crest
            # Could be improved
            typeofdwelling = conf['dwelling']['type'] 
            if typeofdwelling == '4f':
                conf['BuildingEnvelope']['A_s'] = 4.327106037
            elif typeofdwelling == '3f':
                conf['BuildingEnvelope']['A_s'] = 4.862912117
            elif typeofdwelling == '2f':
                conf['BuildingEnvelope']['A_s'] = 2.790283243
            elif typeofdwelling == '1f':
                conf['BuildingEnvelope']['A_s'] = 1.5  
            else:
                sys.exit('invalid dwelling type')
            
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
            result,textoutput = strobe.simulate_scenarios(1,conf)
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
            if not conf['hp']['automatic_sizing']:
                QheatHP = conf['hp']['pnom']
            else:
                QheatHP = HPSizing(conf['BuildingEnvelope'],defaults.fracmaxP) # W
                            
            res_househeat = HouseHeating(conf['BuildingEnvelope'],QheatHP,Tset,Qintgains,temp15min,irr15min,n15min,defaults.heatseas_st,defaults.heatseas_end,ts15min)
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
            
            if conf['ev']['yesno']:
                result_ramp = ramp.EVCharging({x:conf[x] for x in ['sim','ev','members']}, result['occupancy'][n_scen])
                res_ramp_charge_home = result_ramp['charge_profile_home']
                conf['ev']['MainDriver'] = result_ramp['main_driver']
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
            out['input_data'].append(conf)  
        
    else:  # if a previous simulation has to be loaded (only taking the first one if there are more):
        import pickle
        out = pickle.load(open(conf['load_demands'],'rb'))
        
        # updating the configuration with the keys from the loaded demand
        conf['members'] = out['input_data'][0]['members']
        conf['BuildingEnvelope'] =  out['input_data'][0]['BuildingEnvelope']
        conf['loc'] =  out['input_data'][0]['loc']
        conf['ownership'] =  out['input_data'][0]['ownership']
        conf['updated'] = True
        # in DHW:
        for key in ['hloss','pnom','set_point','tcold','tfaucet','type','vol']:
            if conf['dhw'][key] != out['input_data'][0]['dhw'][key]:
                print('Overwirting the DHW key with loaded data: ' + key)
                conf['dhw'][key] = out['input_data'][0]['dhw'][key]
        # in dwelling:
        for key in ['member1','member2','member3','member4','member5','type']:
            if conf['dwelling'][key] != out['input_data'][0]['dwelling'][key]:
                print('Overwirting the dwelling key with loaded data: ' + key)
                conf['dwelling'][key] = out['input_data'][0]['dwelling'][key]
        # in ev:
        for key in ['MainDriver']:
            if key in out['input_data'][0]['ev']:
                conf['ev'][key] = out['input_data'][0]['ev'][key]            
        # in hp:
        for key in ['deadband','pnom','set_point']:
            if conf['hp'][key] != out['input_data'][0]['hp'][key]:
                print('Overwirting the hp key with loaded data: ' + key)
                conf['hp'][key] = out['input_data'][0]['hp'][key]
        # in pv:
        for key in [  'losses', 'tilt', 'azimut', 'powerfactor']:
            if conf['pv'][key] != out['input_data'][0]['pv'][key]:
                print('Overwirting the pv key with loaded data: ' + key)
                conf['pv'][key] = out['input_data'][0]['pv'][key]
        
        recalculate_hp = False
        if recalculate_hp:
            # Thermal parameters of the dwelling
            # Taken from Procebar .xls files
            if 'BuildingEnvelope' not in conf:
                print('randomly selecting house typology')
                conf['BuildingEnvelope'] = ProcebarExtractor(conf['dwelling']['type'],True)
                
            # Rough estimation of solar gains based on data from Crest
            # Could be improved
            typeofdwelling = conf['dwelling']['type'] 
            if typeofdwelling == '4f':
                conf['BuildingEnvelope']['A_s'] = 4.327106037
            elif typeofdwelling == '3f':
                conf['BuildingEnvelope']['A_s'] = 4.862912117
            elif typeofdwelling == '2f':
                conf['BuildingEnvelope']['A_s'] = 2.790283243
            elif typeofdwelling == '1f':
                conf['BuildingEnvelope']['A_s'] = 1.5  
            else:
                sys.exit('invalid dwelling type')       
                
            temp, irr = load_climate_data()
            index1min  = pd.date_range(start='2015-01-01',end='2016-01-01 00:00:00',freq='T')
            index10min = pd.date_range(start='2015-01-01',end='2016-01-01 00:00:00',freq='10T')
            index15min = pd.date_range(start='2015-01-01',end='2016-01-01 00:00:00',freq='15T')
            
            temp1min = pd.Series(data=temp,index=index1min)
            temp15min = temp1min.resample('15Min').mean()
            irr1min = pd.Series(data=irr,index=index1min)
            irr15min = irr1min.resample('15Min').mean()
            

            
            occ = out['occupancy'][0].values
            occupancy_10min = (occ.transpose()==1).sum(axis=0) # when occupancy==1, the person is in the house and not sleeping
            occupancy_10min = (occupancy_10min>0)  # if there is at least one person awake in the house
            occupancy_10min = pd.Series(data=occupancy_10min, index = index10min)
            occupancy_15min = occupancy_10min.reindex(index15min,method='nearest')
            Qintgains = out['results'][0]['InternalGains'].values
            Qintgains = pd.Series(data=Qintgains,index=index1min)
            Qintgains = Qintgains.resample('15Min').mean() 
            n1min  = len(index1min)
            n15min = len(index15min)
            
            ### House heating ###
            
            Tset = np.full(n15min,defaults.T_sp_low) + np.full(n15min,defaults.T_sp_occ-defaults.T_sp_low) * occupancy_15min
            ts15min = 0.25
            
            # Heat pump sizing
            if not conf['hp']['automatic_sizing']:
                QheatHP = conf['hp']['pnom']
            else:
                QheatHP = HPSizing(conf['BuildingEnvelope'],defaults.fracmaxP) # W
                            
            res_househeat = HouseHeating(conf['BuildingEnvelope'],QheatHP,Tset,Qintgains,temp15min,irr15min,n15min,defaults.heatseas_st,defaults.heatseas_end,ts15min)
            Qheat = res_househeat['Qheat']
            
            Eheat = np.zeros(n15min)
            
            for i in range(n15min):
                COP = COP_deltaT(temp15min[i])
                Eheat[i] = Qheat[i]/COP # W
            
            Eheat = pd.Series(data=Eheat,index=index15min)
            Eheat = Eheat.resample('1Min').ffill()
            Eheat = Eheat.reindex(index1min).fillna(method='ffill')
            Eheat = Eheat.to_numpy()
            
            out['results'][0]['HeatPumpPower'] = Eheat    
            out['input_data'][0]['BuildingEnvelope'] = conf['BuildingEnvelope']
            print(Eheat.sum()/60/1000)
            
        recalculate_ev = False
        if recalculate_ev:    
            if conf['ev']['yesno']:
                result_ramp = ramp.EVCharging({x:conf[x] for x in ['sim','ev','members']}, out['occupancy'][0])
                res_ramp_charge_home = result_ramp['charge_profile_home']
                conf['ev']['MainDriver'] = result_ramp['main_driver']
                out['input_data'][0]['ev']['MainDriver'] = result_ramp['main_driver']
                res_ramp_charge_home.loc[res_ramp_charge_home.index[-1],'EVCharging']=0
                out['results'][0]['EVCharging'] = res_ramp_charge_home['EVCharging']* 1000
        else:
            if 'MainDriver' in  out['input_data'][0]['ev']:
                conf['ev']['MainDriver'] = out['input_data'][0]['ev']['MainDriver']
                print(out['input_data'][0]['ev']['MainDriver'])
            else:
                conf['ev']['MainDriver'] = conf['dwelling']['member1']
                out['input_data'][0]['ev']['MainDriver'] =  conf['dwelling']['member1']
        
    if 'save_demands' in conf:
        import pickle
        pickle.dump(out,open(conf['save_demands'],'wb'))

    return out


if __name__ == "__main__":
    """
    Testing the main function and saving the results
    """
    conf,prices,config_full = read_config('inputs/config.xlsx')
    conf['dwelling']['type'] = '4f'
    conf['dwelling']['member1'] = 'FTE'
    conf['dwelling']['member2'] = 'FTE'
    conf['dwelling']['member3'] = 'PTE'
    thermal_parameters = {'Aglazed': 46.489999999999995,
                         'Aopaque': 92.02000000000001,
                         'Afloor': 131.7,
                         'Afloor_heated': 131.72000000000003,
                         'Afloor_tot': 148.5,
                         'volume': 375.34499999999997,
                         'Atotal': 592.65,
                         'Uwalls': 0.48,
                         'Uwindows': 2.75,
                         'ACH_vent': 0.5,
                         'ACH_infl': 0.7,
                         'VentEff': 0.0,
                         'Ctot': 16698806.413587457,
                         'Uavg': 0.7261926799532233,
                         'ref_procebar': 103,
                         'A_s': 4.327106037}
    conf['BuildingEnvelope'] = thermal_parameters  # comment this line to randomly extract the building envelope from procebar
    
    out = compute_demand(conf)
    
    # Heat pump consumption:
    E_hp = out['results'][0]['HeatPumpPower'].sum()/(60*1000)
    print('Heat pump consumption: ' + str(E_hp) + ' kWh')









        