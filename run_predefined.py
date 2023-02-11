   
import os
import time
import pandas as pd 
from simulation import shift_load
from simulation import read_config
import json
import pickle

start_time = time.time()
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

conf,prices,config_full = read_config(__location__ + '/inputs/config.xlsx')


#res = pickle.load(open('simulations/demands_1f_unemployed_cl236_2478_30.pkl','rb'))

#Adjusting the config:
conf['cont']['wetapp'] = 'none'
conf['pv']['yesno'] = True
conf['batt']['yesno'] = False
conf['ev']['yesno'] = True
conf['pv']['yesno'] = True
conf['hp']['loadshift'] = False
conf['dhw']['loadshift'] = False
conf['hp']['yesno'] = False
conf['dhw']['yesno'] = True

conf['dwelling']['type'] = '2f'
conf['dwelling']['member1'] = 'FTE'     #['FTE','PTE','Retired','Unemployed']
conf['dwelling']['member2'] = 'FTE'
conf['dwelling']['member3'] = None
conf['dwelling']['member4'] = None
conf['dwelling']['member5'] = None


thermal_parameters = {'Aglazed': 19.0,
                     'Aopaque': 94.82,
                     'Afloor': 63.900000000000006,
                     'Afloor_heated': 121.39999999999999,
                     'Afloor_tot': 185.6,
                     'volume': 307.15999999999997,
                     'Atotal': 364.65,
                     'Uwalls': 0.48,
                     'Uwindows': 2.75,
                     'ACH_vent': 0.5,
                     'ACH_infl': 0.8,
                     'VentEff': 0.0,
                     'Ctot': 11696565.816875711,
                     'Uavg': 0.6489423680575901,
                     'ref_procebar': 203,
                     'A_s': 4.862912117}
thermal_parameters = {'Aglazed': 24.75, 'Aopaque': 42.65, 'Afloor': 44.379999999999995, 'Afloor_heated': 91.28, 'Afloor_tot': 206.38, 'volume': 255.58399999999995, 'Atotal': 205.68, 'Uwalls': 0.4, 'Uwindows': 2.5, 'ACH_vent': 0.5, 'ACH_infl': 0.7, 'VentEff': 0.0, 'Ctot': 6289720.615173837, 'Uavg': 0.5681747243426633, 'ref_procebar': 304, 'A_s': 2.790283243}
conf['BuildingEnvelope'] = thermal_parameters
# del conf['BuildingEnvelope']



conf['save_demands'] = 'simulations/demands1.pkl'     # adding this key to save the demands in the specified path

#conf['load_demands'] = 'simulations/demands_1f_unemployed_cl236_2478_30ok.pkl'
#conf['load_demands'] = 'simulations/demands_2f_2pers_cl3456_32ok.pkl'
#conf['load_demands'] = 'simulations/demands_3f_2pers_cl1237_34ok.pkl'
conf['load_demands'] = 'simulations/demands_4f_3pers_cl12345_34ok.pkl'
conf_ref = conf.copy()

results,demand_15min,demand_shifted,pflows,input_data = shift_load(conf,prices)

loads = demand_15min.sum()/4
load_noev_nodhw_nohp = loads['StaticLoad'] + loads['WashingMachine'] + loads['TumbleDryer'] + loads['DishWasher'] 
print(results)
print(loads)

#while (load_noev_nodhw_nohp < 3000) or (load_noev_nodhw_nohp > 3800) or (results.loc['selfsuffrate','Valeur'] < 0.3) or (loads['DomesticHotWater'] > 2900):
# while (loads['HeatPumpPower'] > 3800) or (loads['HeatPumpPower'] > 3000):
#     conf = conf_ref.copy()
#     results,demand_15min,demand_shifted,pflows,input_data = shift_load(conf,prices)
#     loads = demand_15min.sum()/4
#     load_noev_nodhw_nohp = loads['StaticLoad'] + loads['WashingMachine'] + loads['TumbleDryer'] + loads['DishWasher'] 
#     print(results)
#     print(loads)
#     print(input_data['BuildingEnvelope'])

summary = pd.DataFrame(index=loads.index)


for i in range(1,5):
    conf['load_demands'] = "inputs/loads_" + str(i) + "f.pkl"
    results,demand_15min,demand_shifted,pflows,input_data = shift_load(conf,prices)
    loads = demand_15min.sum()/4
    summary[str(i) + 'f'] = loads
    


print(summary)

summary.to_csv('consumptions_households.csv')








