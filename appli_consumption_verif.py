# -*- coding: utf-8 -*-
"""
Created on %(date)s


@author: %Ilian

"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta 
from demands import compute_demand
from plots import make_demand_plot



#Consuption of appliance
def appli_consumption (cases,housestypes,N,wanted_case,correction = False):
    
    app_cons={}
    
    for case in cases :
        
        if case in wanted_case :
            print(case)
            #Simulation for each type of accomodation
            out = compute_demand(housestypes[cases[case]['house']],N,factor_gain_sim=factor_gain_sim)
            results = out['results']
            factor_gain = out['factor gain']
            
            #Consumption of each appliance by year
            app_cons_home={}
            columns = cases[case]['columns']
            for k in range(N) : 
                
                sim_index = {}
                if correction == True :
                    dict_corr = {}
                    for simulation in factor_gain :
                        sim_factor=factor_gain[simulation]
                        for appli in sim_factor : 
                            if appli in dict_corr.keys() :
                                dict_corr[appli].append(sim_factor[appli]['factor'][0])
                            else :
                                dict_corr[appli] = sim_factor[appli]['factor']
                for appli in columns :
                    if correction == False :
                        sim_index['Annual consumption by '+appli] = results[k][appli].sum()/60000
                    else :
                        if appli  !='EVCharging' and appli !='InternalGains' :
                            sim_index['Annual consumption by '+appli] = (results[k][appli].sum()/60000)/(dict_corr[appli].sum()/len(dict_corr[appli]))
                if correction == False :
                    sim_index ['DHW'] = results[k]['DomesticHotWater'].sum()/60000
                    sim_index ['HP'] =results[k]['HeatPumpPower'].sum()/60000
                else :
                    
                    sim_index ['DHW'] = (results[k]['DomesticHotWater'].sum()/60000)/(dict_corr['DomesticHotWater'].sum()/len(dict_corr['DomesticHotWater']))
                    sim_index ['HP'] =(results[k]['HeatPumpPower'].sum()/60000)/(dict_corr['HeatPumpPower'].sum()/len(dict_corr['HeatPumpPower']))
                
                app_cons_home['simulation {}'.format(k)] = sim_index
              
            app_cons[case] = app_cons_home
            app_cons[case]['factor gain'] = factor_gain
            
            
                
            print(case + ' is done')
    return(app_cons,out)


#data processing verification
N=50
with open('inputs/cases.json') as json_file : 
    cases = json.load(json_file)
with open('inputs/housetypes.json') as json_file : 
    housetypes = json.load(json_file)
with open('inputs/factor_gain.json') as json_file : 
    factor_gain_sim = json.load(json_file)
    
wanted_case = ['case59','case1','case77']
app_cons,out = appli_consumption (cases,housetypes,N,wanted_case,correction = True)

filename = 'inputs/app_cons_members_correction.json'
with open(filename, 'w',encoding='utf-8') as f:
    json.dump(app_cons, f,ensure_ascii=False, indent=4)