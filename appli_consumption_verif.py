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


import copy




#Consuption of appliance
def appli_consumption (cases,housestypes,N,wanted_case,factor_corr = None,corr = False):
    
    app_cons={}
    correction = {}
    for case in cases :
        
        if case in wanted_case :
            print(case)
            #Simulation for each type of accomodation

            out = compute_demand(housestypes[cases[case]['house']],N,factor_gain_sim=factor_gain_sim)
            results = out['results']
            factor_gain = out['factor gain']

   
            #Correction
            if corr == True :
                dict_corr = {}
                for simulation in factor_gain :
                    sim_factor=factor_gain[simulation]
                    for appli in sim_factor : 
                        if appli in dict_corr.keys() :
                            dict_corr[appli].append(sim_factor[appli]['factor'])
                        else :
                            dict_corr[appli] = [sim_factor[appli]['factor']]
                appliances = list(dict_corr.keys())
                for appli in appliances : 
                    dict_corr [appli] = sum(dict_corr[appli])/len(dict_corr[appli])
                correction [case] = dict_corr
            #Consumption of each appliance by year
            app_cons_home={}
            columns = cases[case]['columns']
            for k in range(N) : 
                
                sim_index = {}
                for appli in columns :
                    sim_index['Annual consumption by '+appli] = results[k][appli].sum()/60000
                
                sim_index ['Annual consumption by DomesticHotWater'] = results[k]['DomesticHotWater'].sum()/60000
                sim_index ['Annual consumption by HeatPumpPower'] =results[k]['HeatPumpPower'].sum()/60000
                
                app_cons_home['simulation {}'.format(k)] = sim_index
            app_cons_home['members'] = members 
            app_cons[case] = app_cons_home
            app_cons[case]['factor gain'] = factor_gain
            
            
                
            print(case + ' is done')

    return(app_cons,out)


#data processing verification

    factor_mean = {}     

    if corr == True :
        factor_mean = {}
        H4f = 0.301
        H2f =  0.488
        H1f = 0.211
        average_coeff = {'case1' : H4f, 'case59' : H2f, 'case77' : H1f}
        for case in correction :
            for appli in correction[case] :
                
                if appli in factor_mean.keys() :
                    
                    factor_mean[appli].append (average_coeff[case]*correction[case][appli])
                else :
                   
                    factor_mean[appli] = [average_coeff[case]*correction[case][appli]]
        for appli in factor_mean :
            factor_mean[appli] = sum(factor_mean[appli])
        
        
    if factor_corr is not None :
        for case in app_cons :
            for simulation in app_cons[case] : 
                if simulation != 'factor gain':
                    sim = app_cons[case][simulation] 
                
                    for appli in factor_mean :
                        sim['Annual consumption by '+appli] = sim['Annual consumption by '+appli]/factor_mean[appli]
    return(app_cons,out,factor_mean,correction)

#Récupération des fichiers nécessaires et définition des entrées



N=100

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


with open('inputs/factor_mean.json') as json_file :
    fm = json.load(json_file)


wanted_case = ['case59','case1','case77']
app_cons,out,fm,correction = appli_consumption (cases,housetypes,N,wanted_case)

filename = 'inputs/app_cons_members.json'
with open(filename, 'w',encoding='utf-8') as f:
    json.dump(app_cons, f,ensure_ascii=False, indent=4)
# filename ='\inputs/factor_mean.json'
# with open(filename, 'w',encoding='utf-8') as f:
#     json.dump(fm, f,ensure_ascii=False, indent=4)
# filename = 'inputs/correction.json'
# with open(filename, 'w',encoding='utf-8') as f:
#     json.dump(correction, f,ensure_ascii=False, indent=4)

>>>>>>> Stashed changes
    
wanted_case = ['case1','case59','case77']

#Simulation et sortie en dictionnaire
app_cons,out, factor_mean,correction = appli_consumption (cases,housetypes,N,wanted_case)

<<<<<<< Updated upstream
filename = 'inputs/app_cons_members.json'
=======
filename = 'inputs/app_cons_members_correction.json'
>>>>>>> Stashed changes
with open(filename, 'w',encoding='utf-8') as f:
    json.dump(app_cons, f,ensure_ascii=False, indent=4)


<<<<<<< Updated upstream
# filename ='inputs/corr_factor.json'
# with open(filename, 'w',encoding='utf-8') as f:
#     json.dump(factor_mean, f,ensure_ascii=False, indent=4)
    
# filename ='inputs/corr_factor_global.json'
# with open(filename, 'w',encoding='utf-8') as f:
#     json.dump(correction, f,ensure_ascii=False, indent=4)
=======
filename ='inputs/corr_factor.json'
with open(filename, 'w',encoding='utf-8') as f:
    json.dump(factor_mean, f,ensure_ascii=False, indent=4)
    
filename ='inputs/corr_factor_global.json'
with open(filename, 'w',encoding='utf-8') as f:
    json.dump(correction, f,ensure_ascii=False, indent=4)
>>>>>>> Stashed changes

    


#Consuption of appliance
# def appli_consumption (cases,housestypes,N,wanted_case,factor_mean = None,corr = True):
    
#     app_cons={}
#     correction = {}
#     for case in cases :
        
#         if case in wanted_case :
#             print(case)
#             #Simulation for each type of accomodation
#             out,members = compute_demand(housestypes[cases[case]['house']],N,factor_gain_sim=factor_gain_sim)
#             results = out['results']
#             factor_gain = out['factor gain']
            
#             #Correction
#             if corr == True :
#                 dict_corr = {}
#                 for simulation in factor_gain :
#                     sim_factor=factor_gain[simulation]
#                     for appli in sim_factor : 
#                         if appli in dict_corr.keys() :
#                             dict_corr[appli].append(sim_factor[appli]['factor'])
#                         else :
#                             dict_corr[appli] = [sim_factor[appli]['factor']]
#                 appliances = list(dict_corr.keys())
#                 for appli in appliances : 
#                     dict_corr [appli] = sum(dict_corr[appli])/len(dict_corr[appli])
#                 correction [case] = dict_corr
                
#             #Consumption of each appliance by year
#             app_cons_home={}
#             columns = cases[case]['columns']
#             for k in range(N) : 
                
#                 sim_index = {}
                
#                 for appli in columns :
#                     sim_index['Annual consumption by '+appli] = results[k][appli].sum()/60000
                
#                 sim_index ['Annual consumption by DomesticHotWater'] = results[k]['DomesticHotWater'].sum()/60000
#                 sim_index ['Annual consumption by HeatPumpPower'] =results[k]['HeatPumpPower'].sum()/60000
                
#                 app_cons_home['simulation {}'.format(k)] = sim_index
#             app_cons_home['members'] = members 
#             app_cons[case] = app_cons_home
#             app_cons[case]['factor gain'] = factor_gain
            
            
                
#             print(case + ' is done')
            
#     if corr == True :
#         factor_mean1 = {}
#         H4f = 0.301
#         H2f =  0.488
#         H1f = 0.211
#         average_coeff = {'case1' : H4f, 'case59' : H2f, 'case77' : H1f}
#         for case in correction :
#             for appli in correction[case] :
                
#                 if appli in factor_mean1.keys() :
                    
#                     factor_mean1[appli].append (average_coeff[case]*correction[case][appli])
#                 else :
                   
#                     factor_mean1[appli] = [average_coeff[case]*correction[case][appli]]
#         for appli in factor_mean1 :
#             factor_mean1[appli] = sum(factor_mean1[appli])
        
        
#     if factor_mean is not None :
#         for case in app_cons :
#             for simulation in app_cons[case] : 
#                 if simulation != 'factor gain' and simulation != 'members':
#                     sim = app_cons[case][simulation] 
                
#                     for appli in factor_mean :
#                         sim['Annual consumption by '+appli] = sim['Annual consumption by '+appli]/factor_mean[appli]
#     if corr ==True :
#         return(app_cons,out,factor_mean1)
#     else :
#         return (app_cons, out, factor_mean)
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
