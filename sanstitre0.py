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
import statistics as stat




def somme_liste_4 (L1,L2,L3,L4)    :
    L=[]
    for i in range (len(L4)) : 
        Lt=[]
        
        for k in range (len(L4[i])):
            A=L1[i]
            B=L2[i]
            C=L3[i]
            D=L4[i]
            Lt.append( A[k]+B[k]+C[k]+D[k] )
        L.append(Lt)
    return (L)

def somme_liste_3 (L1,L2,L3)    :
    L=[]
    for i in range (len(L1)) : 
        L.append( L1[i]+L2[i]+L3[i] )
    return (L)

<<<<<<< Updated upstream
<<<<<<< Updated upstream
file_name = 'inputs/app_cons_members.json'
=======
file_name = 'inputs/app_cons_members_correction.json'
>>>>>>> Stashed changes
with open(file_name) as json_file : 
    appli_consumption = json.load(json_file)

results = {}
for case in appli_consumption:
    average={}
    home = appli_consumption[case]
    factor_t ={}
    for simulation in home :
<<<<<<< Updated upstream
        if simulation != 'factor gain' :
=======
        if simulation != 'factor gain' and  simulation!= 'members':
>>>>>>> Stashed changes
            for appli in home[simulation] :
                
                if appli in average.keys() :
                    average[appli].append(home[simulation][appli])
                else :
                    average[appli] = [home[simulation][appli]]
<<<<<<< Updated upstream
        else :
            factor = home[simulation]
            
            
            for simulation1 in factor :
                for appli in factor[simulation1] : 
                    if appli in factor_t.keys() :
                        
                        factor_t[appli].append(factor[simulation1][appli]['factor'][0])
                    else :
                        factor_t[appli]=factor[simulation1][appli]['factor']
=======
        elif simulation == 'factor gain' :
            factor = home[simulation]
            
            
            for si in factor :
                for appli in factor[si] : 
                    if appli in factor_t.keys() :
                        
                        factor_t[appli].append(factor[si][appli]['factor'])
                    else :
                        factor_t[appli]=[factor[si][appli]['factor']]
>>>>>>> Stashed changes
                        
    average['factor'] = factor_t
    results[case] = average 
    
#Repartition
H4f = 0.301
H2f =  0.488
H1f = 0.211

#Results for each case
DishWasher = []
TumbleDryer = []
WashingMachine = []
StaticLoad = []
DHW = []
DHW_fact=[]
HP = []
HP_fact=[]
StaticLoad_fact=[]
for case in results : 
    DishWasher.append(results[case]['Annual consumption by DishWasher'])
    TumbleDryer.append(results[case]['Annual consumption by TumbleDryer'])
    WashingMachine.append(results[case]['Annual consumption by WashingMachine'])
    StaticLoad.append(results[case]['Annual consumption by StaticLoad'])
<<<<<<< Updated upstream
    DHW.append(results[case]['DHW'])
    HP.append(results[case]['HP'])
=======
    DHW.append(results[case]['Annual consumption by DomesticHotWater'])
    HP.append(results[case]['Annual consumption by HeatPumpPower'])
>>>>>>> Stashed changes
    for key in results[case].keys() :
        if key == 'factor' :
            DHW_fact.append(results[case]['factor']['DomesticHotWater'])
            HP_fact.append(results[case]['factor']['HeatPumpPower'])
            StaticLoad_fact.append(results[case]['factor']['StaticLoad'])
DHW_fact = somme_liste_3( H4f*np.array(DHW_fact[0]),  H2f*np.array(DHW_fact[1]),  H1f*np.array(DHW_fact[2]))  
HP_fact = somme_liste_3( H4f*np.array(HP_fact[0]),  H2f*np.array(HP_fact[1]),  H1f*np.array(HP_fact[2]))
StaticLoad_fact = somme_liste_3( H4f*np.array(StaticLoad_fact[0]),  H2f*np.array(StaticLoad_fact[1]),  H1f*np.array(StaticLoad_fact[2]))
TotalConsumption = somme_liste_4(DishWasher,TumbleDryer,WashingMachine,StaticLoad)    
    

<<<<<<< Updated upstream
=======
=======
# file_name = 'inputs/app_cons_members_correction.json'

# with open(file_name) as json_file : 
#     appli_consumption = json.load(json_file)

# results = {}
# for case in appli_consumption:
#     average={}
#     home = appli_consumption[case]
#     factor_t ={}
#     for simulation in home :
#         if simulation != 'factor gain' and  simulation!= 'members':
#             for appli in home[simulation] :
                
#                 if appli in average.keys() :
#                     average[appli].append(home[simulation][appli])
#                 else :
#                     average[appli] = [home[simulation][appli]]


#         elif simulation == 'factor gain' :
#             factor = home[simulation]
            
            
#             for si in factor :
#                 for appli in factor[si] : 
#                     if appli in factor_t.keys() :
                        
#                         factor_t[appli].append(factor[si][appli]['factor'])
#                     else :
#                         factor_t[appli]=[factor[si][appli]['factor']]

                        
#     average['factor'] = factor_t
#     results[case] = average 
    
# #Repartition
# H4f = 0.301
# H2f =  0.488
# H1f = 0.211

# #Results for each case
# DishWasher = []
# TumbleDryer = []
# WashingMachine = []
# StaticLoad = []
# DHW = []
# DHW_fact=[]
# HP = []
# HP_fact=[]
# StaticLoad_fact=[]
# for case in results : 
#     DishWasher.append(results[case]['Annual consumption by DishWasher'])
#     TumbleDryer.append(results[case]['Annual consumption by TumbleDryer'])
#     WashingMachine.append(results[case]['Annual consumption by WashingMachine'])
#     StaticLoad.append(results[case]['Annual consumption by StaticLoad'])


#     DHW.append(results[case]['Annual consumption by DomesticHotWater'])
#     HP.append(results[case]['Annual consumption by HeatPumpPower'])

#     for key in results[case].keys() :
#         if key == 'factor' :
#             DHW_fact.append(results[case]['factor']['DomesticHotWater'])
#             HP_fact.append(results[case]['factor']['HeatPumpPower'])
#             StaticLoad_fact.append(results[case]['factor']['StaticLoad'])
# DHW_fact = somme_liste_3( H4f*np.array(DHW_fact[0]),  H2f*np.array(DHW_fact[1]),  H1f*np.array(DHW_fact[2]))  
# HP_fact = somme_liste_3( H4f*np.array(HP_fact[0]),  H2f*np.array(HP_fact[1]),  H1f*np.array(HP_fact[2]))
# StaticLoad_fact = somme_liste_3( H4f*np.array(StaticLoad_fact[0]),  H2f*np.array(StaticLoad_fact[1]),  H1f*np.array(StaticLoad_fact[2]))
# TotalConsumption = somme_liste_4(DishWasher,TumbleDryer,WashingMachine,StaticLoad)    
    


>>>>>>> Stashed changes
#Members for each case
# members=[]
# for case in appli_consumption :
#     t=appli_consumption[case]['members']
#     x=[]
#     for sim in t :
#         x.append(t[sim])
#     members.append(x)

<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======

>>>>>>> Stashed changes
#Simulation's display for each case
plt.figure(1)

ax1 = plt.subplot(131)
ax1.title.set_text('Dish Washer consumption ')
ax1.set_xlabel ('Cas utilisé')
ax1.set_ylabel ('Consommation en kWh')
ax1.xaxis.set_ticklabels(['Case 1','Case 60', 'Case 77'])
bp = plt.boxplot(DishWasher, meanline = True,showmeans =True)

ax2 = plt.subplot(132, sharey = ax1)
ax2.title.set_text('TumbleDryer consumption ')
# ax2.set_xlabel ('Cas utilisé')
# ax2.set_ylabel ('Consommation en kWh')
ax2.xaxis.set_ticklabels(['Case 1','Case 60', 'Case 77'])
bp = plt.boxplot(TumbleDryer, meanline = True,showmeans =True)

ax3 = plt.subplot(133, sharey=ax1)
ax3.title.set_text('Washing Machine consumption')
# ax3.set_xlabel ('Cas utilisé')
# ax3.set_ylabel ('Consommation en kWh')
ax3.xaxis.set_ticklabels(['Case 1','Case 60', 'Case 77'])
bp = plt.boxplot(WashingMachine, meanline = True,showmeans =True)

plt.figure(2)
ax4 = plt.subplot(131)
ax4.title.set_text('Domestic Hot Water consumption ')
ax4.set_xlabel ('Cas utilisé')
ax4.set_ylabel ('Consommation en kWh')
ax4.xaxis.set_ticklabels(['Case 1','Case 60', 'Case 77'])
bp = plt.boxplot(DHW, meanline = True,showmeans =True)

ax5 = plt.subplot(132, sharey=ax4)
ax5.title.set_text('Heat Pump consumption ')
# ax5.set_xlabel ('Cas utilisé')
# ax5.set_ylabel ('Consommation en kWh')
ax5.xaxis.set_ticklabels(['Case 1','Case 60', 'Case 77'])
bp = plt.boxplot(HP, meanline = True,showmeans =True)

ax6 = plt.subplot(133, sharey=ax4)
ax6.title.set_text('Total appliance consumption ')
# ax6.set_xlabel ('Cas utilisé')
# ax6.set_ylabel ('Consommation en kWh')
ax6.xaxis.set_ticklabels(['Case 1','Case 60', 'Case 77'])
bp = plt.boxplot(TotalConsumption, meanline = True,showmeans =True)

plt.figure(3)
ax7 = plt.subplot(131)
ax7.title.set_text('Factor gain for DHW ')
ax7.set_xlabel ('Cas utilisé')
ax7.set_ylabel ('Facteur de gain')
ax7.xaxis.set_ticklabels(['Moyenne nationale'])
bp = plt.boxplot(DHW_fact, meanline = True,showmeans =True)

ax8 = plt.subplot(132, sharey = ax7)
ax8.title.set_text('Factor gain for HeatPump Power ')
ax8.xaxis.set_ticklabels(['Moyenne nationale'])
bp = plt.boxplot(HP_fact, meanline = True,showmeans =True)

ax9 = plt.subplot(133, sharey = ax7)
ax9.title.set_text('Factor gain for StaticLoad ')
ax9.xaxis.set_ticklabels(['Moyenne nationale'])
bp = plt.boxplot(HP_fact, meanline = True,showmeans =True)
<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
#Moyenne global de consommation : 
DishWasher_moy = somme_liste_3( H4f*np.array(DishWasher[0]),  H2f*np.array(DishWasher[1]),  H1f*np.array(DishWasher[2]))
DishWasher_moy = np.mean(DishWasher_moy)

Tumbledryer_moy = somme_liste_3( H4f*np.array(TumbleDryer[0]),  H2f*np.array(TumbleDryer[1]),  H1f*np.array(TumbleDryer[2]))
Tumbledryer_moy = np.mean(Tumbledryer_moy)

WashingMachine_moy =  somme_liste_3( H4f*np.array(WashingMachine[0]),  H2f*np.array(WashingMachine[1]),  H1f*np.array(WashingMachine[2]))
WashingMachine_moy = np.mean(WashingMachine_moy)

TotalConsumption_moy = somme_liste_3( H4f*np.array(TotalConsumption[0]),  H2f*np.array(TotalConsumption[1]),  H1f*np.array(TotalConsumption[2]))
TotalConsumption_moy = np.mean(TotalConsumption_moy)

print('For ' +file_name + ' the means for DishWasher, TumbleDryer, WashingMachine and all combined are :'  +str(DishWasher_moy )+',' + str( Tumbledryer_moy )+ ',' + str( WashingMachine_moy )+ ',' +str( TotalConsumption_moy ))
=======

#Moyenne global de consommation : 
# DishWasher_moy = somme_liste_3( H4f*np.array(DishWasher[0]),  H2f*np.array(DishWasher[1]),  H1f*np.array(DishWasher[2]))
# DishWasher_moy = np.mean(DishWasher_moy)

# Tumbledryer_moy = somme_liste_3( H4f*np.array(TumbleDryer[0]),  H2f*np.array(TumbleDryer[1]),  H1f*np.array(TumbleDryer[2]))
# Tumbledryer_moy = np.mean(Tumbledryer_moy)

# WashingMachine_moy =  somme_liste_3( H4f*np.array(WashingMachine[0]),  H2f*np.array(WashingMachine[1]),  H1f*np.array(WashingMachine[2]))
# WashingMachine_moy = np.mean(WashingMachine_moy)

# TotalConsumption_moy = somme_liste_3( H4f*np.array(TotalConsumption[0]),  H2f*np.array(TotalConsumption[1]),  H1f*np.array(TotalConsumption[2]))
# TotalConsumption_moy = np.mean(TotalConsumption_moy)

#print('For ' +file_name + ' the means for DishWasher, TumbleDryer, WashingMachine and all combined are :'  +str(DishWasher_moy )+',' + str( Tumbledryer_moy )+ ',' + str( WashingMachine_moy )+ ',' +str( TotalConsumption_moy ))
>>>>>>> Stashed changes


# plt.plot(DishWasher,label='DishWasher consumption')
# plt.plot(TumbleDryer,label='TumbleDryer consumption')
# plt.plot(WashingMachine,label='WashingMachine consumption')
plt.legend()
plt.show()