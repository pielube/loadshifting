   
import os
import time
import pandas as pd 
from demands import compute_demand
import json

start_time = time.time()
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

#N = 10 # Number of stochastic simulations to be run for the demand curves

filename = __location__ + '/inputs/config.json'
conf = json.load(open(filename))

#Adjusting the config:
conf['cont']['wetapp'] = 'none'
conf['pv']['yesno'] = False
conf['batt']['yesno'] = False
conf['ev']['yesno'] = False
conf['pv']['yesno'] = False
conf['hp']['loadshift'] = False
conf['dhw']['loadshift'] = False

conf['dwelling']['type'] = '1f'

# Source: https://energie.wallonie.be/servlet/Repository/bilan-domestique-et-equivalents-2019-v2.pdf?ID=67309
# pour 2019
conso = {'StaticLoad':2513,             # p 34, cludes wet appliances
         'HeatPump_1f': 3600,           # p 49
         'HeatPump_234f': 6400,
         'WashingMachine':190,          # p 34
         'TumbleDryer':240,
         'DishWasher':190}

# Source: https://energie.wallonie.be/servlet/Repository/bilan-domestique-et-equivalents-2019-v2.pdf?ID=67309
# pour 2019
logements_cadastre = {'2f':456725,   # p24
                      '3f':382354,
                      '4f':518219,
                      '1f':271249}
logements = pd.DataFrame.from_dict(logements_cadastre,columns=['cadastre'],orient='index')

# fraction des logements occupés slon le cadastre et l'ICEDD
# Source: https://energie.wallonie.be/servlet/Repository/bilan-domestique-et-equivalents-2019-v2.pdf?ID=67309
# pour 2019
logements_occupes = 1523/1720
logements['reel'] = logements['cadastre'] * logements_occupes

# proportion de la pop dans chaque type de logement
# source: https://www.cehd.be/media/1160/2018_05_03_chiffrescles2017_final.pdf  p194
logements.loc['4f','pop_frac'] = 0.43
logements.loc['2f','pop_frac'] = 0.44 * logements.loc['2f','cadastre']/(logements.loc['2f','cadastre']+logements.loc['3f','cadastre'])
logements.loc['3f','pop_frac'] = 0.44 * logements.loc['3f','cadastre']/(logements.loc['2f','cadastre']+logements.loc['3f','cadastre'])
logements.loc['1f','pop_frac'] = 0.13

# demographie 2019:
# source: https://energie.wallonie.be/servlet/Repository/bilan-domestique-et-equivalents-2019-v2.pdf?ID=67309
demo = { 'habitants': 3633795,
         'menages': 1581386}
print('taille moyenne des ménages: ' + str(demo['habitants']/demo['menages']))

logements['habitants'] = logements['pop_frac'] * demo['habitants']
logements['taille_moyenne'] = logements['habitants']/logements['reel']

# source: https://energie.wallonie.be/servlet/Repository/bilan-domestique-et-equivalents-2019-v2.pdf?ID=67309
# surface moyenne habitable chauffée 2019
logements.loc['4f','surface'] = 118
logements.loc['2f','surface'] = 99
logements.loc['3f','surface'] = 102
logements.loc['1f','surface'] = 67


#rapport entre surface totale et chauffée pour tous les logements wallons
# source bilan énergétique de la wallonie p 26
ratio_surface_tot_chauffee = 209/101

# conso moyenne de EV, 14770 km en moyenne, 200 kWh/ 1000 km
conso_ev = 14770 * 0.2
print('conommation moyenne véhicule électrique (kWh): ' + str (conso_ev))

N = 20

out = []
data = pd.DataFrame(index=range(N))

for i in range(N):
    res = compute_demand(conf)
    out.append(res)
    conso = res['results'][0].sum(axis=0)/(8760*3.6)
    data.loc[i,'type'] = res['input_data'][0]['dwelling']['type']
    data.loc[i,'Afloor'] = res['input_data'][0]['BuildingEnvelope']['Afloor']
    data.loc[i,'Nmembers'] = len(res['input_data'][0]['members'])
    for key in conso.index:
        data.loc[i,key] = conso[key]
    



exectime = (time.time() - start_time)
print('It all took {:.1f} seconds'.format(exectime))



data.to_csv(conf['dwelling']['type'] + '_stats.csv')
















