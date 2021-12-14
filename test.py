
"""Simulate demand scenarios at building level, for specific household."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import strobe
import ramp
import json
import time
from preprocess import ProcebarExtractor,HouseholdMembers,yearlyprices


"""
Loading inputs
"""

with open('inputs/example.json') as f:
  inputs = json.load(f)
  
# People living in the dwelling
# Taken from StRoBe list
cond1 = 'members' not in inputs
cond2 = 'members' in inputs and inputs['members'] == None
if cond1 or cond2:
    inputs['members'] = HouseholdMembers(inputs['HP']['dwelling_type'])

# Thermal parameters of the dwelling
# Taken from Procebar .xls files

procebinp = ProcebarExtractor(inputs['HP']['dwelling_type'],True)
inputs['HP'] = {**inputs['HP'],**procebinp}  
  
start_time = time.time()

"""
Running the models
"""

# Strobe
# House thermal model + HP
# DHW
result,textoutput = strobe.simulate_scenarios(1,inputs)
n_scen = 0 # Working only with the first scenario

# RAMP-mobility
if inputs['EV']['loadshift']:
    result_ramp = ramp.EVCharging(inputs, result['occupancy'][n_scen])
else:
    result_ramp=pd.DataFrame()

exectime = (time.time() - start_time)/60.
print("Time to run the models: {:.1f} minutes".format(exectime))

"""
Creating dataframe with the results
"""
 
n_steps = np.size(result['StaticLoad'][n_scen,:])
index = pd.date_range(start='2015-01-01 00:00', periods=n_steps, freq='1min')
df = pd.DataFrame(index=index,columns=['StaticLoad','TumbleDryer','DishWasher','WashingMachine','DomesticHotWater','HeatPumpPower','EVCharging'])

result_ramp.loc[df.index[-1],'EVCharging']=0
#df.index.union(result_ramp.index)        # too slow

for key in df.columns:
    if key in result:
        df[key] = result[key][n_scen,:]
    elif key in result_ramp:
        df[key] = result_ramp[key]* 1000
    else:
        df[key] = 0


"""
Load shifting for the appliances
Strategy 1
"""

with open('inputs/economics.json') as g:
  econ = json.load(g)

# Annual energy prices
scenario  = 'CRC2080'
timeslots = econ['timeslots']
prices    = econ['prices']
yprices = yearlyprices(scenario,timeslots,prices)

# Occupancy
occ = result['occupancy'][0][0]
occ = occ[:-1].copy()
occupancy = np.zeros(len(result['StaticLoad'][0,:]))
for i in range(len(occ)):
    for j in range(10):
        occupancy[i*10+j] = occ[i]
occupancy[-1] = occupancy[-2]

# Admissible time windows
admtimewin = np.where(yprices <= prices[scenario]['hollow']/1000,1.,0.)
admtimewin = np.append(admtimewin,yprices[-1])
admtimewin = admtimewin*occupancy
#  admtimewin = [1 if a >= 1 else 0 for a in admtimewin]


def strategy1(app,admtimewin):
    
    app_s  = np.roll(app,1)
    starts = np.where(app-app_s> 1)[0]
    ends   = np.where(app-app_s<-1)[0]
    
    app_n = np.ones(len(app))
    
    for i in range(len(starts)):
         #if admtimewin[starts[i]] == 0 and random.random()<0.5:
        if admtimewin[starts[i]] == 0:
            non_zeros = np.nonzero(admtimewin)[0] # array of indexes of non 0 elements
            distances = np.abs(non_zeros-starts[i]) # array of distances btw non 0 elem and ref
            closest_idx = np.where(distances == np.min(distances))[0]
            newstart = non_zeros[closest_idx][0]
            cyclen = ends[i]-starts[i]
            newend = newstart + cyclen
            app_n[newstart:newend] = app[starts[i]:ends[i]]
        else:
            app_n[starts[i]:ends[i]] = app[starts[i]:ends[i]]
            
    return app_n
            
startshift = time.time()

for app in inputs['appliances']['apps']:
    print(sum(result[app][0,:])/60./1000.)
    app_n = strategy1(result[app][0,:],admtimewin)
    result[app][0,:] = app_n
    print(sum(result[app][0,:])/60./1000.)

execshift = (time.time() - startshift)
print("Time to shift the appliances: {:.1f} seconds".format(execshift))


"""
Quick plotting
"""

rng = pd.date_range(start='2015-08-09',end='2015-08-16',freq='min')
ax = df.loc[rng].plot.area(lw=0)
ax.set(ylabel = "Power [W]")
plt.legend(loc='upper left')

day = '2015-08-14'

rngyear = pd.date_range(start='2015-01-01 00:00:00',end='2016-01-01 00:00:00',freq='T')
dfprices = pd.DataFrame(admtimewin,index=rngyear)

ax = df.loc[day].plot.area(figsize=(8,4),lw=0)
ax1=ax.twinx()
ax1.spines['right'].set_position(('axes', 1.0))
dfprices.loc[day].plot(ax=ax1, color='black')
ax.set(xlabel = 'Time [min]')
ax.set(ylabel = "Power [W]")
plt.legend(loc='upper left')

# hour = '2015-08-11 17:10:00'
# df['WashingMachine'].loc[hour]



        