
"""Simulate demand scenarios at building level, for specific household."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import strobe
import ramp
import json
import time
import random
from preprocess import ProcebarExtractor,HouseholdMembers,yearlyprices
import os


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
Load shifting for the appliances
Strategy 1
"""

# Admissible time windows according to energy prices
with open('inputs/economics.json') as g:
  econ = json.load(g)

scenario  = 'CRC2080'
timeslots = econ['timeslots']
prices    = econ['prices']
yprices = yearlyprices(scenario,timeslots,prices)
admprices = np.where(yprices <= prices[scenario]['hollow']/1000,1.,0.)
admprices = np.append(admprices,yprices[-1])

# Custom admissible windows
admcustom = np.ones(len(admprices))
for i in range(len(admprices)-60):
    if admprices[i]-admprices[i+60] == 1.:
        admcustom[i] = 0
    
# Adimissibile time windows according to occupancy
occ = np.ones(len(result['occupancy'][0][0]))
for i in range(len(inputs['members'])):
    result['occupancy'][0][i] = [1 if a==1 else 0 for a in result['occupancy'][0][i]]
    occ = np.multiply(occ,result['occupancy'][0][i])
occ = occ[:-1].copy()
occupancy = np.zeros(len(result['StaticLoad'][0,:]))
for i in range(len(occ)):
    for j in range(10):
        occupancy[i*10+j] = occ[i]
occupancy[-1] = occupancy[-2]

# Resulting admissibile time windows
admtimewin = admprices*admcustom*occupancy

# Probability of load being shifted
probshift = 1.

def strategy1(app,admtimewin,probshift):
    
    totlen = 0.
    
    app_s  = np.roll(app,1)
    starts = np.where(app-app_s> 1)[0]
    print(str(len(starts)))
    ends   = np.where(app-app_s<-1)[0]
    print(str(len(ends)))
    
    app_n = np.ones(len(app))
    
    for i in range(len(starts)):
        if admtimewin[starts[i]] == 0 and random.random() <= probshift:
            non_zeros = np.nonzero(admtimewin)[0] # array of indexes of non 0 elements
            distances = np.abs(non_zeros-starts[i]) # array of distances btw non 0 elem and ref
            closest_idx = np.where(distances == np.min(distances))[0]
            newstart = non_zeros[closest_idx][0]
            cyclen = ends[i]-starts[i]
            newend = newstart + cyclen
            if newend > len(app)-1:
                newend = len(app)-1
                cyclen = newend-newstart
                app_n[newstart:newend] = app[starts[i]:starts[i]+cyclen]
                totlen = totlen + cyclen
            else:
                app_n[newstart:newend] = app[starts[i]:ends[i]]
                totlen = totlen + cyclen
        else:
            app_n[starts[i]:ends[i]] = app[starts[i]:ends[i]]
            cyclen = ends[i]-starts[i]
            totlen = totlen+ cyclen
            
    print("cycle length: "+str(totlen))
    return app_n
            
startshift = time.time()

result_new = result.copy()
for app in inputs['appliances']['apps']:
    print(app)
    app_n = strategy1(result[app][0,:],admtimewin,probshift)
    result[app+'Shift'] = app_n
    result_new[app] = app_n
    print(sum(result[app][0,:])/60./1000.)
    print(sum(result[app+'Shift'])/60./1000.)
    print(sum(result_new[app])/60./1000.)

execshift = (time.time() - startshift)
print("Time to shift the appliances: {:.1f} seconds".format(execshift))


"""
Creating dataframe with the results
"""

n_steps = np.size(result['StaticLoad'][n_scen,:])
index = pd.date_range(start='2015-01-01 00:00', periods=n_steps, freq='1min')

# Dataframe of original results

df = pd.DataFrame(index=index,columns=['StaticLoad','TumbleDryer','DishWasher','WashingMachine','DomesticHotWater','HeatPumpPower','EVCharging','TumbleDryerShift','DishWasherShift','WashingMachineShift'])
result_ramp.loc[df.index[-1],'EVCharging']=0

for key in df.columns:
    if key in result:
        if key in ['TumbleDryerShift','DishWasherShift','WashingMachineShift']:
            df[key] = result[key]
        else:
            df[key] = result[key][n_scen,:]
    elif key in result_ramp:
        df[key] = result_ramp[key]* 1000
    else:
        df[key] = 0

"""
Quick plotting
"""

ystd = ['StaticLoad','TumbleDryer','DishWasher','WashingMachine','DomesticHotWater','HeatPumpPower','EVCharging']
yshift = ['StaticLoad','TumbleDryerShift','DishWasherShift','WashingMachineShift','DomesticHotWater','HeatPumpPower','EVCharging']

# Week
rng = pd.date_range(start='2015-08-02',end='2015-08-09',freq='min')
ax0 = df.loc[rng].plot.area(y=yshift,lw=0)
ax0.set(ylabel = "Power [W]") 
plt.legend(loc='upper left')

# Day
day = '2015-12-31'

rngyear = pd.date_range(start='2015-01-01 00:00:00',end='2016-01-01 00:00:00',freq='T')
dfprices = pd.DataFrame(admtimewin,index=rngyear)
ax = df.loc[day].plot.area(y=yshift,figsize=(8,4),lw=0,ylim=[0,6250])

ax1=ax.twinx()
ax1.spines['right'].set_position(('axes', 1.0))
dfprices.loc[day].plot(ax=ax1, color='black',legend=False)

ax.set(ylabel = "Power [W]")

# # Custom

# day2 = '2015-08-07'

# rngyear = pd.date_range(start='2015-01-01 00:00:00',end='2016-01-01 00:00:00',freq='T')
# dfprices2 = pd.DataFrame(admtimewin,index=rngyear)

# ax2 = df.loc[day2].plot.area(y=['WashingMachine','WashingMachineShift'],figsize=(8,4),lw=0,stacked=False)
# ax3 = ax2.twinx()
# ax3.spines['right'].set_position(('axes', 1.0))
# dfprices2.loc[day].plot(ax=ax3, color='black',legend=False)

# ax2.set(ylabel = "Power [W]")

# Admissible time windows

day3 = '2015-12-31'

dftimewin = pd.DataFrame(index=index,columns=['occupancy','prices','custom','total'])
dftimewin['occupancy'] = occupancy
dftimewin['prices'] = admprices
dftimewin['custom'] = admcustom
dftimewin['total'] = admtimewin

# y = ['occupancy','prices','custom','total']
y = ['prices']
ax4 = dftimewin.loc[day3].plot(y = y)
# ax4 = dftimewin.loc[day3].plot.area(lw=0)

"""
Saving results for prosumpy
"""

# Aggregating electricity consumptions in one demand

def saveforprosumpy(df,cols,name):
    df = df[cols].sum(axis=1)
    # Resampling at 15 min
    df = df.to_frame()
    df = df.resample('15Min').mean()
    # Extracting ref year used in the simulation
    df.index = pd.to_datetime(df.index)
    year = df.index.year[0]
    # # If ref year is leap remove 29 febr
    # leapyear = calendar.isleap(year)
    # if leapyear:
    #     start_leap = str(year)+'-02-29 00:00:00'
    #     stop_leap = str(year)+'-02-29 23:45:00'
    #     daterange_leap = pd.date_range(start_leap,stop_leap,freq='15min')
    #     df = df.drop(daterange_leap)
    # Remove last row if is from next year
    nye = pd.Timestamp(str(year+1)+'-01-01 00:00:00')
    df = df.drop(nye)
    # # New reference year 2015, to be used with TMY from pvlib
    # start_ref = '2015-01-01 00:00:00'
    # end_ref = '2015-12-31 23:45:00'
    # daterange_ref = pd.date_range(start_ref,end_ref,freq='15min')
    # df = df.set_index(daterange_ref)
    # Saving
    newpath = r'.\simulations'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    filename = 'run_'+name+'.pkl'
    filename = os.path.join(newpath,filename)
    df.to_pickle(filename)
    
names = ['noshift','shift']
columns = [ystd,yshift]

for i in range(2):
    saveforprosumpy(df,columns[i],names[i])











        