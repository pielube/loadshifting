
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
import pickle


"""
Loading inputs
"""

with open('inputs/firstsim/1f_machine0.json') as f:
  inputs = json.load(f)
  

mostrapcurves = [4,1,1]
names = ['1f_machine0','2f_machine0','4f_machine0']

for ii in range(3):

    """
    Loading results
    """
    
    path = r'simulations/firstsim'
    name = names[ii]+'.pkl'
    file = os.path.join(path,name)
    results = pd.read_pickle(file)
    results = results[mostrapcurves[ii]]
    
    name = names[ii]+'_occ.pkl'
    file = os.path.join(path,name)
    resoccupancy = pd.read_pickle(file)
    resoccupancy = resoccupancy[mostrapcurves[ii]]
    
    """
    Load shifting for the appliances
    Strategy 1
    """
    
    # Admissible time windows according to energy prices
    with open('inputs/economics.json') as g:
      econ = json.load(g)
    
    scenario  = 'test'
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
    occ = np.ones(len(resoccupancy[0]))
    for i in range(len(resoccupancy)):
        resoccupancy[i] = [1 if a==1 else 0 for a in resoccupancy[i]]
        occ = np.multiply(occ,resoccupancy[i])
    occ = occ[:-1].copy()
    occupancy = np.zeros(len(results['StaticLoad']))
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
    
    for app in inputs['appliances']['apps']:
        print(app)
        app_n = strategy1(results[app],admtimewin,probshift)
        results[app+'Shift'] = app_n
        print(sum(results[app])/60./1000.)
        print(sum(results[app+'Shift'])/60./1000.)
    
    execshift = (time.time() - startshift)
    print("Time to shift the appliances: {:.1f} seconds".format(execshift))
    
    shift = ['StaticLoad','TumbleDryerShift','DishWasherShift','WashingMachineShift','DomesticHotWater','HeatPumpPower','EVCharging']
    results = results[shift]
    
    path = r'.\simulations\firstsim'
    if not os.path.exists(path):
        os.makedirs(path)
    name = names[ii]+'_shifted.pkl'
    file = os.path.join(path,name)
    with open(file, 'wb') as b:
        pickle.dump(results,b)












        