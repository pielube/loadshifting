#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 10:57:35 2022

@author: sylvain
"""

import pickle
import time
from temp_functions import shift_appliance

with open('examples/shiftinputs.pkl', 'rb') as handle:
    (app,admtimewin,probshift) = pickle.load(handle)
    
# adding non-constant consumption in starts 2,3 and 5 (for testing purposes)
app[2201:2221] = app[2201:2221] /2
app[2438:2448] = app[2438:2448] /2
app[3502:3522] = app[3502:3522] /2

time1 = time.time()

#app_n1,ncyc1,ncycshift1,maxshift1,avgshift1,cycnotshift1,enshift1 = strategy1(app,admtimewin,probshift)

time2 = time.time()

app_n2,ncyc2,ncycshift2,enshift2 = shift_appliance(app,admtimewin,probshift,max_shift=2*60,threshold_window=1,verbose=True)

time3 = time.time()

print('First method took {:.2f} seconds'.format(time2 - time1))
print('Second method took {:.2f} seconds'.format(time3 - time2))
