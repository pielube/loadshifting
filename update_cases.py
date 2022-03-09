#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small utility script to batch-modify the case inputs from the json file

To be adapted each time

@author: sylvain
"""
import json
import os
import defaults

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


inputhpath = __location__ + '/inputs/'
out = {}

# Case description
with open(inputhpath + 'cases.json','r') as f:
    cases = json.load(f)

for c in cases:
    print(c)
    # removing unnecessary keys:
    del cases[c]['WetAppBool']
    del cases[c]['WetAppAutoBool']
    del cases[c]['DHWBool']
    del cases[c]['HeatingBool']
    del cases[c]['EVBool']
    # renaming other keys for more clarity
    cases[c]['WetAppManualShifting'] = cases[c].pop('WetAppManBool')
    cases[c]['PresenceOfPV'] = cases[c].pop('PVBool')
    cases[c]['PresenceOfBattery'] = cases[c].pop('BattBool')

with open(inputhpath + 'cases2.json','w') as f:
    json.dump(cases,f, indent=4)