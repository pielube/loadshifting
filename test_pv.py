#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 14:04:48 2022

@author: Sylvain Quoilin
"""

from pv import pvgis_hist


inputs = {'location':(50.6,5.6,'Europe/Brussels',60,'Etc/GMT-2'),
         'Ppeak':2,
         'losses':12,
         'tilt':35,
         'azimuth':0,
         'year':2015}

pv = pvgis_hist(inputs)

print('Annual production with pvgis: {:.2f} kWh/kWp'.format(pv.sum()/inputs['Ppeak']/4000))