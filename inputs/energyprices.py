# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:31:05 2022

@author: pietro
"""

import pandas as pd

index = pd.date_range(start='2015-01-01',end='2015-12-31 23:59:00',freq='15T')

times = {}
prices_en = {}
prices_grid = {}

names = ['single','double','multi']

times['single']       = [['0:00','23:59']]
prices_en['single']   = [0.1845]
prices_grid['single'] = [0.1588]

times['double']       = [['0:00','6:00'],['6:00','11:00'],['11:00','17:00'],['17:00','22:00'],['22:00','23:59']]
prices_en['double']   = [0.1667,0.2029,0.1667,0.2029,0.1667]
prices_grid['double'] = [0.1142,0.1657,0.1142,0.1657,0.1142]

times['multi']       = [['0:00','6:00'],['6:00','11:00'],['11:00','17:00'],['17:00','22:00'],['22:00','23:59']]
prices_en['multi']   = [0.1667,0.3334,0.08335,0.41675,0.1667]
prices_grid['multi'] = [0.1142,0.1614,0.0906,0.1849,0.1142]

names = ['single','double','multi']

for i in names:
    df = pd.DataFrame(index=index,columns=['energy','grid','sell'],dtype='float64')
    df['sell'] = 0.04
    for j in range(len(times[i])):
        df['energy'][df.between_time(times[i][j][0],times[i][j][1]).index] = prices_en[i][j]
        df['grid'][df.between_time(times[i][j][0],times[i][j][1]).index] = prices_grid[i][j]
        name = i + '_price.csv'
        df.to_csv(name)

















