#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 19:00:43 2022

@author: Sylvain Quoilin
"""

import pandas as pd
    
def input_elprices(dict2,index):
    '''
    Generates a dataframe with the prices
    '''           
    times = {}
    prices_en = {}
    prices_grid = {}
    
    names = ['single','double','multi']
    
    times['single']       = [['0:00','23:59']]
    prices_en['single']   = [dict2['single_en']]
    prices_grid['single'] = [dict2['single_grid']]
    
    times['double']       = [['0:00','6:00'],['6:00','11:00'],['11:00','17:00'],['17:00','22:00'],['22:00','23:59']]
    prices_en['double']   = [dict2['double_en_low'],  dict2['double_en_high'],  dict2['double_en_low'],  dict2['double_en_high'],  dict2['double_en_low']]
    prices_grid['double'] = [dict2['double_grid_low'],dict2['double_grid_high'],dict2['double_grid_low'],dict2['double_grid_high'],dict2['double_grid_low']]
    
    times['multi']       = [['0:00','6:00'],['6:00','11:00'],['11:00','17:00'],['17:00','22:00'],['22:00','23:59']]
    prices_en['multi']   = [dict2['multi_en_midlow'],  dict2['multi_en_midhigh'],  dict2['multi_en_low'],  dict2['multi_en_high'],  dict2['multi_en_midlow']]
    prices_grid['multi'] = [dict2['multi_grid_midlow'],dict2['multi_grid_midhigh'],dict2['multi_grid_low'],dict2['multi_grid_high'],dict2['multi_grid_midlow']]
    
    for i in names:
        
        df = pd.DataFrame(index=index,columns=['energy','grid','sell'],dtype='float64')
        df['sell'] = dict2['sell']
        
        for j in range(len(times[i])):
            df['energy'][df.between_time(times[i][j][0],times[i][j][1]).index] = prices_en[i][j]
            df['grid'][df.between_time(times[i][j][0],times[i][j][1]).index] = prices_grid[i][j]
    return df


if __name__ == '__main__':
    
    filename = './config.xlsx'
    prices =  {'sell': 0.04, 
               'single_en': 0.1845, 
               'single_grid': 0.1705, 
               'double_en_low': 0.1667, 
               'double_en_high': 0.2029, 
               'double_grid_low': 0.1142, 
               'double_grid_high': 0.183, 
               'multi_en_low': 0.08335, 
               'multi_en_midlow': 0.1667, 
               'multi_en_midhigh': 0.3334, 
               'multi_en_high': 0.41675, 
               'multi_grid_low': 0.0829, 
               'multi_grid_midlow': 0.1142, 
               'multi_grid_midhigh': 0.1768, 
               'multi_grid_high': 0.2081}
    index = pd.date_range(start='2015-01-01',end='2015-12-31 23:59:00',freq='1h')

    elprices = input_elprices(prices,index)
    
    elprices.index = [elprices.index.date, elprices.index.hour]
    sellprice = elprices['sell'].unstack()
    gridprice = elprices['grid'].unstack()
    energyprice = elprices['energy'].unstack()
    
    with pd.ExcelWriter('./config.xlsx', engine='openpyxl', mode='a',if_sheet_exists='replace') as writer:  
        sellprice.to_excel(writer, sheet_name='sellprice')
        gridprice.to_excel(writer, sheet_name='gridprice')
        energyprice.to_excel(writer, sheet_name='energyprice')


