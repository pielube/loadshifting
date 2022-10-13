   
import os
import time
import pandas as pd 
from functions import WriteResToExcel
from simulation import shift_load
import json

start_time = time.time()
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

#N = 10 # Number of stochastic simulations to be run for the demand curves


# idx_casestobesim = [0]

filename = __location__ + '/inputs/cases.json'
cases = json.load(open(filename))
idx_casestobesim = range(1,len(cases))   
idx_casestobesim = [1]   

for jjj in idx_casestobesim:
    namecase = 'case'+str(jjj)
    conf = cases[namecase]
    # namecase = 'default'
    
    # load prices:
    prices_filename = os.path.join(os.path.dirname(filename),conf['prices'])
    prices = pd.read_csv(prices_filename,index_col=0)

    results,demand_15min,demand_shifted,pflows = shift_load(conf,prices)
    
    """
    Saving results to Excel
    """   
    house    = conf['dwelling']['type']
    
    # Saving results to excel
    file = __location__ + '/simulations/case_results.xlsx'
    WriteResToExcel(file,conf['dwelling']['type'],results,conf)


exectime = (time.time() - start_time)
print('It all took {:.1f} seconds'.format(exectime))




















