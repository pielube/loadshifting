   
import os
import time
import pandas as pd 
from simulation import shift_load
import json

start_time = time.time()
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


filename = __location__ + '/inputs/cases.json'
cases = json.load(open(filename))
idx_casestobesim = range(1,len(cases))   
idx_casestobesim = [1,2,3]  


namecase = 'default'
conf = cases[namecase]
# load prices:
prices_filename = os.path.join(os.path.dirname(filename),conf['prices'])
prices = pd.read_csv(prices_filename,index_col=0)
res,demand_15min,demand_shifted,pflows = shift_load(conf,prices)
res = res.drop(labels=['Valeur'],axis=1)
 

for jjj in idx_casestobesim:
    namecase = 'case'+str(jjj)
    # namecase = 'default'

    conf = cases[namecase]
    
    # load prices:
    prices_filename = os.path.join(os.path.dirname(filename),conf['prices'])
    prices = pd.read_csv(prices_filename,index_col=0)

    results,demand_15min,demand_shifted,pflows = shift_load(conf,prices)
    
    # Saving results to excel
    file = __location__ + '/simulations/case_results.xlsx'
    with pd.ExcelWriter(file, engine='openpyxl', mode='a',if_sheet_exists='replace') as writer:  
        results.to_excel(writer, sheet_name=namecase)
        
    # Building dataframe for alternative excel file layout
    res[namecase] = results['Valeur']

# Saving results to excel 2
file = __location__ + '/simulations/case_results.xlsx'
with pd.ExcelWriter(file, engine='openpyxl', mode='a',if_sheet_exists='replace') as writer:  
    res.to_excel(writer, sheet_name='test')

exectime = (time.time() - start_time)
print('It all took {:.1f} seconds'.format(exectime))




















