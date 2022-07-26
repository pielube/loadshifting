   
import os
import time
import pandas as pd 
from functions import WriteResToExcel
from simulation import load_config,shift_load


start_time = time.time()
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))



N = 10 # Number of stochastic simulations to be run for the demand curves
# N = 1

idx_casestobesim = [i for i in range(83)]
# idx_casestobesim = [0]

        

for jjj in idx_casestobesim:
    namecase = 'case'+str(jjj+1)
    # namecase = 'default'
    
    conf = load_config(namecase)
    config,pvbatt_param,econ_param,inputs,N = conf['config'],conf['pvbatt_param'],conf['econ_param'],conf['housetype'],conf['N']

    outs = shift_load(config,pvbatt_param,econ_param,inputs,N)
    
    """
    Saving results to Excel
    """   
    house    = config['house']
    
    inputhpath = __location__ + '/inputs/' + econ_param['tariff'] + '.csv'
    with open(inputhpath,'r') as f:
        prices = pd.read_csv(f,index_col=0)
            
    enprices = prices['energy'].to_numpy() # €/kWh
    gridfees = prices['grid'].to_numpy()   # €/kWh
    
    # Saving results to excel
    file = __location__ + '/simulations/test'+house+'.xlsx'
    WriteResToExcel(file,config['sheet'],outs[0],econ_param,enprices,gridfees,config['row'])


exectime = (time.time() - start_time)
print('It all took {:.1f} seconds'.format(exectime))




















