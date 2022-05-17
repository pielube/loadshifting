
import os
import time
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
    config,pvbatt_param,econ_param,tariffs,inputs,N = conf['config'],conf['pvbatt_param'],conf['econ_param'],conf['tariffs'],conf['housetype'],conf['N']

    outs = shift_load(config,pvbatt_param,econ_param,tariffs,inputs,N)
    
    """
    Saving results to Excel
    """
    # TODO
    #   - add column with time horizion EconomicVar['time_horizon']
    #   - add columns with el prices
    #   - add columns with capacity-related prices
    #   - add in previous passages overall electricity shifted (right here set to 0)
    
    house    = config['house']
    scenario = econ_param['scenario']    
    enprices = tariffs['prices']
    gridfees = tariffs['gridfees']
    
    # Saving results to excel
    file = __location__ + '/simulations/test'+house+'.xlsx'
    WriteResToExcel(file,config['sheet'],outs[0],econ_param,enprices[scenario],gridfees[scenario],config['row'])


exectime = (time.time() - start_time)
print('It all took {:.1f} seconds'.format(exectime))




















