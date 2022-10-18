
import json
import pandas as pd
import os
import datetime
import defaults

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))     
     
def read_sheet(file,sheet):
    '''
    function that reads one sheet of the excel config file and outputs it in a dataframe
    '''
    raw = pd.read_excel(file,sheet_name=sheet)
    raw.rename(columns={ raw.columns[0]: "varname" }, inplace = True)
    raw = raw.loc[raw['varname'].notna(),:]
    raw.index = raw['varname']
    return raw[['Variable','Valeur','Description']]


def read_json(filename):
    '''
    Function that read the json config file for the load-shifting library and load the corresponding prices table
    
    Parameters
    ----------
    filename : string
        path to the config file

    Returns
    -------
    dict
    '''
    out = json.load(open(filename))
    prices_filename = os.path.join(os.path.dirname(filename),out['prices'])
    prices = pd.read_csv(prices_filename,index_col=0)
    
    return out,prices


def read_config(filename):
    '''
    Function that read the excel config file for the load-shifting library
    Parameters
    ----------
    filename : string
        path to the config file

    Returns
    -------
    dict
    '''
    out = {}
    config_full = read_sheet(filename,'main')
    out['ownership'] = read_sheet(filename,'ownership')
    out['ownership'] = out['ownership']['Valeur'].to_dict()
    
    sellprice_2D = pd.read_excel(filename,sheet_name='sellprice',index_col=0)
    gridprice_2D = pd.read_excel(filename,sheet_name='gridprice',index_col=0)
    energyprice_2D = pd.read_excel(filename,sheet_name='energyprice',index_col=0)
    
    idx = pd.date_range(start=datetime.datetime(year = sellprice_2D.index[0].year, month = 1, day = 1,hour=0),end = datetime.datetime(year = sellprice_2D.index[0].year, month = 12, day = 31,hour=23),freq='1h')
    
    # turn data into a column (pd.Series)
    sellprice = sellprice_2D.stack()
    sellprice.index = idx
    
    gridprice = gridprice_2D.stack()
    gridprice.index = idx
    
    energyprice = energyprice_2D.stack()
    energyprice.index = idx
    
    prices = pd.DataFrame(index = idx)
    prices['sell'] = sellprice
    prices['energy'] = energyprice
    prices['grid'] = gridprice
    
    config = config_full['Valeur']
    # Transform selected string variables to boolean:
    for x in ['dwelling_washing_machine','dwelling_tumble_dryer','dwelling_dish_washer','hp_yesno','hp_loadshift','hp_automatic_sizing',
              'dhw_yesno','dhw_loadshift','ev_yesno','ev_loadshift','pv_yesno','pv_automatic_sizing','batt_yesno','econ_smart_meter',
              'pv_inverter_automatic_sizing']:
        if config[x] in ['Oui','Yes','yes']:
            config[x] = True
        else:
            config[x] = False
   
    # translate text variables into the standard english form used by the library:
    for key in defaults.translate:
        config[key] = defaults.translate[key][config[key]]
        
    # Add the reference weather-year if not present:
    if 'sim_year' not in config:
        config['sim_year'] = defaults.year
       
    # write the configuration into sub-dictionnaries
    for prefix in ['sim','dwelling','hp','dhw','ev','pv','batt','econ','cont','loc']:
        subset = config[[x.startswith(prefix + '_') for x in config.index]]
        n = len(prefix)+1
        subset.index = [x[n:] for x in subset.index]
        out[prefix] = subset.to_dict()
    
    return out,prices


if __name__ == '__main__':
    
    # load the new, all-included config file:
    filename = 'inputs/config.xlsx'
    conf,prices = read_config(filename)
    
    # write to json/csv input format:
    filename_csv = filename[:-5] + '_prices.csv'
    filename_json = filename[:-5] + '.json'
    prices.to_csv(filename_csv)
    conf['prices'] = os.path.split(filename_csv)[-1]
    json.dump(conf,open(filename_json, "w"),indent=5)
    
    # re-load json config:
    conf_json, prices_csv = read_json(filename_json)
    
    

    
























