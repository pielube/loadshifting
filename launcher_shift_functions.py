
import random
import numpy as np
import pandas as pd
from statistics import mean
from itertools import chain

from prosumpy import dispatch_max_sc
from strobe.RC_BuildingSimulator import Zone

from temp_functions import EconomicAnalysis,EconomicAnalysisRefPV

import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))



# List of functions this .py file should contain

# - EconomicAnalysis
# - MostRapCurve
# - AdmTimeWinShift
# - DHWShiftTariffs
# - DHWShiftSC
# - HouseHeatingShiftTariffs
# - HouseHeatingShiftSC
# - EVShiftTariffs
# - EVShiftSC
# - ResultsAnalysis
# - WriteToExcel

# TODO
# - add description of all functions
# - comment all functions
# - change name of this file

<<<<<<< Updated upstream:launcher_shift_functions.py
@memory.cache
=======

def scale_timeseries(data,index):
    ''' 
    Function that takes a pandas Dataframe as input and interpolates it to the proper datetime index
    '''
    if isinstance(data,pd.Series):
        data_df = pd.DataFrame(data)
    elif isinstance(data,pd.DataFrame):
        data_df = data
    else:
        raise Exception("The input must be a pandas series or dataframe")
    dd = data_df.reindex(data_df.index.union(index)).interpolate(method='time').reindex(index)
    if isinstance(data,pd.Series):
        return dd.iloc[:,0]
    else:
        return dd


<<<<<<< Updated upstream:launcher_shift_functions.py
>>>>>>> Stashed changes:functions.py
=======

>>>>>>> Stashed changes:functions.py
def load_climate_data(datapath = __location__ + '/strobe/Data'):
    '''
    Function that loads the climate data from strobe
    '''
            # ambient data
    temp = np.loadtxt(datapath + '/Climate/temperature.txt')
    irr  = np.loadtxt(datapath + '/Climate/irradiance.txt')   
    return temp,irr


def MostRepCurve(demands,columns,ElPrices,timestep,econ_param):
    
    """
    Choosing most representative curve among a list of demand curves
    based on electricity bill buying all electricity from grid
    hence wiithout PV or batteries
    """

    # Technology parameters required by prosumpy
    # Battery forced to be 0 
    param_tech = {'BatteryCapacity': 0.,
                  'BatteryEfficiency': 1.0,
                  'MaxPower': 0.,
                  'InverterEfficiency': 1.,
                  'timestep': 0.25}
    
    # Technology parameters required by economic analysis
    # PV and battery forced to be 0
    inputs = {'PVCapacity': 0.,
              'BatteryCapacity': 0.}
    
    # Technology costs required economic analysis
    # Not relevant
    Inv = {'FixedPVCost':0.,
           'PVCost_kW':0.,
           'FixedBatteryCost':0.,
           'BatteryCost_kWh':0.,
           'PVLifetime':20.,
           'BatteryLifetime':0.,
           'OM': 0.,
           'FixedControlCost': 0.,
           'AnnualControlCost': 0.} 
    
    results = []
    
    pv = np.zeros(int((len(demands[0])-1)/15))
    date = pd.date_range(start='2015-01-01 00:00:00',end='2015-12-31 23:45:00',freq='15Min')
    pv = pd.DataFrame(data=pv,index=date)
    pv = pv.iloc[:,0]
    
    for ii in range(len(demands)):
        demand = demands[ii][columns]
        demand = demand.sum(axis=1)
        demand = demand/1000. # W to kW
        demand = demand.to_frame()
        demand = demand.resample('15Min').mean() # resampling at 15 min
        demand.index = pd.to_datetime(demand.index)
        year = demand.index.year[0] # extracting ref year used in the simulation
        nye = pd.Timestamp(str(year+1)+'-01-01 00:00:00') # remove last row if is from next year
        demand = demand.drop(nye)
        demand = demand.iloc[:,0]
        
        outputs = dispatch_max_sc(pv,demand,param_tech,return_series=False)
        # print_analysis(pv, demand, param_tech, outputs)
        
        inputs['ACGeneration'] = pv.to_numpy() # should be equal to outputs['inv2grid']+outputs['inv2load']
        inputs['Load'] = demand.to_numpy()
        inputs['ToGrid'] = outputs['inv2grid']
        inputs['FromGrid'] = outputs['grid2load']
        inputs['SC'] = outputs['inv2load']
        inputs['FromBattery'] = outputs['store2inv']
        
        out = EconomicAnalysis(inputs,econ_param,ElPrices,timestep,inputs['Load'])
        results.append(out['ElBill'])
    
    meanelbill = mean(results)
    var = results-meanelbill
    index = min(range(len(var)), key=var.__getitem__)
        
    return index


def AdmTimeWinShift(app,admtimewin,probshift):
   
    ncycshift = 0
    ncycnotshift = 0
    maxshift = 0
    totshift = 0
    enshift = 0.
    
    app_s  = np.roll(app,1)
    starts   = np.where(app-app_s>1)[0]
    ends   = np.where(app-app_s<-1)[0]
    
    app_n = np.zeros(len(app))
    
    for i in range(len(starts)):
        
        if admtimewin[starts[i]] == 1:
            app_n[starts[i]:ends[i]] += app[starts[i]:ends[i]]
        
    for i in range(len(starts)):
        
        if admtimewin[starts[i]] == 0:
            
            if random.random() > probshift:
                app_n[starts[i]:ends[i]] += app[starts[i]:ends[i]]
            else:
                
                ncycshift += 1
                
                non_zeros = np.nonzero(admtimewin)[0] # array of indexes of non 0 elements
                distances = np.abs(non_zeros-starts[i]) # array of distances btw non 0 elem and ref           
                closest_idx = np.where(distances == np.min(distances))[0]
                newstart = non_zeros[closest_idx][0]
                cyclen = ends[i]-starts[i]
                newend = newstart + cyclen
                
                while any(app_n[newstart:newend]):
                    non_zeros = np.delete(non_zeros,closest_idx)
                    if np.size(non_zeros)==0:
                        newstart = starts[i]
                        newend = ends[i]
                        ncycnotshift += 1
                        break
                    distances = np.abs(non_zeros-starts[i])
                    closest_idx = np.where(distances == np.min(distances))[0]
                    newstart = non_zeros[closest_idx][0]
                    cyclen = ends[i]-starts[i]
                    newend = newstart + cyclen
                           
                if newend > len(app)-1:
                    newend = len(app)-1
                    cyclen = newend-newstart
                    app_n[newstart:newend] += app[starts[i]:starts[i]+cyclen]
                else:
                    app_n[newstart:newend] += app[starts[i]:ends[i]]
            
            enshift += np.sum(app_n[newstart:newend])/60.
            maxshift = max(maxshift,abs(newstart-starts[i])/60.)
            totshift += abs(newstart-starts[i])
    
    avgshift = totshift/len(starts)/60.
    app_n=np.where(app_n==0,1,app_n)
    ncyc = len(starts)
    ncycshift = ncycshift - ncycnotshift
    
    if ncycnotshift > 0:
        val = np.sort(np.unique(app_n))
        if np.size(val) > 2:
            indexes = np.where(app_n==val[-1])[0]
            app_n[indexes]=val[-2]
    
    conspre  = sum(app)/60./1000.
    conspost = sum(app_n)/60./1000.
    print("Original consumption: {:.2f}".format(conspre))
    print("Number of cycles: {:}".format(ncyc))
    print("Number of cycles shifted: {:}".format(ncycshift))
    print("Consumption after shifting (check): {:.2f}".format(conspost))
    print("Max shift: {:.2f} hours".format(maxshift))
    print("Avg shift: {:.2f} hours".format(avgshift))
    print("Unable to shift {:} cycles".format(ncycnotshift))
                
    return app_n,enshift

<<<<<<< Updated upstream:launcher_shift_functions.py
<<<<<<< Updated upstream:launcher_shift_functions.py
=======
=======
>>>>>>> Stashed changes:functions.py

def shift_appliance(app,admtimewin,probshift,max_shift=None,threshold_window=0,verbose=False):
    '''
    This function shifts the duty duty cycles of a particular appliances according
    to a vector of admitted time windows.

    Parameters
    ----------
    app : numpy.array
        Original appliance consumption vector to be shifted
    admtimewin : numpy.array
        Vector of admitted time windows, where load should be shifted.
    probshift : float
        Probability (between 0 and 1) of a given cycle to be shifted
    max_shift : int
        Maximum number of time steps over which a duty cycle can be shifted
    threshold_window: float [0,1]
        Share of the average cycle length below which an admissible time window is considered as unsuitable and discarded
    verbose : bool
        Print messages or not. The default is False.

    Returns
    -------
    tuple with the shifted appliance load, the total number of duty cycles and 
    the number of shifted cycles

    '''
    ncycshift = 0                   # initialize the counter of shifted duty cycles
    if max_shift is None:
        max_shift = 24*60                    # maximmum time over which load can be shifted
    
    #remove offset from consumption vector:
    offset = app.min()
    app = app - offset
    
    # check if admtimewin is boolean:
    if not admtimewin.dtype=='bool':
        if (admtimewin>1).any() or (admtimewin<0).any():
            print('WARNING: Some values of the admitted time windows are higher than 1 or lower than 0')
        admtimewin = (admtimewin>0)
    
    # Define the shifted consumption vector for the appliance:
    app_n = np.full(len(app),offset)
    
    # Shift the app consumption vector by one time step:
    app_s  = np.roll(app,1)
    
    # Imposing the extreme values
    app_s[0] = 0; app[-1] = 0
    
    # locate all the points whit a start or a shutdown
    starting_times = (app>0) * (app_s==0)
    stopping_times = (app_s>0) * (app==0)
    
    # List the indexes of all start-ups and shutdowns
    starts   = np.where(starting_times)[0]
    ends   = np.where(stopping_times)[0]
    means = (( starts + ends)/2 ).astype('int')
    lengths = ends - starts
    
    # Define the indexes of each admitted time window
    admtimewin_s = np.roll(admtimewin,1)
    admtimewin_s[0] = False; admtimewin[-1] = False
    adm_starts   = np.where(admtimewin * np.logical_not(admtimewin_s))[0]
    adm_ends   = np.where(admtimewin_s * np.logical_not(admtimewin))[0]
    adm_lengths = adm_ends - adm_starts
    adm_means = (( adm_starts + adm_ends)/2 ).astype('int')
    admtimewin_j = np.zeros(len(app),dtype='int')
    
    # remove all windows that are shorter than the average cycle length:
    tooshort = adm_lengths < lengths.mean() * threshold_window
    adm_means[tooshort] = -max_shift -999999            # setting the mean to a low value makes this window unavailable
    
    for j in range(len(adm_starts)):            # create a time vector with the index number of the current time window
        admtimewin_j[adm_starts[j]:adm_ends[j]] = j

    
    # For all activations events:
    for i in range(len(starts)):
        length = lengths[i]
        
        if admtimewin[starts[i]] and admtimewin[ends[i]]:           # if the whole activation length is within the admitted time windows
            app_n[starts[i]:ends[i]] += app[starts[i]:ends[i]]
            j = admtimewin_j[starts[i]]
            admtimewin[adm_starts[j]:adm_ends[j]] = False       # make the whole time window unavailable for further shifting
            adm_means[j] = -max_shift -999999
            
        else:     # if the activation length is outside admitted windows
            if random.random() > probshift:
                app_n[starts[i]:ends[i]] += app[starts[i]:ends[i]]
            else:
                j_min = np.argmin(np.abs(adm_means-means[i]))          # find the closest admissible time window
                if np.abs(adm_means[j_min]-means[i]) > max_shift:     # The closest time window is too far away, no shifting possible
                    app_n[starts[i]:ends[i]] += app[starts[i]:ends[i]]
                else:
                    ncycshift += 1                                      # increment the counter of shifted cycles
                    delta = adm_lengths[j_min] - length
                    if delta < 0:                                        # if the admissible window is smaller than the activation length
                        t_start = int(adm_starts[j_min] - length/2)
                        t_start = np.minimum(t_start,len(app)-length)    # ensure that there is sufficient space for the whole activation at the end of the vector
                        
                        patch = 0                                      # patch added to deal with negative t_start
                        if t_start < 0:
                            patch = - t_start
                            length += t_start
                            t_start = 0 
                        
                        app_n[t_start:t_start+length] += app[starts[i]+patch:ends[i]] 
                        admtimewin[adm_starts[j_min]:adm_ends[j_min]] = False       # make the whole time window unavailable for further shifting
                        adm_means[j_min] = -max_shift -999999  
                    elif delta < length:                                    # This an arbitrary value
                        delay = random.randrange(1+delta)             # randomize the activation time within the allowed window
                        app_n[adm_starts[j_min]+delay:adm_starts[j_min]+delay+length] += app[starts[i]:ends[i]]
                        admtimewin[adm_starts[j_min]:adm_ends[j_min]] = False       # make the whole time window unavailable for further shifting
                        adm_means[j_min] = -max_shift -999999  
                    else:                                                    # the time window is longer than two times the activation. We split it and keep the first part
                        delay = random.randrange(1+length)                # randomize the activation time within the allowed window
                        app_n[adm_starts[j_min]+delay:adm_starts[j_min]+delay+length] += app[starts[i]:ends[i]]
                        admtimewin[adm_starts[j_min]:adm_starts[j_min]+2*length] = False       # make the first part of the time window unavailable for further shifting
                        adm_starts[j_min] = adm_starts[j_min]+2*length+1                   # Update the size of this time window
                        adm_means[j_min] = (( adm_starts[j_min] + adm_ends[j_min])/2 ).astype('int')
                        adm_lengths[j_min] = adm_ends[j_min] - adm_starts[j_min]
    app = app + offset
    enshift = np.abs(app_n - app).sum()/2
    
    if verbose: 
        if np.abs(app_n.sum() - app.sum())/app.sum() > 0.01:    # check that the total consumption is unchanged
            print('WARNING: the total shifted consumption is ' + str(app_n.sum()) + ' while the original consumption is ' + str(app.sum()))
        print(str(len(starts)) + ' duty cycles detected. ' + str(ncycshift) + ' cycles shifted in time')
        print(str(tooshort.sum()) + ' admissible time windows were discarded because they were too short')
        print('Total shifted energy : {:.2f}% of the total load'.format(enshift/app.sum()*100))

    return app_n,len(starts),ncycshift,enshift



def HPSizing(inputs,fracmaxP):

    if inputs['HP']['HeatPumpThermalPower'] == None:
        # Heat pump sizing
        # External T = -10°C, internal T = 21°C
        House = Zone(window_area=inputs['HP']['Aglazed'],
                     walls_area=inputs['HP']['Aopaque'],
                     floor_area=inputs['HP']['Afloor'],
                     room_vol=inputs['HP']['volume'],
                     total_internal_area=inputs['HP']['Atotal'],
                     u_walls=inputs['HP']['Uwalls'],
                     u_windows=inputs['HP']['Uwindows'],
                     ach_vent=inputs['HP']['ACH_vent'],
                     ach_infl=inputs['HP']['ACH_infl'],
                     ventilation_efficiency=inputs['HP']['VentEff'],
                     thermal_capacitance=inputs['HP']['Ctot'],
                     t_set_heating=21.,
                     max_heating_power=float('inf'))
        
        Tm = (21.-(-10.))/2
        House.solve_energy(0.,0.,-10.,Tm)
        QheatHP = House.heating_demand*fracmaxP
   
    else:
        # Heat pump size given as an input
        QheatHP = inputs['HP']['HeatPumpThermalPower']
    
    return QheatHP

def COP_Tamb(Temp):
    COP = 0.001*Temp**2 + 0.0471*Temp + 2.1259
    return COP

>>>>>>> Stashed changes:functions.py

def DHWShiftTariffs(demand, prices, thresholdprice, param, return_series=False):
    
    """ Tariffs-based battery dispatch algorithm.
    Battery is charged when energy price is below the threshold limit and as long as it is not fully charged.
    It is discharged as soon as the energy price is over the threshold limit and as long as it is not fully discharged.

    Arguments:
        demand (pd.Series): Vector of household consumption, kW
        prices (pd.Series): Vector of energy prices, €/kW
        thresholdprice (float): Price under which energy is bought to be stored in the battery, €/kW
        param (dict): Dictionary with the simulation parameters:
                timestep (float): Simulation time step (in hours)
                BatteryCapacity: Available battery capacity (i.e. only the the available DOD), kWh
                MaxPower: Maximum battery charging or discharging powers (assumed to be equal), kW
        return_series(bool): if True then the return will be a dictionary of series. Otherwise it will be a dictionary of ndarrays.
                        It is reccommended to return ndarrays if speed is an issue (e.g. for batch runs).
    Returns:
        dict: Dictionary of Time series

    """

    bat_size_e_adj = param['BatteryCapacity']
    bat_size_p_adj = param['MaxPower']
    timestep = param['timestep']
    
    # We work with np.ndarrays as they are much faster than pd.Series
    Nsteps = len(demand)
    LevelOfCharge = np.zeros(Nsteps)
    grid2store = np.zeros(Nsteps)
    store2load = np.zeros(Nsteps)

    admprices = np.where(prices <= thresholdprice,1,0)   
    demand1 = demand.to_numpy()

    LevelOfCharge[0] = bat_size_e_adj / 2.
    
    for i in range(1,Nsteps):
        
        if admprices[i] == 1: # low prices
            if LevelOfCharge[i-1] < bat_size_e_adj:  # if battery is full
                grid2store[i] = min((bat_size_e_adj - LevelOfCharge[i-1]) / timestep, bat_size_p_adj-demand[i])
            LevelOfCharge[i] =  LevelOfCharge[i-1]+grid2store[i]*timestep
                
        else: # high prices
            store2load[i] = min((LevelOfCharge[i-1] / timestep),demand[i],bat_size_p_adj)
            LevelOfCharge[i] =  LevelOfCharge[i-1]-store2load[i]*timestep

    grid2load = demand - store2load

    out = {'grid2store': grid2store,
           'grid2load': grid2load,
           'store2load': store2load,
           'LevelOfCharge': LevelOfCharge}
    
    if return_series:
        out_pd = {}
        for k, v in out.items():  # Create dictionary of pandas series with same index as the input demand
            out_pd[k] = pd.Series(v, index=demand.index)
        out = out_pd
        
    return out

<<<<<<< Updated upstream:launcher_shift_functions.py
<<<<<<< Updated upstream:launcher_shift_functions.py
@memory.cache
def HouseHeating(inputs,QheatHP,Tset,Qintgains,Tamb,irr,nminutes,heatseas_st,heatseas_end):
=======
=======
>>>>>>> Stashed changes:functions.py

def HouseHeating(inputs,QheatHP,Tset,Qintgains,Tamb,irr,nminutes,heatseas_st,heatseas_end,ts):
>>>>>>> Stashed changes:functions.py

    # Rough estimation of solar gains based on data from Crest
    # Could be improved
    
    typeofdwelling = inputs['HP']['dwelling_type'] 
    if typeofdwelling == 'Freestanding':
        A_s = 4.327106037
    elif typeofdwelling == 'Semi-detached':
        A_s = 4.862912117
    elif typeofdwelling == 'Terraced':
        A_s = 2.790283243
    elif typeofdwelling == 'Apartment':
        A_s = 1.5   
    Qsolgains = irr * A_s
        
    # Defining the house to be modelled with obtained HP size
    House = Zone(window_area=inputs['HP']['Aglazed'],
                walls_area=inputs['HP']['Aopaque'],
                floor_area=inputs['HP']['Afloor'],
                room_vol=inputs['HP']['volume'],
                total_internal_area=inputs['HP']['Atotal'],
                u_walls=inputs['HP']['Uwalls'],
                u_windows=inputs['HP']['Uwindows'],
                ach_vent=inputs['HP']['ACH_vent']/60,
                ach_infl=inputs['HP']['ACH_infl']/60,
                ventilation_efficiency=inputs['HP']['VentEff'],
                thermal_capacitance=inputs['HP']['Ctot'],
                t_set_heating=Tset[0], #inputs['HP']['Tthermostatsetpoint'],
                max_heating_power=QheatHP)
            
    Qheat = np.zeros(nminutes)
    Tinside = np.zeros(nminutes)

    d1 = 60*24*heatseas_end-1
    d2 = 60*24*heatseas_st-1
    concatenated = chain(range(1,d1), range(d2,nminutes))

    Tair = max(16.,Tamb[0])
    House.t_set_heating = Tset[0]    
    House.solve_energy(Qintgains[0], Qsolgains[0], Tamb[0], Tair)
    Qheat[0]   = House.heating_demand
    Tinside[0] = House.t_air    

    for i in concatenated:
        
        if i == d2:
            Tinside[i-1] = max(16.,Tamb[i-1])

        if Tset[i] != Tset[i-1]:
            House.t_set_heating = Tset[i]    
            
        House.solve_energy(Qintgains[i], Qsolgains[i], Tamb[i], Tinside[i-1])
        Qheat[i]   = House.heating_demand
        Tinside[i] = House.t_air
                       
    return Qheat, Tinside


def EVshift_PV(pv,arrive,leave,starts,ends,idx_athomewindows,LOC_min,LOC_max,param,return_series=False):
    
    """
    Function to shift at-home charging based on PV production
    Charging when PV power is available and LOC < LOC_max or when LOC < LOC_min regardless PV production.
    It requires start and end indexes of at-home time windows and charging events and
    to which at-home time window each charging event belongs. 
    For each at home time window LOC_min is defined as the charge obtained from reference at-home charging events
    and LOC_max as the total consumption of charging events in that at-home time window.
    
    Parameters:
        pv (pandas Series): vector Nsteps long with residual PV production, kW DC
        arrive (numpy array): vector of indexes, start at-home time windows  
        leave  (numpy array): vector of indexes, end   at-home time windows
        starts (numpy array): vector of indexes, start charging at-home time windows
        ends   (numpy array): vector of indexes, end   charging at-home time windows
        idx_athomewindows (numpy array): vector with which at-home window corresponds to each charging window
        LOC_min (numpy array): vector Nsteps long with min LOC, kWh
        LOC_max (numpy array): vector long as the number of at-home time windows with max LOC, kWh
        param (dict): dictionary with charge power [kW], inverter efficiency [-] and timestep [h]
        return_series (bool): if True then the return will be a dictionary of series. 
                              Otherwise it will be a dictionary of ndarrays.
                              It is reccommended to return ndarrays if speed is an issue (e.g. for batch runs).
                              Default is False.

    Returns:
        out (dict): dict with numpy arrays or pandas series with energy fluxes and LOC 
    """
    
    bat_size_p_adj = param['MaxPower']
    n_inv = param['InverterEfficiency']
    timestep = param['timestep']
    
    Nsteps = len(pv)
    pv_np = pv.to_numpy()
     
    pv2inv = np.zeros(Nsteps)
    inv2grid = np.zeros(Nsteps)
    inv2store = np.zeros(Nsteps)
    grid2store = np.zeros(Nsteps)
    LOC = np.zeros(Nsteps)
    
    # Not going twice through the same at-home time window    
    idx_athomewindows,idxs = np.unique(idx_athomewindows,return_index=True)
    LOC_max = LOC_max[idxs]
    
    for i in range(len(idx_athomewindows)): # iter over at-home time windows
        
        LOC[arrive[idx_athomewindows[i]]-1] = 0
        
        for j in range(arrive[idx_athomewindows[i]],leave[idx_athomewindows[i]]): # iter inside at-home time windows
                        
            pv2inv[j] = pv_np[j] # kW
            
            inv2store_t = min(pv2inv[j]*n_inv,bat_size_p_adj) # kW          
            LOC_t = LOC[j-1] + inv2store_t*timestep # kWh
            
            if LOC_t < LOC_min[j]:
                
                inv2store[j]  = inv2store_t # kW
                grid2store[j] = min(bat_size_p_adj-inv2store[j],(LOC_min[j]-LOC_t)/timestep) # kW
                                
                LOC[j] = LOC[j-1] + inv2store[j]*timestep + grid2store[j]*timestep # kWh
            
            elif  LOC_min[j] <= LOC_t <= LOC_max[i]:
                
                inv2store[j]  = inv2store_t # kW
                
                LOC[j] = LOC_t # kWh
                                
            elif LOC_t > LOC_max[i]:
                    
                inv2store[j] = (LOC_max[i]-LOC[j-1]) /timestep # kW
                
                LOC[j] = LOC_max[i] # kWh
   
    inv2grid = pv2inv*n_inv - inv2store # kW
        
    out = {'pv2inv': pv2inv,
           'inv2grid': inv2grid,
           'inv2store': inv2store,
           'grid2store': grid2store,
           'LevelOfCharge': LOC
            }
    
    if return_series:
        out_pd = {}
        for k, v in out.items():  # Create dictionary of pandas series with same index as the input pv
            out_pd[k] = pd.Series(v, index=pv.index)
        out = out_pd
    return out


def EVshift_tariffs(yprices_1min,pricelim,arrive,leave,starts,ends,idx_athomewindows,LOC_min,LOC_max,param,return_series=False):
    
    """
    Function to shift at-home charging based on tariffs
    Charging when energy price <= pricelim and LOC < LOC_max or when LOC < LOC_min regardless of energy price.
    It requires start and end indexes of at-home time windows and charging events and
    to which at-home time window each charging event belongs. 
    For each at home time window LOC_min is defined as the charge obtained from reference at-home charging events
    and LOC_max as the total consumption of charging events in that at-home time window.
    
    
    Parameters:
        yprices_1min (pandas Series): vector Nsteps long with energy prices, €
        arrive (numpy array): vector of indexes, start at-home time windows  
        leave  (numpy array): vector of indexes, end   at-home time windows
        starts (numpy array): vector of indexes, start charging at-home time windows
        ends   (numpy array): vector of indexes, end   charging at-home time windows
        idx_athomewindows (numpy array): vector with which at-home window corresponds to each charging window
        LOC_min (numpy array): vector Nsteps long with min LOC, kWh
        LOC_max (numpy array): vector long as the number of at-home time windows with max LOC, kWh
        param (dict): dictionary with charge power [kW], inverter efficiency [-] and timestep [h]
        return_series (bool): if True then the return will be a dictionary of series. 
                              Otherwise it will be a dictionary of ndarrays.
                              It is reccommended to return ndarrays if speed is an issue (e.g. for batch runs).
                              Default is False.

    Returns:
        out (dict): dict with numpy arrays or pandas series with energy fluxes and LOC 
    """
    
    bat_size_p_adj = param['MaxPower']
    n_inv = param['InverterEfficiency']
    timestep = param['timestep']
    
    Nsteps = len(yprices_1min)
    yprices_1min_np = yprices_1min.to_numpy()

    grid2store = np.zeros(Nsteps)
    LOC = np.zeros(Nsteps)
    
    # Not going twice through the same at-home time window    
    idx_athomewindows,idxs = np.unique(idx_athomewindows,return_index=True)
    LOC_max = LOC_max[idxs]
    
    for i in range(len(idx_athomewindows)): # iter over at-home time windows
        
        LOC[arrive[idx_athomewindows[i]]-1] = 0
        
        for j in range(arrive[idx_athomewindows[i]],leave[idx_athomewindows[i]]): # iter inside at-home time windows
            
            if yprices_1min_np[j] <= pricelim:
                grid2store[j] = min((LOC_max[i]-LOC[j-1])/timestep,bat_size_p_adj) # kW
                
            else:
                if LOC[j-1] < LOC_min[j]:
                    grid2store[j] = min((LOC_min[j]-LOC[j-1])/timestep,bat_size_p_adj) # kW

            LOC[j] = LOC[j-1] + grid2store[j]*timestep # kWh
        
    out = {'grid2store': grid2store,
           'LevelOfCharge': LOC
            }
    
    if return_series:
        out_pd = {}
        for k, v in out.items():  # Create dictionary of pandas series with same index as the input pv
            out_pd[k] = pd.Series(v, index=yprices_1min.index)
        out = out_pd
    return out



def ResultsAnalysis(pv_capacity,batt_capacity,pflows,ElPrices,prices,scenario,econ_param):
    
    # Running prosumpy to get SC and SSR and energy fluxes for economic analysis
    # All shifting must have already been modelled, including battery
    # param_tech is hence defined here and battery forced to be 0
    
    pv,demand_ref = pflows.pv,pflows.demand_noshift
    
    if batt_capacity > 0:
        demand = pflows.demand_shifted
    else:
        demand = pflows.demand_shifted_nobatt
    
    param_tech = {'BatteryCapacity': 0.,
                  'BatteryEfficiency': 1.,
                  'MaxPower': 0.,
                  'InverterEfficiency': 1.,
                  'timestep': 0.25}
    
    res_pspy = dispatch_max_sc(pv,demand,param_tech,return_series=False)

    Epspy = {}
    Epspy['PVCapacity']      = pv_capacity
    Epspy['BatteryCapacity'] = batt_capacity
    Epspy['ACGeneration'] = pv.to_numpy()
    Epspy['Load']         = demand.to_numpy()
    Epspy['ToGrid']       = res_pspy['inv2grid']
    Epspy['FromGrid']     = res_pspy['grid2load']
    Epspy['SC']           = res_pspy['inv2load']
    # Not used by economic analysis and would be all 0 considerng how prosumpy has been used
    # Epspy['FromBattery'] = outputs['store2inv']
    
    timestep = 0.25
    
    res_EA = EconomicAnalysis(Epspy,econ_param,ElPrices,timestep,demand_ref)
    
    # Running prosumpy for reference case with only PV
    # Used in another economic analysis to get NPV, PBP and PI
    # of the only shifting part, considering PV in the ref case
    
    demand_ref_series = pd.Series(data=demand_ref,index=demand.index)
    res_pspy_pv = dispatch_max_sc(pv,demand_ref_series,param_tech,return_series=False)
    
    Epspy_pv = {}
    Epspy_pv['PVCapacity']      = pv_capacity
    Epspy_pv['BatteryCapacity'] = 0.
    Epspy_pv['ACGeneration'] = pv.to_numpy()
    Epspy_pv['Load']         = demand_ref
    Epspy_pv['ToGrid']       = res_pspy_pv['inv2grid']
    Epspy_pv['FromGrid']     = res_pspy_pv['grid2load']
    Epspy_pv['SC']           = res_pspy_pv['inv2load']
    
    res_EA_pv = EconomicAnalysisRefPV(Epspy,econ_param,ElPrices,timestep,Epspy_pv)   
    
    # Preparing function outputs
    
    out = {}

    heel   = np.where(ElPrices == prices[scenario]['heel']/1000,1.,0.)
    hollow = np.where(ElPrices == prices[scenario]['hollow']/1000,1.,0.)
    full   = np.where(ElPrices == prices[scenario]['full']/1000,1.,0.)
    peak   = np.where(ElPrices == prices[scenario]['peak']/1000,1.,0.)
   
    consrefheel   = np.sum(demand_ref*heel)*timestep
    consrefhollow = np.sum(demand_ref*hollow)*timestep
    consreffull   = np.sum(demand_ref*full)*timestep
    consrefpeak   = np.sum(demand_ref*peak)*timestep

    out['PVCapacity']      = res_EA['PVCapacity'] 
    out['BatteryCapacity'] = res_EA['BatteryCapacity'] 
    
    out['CostPV']      = res_EA['CostPV']
    out['CostBattery'] = res_EA['CostBattery']

    out['peakdem'] = np.max(demand)
    
    out['cons_total'] = np.sum(demand)*timestep

    out['cons_heel']   = np.sum(res_pspy['grid2load']*heel)*timestep
    out['cons_hollow'] = np.sum(res_pspy['grid2load']*hollow)*timestep
    out['cons_full']   = np.sum(res_pspy['grid2load']*full)*timestep
    out['cons_peak']   = np.sum(res_pspy['grid2load']*peak)*timestep
        
    out['varperc_cons_heel']   = (out['cons_heel']-consrefheel)/out['cons_total']*100
    out['varperc_cons_hollow'] = (out['cons_hollow']-consrefhollow)/out['cons_total']*100
    out['varperc_cons_full']   = (out['cons_full']-consreffull)/out['cons_total']*100
    out['varperc_cons_peak']   = (out['cons_peak']-consrefpeak)/out['cons_total']*100
    
    out['el_prod']           = np.sum(pv)*timestep
    out['el_selfcons']       = np.sum(res_pspy['inv2load'])*timestep
    out['el_soldtogrid']     = np.sum(res_pspy['inv2grid'])*timestep
    out['el_boughtfromgrid'] = np.sum(res_pspy['grid2load'])*timestep
    
    out['selfsuffrate'] = out['el_selfcons']/out['cons_total']
    out['el_shifted'] = np.abs(pflows.demand_noshift-pflows.demand_shifted).sum()/2/4
    out['losses'] = (pflows.demand_shifted.sum() - pflows.demand_noshift.sum())/4
    
    if out['el_prod'] == 0:
        out['selfconsrate'] = 0
    else:
        out['selfconsrate'] = out['el_selfcons']/out['el_prod']
    
    out['el_soldtogrid_rev']      = res_EA['RevSelling']
    out['el_boughtfromgrid_cost'] = res_EA['CostBuying']
    out['annualgridcosts']        = res_EA['AnnualGridCosts']
    out['el_netexpend']           = res_EA['ElBill']
    out['el_costperkwh']          = res_EA['costpermwh']/1000.
    
    out['PBP'] = res_EA['PBP']
    out['NPV'] = res_EA['NPV']
    out['PI']  = res_EA['PI']
    
    out['PBP_onlyshift'] = res_EA_pv['PBP']
    out['NPV_onlyshift'] = res_EA_pv['NPV']
    out['PI_onlyshift']  = res_EA_pv['PI']

    out['el_netexpend_onlyshift'] = res_EA_pv['ElBill']    
    out['el_netexpend_onlyshift_ref'] = res_EA_pv['ElBill_ref']
    
    return out


def WriteResToExcel(file,sheet,results,econ_param,tariff,row):
    
    df = pd.read_excel(file,sheet_name=sheet,header=0,index_col=0)
    
    df.at[row,'Investissement fixe pilotage [€]'] = econ_param['FixedControlCost']
    df.at[row,'Investissement annuel pilotage [€] (abonnement, … )'] = econ_param['AnnualControlCost']
    df.at[row,"Année d'étude"] = econ_param['time_horizon']
    df.at[row,"Valeur de vendue de l'élec [€/kWh]"] = econ_param['P_FtG']/1000.
    df.at[row,'Heure Talon [1h-7h]  [€/kWh]'] = tariff['heel']/1000.
    df.at[row,'Heure creuse [23h-1h et 7h-10h]  [€/kWh]'] = tariff['hollow']/1000.
    df.at[row,'Heure pleine [10h-18h et 21h-23h]  [€/kWh]'] = tariff['full']/1000.
    df.at[row,'Heure pointe [18h-21h]  [€/kWh]'] = tariff['peak']/1000.
    df.at[row,'Fixe [€/an]'] = econ_param['C_grid_fixed']
    df.at[row,'Capacitaire [€/kW]'] = econ_param['C_grid_kW']
    
    df.at[row,'PV [kWp]'] = results['PVCapacity']
    df.at[row,'Battery [kWh]'] = results['BatteryCapacity']
    
    df.at[row,'Investissement PV [€]'] = results['CostPV'] 
    df.at[row,'Investissement Battery [€]'] = results['CostBattery']
    
    df.at[row,"Capacité d'accès kW"]  = results['peakdem']
    
    df.at[row,'Total  [kWh]']  = results['cons_total']

    df.at[row,'Electricté Produite [kWh] ']  = results['el_prod']
    df.at[row,'Electricté autoconsommée [kWh] ']  = results['el_selfcons']
    df.at[row,'Electricté injectée [kWh] '] = results['el_soldtogrid']
    df.at[row,'Electricté achetée [kWh] '] = results['el_boughtfromgrid']
    
    df.at[row,'Heure Talon [1h-7h] [kWh]'] = results['cons_heel']
    df.at[row,'Heure creuse [23h-1h et 7h-10h] [kWh]']  = results['cons_hollow']
    df.at[row,'Heure pleine [10h-18h et 21h-23h]  [kWh]']  = results['cons_full']
    df.at[row,'Heure pointe [18h-21h]  [kWh]']  =  results['cons_peak']
    
    df.at[row,'Variation Heure Talon [1h-7h]'] = results['varperc_cons_heel']
    df.at[row,'Variation Heure creuse [23h-1h et 7h-10h]'] = results['varperc_cons_hollow']
    df.at[row,'Variation Heure pleine [10h-18h et 21h-23h]'] =  results['varperc_cons_full']
    df.at[row,'Variation Heure pointe [18h-21h]'] = results['varperc_cons_peak']
    
    df.at[row,'Taux autosuff [%]']  = results['selfsuffrate']
    df.at[row,'Taux autocons [%]']  = results['selfconsrate']
    
    df.at[row,'Electricitédéplacée [kWh]'] = results['el_shifted']
    
    df.at[row,'Electricté vendue [€] '] = results['el_soldtogrid_rev']
    df.at[row,'Electricté achetée [€] '] = results['el_boughtfromgrid_cost']
    df.at[row,'Electricté couts du réseau [€]'] = results['annualgridcosts']

    df.at[row,'Elec dépenses [€] = Vend-Ach-CoutsRes'] = results['el_netexpend']
    df.at[row,"Coût de l'électricité [€/kWh]"] = results['el_costperkwh']
        
    df.at[row,'PBP [years]'] = results['PBP']
    df.at[row,'NPV [€]']     = results['NPV']
    df.at[row,'PI [-]']      = results['PI'] 
    
    df.at[row,'PBP_onlyshift [years]'] = results['PBP_onlyshift']
    df.at[row,'NPV_onlyshift [€]']     = results['NPV_onlyshift']
    df.at[row,'PI_onlyshift [-]']      = results['PI_onlyshift'] 
    
    df.to_excel(file,sheet_name=sheet)














