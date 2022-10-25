
import random
import copy

import numpy as np
import pandas as pd
from statistics import mean
from itertools import chain

from strobe.Data.Households import households
from RC_BuildingSimulator import Zone

from economics import CashFlows,FinancialMetrics
import defaults


import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

from joblib import Memory
memory = Memory(__location__ + '/cache/', verbose=defaults.verbose)


def read_sheet(file,sheet):
    '''
    function that reads one sheet of the excel config file and outputs it in a dataframe
    '''
    raw = pd.read_excel(file,sheet_name=sheet)
    raw.rename(columns={ raw.columns[0]: "varname" }, inplace = True)
    raw = raw.loc[raw['varname'].notna(),:]
    raw.index = raw['varname']
    return raw[['Variable','Valeur','Description']]


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


@memory.cache
def load_climate_data(datapath = __location__ + '/strobe/Data'):
    '''
    Function that loads the climate data from strobe
    '''
            # ambient data
    temp = np.loadtxt(datapath + '/Climate/temperature.txt')
    irr  = np.loadtxt(datapath + '/Climate/irradiance.txt')   
    return temp,irr

def ProcebarExtractor(buildtype,wellinsulated):
    
    """
    Given the building type, input data required by the 5R1C model 
    are obtained based on a simple elaboration of Procebar data.
    Thermal and geometric characteristics are randomly picked from 
    Procebar data according to Procebar's probability distribution
    of the given building type to have such characteristics
    
    input:
    buildtype   str defining building type (according to Procebar types('Freestanding','Semi-detached','Terraced','Apartment'))
    wellinsulated   bool if true only well insulated houses considered (according to column fitforHP in the excel file)
    
    output:
    output      dict with params needed by the 5R1C model
    """

    
    # Opening building stock excel file
    # Selecting type of building wanted
    # Getting random (weighted) house thermal parameters
    # Getting corresponding reference geometry
                
    filename1 = __location__ + '/inputs/Building_Stock_arboresence_SG_130118_EME.xls'
    sheets1 = ['Freestanding','Semi-detached','Terraced','Apartment']

    buildtype = defaults.convert_building[buildtype]
    data1 = pd.read_excel (filename1,sheet_name=sheets1,header=0)
        
    df = data1[buildtype]
    df.columns = df.columns.str.rstrip()
    
    df["Occurence"].replace({np.nan: 0, -1: 0}, inplace=True)
    df['fitforHP'].replace({np.nan: 0, -1: 0}, inplace=True)

    if wellinsulated:
        df["Occurence"]=df["Occurence"]*df['fitforHP']
    totprob = df["Occurence"].sum()
    df["Occurence"] = df["Occurence"]/totprob
    
    rndrow = df["Occurence"].sample(1,weights=df["Occurence"])
    rowind = rndrow.index.values[0]
    rowgeom = df.iloc[rowind]['Geometry reference']
    
    # Opening geometry excel file
    # Getting geometry parameters based on reference geometry just obtained
    
    filename2 = __location__ + '/inputs/Arborescence_geometry_SG_130118.xls'
    sheets2 = [101,102,103,104,201,202,203,204,301,302,303,304,401,402,403,404]
    sheets2 = [str(i) for i in sheets2]
    data2 = pd.read_excel (filename2,sheet_name=sheets2)
    
    df2 = data2[str(rowgeom)]
    
    df3 = df2.iloc[0:7].iloc[:,0:2]
    df3 = df3.set_index(df3.iloc[:,0])
    df3 = df3.drop('General characteristics',axis=1)
    
    df4 = df2.iloc[9:16].iloc[:,0:3]
    df4.columns = df4.iloc[0]
    df4 = df4.drop(9)
    df4 = df4.reset_index(drop=True)
    df4 = df4.set_index(df4.iloc[:,0])
    df4 = df4.drop(np.nan,axis=1)
    
    df5 = df2.iloc[18:26].iloc[:,0:7]
    df5.columns = df5.iloc[0]
    df5 = df5.drop(18)
    df5 = df5.reset_index(drop=True)
    df5 = df5.set_index(df5.iloc[:,0])
    df5 = df5.drop(np.nan,axis=1)
    
    # Obtaining the parameters needed by the RC model
    
    heatedareas1 = ['Life','Night','Kitchen','Bathroom']
    heatedareas2 = ['Alife','Anight','Akitchen','Abathroom']
    
    Awindows  = df5[heatedareas1].loc['Awind'].sum()  # m2
    Awalls    = df5[heatedareas1].loc['Awall'].sum()  # m2
    Aroof     = df5[heatedareas1].loc['Aroof'].sum()  # m2
    Afloor    = df5[heatedareas1].loc['Afloor'].sum() # m2
    Ainternal = df5[heatedareas1].loc['Aint'].sum()   # m2
    
    volume = df4['Volume [m3]'].loc[heatedareas2].sum() # m3
    
    Atotal = max((Awindows + Awalls + Aroof + Afloor + Ainternal),Afloor*4.5) # m2  Afloor*4.5 from ISO13790 under eq. 9
    
    Uwalls = df.iloc[rowind]['U_Wall']     # W/(m2K)
    Uwindows = df.iloc[rowind]['U_Window'] # W/(m2K)
    Uroof = df.iloc[rowind]['U_Roof']      # W/(m2K)
    Ufloor = df.iloc[rowind]['U_Floor']    # W/(m2K)
    
    Ctot = df.iloc[rowind]['C_Roof']   *Aroof  + \
           df.iloc[rowind]['C_Wall']   *Awalls + \
           df.iloc[rowind]['C_Floor']  *Afloor + \
           df.iloc[rowind]['C_Window'] *Awindows # J/K
           
    ACH_vent = 0.5 # Air changes per hour through ventilation [Air Changes Per Hour]
    ACH_infl = 0.0 # Air changes per hour through infiltration [Air Changes Per Hour]
    VentEff = 0. # The efficiency of the heat recovery system for ventilation. Set to 0 if there is no heat recovery []
    
    # U average for choosing HP type
    Uavg = (Uwalls*Awalls + Uwindows*Awindows + Uroof*Aroof + Ufloor*Afloor) / (Awalls + Awindows + Aroof + Afloor)

    
    outputs = {
        'Aglazed': Awindows,
        'Aopaque': Awalls,
        'Afloor': Afloor,
        'volume': volume,
        'Atotal': Atotal,
        'Uwalls': Uwalls,
        'Uwindows': Uwindows,
        'ACH_vent': ACH_vent,
        'ACH_infl': ACH_infl,
        'VentEff': VentEff,
        'Ctot': Ctot,
        'Uavg': Uavg
        }
    
    return outputs


def HouseholdMembers(conf):
    
    """
    Function to pick household members composition from Strobe's list
    If input is None, composition randomly picked
    If input is int, composition randomly picked from list with given size
    If input is list, same list is given as output
    
    input:
    members  can be None, int or list
    
    output:
    out      list of dwelling members

    """
    config = conf['dwelling']
    
    N = 0           # number of occupants
    N_random = 0    # number of occupants to be randomly determined
    defined = []
    adults = ['FTE','PTE','Retired','Unemployed']
    
    for key in ['member1','member2','member3','member4','member5']:
        if config[key] is not None:
            N +=1
            if config[key] == 'Random':
                N_random += 1
            else:
                defined.append(config[key])
    if N == 0:              # if no occupants are defined, we randomize their number and type by picking from the strobe list
        print('No occupant defined. Selecting a random number of occupants')
        finished = False
        while not finished: 
            subset = {key: value for key, value in households.items()}
            out = random.choice(list(subset.values()))
            finished = not set(out).isdisjoint(adults)
    elif N == N_random:      # All occupants are random but their number is defined. Picking from the list (restricted to this number)
        finished = False
        while not finished: 
            subset = {key: value for key, value in households.items() if np.size(value) == N}
            out = random.choice(list(subset.values()))
            finished = not set(out).isdisjoint(adults)        
    elif N_random == 0:      # all occupants are defined
        out = defined
    elif N_random < N:       # some occupants are defined, others not. Randomly picking the remaining ones
        out = defined
        for i in range(N_random):
            out.append(random.choice(adults))
    
    return out


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

@memory.cache
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



def HPSizing(config_envelope,fracmaxP,T_in=21,T_amb=-10):
    '''
    Function that size the HP as a fraction of the maximum heat demand

    Parameters
    ----------
    config_envelope : dict
        Dictionnary with the envelope parameters.
    fracmaxP : float
        Fraction of the maximum thermal load to be covered by the heat pumps.

    Returns
    -------
    QheatHP : TYPE
        Heat pump nominal thermal power.

    '''
    # Walls
    walls_area=config_envelope['Aopaque']
    u_walls=config_envelope['Uwalls']
    # Windows
    window_area=config_envelope['Aglazed']
    u_windows=config_envelope['Uwindows']
    # Total UA given by walls and windows
    UA = u_walls*walls_area + u_windows*window_area
    # Air changes
    room_vol=config_envelope['volume']
    ach_infl=config_envelope['ACH_infl']
    ach_vent=config_envelope['ACH_vent']
    ach_tot = ach_infl + ach_vent
    ventilation_efficiency=config_envelope['VentEff']
    b_ek = (1 - (ach_vent / (ach_tot)) * ventilation_efficiency)
    # Static heat demand in sizing conditions
    QheatHP = UA*(T_in-T_amb) + 1200*b_ek*room_vol*(ach_tot/3600)*(T_in-T_amb)
    # Fraction of heat demand to be considered for sizing
    QheatHP = QheatHP*fracmaxP
        
    return QheatHP



def COP_Tamb(temp):
    
    """
    DEPRECATED
    Generic COP as a function of ambient temperature
    from:
    missing
    """
    
    COP = 0.001*temp**2 + 0.0471*temp + 2.1259
    
    return COP

def COP_deltaT(temp):
    
    """
    COP for air-water residential heat pumps
    as a function of ambient and water temperatures
    from:
    Staffell, Iain, et al. "A review of domestic heat pumps." Energy & Environmental Science 5.11 (2012): 9291-9306.
    
    ok for 15°C < deltaT < 60°C
    direct air heating: 25–35 °C
    underfloor heating: 30–45 °C
    large-area radiators: 45–60 °C
    conventional radiators: 60–75 °C
    """
    Tw = 45.
    deltaT = Tw - temp
    COP = 6.81 - 0.121*deltaT + 0.000630*deltaT**2
    
    return COP


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

@memory.cache
def HouseHeating(config_dwelling,QheatHP,Tset,Qintgains,Tamb,irr,nsteps,heatseas_st,heatseas_end,ts):


    Qsolgains = irr * config_dwelling['A_s']
        
    # Defining the house to be modelled with obtained HP size
    House = Zone(window_area=config_dwelling['Aglazed'],
                walls_area=config_dwelling['Aopaque'],
                floor_area=config_dwelling['Afloor'],
                room_vol=config_dwelling['volume'],
                total_internal_area=config_dwelling['Atotal'],
                u_walls=config_dwelling['Uwalls'],
                u_windows=config_dwelling['Uwindows'],
                ach_vent=config_dwelling['ACH_vent'],
                ach_infl=config_dwelling['ACH_infl'],
                ventilation_efficiency=config_dwelling['VentEff'],
                thermal_capacitance=config_dwelling['Ctot'],
                t_set_heating=Tset[0],
                max_heating_power=QheatHP,
                ts=ts)
            
    Qheat = np.zeros(nsteps)
    Tinside = np.zeros(nsteps)
    Tm = np.zeros(nsteps)

    d1 = int(1/ts)*24*heatseas_end-1
    d2 = int(1/ts)*24*heatseas_st-1
    concatenated = chain(range(1,d1), range(d2,nsteps))

    Tm[0] = 15.
    House.t_set_heating = Tset[0]    
    House.solve_energy(Qintgains[0], Qsolgains[0], Tamb[0], Tm[0])
    Qheat[0]   = House.heating_demand
    Tinside[0] = House.t_air

    for i in concatenated:
        
        if i == d2:
            Tm[i-1] = 15.

        if Tset[i] != Tset[i-1]:
            House.t_set_heating = Tset[i]    
            
        House.solve_energy(Qintgains[i], Qsolgains[i], Tamb[i], Tm[i-1])
        Qheat[i]   = House.heating_demand
        Tinside[i] = House.t_air
        Tm[i] = House.t_m
                       
    
    out = {'Qheat': Qheat,'Tinside':Tinside,'Tm':Tm}
    
    return out


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
        yprices_1min (numpy array): vector Nsteps long with energy prices, €
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
    timestep = param['timestep']
    
    Nsteps = len(yprices_1min)
    yprices_1min_np = yprices_1min

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
        for k, v in out.items():  # Create dictionary of pandas series
            index1min = pd.date_range(start='2015-01-01',end='2015-12-31 23:59:00',freq='T')
            out_pd[k] = pd.Series(v, index=index1min)
        out = out_pd
    return out


def ResultsAnalysis(conf,prices,pflows):
    '''
    Function that analyses the results from the load shifting library 
    The results are input through the pflows dataframe
    the function returns some important energy and economic indicators
    '''
    
    ts = conf['sim']['ts']
    
    demand_tot = pflows['demand_noshift'].sum() * ts
    SC = np.minimum(pflows['pv'], pflows['demand_shifted'])
    
    CF= CashFlows(conf,prices,pflows['fromgrid'],pflows['togrid'])
    
    import copy
    # case 0: without PV, battery or load shifting
    conf_0 = copy.deepcopy(conf)
    conf_0['pv']['yesno'] = False
    conf_0['pv']['ppeak'] = 0
    conf_0['batt']['yesno'] = False
    conf_0['batt']['capacity'] = 0
    conf_0['cont']['wetapp'] = 'none'
    conf_0['ev']['loadshift'] = False
    conf_0['hp']['loadshift'] = False
    conf_0['dhw']['loadshift'] = False
    CF_0 = CashFlows(conf_0,prices,pflows['demand_noshift'],pd.Series(0,index=pflows['demand_noshift'].index))
    
    # case1: with PV, without battery and without load shifting
    conf_1 = copy.deepcopy(conf)
    if not conf_1['pv']['yesno']:
        CF_1 = CF_0
    else:
        conf_1['batt']['yesno'] = False
        conf_1['batt']['capacity'] = 0
        conf_1['cont']['wetapp'] = 'none'
        conf_1['ev']['loadshift'] = False
        conf_1['hp']['loadshift'] = False
        conf_1['dhw']['loadshift'] = False
        CF_1 = CashFlows(conf_1,prices,np.maximum(0,pflows['demand_noshift']-pflows['pv']),np.maximum(0,pflows['pv'] - pflows['demand_noshift']))
        
    # case2: with PV, with load shifting, without battery
    conf_2 = copy.deepcopy(conf)
    conf_2['batt']['yesno'] = False
    conf_2['batt']['capacity'] = 0   
    CF_2 = CashFlows(conf_2,prices,np.maximum(0,pflows['demand_shifted_nobatt']-pflows['pv']),np.maximum(0,pflows['pv'] - pflows['demand_shifted_nobatt']))
    
    # case3: with PV, with battery, without load shifting
    conf_3 = copy.deepcopy(conf)
    conf_3['cont']['wetapp'] = 'none'
    conf_3['ev']['loadshift'] = False
    conf_3['hp']['loadshift'] = False
    conf_3['dhw']['loadshift'] = False
    CF_3 = CashFlows(conf_3,prices,np.maximum(0,pflows['demand_noshift']-pflows['pv'] - pflows['BatteryGeneration']),np.maximum(0,pflows['pv'] - pflows['demand_noshift'] - pflows['BatteryConsumption']))
    
    econ = pd.DataFrame()
    # profitability of adding a PV installation to the household:
    out = FinancialMetrics(conf['econ']['wacc'],CF_1.CashFlows - CF_0.CashFlows)
    econ['PV installation'] = pd.Series(out)
    
    # profitability of adding load shifting to the household with PV:
    out = FinancialMetrics(conf['econ']['wacc'],CF_2.CashFlows - CF_1.CashFlows)
    econ['Adding load-shifting to existing PV'] = pd.Series(out)
    
    # profitability of adding a battery to the household with PV:
    out = FinancialMetrics(conf['econ']['wacc'],CF_3.CashFlows - CF_1.CashFlows)
    econ['Adding a battery to existing PV'] = pd.Series(out)
    
    # profitability of the overall system compared to a household without PV:
    out = FinancialMetrics(conf['econ']['wacc'],CF.CashFlows - CF_0.CashFlows)
    econ['Full system (PV+batt+control)'] = pd.Series(out)        
    
    results = pd.DataFrame(columns=['Description','Valeur'])

    annuity_factor = conf['econ']['wacc'] / ( 1 - (1+conf['econ']['wacc'])**(-conf['econ']['time_horizon']))
    NPV = FinancialMetrics(conf['econ']['wacc'],CF['CashFlows'])['NPV']
    results.loc['LCOE','Valeur'] = -NPV/(demand_tot/annuity_factor) * 1000           # to be checked
    results.loc['LCOE','Description'] = "Coût actualisé de l'électricité (LCOE, en €/MWh)"
    results.loc['peakdem','Valeur'] = np.max(pflows['demand_noshift'] )       
    results.loc['peakdem','Description'] = "Demande maximale (kW)"    
    results.loc['annual_load','Valeur'] = demand_tot
    results.loc['annual_load','Description'] = "Demande annuelle (kWh)"        
    results.loc['PVCapacity','Valeur'] = conf['pv']['ppeak']
    results.loc['PVCapacity','Description'] = "Puissance crète du système PV (kWc)"     
    results.loc['el_prod','Valeur'] = pflows.pv.sum()*ts
    results.loc['el_prod','Description'] = "Electricité produite (kWh)"  
    results.loc['el_selfcons','Valeur'] = SC.sum() * ts
    results.loc['el_selfcons','Description'] = "Electricité autoconsommée (kWh)"  
    results.loc['el_soldtogrid','Valeur'] = pflows['togrid'].sum() * ts
    results.loc['el_soldtogrid','Description'] = "Electricité reinjectée sur le réseau (kWh)"  
    results.loc['el_boughtfromgrid','Valeur'] = pflows['fromgrid'].sum() * ts
    results.loc['el_boughtfromgrid','Description'] = "Electricité achetée au réseau (kWh)"  
    results.loc['selfsuffrate','Valeur'] = results.loc['el_selfcons','Valeur']/demand_tot
    results.loc['selfsuffrate','Description'] = "SSR"  
    if results.loc['el_prod','Valeur'] == 0:
        results.loc['selfconsrate','Valeur'] = 0
    else:
        results.loc['selfconsrate','Valeur'] = results.loc['el_selfcons','Valeur']/results.loc['el_prod','Valeur']
    results.loc['selfconsrate','Description'] = "SCR"  
    results.loc['el_shifted','Valeur'] = np.abs(pflows.demand_noshift-pflows.demand_shifted).sum()/2 * ts
    results.loc['el_shifted','Description'] = "Quantité de charge déplacée (kWh)"  
    results.loc['losses','Valeur'] = (pflows.demand_shifted.sum() - pflows.demand_noshift.sum()) * ts
    results.loc['losses','Description'] = "Pertes énergétiques liées au déplacement de charge (kWh)"  
    
    results.loc['inv_PV','Valeur'] = -CF.loc[0,'Inv_PV'] - CF.loc[1,'Inv_Invert']
    results.loc['inv_PV','Description'] = "Coût d'investissement - Système PV (€)"      
    results.loc['inv_batt','Valeur'] = -CF.loc[0,'Inv_Batt'] 
    results.loc['inv_batt','Description'] = "Coût d'investissement - Batterie (€)"      
    results.loc['inv_cont','Valeur'] = -CF.loc[0,'Inv_Control'] 
    results.loc['inv_cont','Description'] = "Coût d'investissement - Système de pilotage (€)"       
    
    results.loc['el_bill','Valeur'] = -CF.loc[1,'CostBfG'] - CF.loc[1,'IncomeStG'] - CF.loc[1,'CostStG']
    results.loc['el_bill','Description'] = "Facture annuelle nette d'électricité (€)"         
    results.loc['el_stg','Valeur'] = CF.loc[1,'IncomeStG'] 
    results.loc['el_stg','Description'] = "Bénéfices annuels de la revente au réseau (€)"       
    
    results.loc['pbp_pv','Valeur'] = econ.loc['PBP','PV installation']
    results.loc['pbp_pv','Description'] = "Temps de retour du système PV seul (années)"        
    results.loc['pbp_batt','Valeur'] = econ.loc['PBP','Adding a battery to existing PV']
    results.loc['pbp_batt','Description'] = "Temps de retour de l'ajout d'une batterie au PV (années)"   
    results.loc['pbp_cont','Valeur'] = econ.loc['PBP','Adding load-shifting to existing PV']
    results.loc['pbp_cont','Description'] = "Temps de retour de l'ajout du pilotage/déplacement de charge au PV (années)"  
    results.loc['pbp_all','Valeur'] = econ.loc['PBP','Full system (PV+batt+control)']
    results.loc['pbp_all','Description'] = "Temps de retour de système complet (avec PV, batterie et pilotage si définis) (années)"      
    
    return results, econ

    
def scale_vector(vec_in,N,silent=False):
    ''' 
    Function that scales a numpy vector or Pandas Series to the desired length
    
    :param vec_in: Input vector
    :param N: Length of the output vector
    :param silent: Set to True to avoid verbosity
    '''    
    N_in = len(vec_in)
    if type(N) != int:
        N = int(N) 
        if not silent:
            print('Converting Argument N to int: ' + str(N))
    if N > N_in:
        if np.mod(N,N_in)==0:
            if not silent:
                print('Target size is a multiple of input vector size. Repeating values')
            vec_out = np.repeat(vec_in,N/N_in)
        else:
            if not silent:
                print('Target size is larger but not a multiple of input vector size. Interpolating')
            vec_out = np.interp(np.linspace(start=0,stop=N_in,num=N),range(N_in),vec_in)
    elif N == N_in:
        print('Target size is iqual to input vector size. Not doing anything')
        vec_out = vec_in
    else:
        if np.mod(N_in,N)==0:
            if not silent:
                print('Target size is entire divisor of the input vector size. Averaging')
            vec_out = np.zeros(N)
            for i in range(N):
                vec_out[i] = np.mean(vec_in[i*N_in/N:(i+1)*N_in/N])
        else:
            if not silent:
                print('Target size is lower but not a divisor of the input vector size. Interpolating')
            vec_out = np.interp(np.linspace(start=0,stop=N_in,num=N),range(N_in),vec_in)
    return vec_out  


def MostRepCurve(conf,prices,demands,columns,timestep):
    
    """
    Choosing most representative curve among a list of demand curves
    based on electricity bill buying all electricity from grid
    hence wiithout PV or batteries
    """
    
    # Input parameters required by economic analysis
    # PV and battery forced to be 0
    
    conf2 = copy.deepcopy(conf)   # we need to make a deep copy of the nested disctionnary, or it will change the values. A shallow copy is not sufficient
    
    conf2['pv']['ppeak'] = 0.
    conf2['batt']['capacity'] = 0.
    conf2['inverter_pmax'] = 0.
    conf2['sim']['ts'] = 0.25
    
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
        
        pflows = pd.DataFrame(index=pv.index)
        pflows['pv'] = np.zeros(len(date)) 
        pflows['demand_noshift']         = demand.to_numpy()
        pflows['togrid']       = np.zeros(len(date)) 
        pflows['fromgrid']     = demand.to_numpy() 
        pflows['demand_shifted']  = pflows['demand_noshift']
        pflows['demand_shifted_nobatt'] = pflows['demand_noshift']
        pflows['BatteryGeneration'] = 0
        pflows['BatteryConsumption'] = 0
        
        out,econ = ResultsAnalysis(conf,prices,pflows)
        results.append(out['Valeur']['el_bill'])
    
    meanelbill = mean(results)
    var = results-meanelbill
    index = min(range(len(var)), key=var.__getitem__)
        
    return index


