# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 11:46:20 2021
@author: pietro
"""


import os
import numpy as np
import datetime
import random
from .residential import Household

import pathlib
strobepath = pathlib.Path(__file__).parent.parent.resolve()
datapath = os.path.join(strobepath,'Data')

from strobe.Data.Dwellings import dwellings

def convert_occupancy(occ):
    
    """Convert occupancy as number of inhabitants currently in house."""
    
    for i in range(len(occ)):
        arr = occ[i]
        arr[arr < 3] = 1 # active (1) or sleeping (2) are both present =1
        arr[arr == 3] = 0 # absent (3) =0
    return sum(occ)


def simulate_scenarios(n_scen,inputs):
    
    """Simulate scenarios of demands during ndays.
    Parameters
    ----------
    n_scen : int
        Number of scenarios to generate
    year : int
        Year to consider
        
    ndays : int
        Number of days to consider
    members : str
        Member list of household
    Returns
    -------
    elec : numpy array, shape (n_scen, nminutes)
        Electrical demands scenarios, sampled at
        minute time-step
    mDHW : numpy array, shape (n_scen, nminutes)
        DHW demands scenarios, sampled at
        minute time-step
    occupancy : numpy array, shape (n_scen, tenminutes)
        DHW demands scenarios, sampled at a
        10 minute time-step
    """
    year = inputs['year']
    ndays = inputs['ndays']

    nminutes = ndays * 1440 + 1
    ntenm = ndays * 144 + 1
    
    Tamb, irr = ambientdata()
    
    family = Household(**inputs)

    # define arrays storing the scenarios:
    elec = np.zeros((n_scen, nminutes))
    
    pstatic = np.zeros((n_scen, nminutes))
    ptd     = np.zeros((n_scen, nminutes))
    pdw     = np.zeros((n_scen, nminutes))
    pwm     = np.zeros((n_scen, nminutes))
    
    mDHW  = np.zeros((n_scen, nminutes))
    
    Qrad = np.zeros((n_scen, nminutes))
    Qcon = np.zeros((n_scen, nminutes))

    occupancy = np.zeros((n_scen, ntenm))   # occupance has a time resolution of 10 min!
    
    Qspace = np.zeros((n_scen, nminutes))
    Wdot_hp = np.zeros((n_scen, nminutes))
    Qeb= np.zeros((n_scen, nminutes))

    textoutput = []
    for i in range(n_scen):
        print("Generating scenario {}".format(i))
        family.simulate(year, ndays)

        # aggregate scenarios:
        elec[i, :] = family.P
        
        pstatic[i, :] = family.Pst
        ptd[i, :]     = family.Ptd
        pdw[i, :]     = family.Pdw
        pwm[i, :]     = family.Pwm
        
        mDHW[i, :]  = family.mDHW
        
        Qrad[i,:] = family.QRad
        Qcon[i,:] = family.QCon
        
        occupancy[i, :] = convert_occupancy(family.occ)
        
        textoutput += ['']
        textoutput += ["Generating scenario {}".format(i)]
        textoutput += family.textoutput
        
        # House heating model
        timersetting = HeatingTimer(inputs)
        Qspace[i,:],Temitter = HouseThermalModel(inputs,nminutes,Tamb,irr,family.QRad+family.QCon,timersetting)
        thermal_load = int(sum(Qspace[i,:])/1000./60.)
        print(' - Thermal demand for space heating is ',thermal_load,' kWh')
        textoutput.append(' - Thermal demand for space heating is '+ str(thermal_load) + ' kWh')
        
        # Annual load from appliances
        E_app = int(np.sum(family.P)/60/1000)
        print(' - Receptacle load (including lighting) is %s kWh' % str(E_app))
        textoutput.append(' - Receptacle (plugs + lighting) load is %s kWh' % str(E_app))
        
        # Annual load from dryer
        E_td = int(np.sum(family.Ptd)/60/1000)
        print(' - Load from tumble dryer is %s kWh' % str(E_td))
        textoutput.append(' - Load from tumble dryer is %s kWh' % str(E_td))
        
        # Annual load from washing machine
        E_wm = int(np.sum(family.Pwm)/60/1000)
        print(' - Load from washing machine is %s kWh' % str(E_wm))
        textoutput.append(' - Load from washing machine is %s kWh' % str(E_wm))
        
        # Annual load from dish washer
        E_dw = int(np.sum(family.Pdw)/60/1000)
        print(' - Load from dish washer is %s kWh' % str(E_dw))
        textoutput.append(' - Load from dish washer is %s kWh' % str(E_dw))
        
        # Heat pump electric load
        if inputs['HeatPump']:
            Wdot_hp[i,:] = ElLoadHP(Tamb,Qspace[i,:])
            E_hp = int(sum(Wdot_hp[i,:])/1000./60.)
            print(' - Heat pump consumption: ',E_hp,' kWh')
            textoutput.append(' - Heat pump consumption: ' + str(E_hp) + ' kWh')
        else:
            Wdot_hp[i,:] = 0
            E_hp = 0
        
        # Electric boiler and hot water tank
        if inputs['ElectricBoiler']:
            Qeb[i,:] = HotWaterTankModel(inputs,family.mDHW,family.sh_day)
            E_eb = int(sum(Qeb[i,:])/1000./60.)
            print(' - Electrical boiler consumption: ',E_eb,' kWh')
            textoutput.append(' - Electrical boiler consumption: ' + str(E_eb) +' kWh')
        else:
            Qeb[i,:] = 0
            E_eb = 0
        
        E_total = E_app + E_td + E_wm + E_dw + E_hp + E_eb
        print(' - Total annual load: ',E_total,' kWh')
        textoutput.append(' - Total annual load: ' + str(E_total) + ' kWh')   
        

    result={
        'ElectricalLoad':elec,
        'StaticLoad':pstatic,
        'TumbleDryer':ptd, 
        'DishWasher':pdw, 
        'WashingMachine':pwm, 
        'ElectricalBoiler':Qeb,
        'SpaceHeating':Qspace,
        'HeatPumpPower':Wdot_hp,
        'InternalGains':Qrad+Qcon,
        'mDHW':mDHW, 
        'occupancy':occupancy}

    return result,textoutput



def HotWaterTankModel(inputs,mDHW,Tbath):
    
    """
    Electric boiler and water tank
    
    In:
    inputs    dictionary with input data from JSON
    mDHW      tap water required l/min [every min]
    Tbath     temperature in the room where the boiler is stored [every 10 min]
    
    Out: 
    power consumption [every min]
    
    hp. eff boiler = 100%
    """
    
    tstep   = 60.  # s
    
    Ttarget = inputs['Ttarget'] #°C
    Tcw     = inputs['Tcw']     #°C
    Vcyl    = inputs['Vcyl']    # l
    Hloss   = inputs['Hloss']  # W/K
    
    phi_t = np.zeros(np.size(mDHW))
    phi_a = np.zeros(np.size(mDHW))
    
    # Inizialization
    Tcyl = 60. + random.random()*2. #°C

    resV = mDHW / 1000. / 60. # from l/min to m3/s
    resM = resV * 1000.       # from m3/s to kg/s
    resH = resM * 4200.       # from kg/s to W/K, cp = 4200. J/kg/K
      
    Ccyl = Vcyl * 1000. /1000. * 4200. # J/K
    
    for i in range(np.size(mDHW)):
        
        j = int(i/10)
        
        phi_t[i] = Ccyl/tstep * (Ttarget-Tcyl) + resH[i] * (Tcyl-Tcw) + Hloss * (Tcyl-Tbath[j])
        phi_a[i] = max([0.,min([2000.,phi_t[i]])])
        deltaTcyl = (tstep/Ccyl) * (Hloss*Tbath[j] - (Hloss+resH[i])*Tcyl + resH[i]*Tcw + phi_a[i])
        Tcyl += deltaTcyl        
      
    return phi_a


def HouseThermalModel(inputs,ressize,To,Go, phi_c,timersetting):
    
    """
    inputs            dictionary with input data from JSON
    ressize           size of the reuslts vector
    To        °C      array of outdoor temperature
    Go        Wm-2    array of outdoor global radiation (horizontal)
    phi_c     W       casual thermal gains Qrad+Qconv, missing component from occupants
    day_of_week       array sized 365 or 366 with days of the week (0-6)
    typeofdwelling    string with the name of the type of dwelling
    """

    tstep = 60.             # s
    
    Tthermostatsetpoint = inputs['Tthermostatsetpoint'] #°C to be updated with data from Strobe or distrib probability
    ThermostatDeadband  = inputs['ThermostatDeadband']  #°C
    Tem_target          = inputs['Tem_target']          #°C
    Temittersetpoint    = inputs['Temittersetpoint']    #°C
    EmitterDeadband     = inputs['EmitterDeadband']     #°C

    typeofdwelling = inputs['dwelling_type']

    # Dwelling parameters
    
    A_s = dwellings[typeofdwelling]['A_s']    # m2 Global irradiance multiplier       
    Hv  = dwellings[typeofdwelling]['Hv']    # W/K Thermal transfer coefficient representing ventilation heat loss between outside air and internal building thermal capacitance
    Hob = dwellings[typeofdwelling]['Hob']   # W/K Thermal transfer coefficient between outside air and external building thermal capacitance
    Hbi = dwellings[typeofdwelling]['Hbi']   # W/K Thermal transfer coefficient between external building thermal capacitance and internal building thermal capacitance
    Hem = dwellings[typeofdwelling]['Hem']   # W/K Heat transfer coefficient of heat emitters
    Cb  = dwellings[typeofdwelling]['Cb']    # J/K External building thermal capacitance (Building thermal mass)
    Ci  = dwellings[typeofdwelling]['Ci']    # J/K Internal building thermal capacitance (Indoor air thermal mass)
    Cem = dwellings[typeofdwelling]['Cem']   # J/K Thermal capacitance of heat emitters (Heat emitter and cooler thermal masses)

       
    # Temperatures inizialization

    Ti = min([max([19.,To[0]]),25.]) + random.random()*2. #°C
    Tem = Ti  #°C
    Tem_target += EmitterDeadband #°C
    Tb = max(16.,To[0])  + random.random()*2. #°C

    # Space heating vector inizialization

    phi_h_space        = np.zeros(ressize)
    Tem_test =  np.zeros(ressize)
    
    
    # Space thermostat inizialization
    if Ti > Tthermostatsetpoint:
        SpaceThermostatState = False
    else:
        SpaceThermostatState = True

    # Emitter thermostat inizialization
    if Tem > Tem_target:
        EmitterThermostatState = False
    else:
        EmitterThermostatState = True           
    
    

    for i in range(ressize):
        
        Tem_test[i] = Tem
        
        if SpaceThermostatState == True  and Ti < (Tthermostatsetpoint + ThermostatDeadband) or \
            SpaceThermostatState == False and Ti < (Tthermostatsetpoint - ThermostatDeadband):
               
            SpaceThermostatState = True
        else:
            SpaceThermostatState = False


        if EmitterThermostatState == True  and Tem < (Temittersetpoint + EmitterDeadband) or \
            EmitterThermostatState == False and Tem < (Temittersetpoint - EmitterDeadband):
               
            EmitterThermostatState = True
        else:
            EmitterThermostatState = False
            
        SpaceTimerState = timersetting[i]
                           
        SpaceHeatingOnOff = SpaceThermostatState * SpaceTimerState * EmitterThermostatState

        if SpaceHeatingOnOff:
            phi_h_space_target = Cem/tstep * (Tem_target - Tem) + Hem * (Tem - Ti)
            phi_h_space[i] = max([0.,min([inputs['HeatPumpThermalPower'],phi_h_space_target])])
    
        
        phi_s = A_s * Go[i]
        
        deltaTb = tstep/Cb * (- (Hob+Hbi)*Tb + Hbi*Ti + Hob*To[i])
        deltaTi = tstep/Ci * (Hbi*Tb - (Hv+Hbi+Hem)*Ti + Hv*To[i] + Hem*Tem + phi_s + phi_c[i])     
        deltaTem = tstep/Cem * (Hem*Ti - Hem*Tem + phi_h_space[i])
        
        Tb  += deltaTb
        Ti  += deltaTi
        Tem += deltaTem

    
    return phi_h_space,Tem_test


        
def ambientdata():
    temp = np.loadtxt(datapath + '/Climate/temperature.txt')
    temp=np.append(temp,temp[-24*60:]) # add december 31 to end of year in case of leap year
    irr = np.loadtxt(datapath + '/Climate/irradiance.txt')
    irr=np.append(irr,irr[-24*60:]) # add december 31 to end of year in case of leap year
    return temp,irr


def TransitionProbMatrix():
    time_transprob = np.loadtxt(datapath+'/Heating/timer.txt', dtype = float)
    return time_transprob


def HeatingTimer(inputs):

    year = inputs['year']
    nday = inputs['ndays']    

    time_transprob = TransitionProbMatrix()

    # Days of the year
    fdoy = datetime.datetime(year-1,12,31).weekday()
    fweek = list(range(7))[fdoy:]
    day_of_week = (fweek+53*list(range(7)))[:nday]


    # Weekday timer
    
    column = 0
    
    # Inizialization. For weekdays: prob heating on at 00:00 is 0.09
    
    timersetting_wd = np.zeros(48)
    rnd = random.random()
    if rnd <0.09:
        timersetting_wd[0] = True
    else:
        timersetting_wd[0] = False


    for i in range(47):
        currentstate = timersetting_wd[i]
        row = int(i*2 + currentstate)
        rnd = random.random()
        if rnd < time_transprob[row,column]:
            nextstate = 0
        else:
            nextstate = 1
        timersetting_wd[i+1] = nextstate
    
    
    # Weekend timer
    
    column = 2
    
    # Inizialization. For weekends: prob heating on at 00:00 is 0.10
    
    timersetting_we = np.zeros(48)
    rnd = random.random()
    if rnd <0.09:
        timersetting_we[0] = True
    else:
        timersetting_we[0] = False
    
    for i in range(47):
        currentstate = timersetting_we[i]
        row = int(i*2 + currentstate)
        rnd = random.random()
        if rnd < time_transprob[row,column]:
            nextstate = 0
        else:
            nextstate = 1
        timersetting_we[i+1] = nextstate
    
    # 30 min to 1 min array
    
    timersetting_wd_min = np.zeros(1440)
    timersetting_we_min = np.zeros(1440)
    
    for i in range(np.size(timersetting_wd)):
        for j in range(30):
            timersetting_wd_min[i*30+j] = timersetting_wd[i]
            timersetting_we_min[i*30+j] = timersetting_we[i]
    
    # whole year timer array
    
    ressize = 527041 # should be defined once and for all not here
    timersetting = np.zeros(ressize)
    
    for i in range(np.size(day_of_week)):
        if day_of_week[i] <= 4:
            for j in range(np.size(timersetting_wd_min)):
                timersetting[i*np.size(timersetting_wd_min)+j]=timersetting_wd_min[j]
        else:
            for k in range(np.size(timersetting_we_min)):
                timersetting[i*np.size(timersetting_we_min)+k]=timersetting_we_min[k]  
    
    return timersetting


def ElLoadHP(temp,phi_h_space):
    phi_hp= np.zeros(np.size(phi_h_space))
    for i in range(np.size(phi_h_space)):
        COP = 0.001*temp[i]**2 + 0.0471*temp[i] + 2.1259
        phi_hp[i] = phi_h_space[i]/COP
    return phi_hp



