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
from ..RC_BuildingSimulator import Zone


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

    # Define arrays storing the scenarios
    # Total receptacle loads and lights
    elec = np.zeros((n_scen, nminutes))
    # Static demand and loadshifting appliances
    pstatic = np.zeros((n_scen, nminutes))
    ptd     = np.zeros((n_scen, nminutes))
    pdw     = np.zeros((n_scen, nminutes))
    pwm     = np.zeros((n_scen, nminutes))
    # Domestic hot water - Water consumption
    mDHW  = np.zeros((n_scen, nminutes))
    # Internal heat gains from receptacle load and lights
    Qrad = np.zeros((n_scen, nminutes))
    Qcon = np.zeros((n_scen, nminutes))
    # Occupancy
    occupancy = []   # occupance has a time resolution of 10 min!
    # Space heating - Heat demand and electricity consumption
    Qspace = np.zeros((n_scen, nminutes))
    Wdot_hp = np.zeros((n_scen, nminutes))
    # Domestic hot water - Electricity consumption
    Qeb= np.zeros((n_scen, nminutes))

    textoutput = []
    
    members = []
    
    for i in range(n_scen):
        
        
        """ 
        Receptacle loads, lights, occupancy, internal heat gains from apps and lights, water consumption 
        Models used:
        - Strobe
        """
        
        print("Generating scenario {}".format(i))
        family.simulate(year, ndays)

        # Total receptacle loads and lights
        elec[i, :] = family.P
        # Static part of the demand
        # WM,TD and DW added later if not considered for load shifting
        pstatic[i, :] = family.Pst
        # Washing machine
        if inputs['appliances']['loadshift'] and 'WashingMachine' in inputs['appliances']['apps']:
            ptd[i, :] = family.Ptd
        else:
            pstatic[i, :] = pstatic[i, :] + family.Ptd        
        # Tumble dryer
        if inputs['appliances']['loadshift'] and 'TumbleDryer' in inputs['appliances']['apps']:
            pwm[i, :] = family.Pwm
        else:
            pstatic[i, :] = pstatic[i, :] + family.Pwm
        # Dish washer
        if inputs['appliances']['loadshift'] and 'DishWasher' in inputs['appliances']['apps']:
            pdw[i, :] = family.Pdw
        else:
            pstatic[i, :] = pstatic[i, :] + family.Pdw            
        # Domestic hot water consumption
        mDHW[i, :]  = family.mDHW
        # Internal heat gains from appliances (radiative and convective)
        Qrad[i,:] = family.QRad
        Qcon[i,:] = family.QCon
        # Occupancy
        occupancy.append(family.occ)
        
        textoutput += ['']
        textoutput += ["Generating scenario {}".format(i)]
        textoutput += family.textoutput
        
        #members.append(family.members)
        members.append([f for f in family.members if f!='U12'])        # TODO check if the U12 (children under 12) should really be removed from the occupancy profile in residential.py
        
        # Annual load from appliances
        E_app = int(np.sum(family.P)/60/1000)
        print(' - Receptacle load (including lighting) is %s kWh' % str(E_app))
        textoutput.append(' - Receptacle (plugs + lighting) load is %s kWh' % str(E_app))
        
        # Annual load from washing machine
        E_wm = int(np.sum(family.Pwm)/60/1000)
        print(' - Load from washing machine is %s kWh' % str(E_wm))
        textoutput.append(' - Load from washing machine is %s kWh' % str(E_wm))
        
        # Annual load from dryer
        E_td = int(np.sum(family.Ptd)/60/1000)
        print(' - Load from tumble dryer is %s kWh' % str(E_td))
        textoutput.append(' - Load from tumble dryer is %s kWh' % str(E_td))
        
        # Annual load from dish washer
        E_dw = int(np.sum(family.Pdw)/60/1000)
        print(' - Load from dish washer is %s kWh' % str(E_dw))
        textoutput.append(' - Load from dish washer is %s kWh' % str(E_dw))
        
        
        """
        Electricity consumption for domestic hot water using electric boiler or HP
        Models used:
        - Simple electric boiler model or simple HP model (both with hot water tank)
        """
        
        # Electric boiler with hot water tank
        if inputs['DHW']['loadshift']:
            Qeb[i,:] = DomesticHotWater(inputs,family.mDHW,Tamb,family.sh_day)
            E_eb = int(sum(Qeb[i,:])/1000./60.)
            print(' - Domestic hot water electricity consumption: ',E_eb,' kWh')
            textoutput.append(' - Domestic hot water electricity consumption: ' + str(E_eb) +' kWh')
        else:
            Qeb[i,:] = 0
            E_eb = 0
            print(' - Domestic hot water electricity consumption: ',E_eb,' kWh')
            textoutput.append(' - Domestic hot water electricity consumption: ' + str(E_eb) +' kWh')
        
        """
        House thermal demand and heat pump electricity consumption
        Models used:
        - CREST or 5R1C
        - Simple HP model
        """
        if inputs['HP']['loadshift']:
            if inputs['HP']['model'] == 'CREST':
                
                # Thermostat timer setting
                # 1) According to CREST
                # timersetting = HeatingTimer(inputs)
                # 2) No timer
                # ressize = 527041 # should be defined once and for all not here
                # timersetting = np.ones(ressize)
                # 3) According to Aerts
                timersetting = AertsThermostatTimer(ndays)
                
                # Thermostat temperature setting according to Aerts
                # Look inside HouseThermalModel() if actually used, around line 310
                Tthermostat  = AertsThermostatTemp(occupancy)
    
                Qspace[i,:],Temitter = HouseThermalModel(inputs,nminutes,Tamb,irr,family.QRad+family.QCon,timersetting,Tthermostat)
                thermal_load = int(sum(Qspace[i,:])/1000./60.)
                print(' - Thermal demand for space heating is ',thermal_load,' kWh')
                textoutput.append(' - Thermal demand for space heating is '+ str(thermal_load) + ' kWh')
                
            elif inputs['HP']['model'] == '5R1C':
            
                # R51C model
                Qspace[i,:],QheatHP = HouseThermalModel5R1C(inputs,nminutes,Tamb,irr,family.QRad+family.QCon,occupancy[0])
                thermal_load = int(sum(Qspace[i,:])/1000./60.)
                QheatHP = int(QheatHP)
                print(' - Heat pump size is ',QheatHP,' kW (heat)')
                textoutput.append(' - Heat pump size is ' + str(QheatHP)+ ' kW (heat)')
                print(' - Thermal demand for space heating is ',thermal_load,' kWh')
                textoutput.append(' - Thermal demand for space heating is '+ str(thermal_load) + ' kWh')
            
            else:
                Qspace[i,:] = 0
                thermal_load = 0
                print('WARNING: wrong house thermal model option - Thermal demand forced to be null')
                print(' - Thermal demand for space heating is ',thermal_load,' kWh')
                textoutput.append(' - Thermal demand for space heating is '+ str(thermal_load) + ' kWh')            
        else:
            Qspace[i,:] = 0
            thermal_load = 0
            print(' - Thermal demand for space heating is ',thermal_load,' kWh')
            textoutput.append(' - Thermal demand for space heating is '+ str(thermal_load) + ' kWh')
        
        # Heat pump electric load
        if inputs['HP']['loadshift']:
            Wdot_hp[i,:] = ElLoadHP(Tamb,Qspace[i,:])
            E_hp = int(sum(Wdot_hp[i,:])/1000./60.)
            print(' - Heat pump consumption: ',E_hp,' kWh')
            textoutput.append(' - Heat pump consumption: ' + str(E_hp) + ' kWh')
        else:
            Wdot_hp[i,:] = 0
            E_hp = 0
            print(' - Heat pump consumption: ',E_hp,' kWh')
            textoutput.append(' - Heat pump consumption: ' + str(E_hp) + ' kWh')
        

        """
        Total electricity demand
        """
        E_total = E_app + E_wm + E_td + E_dw + E_hp + E_eb
        print(' - Total annual load: ',E_total,' kWh')
        textoutput.append(' - Total annual load: ' + str(E_total) + ' kWh')   
        

    result={
        'ElectricalLoad':elec,
        'StaticLoad':pstatic,
        'TumbleDryer':ptd, 
        'DishWasher':pdw, 
        'WashingMachine':pwm, 
        'DomesticHotWater':Qeb,
        'SpaceHeating':Qspace,
        'HeatPumpPower':Wdot_hp,
        'InternalGains':Qrad+Qcon,
        'mDHW':mDHW, 
        'occupancy':occupancy,
        'members':members}

    return result,textoutput



def DomesticHotWater(inputs,mDHW,Tamb,Tbath):
    
    """
    Domestic hot water heater
    Can be both 
    
    In:
    inputs    dictionary with input data from JSON
    mDHW      tap water required l/min [every min]
    Tamb      ambient temperature [every min]
    Tbath     temperature in the room where the boiler is stored [every 10 min]
    
    Out: 
    electrical power consumption [every min]
    
    """
    
    tstep   = 60.  # s

    PowerElMax = inputs['DHW']['PowerElMax']   # W  
    Ttarget    = inputs['DHW']['Ttarget'] #°C
    Tcw        = inputs['DHW']['Tcw']     #°C
    Vcyl       = inputs['DHW']['Vcyl']    # l
    Hloss      = inputs['DHW']['Hloss']  # W/K
    
    phi_t = np.zeros(np.size(mDHW))
    phi_a = np.zeros(np.size(mDHW))
    
    # Inizialization
    Tcyl = 60. + random.random()*2. #°C
    resV = mDHW / 1000. / 60. # from l/min to m3/s
    resM = resV * 1000.       # from m3/s to kg/s
    resH = resM * 4200.       # from kg/s to W/K, cp = 4200. J/kg/K
    Ccyl = Vcyl * 1000. /1000. * 4200. # J/K

    if inputs['DHW']['type'] == 'ElectricBoiler':
        
        for i in range(np.size(mDHW)):
            
            j = int(i/10)
            
            eff = 1.
            
            phi_t[i] = Ccyl/tstep * (Ttarget-Tcyl) + resH[i] * (Tcyl-Tcw) + Hloss * (Tcyl-Tbath[j])
            phi_a[i] = phi_t[i]/eff
            phi_a[i] = max([0.,min([PowerElMax,phi_a[i]])])
            deltaTcyl = (tstep/Ccyl) * (Hloss*Tbath[j] - (Hloss+resH[i])*Tcyl + resH[i]*Tcw + phi_a[i])
            Tcyl += deltaTcyl
            
    elif inputs['DHW']['type'] == 'HeatPump':
        
        for i in range(np.size(mDHW)):
            
            j = int(i/10)
            
            phi_t[i] = Ccyl/tstep * (Ttarget-Tcyl) + resH[i] * (Tcyl-Tcw) + Hloss * (Tcyl-Tbath[j])
            phi_a[i] = phi_t[i]/COP_Tamb(Tamb[i])
            phi_a[i] = max([0.,min([PowerElMax,phi_a[i]])])
            deltaTcyl = (tstep/Ccyl) * (Hloss*Tbath[j] - (Hloss+resH[i])*Tcyl + resH[i]*Tcw + phi_a[i])
            Tcyl += deltaTcyl
      
    return phi_a


def HouseThermalModel(inputs,ressize,To,Go, phi_c,timersetting,Tthermostat):
    
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
    
    Tthermostatsetpoint = inputs['HP']['Tthermostatsetpoint'] #°C to be updated with data from Strobe or distrib probability
    ThermostatDeadband  = inputs['HP']['ThermostatDeadband']  #°C
    Temittersetpoint    = inputs['HP']['Temittersetpoint']    #°C
    EmitterDeadband     = inputs['HP']['EmitterDeadband']     #°C

    typeofdwelling = inputs['HP']['dwelling_type']

    # Dwelling parameters
        
    Hob = dwellings[typeofdwelling]['Hob']   # W/K Thermal transfer coefficient between outside air and external building thermal capacitance
    Hbi = dwellings[typeofdwelling]['Hbi']   # W/K Thermal transfer coefficient between external building thermal capacitance and internal building thermal capacitance
    Cb  = dwellings[typeofdwelling]['Cb']    # J/K External building thermal capacitance (Building thermal mass)
    Ci  = dwellings[typeofdwelling]['Ci']    # J/K Internal building thermal capacitance (Indoor air thermal mass)
    A_s = dwellings[typeofdwelling]['A_s']   # m2  Global irradiance multiplier       
    # Nvent = 1.0     # 1/h  Ventilation rate
    # A_liv = 136.    # m2
    # L_liv = 4.2     # m
    Hv  = dwellings[typeofdwelling]['Hv']    # W/K Thermal transfer coefficient representing ventilation heat loss between outside air and internal building thermal capacitance

    Tin_design = 20. # sizing temperatures
    Tout_design = -2. # sizing temperatures
    phi_design = (1/(1/Hob + 1/Hbi)+Hv)*(Tin_design-Tout_design)
    Hem = phi_design/(Temittersetpoint-Tin_design)
    mem = 14.*phi_design/1000.
    Cem = mem*4200.
       
    # Temperatures inizialization

    Ti = min([max([19.,To[0]]),25.]) + random.random()*2. #°C
    Tem = Ti  #°C
    Tem_target = Temittersetpoint + EmitterDeadband #°C
    Tb = max(16.,To[0])  + random.random()*2. #°C

    # Space heating vector inizialization

    phi_h_space = np.zeros(ressize)
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
            SpaceThermostatState == False and Ti < (Tthermostatsetpoint - ThermostatDeadband): # Tthermostat[i]
               
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
        phi_hp[i] = phi_h_space[i]/COP_Tamb(temp[i])
    return phi_hp


def COP_Tamb(Temp):
    COP = 0.001*Temp**2 + 0.0471*Temp + 2.1259
    return COP


def AertsThermostatTemp(occupancy):
    
    """
    Function that returns thermostat temperature setting based on occupancy.
    20°C if someone is at home
    15°C otherwise
    
    occupancy : numpy array, shape (n_scen, tenminutes)
    DHW demands scenarios, sampled at a
    10 minute time-step
    
    Tthermostat: numpy array, shape (nminutes)
    Thermostat temperature setting, one minute timestep

    """
    
    # Length of 1-min array based on 10-min array 
    ntenmin = len(occupancy[0][0]) 
    nmin = (ntenmin-1)*10+1 # strobe uses N+1 sized vectors
    # Initializing thermostat Ts array
    Tthermostat = np.zeros(nmin)
    # Initializing resulting occupancy 
    occupancy_tot = np.zeros(ntenmin)
    
    # occupancy_tot[i] = 1 if at least 1 person at home, otherwise 0
    for i in range(len(occupancy[0])):
        occupancy_tot = np.multiply(occupancy_tot,np.where(occupancy[0][i] >= 3,0,1))
    
    # Thermostat = 20. if someone at home, otherwise 15.
    for i in range(len(occupancy_tot)-1):
        for j in range(10):
            if occupancy_tot[i] > 0:
                Tthermostat[i*10+j]=20.
            else:
                Tthermostat[i*10+j]=15.
    
    # Last element would otherwise be 0.
    Tthermostat[nmin-1] = Tthermostat[nmin-2] 
    
    return Tthermostat

    
def AertsThermostatTimer(ndays):
    
    timer_day = np.zeros(1440)
    
    for i in range(len(timer_day)):
        if (360 <= i <= 1380):
            timer_day[i] = 1
    timer_year = np.tile(timer_day,ndays)
    timer_year = np.append(timer_year,timer_year[len(timer_year)-1])
    
    return timer_year



def HouseThermalModel5R1C(inputs,nminutes,Tamb,irr,Qintgains,occupancys):

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
                     ach_vent=inputs['HP']['ACH_vent']/60,
                     ach_infl=inputs['HP']['ACH_infl']/60,
                     ventilation_efficiency=inputs['HP']['VentEff'],
                     thermal_capacitance=inputs['HP']['Ctot'],
                     t_set_heating=21.,
                     max_heating_power=float('inf'))
        Tair = 21.
        House.solve_energy(0.,0.,-10.,Tair)
        QheatHP = House.heating_demand*0.8
        # Ttemp = House.t_m_next this should be the T that would be reached with no heating
        # Tair = House.t_air # T actually reached in the house (in this case should be = to initial Tair)
        
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
                    t_set_heating=inputs['HP']['Tthermostatsetpoint'],
                    max_heating_power=QheatHP)
        
    else:
        # Heat pump size given as an input
        # directly defining the house to be modelled
        QheatHP = inputs['HP']['HeatPumpThermalPower']
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
                    t_set_heating=inputs['HP']['Tthermostatsetpoint'],
                    max_heating_power=QheatHP)  
            
    n10min = int(nminutes/10.)
    n1min  = nminutes
    
    occ = np.zeros(n10min)
    for i in range(len(occupancys)):
        singlehouseholdocc = [1 if a==1 else 0 for a in occupancys[i][:-1]]
        occ += singlehouseholdocc
    occ = [1 if a >=1 else 0 for a in occ]    
    occupancy = np.zeros(n1min)
    for i in range(n10min):
        for j in range(10):
            occupancy[i*10+j] = occ[i]

    Tset = [20. if a == 1 else 15. for a in occupancy] # °C
    Tset = np.array(Tset)
    Tair = max(16.,Tamb[0])  + random.random()*2. #°C
    Qheat = np.zeros(nminutes)
    
    Tinside = np.zeros(nminutes)

    for i in range(nminutes):
        
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
            t_set_heating=Tset[i],
            max_heating_power=QheatHP)
        
        House.solve_energy(Qintgains[i], Qsolgains[i], Tamb[i], Tair)
        Tair      = House.t_air
        Qheat[i] = House.heating_demand
        
        Tinside[i] = Tair
        
        # Heating season
        if 60*24*151 < i < 60*24*244:
            Qheat[i] = 0
    
    Twhenon    = Tinside*occupancy # °C
    Twhenon_hs = Twhenon[np.r_[0:60*24*151,60*24*244:-1]] # °C
    whenon     = np.nonzero(Twhenon_hs)
    Twhenon_hs_mean = np.mean(Twhenon_hs[whenon]) # °C
    Twhenon_hs_min  = np.min(Twhenon_hs[whenon])  # °C
    Twhenon_hs_max  = np.max(Twhenon_hs[whenon])  # °C
    
    print('Average T: {:.2f}°C'.format(Twhenon_hs_mean))
    
    return Qheat, QheatHP
        

