# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 11:46:20 2021
@author: pietro
"""


import os
import numpy as np
import random
from .residential import Household

import pathlib
strobepath = pathlib.Path(__file__).parent.parent.resolve()
datapath = os.path.join(strobepath,'Data')



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
        Total electricity demand
        """
        E_total = E_app + E_wm + E_td + E_dw + E_eb
        print(' - Total annual load: ',E_total,' kWh')
        textoutput.append(' - Total annual load: ' + str(E_total) + ' kWh')   
        

    result={
        'ElectricalLoad':elec,
        'StaticLoad':pstatic,
        'TumbleDryer':ptd, 
        'DishWasher':pdw, 
        'WashingMachine':pwm, 
        'DomesticHotWater':Qeb,
        'InternalGains':Qrad+Qcon,
        'mDHW':mDHW, 
        'occupancy':occupancy,
        'members':members}

    return result,textoutput


def ambientdata():
    temp = np.loadtxt(datapath + '/Climate/temperature.txt')
    temp=np.append(temp,temp[-24*60:]) # add december 31 to end of year in case of leap year
    irr = np.loadtxt(datapath + '/Climate/irradiance.txt')
    irr=np.append(irr,irr[-24*60:]) # add december 31 to end of year in case of leap year
    return temp,irr

def ElLoadHP(temp,phi_h_space):
    phi_hp= np.zeros(np.size(phi_h_space))
    for i in range(np.size(phi_h_space)):   
        phi_hp[i] = phi_h_space[i]/COP_Tamb(temp[i])
    return phi_hp


def COP_Tamb(Temp):
    COP = 0.001*Temp**2 + 0.0471*Temp + 2.1259
    return COP


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
    Tfaucet    = inputs['DHW']['Tfaucet'] #°C
    Tcw        = inputs['DHW']['Tcw']     #°C
    Vcyl       = inputs['DHW']['Vcyl']    # l
    Hloss      = inputs['DHW']['Hloss']  # W/K
    
    phi_t = np.zeros(np.size(mDHW))
    phi_a = np.zeros(np.size(mDHW))
    
    # Inizialization
    Tcyl = 60. + random.random()*2. #°C
    m_hot = mDHW * (Tfaucet-Tcw)/(Ttarget-Tcw) # DHW is used at Tfaucet which is obtained mixing water from boiler and acqueduct
    resV = m_hot / 1000. / 60. # from l/min to m3/s
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

