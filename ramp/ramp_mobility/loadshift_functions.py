
import numpy as np
import pandas as pd
import random
from ramp.ramp_mobility.core_model.stochastic_process_mobility import Stochastic_Process_Mobility
from ramp.ramp_mobility.core_model.charging_process import Charging_Process
from ramp.ramp_mobility.post_process import post_process as pp

import pathlib, os
ramppath = pathlib.Path(__file__).parent.parent.resolve()
datapath = os.path.join(ramppath,'database')

def MainDriver(inputs,members):
    plausibleMDs = ['FTE','PTE','Retired','Unemployed'] # list of plausible main drivers
    plausibleMDsinDwelling = [value for value in members if value in plausibleMDs] # list of plausible main drivers among house members
    MD = random.choice(plausibleMDsinDwelling) # main driver
    return MD


def EVCharging(inputs,members,occupancy):
    
    MD = MainDriver(inputs,members)
    MDathome = occupancy[members.index(MD)]
    
    ndays = inputs['ndays']
    nminutes = ndays * 1440 + 1 
    MDathome_min = np.zeros(nminutes)
    
    for i in range(len(MDathome)-1):
        for j in range(10):
            if MDathome[i] in (1,2):
                MDathome_min[i*10+j]=1
            else:
                MDathome_min[i*10+j]=0
    
    # Inputs definition
    
    full_year = True  # Choose if simulating the whole year (True) or not (False), if False, the console will ask how many days should be simulated.
    
    countries = ['BE']
    
    for c in countries:
        # Define folder where results are saved, it will be:
        # "results/inputfile/simulation_name" leave simulation_name False (or "")
        # to avoid the creation of the additional folder
        inputfile = f'Europe/{c}'
        simulation_name = ''
        
        # Define country and year to be considered when generating profiles
        country = f'{c}'
        year = inputs['year']
        
        # Define attributes for the charging profiles
        charging_mode = 'Uncontrolled' # Select charging mode (Uncontrolled', 'Night Charge', 'RES Integration', 'Perfect Foresight')
        logistic = False # Select the use of a logistic curve to model the probability of charging based on the SOC of the car
        infr_prob = 0.8 # Probability of finding the infrastructure when parking ('piecewise', number between 0 and 1)
        Ch_stations = ([3.7, 11, 120], [1.0, 0., 0.]) # [0.6, 0.3, 0.1] Define nominal power of charging stations and their probability 
        
        #inputfile for the temperature data: 
        inputfile_temp = datapath + "/temp_ninja_pop_1980-2019.csv"
        
        ## If simulating the RES Integration charging strategy, a file with the residual load curve should be included in the folder
        try:
            inputfile_residual_load = datapath + "/residual_load/residual_load_{c}.csv"
            residual_load = pd.read_csv(inputfile_residual_load, index_col = 0)
        except FileNotFoundError:      
            residual_load = pd.DataFrame(0, index=range(1), columns=range(1))
    
        # Call the functions for the simulation
        
        # Simulate the mobility profile 
        (Profiles_list, Usage_list, User_list, Profiles_user_list, dummy_days
         ) = Stochastic_Process_Mobility(inputfile, country, year, full_year)
        
        # Post-processes the results and generates plots
        Profiles_avg, Profiles_list_kW, Profiles_series = pp.Profile_formatting(
            Profiles_list)
        Usage_avg, Usage_series = pp.Usage_formatting(Usage_list)
        Profiles_user = pp.Profiles_user_formatting(Profiles_user_list)
        
        # Create a dataframe with the profile
        Profiles_df = pp.Profile_dataframe(Profiles_series, year) 
        Usage_df = pp.Usage_dataframe(Usage_series, year)
        
        # Time zone correction for profiles and usage
        Profiles_utc = pp.Time_correction(Profiles_df, country, year) 
        Usage_utc = pp.Time_correction(Usage_df, country, year)    
            
        # Add temperature correction to the Power Profiles 
        # To be done after the UTC correction because the source data for Temperatures have time in UTC
        temp_profile = pp.temp_import(country, year, inputfile_temp) #Import temperature profiles, change the default path to the custom one
        Profiles_temp = pp.Profile_temp(Profiles_utc, year = year, temp_profile = temp_profile)
        
        # Resampling the UTC Profiles
        Profiles_temp_h = pp.Resample(Profiles_temp)
        
        # ?    
        Profiles_user_temp = pp.Profile_temp_users(Profiles_user, temp_profile,
                                                    year, dummy_days)
     
        # Charging process function: if no problem is detected, only the cumulative charging profile is calculated. Otherwise, also the user specific quantities are included. 
        (Charging_profile, Ch_profile_user, SOC_user) = Charging_Process(
            Profiles_user_temp, User_list, country, year,dummy_days, 
            residual_load, charging_mode, logistic, infr_prob, Ch_stations)
    
        MDathome_min = np.delete(MDathome_min,-1)
        Charging_profile_home = np.multiply(Charging_profile,MDathome_min)         
    
        Charging_profile_df = pp.Ch_Profile_df(Charging_profile, year)
        Charging_profile_home_df = pp.Ch_Profile_df(Charging_profile_home, year)
                
        # Postprocess of charging profiles 
        Charging_profiles_utc = pp.Time_correction(Charging_profile_df, 
                                                    country, year)
        Charging_profile_home_df= Charging_profile_home_df.rename(columns={'Charging Profile': 'EVCharging'})

    return Charging_profile_home_df