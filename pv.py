
import os
import numpy as np
import pandas as pd
import pvlib
import defaults

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

from joblib import Memory
memory = Memory(__location__ + '/cache/', verbose=defaults.verbose)

# Geographic location - Liège
# Weather -  TMY (2006-2016)
coordinates = (50.6,5.6,'Europe/Brussels',60,'Etc/GMT-2')
surface_tilt = 35.


def pvlib_detailed(coordinates,surface_tilt):
    
    """
    PV production as taken from pvlib example
    
    Parameters
    ----------
    coordinates : tuple
        latitude, longitude, name, altitude, timezone
    surface_tilt : float
        PV panels surface tilt [degrees]

    Returns
    -------
    ac_15min : series
        pandas 15min timeseries AC power production [kW/kWp]
    losses : float
        average inverter losses [-]
    dc_peak['p_mp']: float
        DC peak power production from paanel [W]

    """
    
    latitude, longitude, name, altitude, timezone = coordinates
    weather = pvlib.iotools.get_pvgis_tmy(latitude, longitude, map_variables=True)[0]
    weather.index.name = "utc_time"
    
    # PV modules and inverter
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
    inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    # Defining the system
    # NB here azimuth 0 = north 180 = south
    system = {'module': module, 'inverter': inverter,'surface_azimuth': 180}   
    system['surface_tilt'] = surface_tilt
    # Calculating production
    solpos = pvlib.solarposition.get_solarposition(
        time=weather.index,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        temperature=weather["temp_air"],
        pressure=pvlib.atmosphere.alt2pres(altitude),)
    
    dni_extra = pvlib.irradiance.get_extra_radiation(weather.index)
    airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])
    pressure = pvlib.atmosphere.alt2pres(altitude)
    am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
    aoi = pvlib.irradiance.aoi(
        system['surface_tilt'],
        system['surface_azimuth'],
        solpos["apparent_zenith"],
        solpos["azimuth"],)
    
    total_irradiance = pvlib.irradiance.get_total_irradiance(
        system['surface_tilt'],
        system['surface_azimuth'],
        solpos['apparent_zenith'],
        solpos['azimuth'],
        weather['dni'],
        weather['ghi'],
        weather['dhi'],
        dni_extra=dni_extra,
        model='haydavies',)
    
    cell_temperature = pvlib.temperature.sapm_cell(
        total_irradiance['poa_global'],
        weather["temp_air"],
        weather["wind_speed"],
        **temperature_model_parameters,)
    
    effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
        total_irradiance['poa_direct'],
        total_irradiance['poa_diffuse'],
        am_abs,
        aoi,
        module,)
    
    dc = pvlib.pvsystem.sapm(effective_irradiance, cell_temperature, module) # Wh = W (since we have 1h timestep)
    ac = pvlib.inverter.sandia(dc['v_mp'], dc['p_mp'], inverter) # Wh = W (since we have 1h timestep)
    
    # Estimating mean inverter efficiency
    ac1 = ac.to_numpy()
    dc1 = dc['p_mp'].to_numpy()
    nonzero = np.where(dc1)
    eff = np.divide(ac1[nonzero],dc1[nonzero])
    eff = [a if a >0. else 0. for a in eff]
    eff = np.array(eff)
    eff_m = np.average(eff,weights=ac1[nonzero])
    losses = (1.-eff_m)*100.
    
    # Peak (nominal) production
    # Effective peak starting from peak production definition (taken from PVGIS)
    irr_dir_ref = 1000. # direct irradiance
    irr_diff_ref = 0.   # diffused irradiance
    AM_ref = 1.5        # absolute air mass
    aoi_ref = 0.        # angle of incidence
    Tref = 25.          # ambient reference temperature
    eff_peak_irr = pvlib.pvsystem.sapm_effective_irradiance(irr_dir_ref,irr_diff_ref,AM_ref,aoi_ref,module)
    dc_peak = pvlib.pvsystem.sapm(eff_peak_irr,Tref,module)
    ac_peak = pvlib.inverter.sandia(dc_peak['v_mp'], dc_peak['p_mp'], inverter)
    
    # Adapting pvlib results to be used by prosumpy
    # Reference year 2015 to handle in an easier way the TMY
    # Considering the array to be composed by power values
    
    ac_np = ac.to_numpy()
    ac_np = [a if a>0. else 0. for a in ac_np]
    ac_np = np.array(ac_np)
    index60min = pd.date_range(start='2015-01-01 00:00:00',end='2015-12-31 23:00:00',freq='60T')
    index15min = pd.date_range(start='2015-01-01 00:00:00',end='2015-12-31 23:45:00',freq='15T')
    ac_60min = pd.Series(data=ac_np,index=index60min)
    ac_15min = ac_60min.reindex(ac_60min.index.union(index15min)).interpolate(method='time').reindex(index15min)    
    
    # Adimensionalized wrt peak power
    ac_15min = ac_15min/dc_peak['p_mp'] # W/Wp
    
    return ac_15min,losses,dc_peak['p_mp']

@memory.cache
def pvgis_hist(inputs,loc):
    """
    PV production taken from PVGIS data

    Parameters
    ----------
    inputs : dictionary
        'location': tuple latitude,longitude,name,altitude,timezone
        'Ppeak': float peak DC power [kWp]
        'year': int year used for output data and to get data from PVGIS if TMY = False
        'losses': float losses in cables, power inverters, dirt (sometimes snow), over the years loss of power [%]
        'tilt':  float surface tilt [deg]
        'azimuth': float azimuth angle 0 = south, 180 = north [deg]
        'TMY': bool true if data of TMY is to be used

    Returns
    -------
    pv_15min : series
        pandas 15 min timeseries of AC power production [kW/kWp]

    """


    latitude,longitude,name,altitude,timezone = loc['latitude'],loc['longitude'],loc['name'],loc['altitude'],loc['timezone']
    peakp = inputs['ppeak']
    year = defaults.year
    losses = inputs['losses']
    tilt = inputs['tilt']
    azimuth = inputs['azimut']
    tmybool = True
    
    index60min = pd.date_range(start=str(year)+'-01-01 00:00:00',end=str(year)+'-12-31 23:00:00',freq='60T')
    index15min = pd.date_range(start=str(year)+'-01-01 00:00:00',end=str(year)+'-12-31 23:45:00',freq='15T')
    
    if tmybool:
        weather = pvlib.iotools.get_pvgis_tmy(latitude, longitude, map_variables=True)[0]
        refindex = weather.index
        refindex = refindex.shift(10,'T')
    else:
        refindex = pd.date_range(start=str(year)+'-01-01 00:00:00',end=str(year)+'-12-31 23:00:00',freq='60T',tz='utc')
        refindex = refindex.shift(10,'T')
    
    # Actual production calculation (extract all available data points)
    # NB here azimuth 0 = south 180 = north
    res = pvlib.iotools.get_pvgis_hourly(latitude,longitude,surface_tilt=tilt,surface_azimuth=azimuth,pvcalculation=True,peakpower=peakp,loss=losses)
    
    # Index to select TMY relevant data points
    pv = res[0]['P']
    pv = pv[refindex]
    pv.index = index60min
    
    # Resampling at 15 min
    pv_15min = pv.reindex(pv.index.union(index15min)).interpolate(method='time').reindex(index15min)/1000./peakp
    
    return pv_15min




if __name__ == "__main__":
    
    """
    Testing differences
    """
    
    ac_15min, losses, dc_peak = pvlib_detailed(coordinates,surface_tilt)
    
    inp_test = {'location':coordinates,
                'Ppeak': dc_peak/1000.,
                'losses':losses,
                'tilt': surface_tilt,
                'azimuth':0,
                'year':2015,
                'TMY': True}
    
    pv_15min = pvgis_hist(inp_test)
    
    sum_pvlib = np.sum(ac_15min)/4
    print('Annual production with pvlib example system: {:.2f} kWh/kWp'.format(sum_pvlib))
    sum_pvgis = np.sum(pv_15min)/4
    print('Annual production with pvgis: {:.2f} kWh/kWp'.format(sum_pvgis))
    diff = (sum_pvgis-sum_pvlib)/sum_pvgis*100.
    print("Difference in total production: {:.2f}%".format(diff))
    diff_y = np.sum(pv_15min-ac_15min)/4.
    print("Sum of differences throughout whole year: {:.2f} kWh/kWp".format(diff_y))
    diff_y_abs = np.sum(abs(pv_15min-ac_15min))/4.
    print("Sum of absolute value of differences throughout whole year: {:.2f} kWh/kWp".format(diff_y_abs))
    
    """
    PV data to be saved
    """
    
    inp_save = {'location':coordinates,
                'Ppeak': 1.0,
                'losses': 14.,
                'tilt':35.,
                'azimuth':0,
                'year':2015,
                'TMY': True}

    pv_15min2 = pvgis_hist(inp_save)   
    sum_pvgis2 = np.sum(pv_15min2)/4
    print(sum_pvgis2)
    
    # path = r'./simulations'
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # filename = 'pv.pkl'
    # filename = os.path.join(path,filename)
    # pv_15min2.to_pickle(filename)










