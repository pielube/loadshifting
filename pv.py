
import os
import pandas as pd
import pvlib

"""
PV production as taken from pvlib example
Production of panels placed in Liège
TMY (2006-2016)
"""

# Geographic location
coordinates = (50.6, 5.6, 'Europe/Brussels', 60, 'Etc/GMT-2')
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
system = {'module': module, 'inverter': inverter,'surface_azimuth': 180}   
system['surface_tilt'] = latitude
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

dc = pvlib.pvsystem.sapm(effective_irradiance, cell_temperature, module)
ac = pvlib.inverter.sandia(dc['v_mp'], dc['p_mp'], inverter) # Wh = W (since we have 1h timestep)

# Peak (nominal) production
irr_dir_ref = 1000. # direct irradiance
irr_diff_ref = 0. # diffused irradiance
AM_ref = 1.5 # absolute air mass
aoi_ref = 0. # angle of incidence
Tref = 25. # ambient reference temperature
eff_peak_irr = pvlib.pvsystem.sapm_effective_irradiance(irr_dir_ref,irr_diff_ref,AM_ref,aoi_ref,module)
dc_peak = pvlib.pvsystem.sapm(eff_peak_irr,Tref,module)
ac_peak = pvlib.inverter.sandia(dc_peak['v_mp'], dc_peak['p_mp'], inverter)
# Adapting pvlib results to be used by prosumpy
# Reference year 2015 to handle in an easier way the TMY
# Considering the array to be composed by power values
ac = ac.to_frame()
date = pd.date_range(start='2015-01-01 00:00:00',end='2015-12-31 23:45:00',freq='H')
ac = ac.set_index(date)
# Resampling at 15 min
ac = ac.resample('15Min').mean()
ac.loc['2015-12-31 23:15:00'] = [None]
ac.loc['2015-12-31 23:30:00'] = [None]
ac.loc['2015-12-31 23:45:00'] = [None]
ac = ac.fillna(method='ffill') # W
# Adimensionalized wrt peak power
ac = ac/ac_peak #W/Wp
# Saving results
path = r'.\simulations'
if not os.path.exists(path):
    os.makedirs(path)
filename = 'pv.pkl'
filename = os.path.join(path,filename)
ac.to_pickle(filename)















