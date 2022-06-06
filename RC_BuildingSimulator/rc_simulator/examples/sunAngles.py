import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Set root folder one level up, just for this example
mainPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, mainPath)

from radiation import Location

matplotlib.style.use('ggplot')

Zurich = Location(epwfile_path=os.path.join(
    mainPath, 'auxiliary', 'Zurich-Kloten_2013.epw'))


Zurich.calc_sun_position(latitude_deg=47.480, longitude_deg=8.536, year=2015, hoy=3708)


Azimuth = []
Altitude = []
Sunnyhoy = []

for hoy in range(8760):
    sun = Zurich.calc_sun_position(
        latitude_deg=47.480, longitude_deg=8.536, year=2015, hoy=hoy)
    Altitude.append(sun[0])
    Azimuth.append(sun[1])
    Sunnyhoy.append(hoy + 1)

sunPosition = pd.read_csv(os.path.join(
    mainPath, 'auxiliary', 'SunPosition.csv'), skiprows=1)

transSunPos = sunPosition.transpose()
hoy_check = transSunPos.index.tolist()
hoy_check = [float(ii) for ii in hoy_check]
Azimuth_check = (180 - transSunPos[1]).tolist()

Altitude_check = transSunPos[0].tolist()

plt.style.use('ggplot')

plt.plot(Sunnyhoy, Azimuth, hoy_check, Azimuth_check,
         Sunnyhoy, Altitude, hoy_check, Altitude_check)
plt.legend(['Azimuth', 'Azimuth Check', 'Altitude', 'Altitude_check'])

plt.show()
