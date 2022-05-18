"""
Exmaple code to answer question in Issue #42
Is it possible for a room to define more than one outer surface (wall)
"""
import sys
import os
import numpy as np

# Set root folder one level up, just for this example
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from building_physics import Zone  # Importing Zone Class

# Example Inputs
t_air = 10
t_m_prev = 20
internal_gains = 10  # Internal heat gains, in Watts
# Solar heat gains after transmitting through the winow [Watts]
solar_gains = 200
ill = 4400  # Illuminance after transmitting through the window [Lumens]
occupancy = 0.1  # Occupancy for the timestep [people/hour/square_meter]

"""
Note, if you don't have the same outside temperature of the varying walls,
such as one wall connecting to a basement, then this can be approximated
by modifying the u-value of that particular wall proportionately to the
percentage damping of the temperature that surface is exposed to relative
to the outer temperature
"""

# Define the areas of the outer surfaces
roof_area = 10  # m2
outer_wall_area = 40  # m2
# Define the u values of the outer surfaces
roof_u_value = 0.2
outer_wall_u_value = 0.3

walls_area = roof_area + outer_wall_area

u_walls = (outer_wall_area * outer_wall_u_value + roof_area * roof_u_value)/walls_area

print(u_walls)

# Initialise an instance of the Zone. See ZonePhysics.py to see the default values
Office = Zone(window_area=4.0,
              walls_area=walls_area,
              floor_area=35.0,
              room_vol=105,
              total_internal_area=142.0,
              lighting_load=11.7,
              lighting_control=300.0,
              lighting_utilisation_factor=0.45,
              lighting_maintenance_factor=0.9,
              u_walls=u_walls,
              u_windows=1.1,
              ach_vent=1.5,
              ach_infl=0.5,
              ventilation_efficiency=0.6,
              thermal_capacitance_per_floor_area=165000,
              t_set_heating=20.0,
              t_set_cooling=26.0,
              max_cooling_energy_per_floor_area=-np.inf,
              max_heating_energy_per_floor_area=np.inf,)

# Solve for Zone energy
Office.solve_energy(internal_gains, solar_gains, t_air, t_m_prev)

# Solve for Zone lighting
Office.solve_lighting(ill, occupancy)

print(Office.t_m)  # Printing Room Temperature of the medium

print(Office.lighting_demand)  # Print Lighting Demand
print(Office.energy_demand)  # Print heating/cooling loads

# Example of how to change the set point temperature after running a simulation
Office.theta_int_h_set = 20.0

# Solve again for the new set point temperature
Office.solve_energy(internal_gains, solar_gains, t_air, t_m_prev)

print(Office.t_m)  # Print the new internal temperature

# Print a boolean of whether there is a heating demand
print(Office.has_heating_demand)
