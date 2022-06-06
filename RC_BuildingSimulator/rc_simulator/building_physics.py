"""
Physics Required to calculate sensible space heating and space cooling loads, and space lighting loads
EN-13970

The equations presented here is this code are derived from ISO 13790 Annex C,
Methods are listed in order of apperance in the Annex

Daylighting is based on methods in The Environmental Science Handbook, S V Szokolay

HOW TO USE

::

    from buildingPhysics import Zone  #Importing Zone Class
    office = Zone()  #Set an instance of the class
    office.solve_energy(internal_gains, solar_gains, t_out, t_m_prev) #Solve for Heating
    office.solve_lighting(illumination, occupancy) #Solve for Lighting

VARIABLE DEFINITION
    internal_gains: Internal Heat Gains [W]
    solar_gains: Solar Heat Gains after transmitting through the window [W]
    t_out: Outdoor air temperature [C]
    t_m_prev: Thermal mass temperature from the previous time step
    ill: Illuminance transmitting through the window [lumen]
    occupancy: Occupancy [people]

    t_m_next: Medium temperature of the next time step [C]
    t_m: Some weird average between the previous and current time-step of the medium  [C] #TODO: Check this

    Inputs to the 5R1C model:
    c_m: Thermal Capacitance of the medium [J/K]
    h_tr_is: Heat transfer coefficient between the air and the inside surface [W/K]
    h_tr_w: Heat transfer from the outside through windows, doors [W/K]
    H_tr_ms: Heat transfer coefficient between the internal surface temperature and the medium [W/K]
    h_tr_em: Heat conductance from the outside through opaque elements [W/K]
    h_ve_adj: Ventilation heat transmission coefficient [W/K]

    phi_m_tot: see formula for the calculation, eq C.5 in standard [W]
    phi_m: Combination of internal and solar gains directly to the medium [W]
    phi_st: combination of internal and solar gains directly to the internal surface [W]
    phi_ia: combination of internal and solar gains to the air [W]
    energy_demand: Heating and Cooling of the Supply air [W]

    h_tr_1: combined heat conductance, see function for definition [W/K]
    h_tr_2: combined heat conductance, see function for definition [W/K]
    h_tr_3: combined heat conductance, see function for definition [W/K]

INPUT PARAMETER DEFINITION
    window_area: Area of the Glazed Surface in contact with the outside [m2]
    walls_area: Area of all envelope surfaces, including windows in contact with the outside
    floor_area : floor area of zone [m2]
    room_vol: volume of interior zone [m3]
    total_internal_area: area of all surfaces facing the room. A_t [m2]
    lighting_load: Lighting Load [W/m2]
    lighting_control: Lux threshold at which the lights turn on [Lx]
    u_walls: U value of opaque surfaces  [W/m2K]
    u_windows: U value of glazed surfaces [W/m2K]
    ach_vent: Air changes per hour through ventilation [Air Changes Per Hour]
    ach_infl: Air changes per hour through infiltration [Air Changes Per Hour]
    ventilation_efficiency: The efficiency of the heat recovery system for ventilation. Set to 0 if there is no heat
        recovery []
    thermal_capacitance_per_floor_area: Thermal capacitance of the room per floor area [J/m2K]
    t_set_heating : Thermal heating set point [C]
    t_set_cooling: Thermal cooling set point [C]
    max_cooling_energy_per_floor_area: Maximum cooling load. Set to -np.inf for unrestricted cooling [C]
    max_heating_energy_per_floor_area: Maximum heating load per floor area. Set to no.inf for unrestricted heating [C]
    heating_supply_system: The type of heating system. Choices are DirectHeater, ResistiveHeater, HeatPumpHeater.
        Direct heater has no changes to the heating demand load, a resistive heater takes an efficiency into account,
        HeatPumpHeatercalculates a COP based on the outdoor and system supply temperature
    cooling_supply_system: The type of cooling system. Choices are DirectCooler HeatPumpCooler.
        DirectCooler has no changes to the cooling demand load,
        HeatPumpCooler calculates a COP based on the outdoor and system supply temperature
    heating_emission_system: How the heat is distributed to the building
    cooling_emission_system: How the cooling energy is distributed to the building

"""

from . import supply_system
from . import emission_system

__authors__ = "Prageeth Jayathissa"
__copyright__ = "Copyright 2016, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Gabriel Happle, Justin Zarb, Michael Fehr"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Prageeth Jayathissa"
__email__ = "p.jayathissa@gmail.com"
__status__ = "production"


class Zone(object):
    '''Sets the parameters of the zone. '''

    def __init__(self,
                 window_area=14.0,
                 walls_area=130.0,
                 floor_area=90.0,
                 room_vol=280.0,
                 total_internal_area=105.0,
                 u_walls=3.0,
                 u_windows=2.75,
                 ach_vent=1.5,
                 ach_infl=0.5,
                 ventilation_efficiency=0.6,
                 thermal_capacitance=5775000.0,
                 t_set_heating=20.0,
                 t_set_cooling=26.0,
                 heating_supply_system=supply_system.HeatPumpWater,
                 cooling_supply_system=supply_system.HeatPumpAir,
                 heating_emission_system=emission_system.NewRadiators,
                 cooling_emission_system=emission_system.AirConditioning,
                 max_heating_power=2000.):

        # Zone Dimensions
        self.window_area = window_area  # [m2] Window Area

        # Calculated Properties
        self.floor_area = floor_area  # [m2] Floor Area
        # [m2] Effective Mass Area assuming a medium weight zone #12.3.1.2
        self.mass_area = self.floor_area * 2.5
        self.room_vol = room_vol  # [m3] Room Volume
        self.total_internal_area = total_internal_area
        # A_t or At is defined as being the area of all surfaces facing the room.
        self.A_t = self.total_internal_area

        # Single Capacitance  5 conductance Model Parameters
        # [kWh/K] Room Capacitance. Default based on ISO standard 12.3.1.2 for medium heavy zones
        self.c_m = thermal_capacitance
        # Conductance of opaque surfaces to exterior [W/K]
        self.h_tr_em = u_walls * walls_area
        # Conductance to exterior through glazed surfaces [W/K], based on
        # U-wert of 1W/m2K
        self.h_tr_w = u_windows * window_area

        # Determine the ventilation conductance
        ach_tot = ach_infl + ach_vent  # Total Air Changes Per Hour
        # temperature adjustment factor taking ventilation and infiltration
        # [ISO: E -27]
        b_ek = (1 - (ach_vent / (ach_tot)) * ventilation_efficiency)
        self.h_ve_adj = 1200 * b_ek * self.room_vol * \
            (ach_tot / 60) /60  # Conductance through ventilation [W/M]
        # transmittance from the internal air to the thermal mass of the
        # zone
        self.h_tr_ms = 9.1 * self.mass_area
        # Conductance from the conditioned air to interior zone surface
        self.h_tr_is = self.total_internal_area * 3.45

        # Thermal set points
        self.t_set_heating = t_set_heating
        self.t_set_cooling = t_set_cooling

        # Thermal Properties
        self.has_heating_demand = True  # Boolean for if heating is required
        self.has_cooling_demand = False  # Boolean for if cooling is required
        self.max_cooling_energy = -float("inf")  # max cooling load (W/m2)
        self.max_heating_energy = max_heating_power #2000. #float("inf")

        # Zone System Properties
        self.heating_supply_system = heating_supply_system
        self.cooling_supply_system = cooling_supply_system
        self.heating_emission_system = heating_emission_system
        self.cooling_emission_system = cooling_emission_system

    @property
    def h_tr_1(self):
        """
        Definition to simplify calc_phi_m_tot
        # (C.6) in [C.3 ISO 13790]
        """
        return 1.0 / (1.0 / self.h_ve_adj + 1.0 / self.h_tr_is)

    @property
    def h_tr_2(self):
        """
        Definition to simplify calc_phi_m_tot
        # (C.7) in [C.3 ISO 13790]
        """
        return self.h_tr_1 + self.h_tr_w

    @property
    def h_tr_3(self):
        """
        Definition to simplify calc_phi_m_tot
        # (C.8) in [C.3 ISO 13790]
        """
        return 1.0 / (1.0 / self.h_tr_2 + 1.0 / self.h_tr_ms)

    @property
    def t_opperative(self):
        """
        The opperative temperature is a weighted average of the air and mean radiant temperatures.
        It is not used in any further calculation at this stage
        # (C.12) in [C.3 ISO 13790]
        """
        return 0.3 * self.t_air + 0.7 * self.t_s

    def solve_energy(self, internal_gains, solar_gains, t_out, t_m_prev):
        """
        Calculates the heating and cooling consumption of a building for a set timestep

        :param internal_gains: internal heat gains from people and appliances [W]
        :type internal_gains: float
        :param solar_gains: solar heat gains [W]
        :type solar_gains: float
        :param t_out: Outdoor air temperature [C]
        :type t_out: float
        :param t_m_prev: Previous air temperature [C]
        :type t_m_prev: float

        :return: self.heating_demand, space heating demand of the building
        :return: self.heating_sys_electricity, heating electricity consumption
        :return: self.heating_sys_fossils, heating fossil fuel consumption
        :return: self.cooling_demand, space cooling demand of the building
        :return: self.cooling_sys_electricity, electricity consumption from cooling
        :return: self.cooling_sys_fossils, fossil fuel consumption from cooling
        :return: self.electricity_out, electricity produced from combined heat pump systems
        :return: self.sys_total_energy, total exergy consumed (electricity + fossils) for heating and cooling
        :return: self.heating_energy, total exergy consumed (electricity + fossils) for heating
        :return: self.cooling_energy, total exergy consumed (electricity + fossils) for cooling
        :return: self.cop, Coefficient of Performance of the heating or cooling system
        :rtype: float

        """
        # Main File

        # check demand, and change state of self.has_heating_demand, and self._has_cooling_demand
        self.has_demand(internal_gains, solar_gains, t_out, t_m_prev)

        if not self.has_heating_demand and not self.has_cooling_demand:

            # no heating or cooling demand
            # calculate temperatures of building R-C-model and exit
            # --> rc_model_function_1(...)
            self.energy_demand = 0

            self.heating_demand = 0  # Energy required by the zone
            self.cooling_demand = 0  # Energy surplus of the zone
            # Energy (in electricity) required by the supply system to provide
            # HeatingDemand
            self.heating_sys_electricity = 0
            # Energy (in fossil fuel) required by the supply system to provide
            # HeatingDemand
            self.heating_sys_fossils = 0
            # Energy (in electricity) required by the supply system to get rid
            # of CoolingDemand
            self.cooling_sys_electricity = 0
            # Energy (in fossil fuel) required by the supply system to get rid
            # of CoolingDemand
            self.cooling_sys_fossils = 0
            # Electricity produced by the supply system (e.g. CHP)
            self.electricity_out = 0
            # Set COP to nan if no heating or cooling is required
            self.cop = float('nan')

        else:

            # has heating/cooling demand

            # Calculates energy_demand used below
            self.calc_energy_demand(
                internal_gains, solar_gains, t_out, t_m_prev)

            self.calc_temperatures_crank_nicolson(
                self.energy_demand, internal_gains, solar_gains, t_out, t_m_prev)
            # calculates the actual t_m resulting from the actual heating
            # demand (energy_demand)

            # Calculate the Heating/Cooling Input Energy Required

            supply_director = supply_system.SupplyDirector()  # Initialise Heating System Manager

            if self.has_heating_demand:
                supply_director.set_builder(self.heating_supply_system(load=self.energy_demand,
                                            t_out=t_out,
                                            heating_supply_temperature=self.heating_supply_temperature,
                                            cooling_supply_temperature=self.cooling_supply_temperature,
                                            has_heating_demand=self.has_heating_demand,
                                            has_cooling_demand=self.has_cooling_demand))
                supplyOut = supply_director.calc_system()
                # All Variables explained underneath line 467
                self.heating_demand = self.energy_demand
                self.heating_sys_electricity = supplyOut.electricity_in
                self.heating_sys_fossils = supplyOut.fossils_in
                self.cooling_demand = 0
                self.cooling_sys_electricity = 0
                self.cooling_sys_fossils = 0
                self.electricity_out = supplyOut.electricity_out

            elif self.has_cooling_demand:
                supply_director.set_builder(self.cooling_supply_system(load=self.energy_demand * (-1),
                                            t_out=t_out,
                                            heating_supply_temperature=self.heating_supply_temperature,
                                            cooling_supply_temperature=self.cooling_supply_temperature,
                                            has_heating_demand=self.has_heating_demand,
                                            has_cooling_demand=self.has_cooling_demand))
                supplyOut = supply_director.calc_system()
                self.heating_demand = 0
                self.heating_sys_electricity = 0
                self.heating_sys_fossils = 0
                self.cooling_demand = self.energy_demand
                self.cooling_sys_electricity = supplyOut.electricity_in
                self.cooling_sys_fossils = supplyOut.fossils_in
                self.electricity_out = supplyOut.electricity_out

            self.cop = supplyOut.cop

        self.sys_total_energy = self.heating_sys_electricity + self.heating_sys_fossils + \
            self.cooling_sys_electricity + self.cooling_sys_fossils
        self.heating_energy = self.heating_sys_electricity + self.heating_sys_fossils
        self.cooling_energy = self.cooling_sys_electricity + self.cooling_sys_fossils

    # TODO: rename. this is expected to return a boolean. instead, it changes state??? you don't want to change state...
    # why not just return has_heating_demand and has_cooling_demand?? then call the function "check_demand"
    # has_heating_demand, has_cooling_demand = self.check_demand(...)
    def has_demand(self, internal_gains, solar_gains, t_out, t_m_prev):
        """
        Determines whether the building requires heating or cooling
        Used in: solve_energy()

        # step 1 in section C.4.2 in [C.3 ISO 13790]
        """

        # set energy demand to 0 and see if temperatures are within the comfort
        # range
        energy_demand = 0
        # Solve for the internal temperature t_Air
        self.calc_temperatures_crank_nicolson(
            energy_demand, internal_gains, solar_gains, t_out, t_m_prev)

        # If the air temperature is less or greater than the set temperature,
        # there is a heating/cooling load
        if self.t_air < self.t_set_heating:
            self.has_heating_demand = True
            self.has_cooling_demand = False
        elif self.t_air > self.t_set_cooling:
            self.has_cooling_demand = True
            self.has_heating_demand = False
        else:
            self.has_heating_demand = False
            self.has_cooling_demand = False

    def calc_temperatures_crank_nicolson(self, energy_demand, internal_gains, solar_gains, t_out, t_m_prev):
        """
        Determines node temperatures and computes derivation to determine the new node temperatures
        Used in: has_demand(), solve_energy(), calc_energy_demand()
        # section C.3 in [C.3 ISO 13790]
        """

        self.calc_heat_flow(t_out, internal_gains, solar_gains, energy_demand)

        self.calc_phi_m_tot(t_out)

        # calculates the new bulk temperature POINT from the old one
        self.calc_t_m_next(t_m_prev)

        # calculates the AVERAGE bulk temperature used for the remaining
        # calculation
        self.calc_t_m(t_m_prev)

        self.calc_t_s(t_out)

        self.calc_t_air(t_out)

        return self.t_m, self.t_air, self.t_opperative

    def calc_energy_demand(self, internal_gains, solar_gains, t_out, t_m_prev):
        """
        Calculates the energy demand of the space if heating/cooling is active
        Used in: solve_energy()
        # Step 1 - Step 4 in Section C.4.2 in [C.3 ISO 13790]
        """

        # Step 1: Check if heating or cooling is needed
        # (Not needed, but doing so for readability when comparing with the standard)
        # Set heating/cooling to 0
        energy_demand_0 = 0
        # Calculate the air temperature with no heating/cooling
        t_air_0 = self.calc_temperatures_crank_nicolson(
            energy_demand_0, internal_gains, solar_gains, t_out, t_m_prev)[1]

        # Step 2: Calculate the unrestricted heating/cooling required

        # determine if we need heating or cooling based based on the condition
        # that no heating or cooling is required
        if self.has_heating_demand:
            t_air_set = self.t_set_heating
        elif self.has_cooling_demand:
            t_air_set = self.t_set_cooling
        else:
            raise NameError(
                'heating function has been called even though no heating is required')

        # Set a heating case where the heating load is 10x the floor area (10
        # W/m2)
        energy_floorAx10 = 10 * self.floor_area

        # Calculate the air temperature obtained by having this 10 W/m2
        # setpoint
        t_air_10 = self.calc_temperatures_crank_nicolson(
            energy_floorAx10, internal_gains, solar_gains, t_out, t_m_prev)[1]

        # Determine the unrestricted heating/cooling off the building
        self.calc_energy_demand_unrestricted(
            energy_floorAx10, t_air_set, t_air_0, t_air_10)

        # Step 3: Check if available heating or cooling power is sufficient
        if self.max_cooling_energy <= self.energy_demand_unrestricted <= self.max_heating_energy:

            self.energy_demand = self.energy_demand_unrestricted
            self.t_air_ac = t_air_set  # not sure what this is used for at this stage TODO

        # Step 4: if not sufficient then set the heating/cooling setting to the
        # maximum
        # necessary heating power exceeds maximum available power
        elif self.energy_demand_unrestricted > self.max_heating_energy:

            self.energy_demand = self.max_heating_energy

        # necessary cooling power exceeds maximum available power
        elif self.energy_demand_unrestricted < self.max_cooling_energy:

            self.energy_demand = self.max_cooling_energy

        else:
            self.energy_demand = 0
            raise ValueError('unknown radiative heating/cooling system status')

        # calculate system temperatures for Step 3/Step 4
        self.calc_temperatures_crank_nicolson(
            self.energy_demand, internal_gains, solar_gains, t_out, t_m_prev)

    def calc_energy_demand_unrestricted(self, energy_floorAx10, t_air_set, t_air_0, t_air_10):
        """
        Calculates the energy demand of the system if it has no maximum output restrictions
        # (C.13) in [C.3 ISO 13790]


        Based on the Thales Intercept Theorem.
        Where we set a heating case that is 10x the floor area and determine the temperature as a result
        Assuming that the relation is linear, one can draw a right angle triangle.
        From this we can determine the heating level required to achieve the set point temperature
        This assumes a perfect HVAC control system
        """
        self.energy_demand_unrestricted = energy_floorAx10 * \
            (t_air_set - t_air_0) / (t_air_10 - t_air_0)

    def calc_heat_flow(self, t_out, internal_gains, solar_gains, energy_demand):
        """
        Calculates the heat flow from the solar gains, heating/cooling system, and internal gains into the building

        The input of the building is split into the air node, surface node, and thermal mass node based on
        on the following equations

        #C.1 - C.3 in [C.3 ISO 13790]

        Note that this equation has diverged slightly from the standard
        as the heating/cooling node can enter any node depending on the
        emission system selected

        """

        # Calculates the heat flows to various points of the building based on the breakdown in section C.2, formulas C.1-C.3
        # Heat flow to the air node
        self.phi_ia = 0.5 * internal_gains
        # Heat flow to the surface node
        self.phi_st = (1 - (self.mass_area / self.A_t) -
                       (self.h_tr_w / (9.1 * self.A_t))) * (0.5 * internal_gains + solar_gains)
        # Heatflow to the thermal mass node
        self.phi_m = (self.mass_area / self.A_t) * \
            (0.5 * internal_gains + solar_gains)

        # We call the EmissionDirector to modify these flows depending on the
        # system and the energy demand
        emDirector = emission_system.EmissionDirector()

        # Set the emission system to the type specified by the user
        if energy_demand > 0:
            emDirector.set_builder(self.heating_emission_system(
                energy_demand=energy_demand))
        else:
            emDirector.set_builder(self.cooling_emission_system(
                energy_demand=energy_demand))
        # Calculate the new flows to each node based on the heating/cooling system
        flows = emDirector.calc_flows()

        # Set modified flows to building object
        self.phi_ia += flows.phi_ia_plus
        self.phi_st += flows.phi_st_plus
        self.phi_m += flows.phi_m_plus

        # Set supply temperature to building object
        self.heating_supply_temperature = flows.heating_supply_temperature
        self.cooling_supply_temperature = flows.cooling_supply_temperature

    def calc_t_m_next(self, t_m_prev):
        """
        Primary Equation, calculates the temperature of the next time step
        # (C.4) in [C.3 ISO 13790]
        """

        self.t_m_next = ((t_m_prev * ((self.c_m / 60.0) - 0.5 * (self.h_tr_3 + self.h_tr_em))) +
                         self.phi_m_tot) / ((self.c_m / 60.0) + 0.5 * (self.h_tr_3 + self.h_tr_em))

    def calc_phi_m_tot(self, t_out):
        """
        Calculates a global heat transfer. This is a definition used to simplify equation
        calc_t_m_next so it's not so long to write out
        # (C.5) in [C.3 ISO 13790]
        # h_ve = h_ve_adj and t_supply = t_out [9.3.2 ISO 13790]
        """

        t_supply = t_out  # ASSUMPTION: Supply air comes straight from the outside air

        self.phi_m_tot = self.phi_m + self.h_tr_em * t_out + \
            self.h_tr_3 * (self.phi_st + self.h_tr_w * t_out + self.h_tr_1 *
                           ((self.phi_ia / self.h_ve_adj) + t_supply)) / self.h_tr_2

    def calc_t_m(self, t_m_prev):
        """
        Temperature used for the calculations, average between newly calculated and previous bulk temperature
        # (C.9) in [C.3 ISO 13790]
        """
        self.t_m = (self.t_m_next + t_m_prev) / 2.0

    def calc_t_s(self, t_out):
        """
        Calculate the temperature of the inside room surfaces
        # (C.10) in [C.3 ISO 13790]
        # h_ve = h_ve_adj and t_supply = t_out [9.3.2 ISO 13790]
        """

        t_supply = t_out  # ASSUMPTION: Supply air comes straight from the outside air

        self.t_s = (self.h_tr_ms * self.t_m + self.phi_st + self.h_tr_w * t_out + self.h_tr_1 *
                    (t_supply + self.phi_ia / self.h_ve_adj)) / \
                   (self.h_tr_ms + self.h_tr_w + self.h_tr_1)

    def calc_t_air(self, t_out):
        """
        Calculate the temperature of the air node
        # (C.11) in [C.3 ISO 13790]
        # h_ve = h_ve_adj and t_supply = t_out [9.3.2 ISO 13790]
        """

        t_supply = t_out

        # Calculate the temperature of the inside air
        self.t_air = (self.h_tr_is * self.t_s + self.h_ve_adj *
                      t_supply + self.phi_ia) / (self.h_tr_is + self.h_ve_adj)
