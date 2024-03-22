import copy
import pandas as pd

class ObjectBase(object):

    def __setattr__(self, attribute_name, value):

        # Set attribute value.
        super().__setattr__(attribute_name, value)

    def __repr__(self) -> str:
        """Obtain string representation."""

        # Obtain attributes.
        attributes = vars(self)

        # Obtain representation string.
        repr_string = ""
        for attribute_name in attributes:
            repr_string += f"{attribute_name} = \n{attributes[attribute_name]}\n"

        return repr_string

    def copy(self):
        """Return a copy of this object. A new object will be created with a copy of the calling object’s attributes.
        Modifications to the attributes of the copy will not be reflected in the original object.
        """

        return copy.deepcopy(self)


class HouseholdData(ObjectBase):
    # Attributes of data object:
    delta_t: int
    timesteps: pd.DatetimeIndex
    timesteps_freq: pd.offsets.Minute
    carrier_limitation: pd.DataFrame
    solar_irradiance: pd.DataFrame
    wind_speed: pd.DataFrame
    electricity_price: pd.DataFrame
    gas_price: pd.DataFrame
    heat_price: pd.DataFrame
    hydrogen_price: pd.DataFrame
    inflexible_electricity_demand: pd.DataFrame
    app_data: pd.DataFrame
    # flexible_electricity_demand: pd.DataFrame
    inflexible_heat_demand: pd.DataFrame
    inflexible_gas_demand: pd.DataFrame
    inflexible_hydrogen_demand: pd.DataFrame
    ambient_temperature: pd.DataFrame
    pv_generator: pd.DataFrame
    electric_vehicle: pd.DataFrame
    energy_storage_system: pd.DataFrame
    time_shiftable_load: pd.DataFrame
    thermostatically_electric_appliance: pd.DataFrame
    chp: pd.DataFrame
    heat_pump: pd.DataFrame
    house_heating_system: pd.DataFrame
    gas_boiler: pd.DataFrame
    electric_boiler: pd.DataFrame
    solar_thermal: pd.DataFrame
    thermal_storage_system: pd.DataFrame
    wind_turbine: pd.DataFrame
    electrolyzer: pd.DataFrame
    hydrogen_storage_system: pd.DataFrame
    fuel_cell: pd.DataFrame
    outdoor_temperature: pd.Series
    room_temperature_sp: int
    domestic_hot_water_temperature_sp: int

    def __init__(self, data_path: dict):
        # Obtain electric grid data.
        time_series_data_csv = pd.read_csv(data_path['time_series'], index_col=[0], skipinitialspace=True)
        time_series_data_csv.index = pd.to_datetime(time_series_data_csv.index)
        time_series_data_csv.index.freq = '15T'
        appliance_data = pd.read_csv(data_path['appliance_data'], index_col=[0, 1], skipinitialspace=True)
        self.app_data = appliance_data
        # heat_pump_data = pd.read_csv(data_path['heat_pump_data'], index_col=[0], skipinitialspace=True)
        # heat_pump_data.index = pd.to_datetime(heat_pump_data.index).tz_localize(None)
        # heat_pump_data = heat_pump_data.resample('15T').asfreq()
        # heat_pump_data.drop(heat_pump_data.index[-1], inplace=True)
        # time_series_data_csv.index = heat_pump_data.index
        # self.outdoor_temperature = heat_pump_data['T_od']
        # self.room_temperature_sp = heat_pump_data['T_room_sp']
        self.domestic_hot_water_temperature_sp = 55
        self.solar_irradiance = time_series_data_csv['solar_irradiance']
        self.wind_speed = time_series_data_csv['wind_speed']
        self.electricity_price = time_series_data_csv['electricity_price']
        self.gas_price = time_series_data_csv['gas_price']
        self.heat_price = time_series_data_csv['heat_price']
        self.hydrogen_price = time_series_data_csv['hydrogen_price']
        # self.flexible_electricity_demand = time_series_data_csv['flex_elec_demand']
        self.inflexible_electricity_demand = time_series_data_csv['inflex_elec_demand']
        self.inflexible_heat_demand = time_series_data_csv['inflex_heat_demand']
        self.inflexible_gas_demand = time_series_data_csv['inflex_gas_demand']
        self.inflexible_hydrogen_demand = time_series_data_csv['inflex_hydrogen_demand']
        self.ambient_temperature = time_series_data_csv['ambient_temperature']
        self.pv_generator = appliance_data.loc['pv_generator']['value']
        self.electric_vehicle = appliance_data.loc['electric_vehicle_charger']['value']
        self.energy_storage_system = appliance_data.loc['energy_storage_system']['value']
        self.time_shiftable_load = appliance_data.loc['time_shiftable_load'][['value', 'power', 'duration']]
        # self.thermostatically_electric_appliance = appliance_data.loc['air_conditioner']['value']
        # self.chp = appliance_data.loc['chp']['value']
        self.heat_pump = appliance_data.loc['heat_pump']['value']
        self.house_thermal_system = appliance_data.loc['house_thermal_system']['value']
        self.gas_boiler = appliance_data.loc['gas_boiler']['value']
        self.electric_boiler = appliance_data.loc['electric_boiler']['value']
        self.solar_thermal = appliance_data.loc['solar_thermal']['value']
        self.thermal_storage_system = appliance_data.loc['thermal_storage_system']['value']
        # self.wind_turbine = appliance_data.loc['wind_turbine']['value']
        self.electrolyzer = appliance_data.loc['electrolyzer']['value']
        self.hydrogen_storage_system = appliance_data.loc['hydrogen_storage_system']['value']
        self.fuel_cell = appliance_data.loc['fuel_cell']['value']
        self.carrier_limitation = appliance_data.loc['carrier_limitation']['value']
        self.timesteps = time_series_data_csv.index
        self.timesteps_freq = self.timesteps.freq
        self.delta_t = self.timesteps_freq / pd.Timedelta('1h')


# data_path = {'time_series': 'household_time_series_data.csv', 'appliance_data': 'house_hold_appliance_data.csv'}
# household_data = HouseholdData(data_path)
# print('Done!')
