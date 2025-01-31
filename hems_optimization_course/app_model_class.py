import inspect
import sys
import typing
from pyomo.environ import *
from pyaugmecon import PyAugmecon
from data_interface import HouseholdData


class HouseModel(object):
    model: ConcreteModel
    app_type: str = None
    household_data: HouseholdData
    domestic_electric_power_balance_element_list: list
    electric_export_power_balance_element_list: list
    domestic_heating_power_balance_element_list: list
    domestic_natural_gas_balance_element_list: list
    heating_export_power_balance_element_list: list

    def __init__(self, household_data: HouseholdData):
        self.household_data = household_data
        self.timesteps = household_data.timesteps
        self.timestep_interval = household_data.timesteps_freq
        self.natural_gas_lower_heating_value = 10.16  # [kWh/m3]
        self.model = ConcreteModel()

    def initialize_model(self):
        self.model.delta_t = self.household_data.delta_t
        self.model.timesteps = Set(ordered=True, initialize=self.household_data.timesteps)
        self.model.timesteps_not0 = self.model.timesteps - [self.model.timesteps.first()]

        # Prices
        self.model.power_buy_price = Param(
            self.model.timesteps, within=Reals,
            initialize=self.household_data.electricity_price
        )
        self.model.power_sell_price = Param(
            self.model.timesteps, within=Reals,
            initialize=0.9 * self.household_data.electricity_price
        )
        self.model.heating_buy_price = Param(
            self.model.timesteps, within=Reals,
            initialize=self.household_data.heat_price
        )
        self.model.heating_sell_price = Param(
            self.model.timesteps, within=Reals,
            initialize=0.8 * self.household_data.heat_price
        )
        self.model.gas_buy_price = Param(
            self.model.timesteps, within=Reals,
            initialize=self.household_data.gas_price
        )

        # CO2 intenisity factor:
        self.model.co2_intensity = Param(within=NonNegativeReals, initialize=0.15)

        # LHV of nautal gas
        self.model.gas_lhv = Param(within=NonNegativeReals, initialize=self.natural_gas_lower_heating_value)

        # Import and export power variables define
        self.model.electric_power_imported = Var(self.model.timesteps, within=NonNegativeReals, initialize=0)
        self.model.electric_power_exported = Var(self.model.timesteps, within=NonPositiveReals, initialize=0)
        self.model.heating_power_imported = Var(self.model.timesteps, within=NonNegativeReals, initialize=0)
        self.model.heating_power_exported = Var(self.model.timesteps, within=NonPositiveReals, initialize=0)
        self.model.natural_gas_imported = Var(self.model.timesteps, within=NonNegativeReals, initialize=0)

        # External network limitations:
        self.model.electric_power_import_export_limit = Param(
            within=NonNegativeReals,
            initialize=self.household_data.carrier_limitation['electric'])  # [kW]
        self.model.heating_power_import_export_limit = Param(
            within=NonNegativeReals,
            initialize=self.household_data.carrier_limitation['heat'])  # [kW]
        self.model.natural_gas_import_limit = Param(
            within=NonNegativeReals,
            initialize=self.household_data.carrier_limitation['gas'])  # [m3]

        self.model.power_buy_binary = Var(self.model.timesteps, within=Binary, initialize=0)
        self.model.heating_buy_binary = Var(self.model.timesteps, within=Binary, initialize=0)

        # Loads including electricity, heating and gas
        # Electric load:
        self.model.inflexible_power_demand = Param(
            self.model.timesteps, within=NonPositiveReals,
            initialize=-1 * self.household_data.inflexible_electricity_demand)
        # Heating load
        self.model.domestic_hot_water_heating_demand = Param(
            self.model.timesteps, within=NonPositiveReals,
            initialize=-1 * self.household_data.inflexible_heat_demand
        )
        # Natural gas load
        self.model.inflexible_gas_demand = Param(
            self.model.timesteps, within=NonPositiveReals,
            initialize=-1 * self.household_data.inflexible_gas_demand
        )

        # Power balance list of equations
        self.domestic_electric_power_balance_element_list = list()
        self.domestic_electric_power_balance_element_list.append(self.model.electric_power_imported)
        self.domestic_electric_power_balance_element_list.append(self.model.inflexible_power_demand)

        self.electric_export_power_balance_element_list = list()
        self.electric_export_power_balance_element_list.append(self.model.electric_power_exported)

        self.domestic_heating_power_balance_element_list = list()
        self.domestic_heating_power_balance_element_list.append(self.model.heating_power_imported)
        self.domestic_heating_power_balance_element_list.append(self.model.domestic_hot_water_heating_demand)

        self.heating_export_power_balance_element_list = list()
        self.heating_export_power_balance_element_list.append(self.model.heating_power_exported)

        self.domestic_natural_gas_balance_element_list = list()
        self.domestic_natural_gas_balance_element_list.append(self.model.inflexible_gas_demand)
        self.domestic_natural_gas_balance_element_list.append(self.model.natural_gas_imported)

    def construct_energy_balance_constraint(self):
        def electric_domestic_power_balance_expr_rule(model, t):
            return sum(element[t] for element in self.domestic_electric_power_balance_element_list) == 0

        self.model.electric_domestic_power_balance_constraint = Constraint(
            self.model.timesteps,
            rule=electric_domestic_power_balance_expr_rule
        )

        def electric_export_power_balance_expr_rule(model, t):
            return sum(element[t] for element in self.electric_export_power_balance_element_list) == 0

        self.model.electric_export_power_balance_constraint = Constraint(
            self.model.timesteps,
            rule=electric_export_power_balance_expr_rule
        )

        # Limit energy export and import from the electric network
        def household_electric_power_import_limit_constraint_rule(model, t):
            return (
                    model.electric_power_imported[t]
                    <=
                    model.electric_power_import_export_limit * model.power_buy_binary[t]
            )

        self.model.household_electric_power_import_limit_constraint = Constraint(
            self.model.timesteps, rule=household_electric_power_import_limit_constraint_rule
        )

        def household_electric_power_export_limit_constraint_rule(model, t):
            return (
                    model.electric_power_exported[t]
                    >=
                    -1 * model.electric_power_import_export_limit * (1-model.power_buy_binary[t])
            )

        self.model.household_electric_power_export_limit_constraint = Constraint(
            self.model.timesteps, rule=household_electric_power_export_limit_constraint_rule
        )

        # ------------------------ Heating carrier -----------------------------------

        def heating_domestic_power_balance_expr_rule(model, t):
            return sum(element[t] for element in self.domestic_heating_power_balance_element_list) == 0

        self.model.heating_domestic_power_balance_constraint = Constraint(
            self.model.timesteps,
            rule=heating_domestic_power_balance_expr_rule
        )

        def heating_export_power_balance_expr_rule(model, t):
            return sum(element[t] for element in self.heating_export_power_balance_element_list) == 0

        self.model.heating_domestic_hot_water_power_balance_constraint = Constraint(
            self.model.timesteps,
            rule=heating_export_power_balance_expr_rule
        )

        # Limit energy export and import from the heating network
        def household_heating_power_import_limit_constraint_rule(model, t):
            return (
                    model.heating_power_imported[t]
                    <=
                    model.heating_power_import_export_limit * model.heating_buy_binary[t]
            )

        self.model.household_heating_power_import_limit_constraint = Constraint(
            self.model.timesteps, rule=household_heating_power_import_limit_constraint_rule
        )

        def household_heating_power_export_limit_constraint_rule(model, t):
            return (
                    model.heating_power_exported[t]
                    >=
                    -1 * model.heating_power_import_export_limit * (1 - model.heating_buy_binary[t])
            )

        self.model.household_heating_power_export_limit_constraint = Constraint(
            self.model.timesteps, rule=household_heating_power_export_limit_constraint_rule
        )

        # ------------------- Natural gas carrier ---------------------------------
        def domestic_natural_gas_balance_expr_rule(model, t):
            return sum(element[t] for element in self.domestic_natural_gas_balance_element_list) == 0

        self.model.domestic_natural_gas_balance_constraint = Constraint(
            self.model.timesteps,
            rule=domestic_natural_gas_balance_expr_rule
        )

        def household_natural_gas_import_limit_constraint_rule(model, t):
            return model.natural_gas_imported[t] <= model.natural_gas_import_limit

        self.model.household_natural_gas_import_limit_constraint = Constraint(
            self.model.timesteps, rule=household_natural_gas_import_limit_constraint_rule
        )

    def construct_single_objective_function(self):
        def cost_min_rule(model):
            return (
                    sum(
                        (model.power_buy_price[t] * model.electric_power_imported[t] + model.power_sell_price[t] *
                         model.electric_power_exported[t])
                        +
                        (model.heating_buy_price[t] * model.heating_power_imported[t] + model.heating_sell_price[t] *
                         model.heating_power_exported[t])
                        +
                        (model.natural_gas_imported[t] * model.gas_buy_price[t])
                        for t in model.timesteps
                    ) * model.delta_t
            )

        self.model.OF = Objective(expr=cost_min_rule, sense=minimize)

    def construct_multi_objective_functions(self):
        def minimize_electricity_cost_rule(model):
            return (
                    sum(
                        (model.power_buy_price[t] * model.electric_power_imported[t] + model.power_sell_price[t] *
                         model.electric_power_exported[t])
                        +
                        (model.heating_buy_price[t] * model.heating_power_imported[t] + model.heating_sell_price[t] *
                         model.heating_power_exported[t])
                        +
                        (model.natural_gas_imported[t] * model.gas_buy_price[t])
                        for t in model.timesteps
                    ) * model.delta_t
            )

        def min_co2_rule(model):
            return (
                    model.co2_intensity * model.delta_t
                    * sum(
                            model.electric_power_imported[t] +
                            model.heating_power_imported[t] +
                            model.natural_gas_imported[t] * model.gas_lhv
                            for t in model.timesteps
                    )
            )

        self.model.obj_list = ObjectiveList()
        self.model.obj_list.add(expr=minimize_electricity_cost_rule(self.model), sense=minimize)
        self.model.obj_list.add(expr=min_co2_rule(self.model), sense=minimize)


class AppModel(object):
    household_data: HouseholdData
    house_model: HouseModel
    model: ConcreteModel
    app_type: str = None

    def __init__(self, household_data: HouseholdData, house_model: HouseModel):
        self.household_data = household_data
        self.house_model = house_model
        self.model = house_model.model
        self.natural_gas_lower_heating_value = 10.16  # [kWh/m3]


class PhotovoltaicSystem(AppModel):
    app_type = "pv_generator"

    def __init__(self, household_data: HouseholdData, house_model: HouseModel):
        super().__init__(household_data, house_model)

        self.model.pv_power_generation = Param(
            self.model.timesteps, within=NonNegativeReals,
            initialize=(self.household_data.pv_generator['area'] * self.household_data.pv_generator['efficiency']
                        * self.household_data.pv_generator['performance_ratio']
                        * self.household_data.solar_irradiance / 1000)
        )

        self.model.pv_power_domestic_use = Var(
            self.model.timesteps, within=NonNegativeReals, initialize=0
        )  # PV power used in the household
        self.model.pv_power_export = Var(
            self.model.timesteps, within=NonNegativeReals, initialize=0
        )  # PV power sold

        def pv_balance_rule(model, t):
            return model.pv_power_domestic_use[t] + model.pv_power_export[t] == model.pv_power_generation[t]

        self.model.pv_balance_constraint = Constraint(self.model.timesteps,
                                                      rule=pv_balance_rule)

        self.house_model.domestic_electric_power_balance_element_list.append(
            self.model.pv_power_domestic_use)
        self.house_model.electric_export_power_balance_element_list.append(self.model.pv_power_export)


class SolarThermalSystem(AppModel):
    app_type = "solar_thermal"

    def __init__(self, household_data: HouseholdData, house_model: HouseModel):
        super().__init__(household_data, house_model)

        self.model.st_heating_power_generation = Param(
            self.model.timesteps, within=NonNegativeReals,
            initialize=(self.household_data.solar_thermal['area']
                        * self.household_data.solar_thermal['efficiency']
                        * self.household_data.solar_irradiance / 1000
                        )
        )

        self.model.st_heating_power_domestic_use = Var(
            self.model.timesteps, within=NonNegativeReals, initialize=0
        )
        self.model.st_heating_power_export = Var(
            self.model.timesteps, within=NonNegativeReals, initialize=0
        )

        def st_balance_rule(model, t):
            return (
                    model.st_heating_power_domestic_use[t]
                    + model.st_heating_power_export[t]
                    == model.st_heating_power_generation[t]
            )

        self.model.st_balance_constraint = Constraint(self.model.timesteps, rule=st_balance_rule)

        self.house_model.domestic_heating_power_balance_element_list.append(
            self.model.st_heating_power_domestic_use)
        self.house_model.heating_export_power_balance_element_list.append(self.model.st_heating_power_export)


class Storage(AppModel):
    app_type = "energy_storage_system"

    def __init__(self, household_data: HouseholdData, house_model: HouseModel):
        # ESS parameters
        super().__init__(household_data, house_model)

        self.model.storage_charging_rate = Param(
            within=NonNegativeReals, initialize=self.household_data.energy_storage_system['charging_rate']
        )  # [kW] rated charging power
        self.model.storage_discharging_rate = Param(
            within=NonNegativeReals, initialize=self.household_data.energy_storage_system['discharging_rate']
        )  # [kW] rated discharging power
        self.model.storage_minimum_soe = Param(
            within=NonNegativeReals, initialize=self.household_data.energy_storage_system['min_soe']
        )  # [kWh] min state-of-energy
        self.model.storage_maximum_soe = Param(
            within=NonNegativeReals, initialize=self.household_data.energy_storage_system['max_soe']
        )  # [kWh] max state-of-energy
        self.model.ess_soe_ini = Param(
            within=NonNegativeReals, initialize=self.household_data.energy_storage_system['ini_soe']
        )  # [kWh] initial state-of-energy at t=0
        self.model.ess_eta_ch = Param(
            within=NonNegativeReals, initialize=self.household_data.energy_storage_system['ch_eta']
        )  # [/] charging efficiency
        self.model.ess_eta_dch = Param(
            within=NonNegativeReals, initialize=self.household_data.energy_storage_system['dch_eta']
        )  # [/] discharging efficiency
        # ESS variables
        self.model.storage_charging_power = Var(
            self.model.timesteps, within=NonPositiveReals, initialize=0
        )  # ESS charging power
        self.model.storage_discharging_power = Var(
            self.model.timesteps, within=NonNegativeReals, initialize=0
        )  # ESS dischargin power
        self.model.storage_soe = Var(
            self.model.timesteps, within=NonNegativeReals, initialize=0
        )  # ESS state-of-energy
        self.model.storage_power_export = Var(
            self.model.timesteps, within=NonNegativeReals, initialize=0
        )  # ESS power sold
        self.model.storage_domestic_power_used = Var(
            self.model.timesteps, within=NonNegativeReals, initialize=0
        )  # ESS power used in the household
        self.model.storage_binary = Var(
            self.model.timesteps, within=Binary, initialize=0
        )  # binary variable: 1-ESS charging, 0 otherwise

        # ESS constraints start ------------------------------------------------------------------------------------------
        # Bounds rule for charging power for ESS
        def bounds_ess_ch_rule(model, t):
            return -1 * model.storage_binary[t] * model.storage_charging_rate <= model.storage_charging_power[t]

        self.model.ess_ch_bound = Constraint(self.model.timesteps, rule=bounds_ess_ch_rule)

        # Bounds rule for discharging power for ESS
        def bounds_ess_dis_rule(model, t):
            return (1 - model.storage_binary[t]) * model.storage_discharging_rate >= model.storage_discharging_power[t]

        self.model.ess_dis_bound = Constraint(self.model.timesteps, rule=bounds_ess_dis_rule)

        def fix_ess_dis_at_end_rule(model):
            return model.storage_discharging_power[model.timesteps.last()] == 0

        self.model.fix_ess_dis_at_end = Constraint(rule=fix_ess_dis_at_end_rule)

        # ESS minimum SOE
        def bounds_ess_soe_rule_1(model, t):
            return model.storage_minimum_soe <= model.storage_soe[t]

        self.model.ess_soe_bound_1 = Constraint(self.model.timesteps,
                                                rule=bounds_ess_soe_rule_1)

        # ESS maximum SOE
        def bounds_ess_soe_rule_2(model, t):
            return model.storage_soe[t] <= model.storage_maximum_soe

        self.model.ess_soe_bound_2 = Constraint(self.model.timesteps,
                                                rule=bounds_ess_soe_rule_2)

        # ESS SOE update
        def ess_soe_balance_rule(model, t):
            return (model.storage_soe[t] == model.storage_soe[model.timesteps.prev(t)]
                    + (-1 * model.storage_charging_power[model.timesteps.prev(t)]
                       * model.ess_eta_ch - model.storage_discharging_power[model.timesteps.prev(t)]) * model.delta_t)

        self.model.ess_soe_balance = Constraint(self.model.timesteps_not0,
                                                rule=ess_soe_balance_rule)

        def ess_soe_ini_balance_rule(model):
            return model.storage_soe[model.timesteps.first()] == model.ess_soe_ini

        self.model.ess_soe_ini_balance = Constraint(rule=ess_soe_ini_balance_rule)

        # ESS balance for discharged power
        def ess_balance_rule(model, t):
            return (model.storage_domestic_power_used[t] + model.storage_power_export[t]
                    == model.storage_discharging_power[t] * model.ess_eta_dch)

        self.model.ess_balance = Constraint(self.model.timesteps, rule=ess_balance_rule)

        # ESS SOE at t=T, end of simulation period
        def ess_eod_rule(model):
            return model.storage_soe[model.timesteps.last()] == model.ess_soe_ini

        self.model.ess_eod = Constraint(rule=ess_eod_rule)

        self.house_model.domestic_electric_power_balance_element_list.append(
            self.model.storage_charging_power)
        self.house_model.domestic_electric_power_balance_element_list.append(
            self.model.storage_domestic_power_used)
        self.house_model.electric_export_power_balance_element_list.append(self.model.storage_power_export)


class ThermalStorage(AppModel):
    app_type = "thermal_storage_system"

    def __init__(self, household_data: HouseholdData, house_model: HouseModel):
        # ESS parameters
        super().__init__(household_data, house_model)

        self.model.thermal_storage_charging_rate = Param(
            within=NonNegativeReals, initialize=self.household_data.thermal_storage_system['charging_rate']
        )  # [kW] rated charging power
        self.model.thermal_storage_discharging_rate = Param(
            within=NonNegativeReals, initialize=self.household_data.thermal_storage_system['discharging_rate']
        )  # [kW] rated discharging power
        self.model.thermal_storage_minimum_soe = Param(
            within=NonNegativeReals, initialize=self.household_data.thermal_storage_system['min_soh']
        )  # [kWh] min state-of-energy
        self.model.thermal_storage_maximum_soe = Param(
            within=NonNegativeReals, initialize=self.household_data.thermal_storage_system['max_soh']
        )  # [kWh] max state-of-energy
        self.model.thermal_ess_soe_ini = Param(
            within=NonNegativeReals, initialize=self.household_data.thermal_storage_system['ini_soh']
        )  # [kWh] initial state-of-energy at t=0
        self.model.thermal_ess_eta_ch = Param(
            within=NonNegativeReals, initialize=self.household_data.thermal_storage_system['ch_eta']
        )  # [/] charging efficiency
        self.model.thermal_ess_eta_dch = Param(
            within=NonNegativeReals, initialize=self.household_data.thermal_storage_system['dch_eta']
        )  # [/] discharging efficiency
        # ESS variables
        self.model.thermal_storage_charging_power = Var(
            self.model.timesteps, within=NonPositiveReals, initialize=0
        )  # ESS charging power
        self.model.thermal_storage_discharging_power = Var(
            self.model.timesteps, within=NonNegativeReals, initialize=0
        )  # ESS dischargin power
        self.model.thermal_storage_soe = Var(
            self.model.timesteps, within=NonNegativeReals, initialize=0
        )  # ESS state-of-energy
        self.model.thermal_storage_power_export = Var(
            self.model.timesteps, within=NonNegativeReals, initialize=0
        )  # ESS power sold
        self.model.thermal_storage_domestic_power_used = Var(
            self.model.timesteps, within=NonNegativeReals, initialize=0
        )  # ESS power used in the household
        self.model.thermal_storage_binary = Var(
            self.model.timesteps, within=Binary, initialize=0
        )  # binary variable: 1-ESS charging, 0 otherwise

        # ESS constraints start ------------------------------------------------------------------------------------------
        # Bounds rule for charging power for ESS
        def bounds_thermal_ess_ch_rule(model, t):
            return (
                    -1 * model.thermal_storage_binary[t]
                    * model.thermal_storage_charging_rate
                    <= model.thermal_storage_charging_power[t]
            )

        self.model.thermal_ess_ch_bound = Constraint(self.model.timesteps, rule=bounds_thermal_ess_ch_rule)

        # Bounds rule for discharging power for ESS
        def bounds_thermal_ess_dis_rule(model, t):
            return (
                    (1 - model.thermal_storage_binary[t])
                    * model.thermal_storage_discharging_rate
                    >= model.thermal_storage_discharging_power[t]
            )

        self.model.thermal_ess_dis_bound = Constraint(self.model.timesteps, rule=bounds_thermal_ess_dis_rule)

        def fix_thermal_ess_dis_at_end_rule(model):
            return model.thermal_storage_discharging_power[model.timesteps.last()] == 0

        self.model.fix_thermal_ess_dis_at_end = Constraint(rule=fix_thermal_ess_dis_at_end_rule)

        # ESS minimum SOE
        def bounds_thermal_ess_soe_rule_1(model, t):
            return model.thermal_storage_minimum_soe <= model.thermal_storage_soe[t]

        self.model.ess_thermal_soe_bound_1 = Constraint(self.model.timesteps, rule=bounds_thermal_ess_soe_rule_1)

        # ESS maximum SOE
        def bounds_thermal_ess_soe_rule_2(model, t):
            return model.thermal_storage_soe[t] <= model.thermal_storage_maximum_soe

        self.model.ess_thermal_soe_bound_2 = Constraint(self.model.timesteps, rule=bounds_thermal_ess_soe_rule_2)

        # ESS SOE update
        def thermal_ess_soe_balance_rule(model, t):
            return (model.thermal_storage_soe[t] == model.thermal_storage_soe[model.timesteps.prev(t)]
                    + (-1 * model.thermal_storage_charging_power[model.timesteps.prev(t)]
                       * model.thermal_ess_eta_ch - model.thermal_storage_discharging_power[model.timesteps.prev(t)])
                    * model.delta_t
                    )

        self.model.thermal_ess_soe_balance = Constraint(self.model.timesteps_not0, rule=thermal_ess_soe_balance_rule)

        def thermal_ess_soe_ini_balance_rule(model):
            return model.thermal_storage_soe[model.timesteps.first()] == model.thermal_ess_soe_ini

        self.model.thermal_ess_soe_ini_balance = Constraint(rule=thermal_ess_soe_ini_balance_rule)

        # ESS balance for discharged power
        def thermal_ess_balance_rule(model, t):
            return (model.thermal_storage_domestic_power_used[t] + model.thermal_storage_power_export[t]
                    == model.thermal_storage_discharging_power[t] * model.thermal_ess_eta_dch)

        self.model.thermal_ess_balance = Constraint(self.model.timesteps, rule=thermal_ess_balance_rule)

        # ESS SOE at t=T, end of simulation period
        def thermal_ess_eod_rule(model):
            return model.thermal_storage_soe[model.timesteps.last()] == model.thermal_ess_soe_ini

        self.model.thermal_ess_eod = Constraint(rule=thermal_ess_eod_rule)

        self.house_model.domestic_heating_power_balance_element_list.append(
            self.model.thermal_storage_charging_power)
        self.house_model.domestic_heating_power_balance_element_list.append(
            self.model.thermal_storage_domestic_power_used)
        self.house_model.heating_export_power_balance_element_list.append(self.model.thermal_storage_power_export)


class HouseThermalSystem(AppModel):
    app_type = "house_thermal_system"

    def __init__(self, household_data: HouseholdData, house_model: HouseModel):
        super().__init__(household_data, house_model)

        self.model.house_thermal_system_resistance = Param(
            within=NonNegativeReals, initialize=self.household_data.house_thermal_system['resistance']
        )
        self.model.house_thermal_system_capacitance = Param(
            within=NonNegativeReals, initialize=self.household_data.house_thermal_system['capacitance']
        )
        self.model.house_thermal_system_ini_temp = Param(
            within=NonNegativeReals, initialize=self.household_data.house_thermal_system['temp_ini']
        )
        self.model.house_thermal_system_band_temp = Param(
            within=NonNegativeReals, initialize=self.household_data.house_thermal_system['band_temp']
        )
        self.model.house_thermal_system_max_temp = Param(
            within=NonNegativeReals, initialize=self.household_data.house_thermal_system['temp_max_sp']
        )
        self.model.house_thermal_system_min_temp = Param(
            within=NonNegativeReals, initialize=self.household_data.house_thermal_system['temp_min_sp']
        )
        self.model.outdoor_temperature = Param(
            self.model.timesteps, within=Reals, initialize=self.household_data.ambient_temperature
        )

        # Sets:
        self.model.timesteps_heating_system_active = Set(
            ordered=True,
            initialize=self.household_data.ambient_temperature[self.household_data.ambient_temperature < 22].index
        )
        self.model.timesteps_heating_system_not_active = (
                self.model.timesteps - self.model.timesteps_heating_system_active
        )

        # Variables:
        self.model.house_temperature = Var(
            self.model.timesteps, within=NonNegativeReals, initialize=0
        )
        self.model.house_thermal_system_heating = Var(
            self.model.timesteps, within=NonPositiveReals, initialize=0
        )
        self.model.house_heating_binary = Var(
            self.model.timesteps, within=Binary, initialize=0
        )

        # Constraints:
        def house_temp_change_rule(model, t):
            return (
                    model.house_temperature[t] == model.house_temperature[model.timesteps.prev(t)] *
                    (
                            1 - model.delta_t
                            / (model.house_thermal_system_resistance * model.house_thermal_system_capacitance)
                    )
                    + (
                            model.outdoor_temperature[model.timesteps.prev(t)]
                            / (model.house_thermal_system_resistance * model.house_thermal_system_capacitance)
                            + (
                                    -1.0 * model.house_thermal_system_heating[model.timesteps.prev(t)]
                            ) / model.house_thermal_system_capacitance
                    ) * model.delta_t
            )

        self.model.house_temp_change = Constraint(
            self.model.timesteps_not0, rule=house_temp_change_rule
        )

        # Temperature change at t=0
        def house_temp_change_ini_rule(model):
            return (
                    model.house_temperature[model.timesteps.first()]
                    ==
                    # model.house_thermal_system_ini_temp
                model.house_temperature[model.timesteps.last()]
            )

        self.model.house_ini_change = Constraint(rule=house_temp_change_ini_rule)

        def house_heating_heat_final_value_rule(model):
            return model.house_thermal_system_heating[model.timesteps.last()] == 0

        self.model.house_heating_heat_final_value_eq = Constraint(rule=house_heating_heat_final_value_rule)

        def house_heating_system_heating_not_active_rule(model, t):
            return model.house_heating_binary[t] == 0

        self.model.house_heating_system_heating_not_active_eq = Constraint(
            self.model.timesteps_heating_system_not_active, rule=house_heating_system_heating_not_active_rule)

        def house_heating_system_heat_max_rule(model, t):
            return model.house_thermal_system_heating[t] >= -5 * model.house_heating_binary[t]

        self.model.house_heating_system_heat_max_eq = Constraint(
            self.model.timesteps, rule=house_heating_system_heat_max_rule)

        # minimum temperature bound
        def house_min_temp_rule(model, t):
            return (
                    model.house_temperature[t]
                    >=
                    model.house_thermal_system_min_temp
                    - model.house_thermal_system_band_temp
            )

        self.model.house_min_temp = Constraint(
            self.model.timesteps_heating_system_active, rule=house_min_temp_rule)

        # maximum temperature bound
        def house_max_temp_rule(model, t):
            return (
                    model.house_temperature[t]
                    <=
                    model.house_thermal_system_max_temp + model.house_thermal_system_band_temp
            )

        self.model.house_max_temp_eq = Constraint(
            self.model.timesteps, rule=house_max_temp_rule)


        self.house_model.domestic_heating_power_balance_element_list.append(
            self.model.house_thermal_system_heating
        )

class HeatPump(AppModel):
    app_type = "heat_pump"

    def __init__(self, household_data: HouseholdData, house_model: HouseModel):
        super().__init__(household_data, house_model)

        # Parameters:
        self.model.heat_pump_cop = Param(within=NonNegativeReals, initialize=self.household_data.heat_pump['cop'])
        self.model.heat_pump_max_power = Param(
            within=NonNegativeReals, initialize=self.household_data.heat_pump['max_power']
        )
        # Variables:
        self.model.heat_pump_power_to_heating = Var(self.model.timesteps, within=NonPositiveReals, initialize=0)
        self.model.heat_pump_heat = Var(self.model.timesteps, within=NonNegativeReals, initialize=0)
        self.model.heat_pump_heat_used = Var(self.model.timesteps, within=NonNegativeReals, initialize=0)
        self.model.heat_pump_heat_sold = Var(self.model.timesteps, within=NonNegativeReals, initialize=0)

        # Constraints:
        def heat_pump_heat_rule(model, t):
            return model.heat_pump_heat[t] == model.heat_pump_power_to_heating[t] * -1 * model.heat_pump_cop

        self.model.heat_pump_heat_eq = Constraint(self.model.timesteps, rule=heat_pump_heat_rule)

        def heat_pump_heat_use_sell_rule(model, t):
            return model.heat_pump_heat[t] == model.heat_pump_heat_used[t] + model.heat_pump_heat_sold[t]

        self.model.heat_pump_heat_use_sell_eq = Constraint(self.model.timesteps, rule=heat_pump_heat_use_sell_rule)

        def heat_pump_max_power_rule(model, t):
            return model.heat_pump_power_to_heating[t] >= -1 * model.heat_pump_max_power

        self.model.heat_pump_max_power_eq = Constraint(self.model.timesteps, rule=heat_pump_max_power_rule)

        self.house_model.domestic_electric_power_balance_element_list.append(self.model.heat_pump_power_to_heating)
        self.house_model.domestic_heating_power_balance_element_list.append(self.model.heat_pump_heat_used)
        self.house_model.heating_export_power_balance_element_list.append(self.model.heat_pump_heat_sold)


class TimeShiftableElectricLoad(AppModel):
    app_type = "time_shiftable_load"

    def __init__(self, household_data: HouseholdData, house_model: HouseModel):
        super().__init__(household_data, house_model)

        # ph_list = ['ph' + str(p) for p in range(1, len(input_param['drye_p']) + 1)]   # list for set of phases for the appliances operation
        self.model.tsl_phases = Set(
            ordered=True, initialize=self.household_data.time_shiftable_load.index.drop(['operating']))
        self.model.tsl_phase_power_consumption = Param(
            self.model.tsl_phases, within=NonPositiveReals,
            initialize=-1 * self.household_data.time_shiftable_load.loc[
                self.model.tsl_phases, 'power'])  # parameter for the power demand per operational phase
        # Variables
        self.model.tsl_operation_phase_status = Var(
            self.model.timesteps, self.model.tsl_phases, within=Binary, initialize=0)  # phase p in duration
        self.model.tsl_starting_phase_status = Var(
            self.model.timesteps, self.model.tsl_phases, within=Binary, initialize=0)  # phase p start
        self.model.tsl_ending_phase_status = Var(
            self.model.timesteps, self.model.tsl_phases, within=Binary, initialize=0)  # phase p end
        self.model.tsl_power_consumption = Var(
            self.model.timesteps, within=NonPositiveReals, initialize=0)

        # Power demand depending on current phase
        def ts_power_phase_rule(model, t):
            return (
                    model.tsl_power_consumption[t] ==
                    sum(
                        (model.tsl_phase_power_consumption[ph] * model.tsl_operation_phase_status[t, ph])
                        for ph in model.tsl_phases
                    )
            )

        self.model.ts_power_phase = Constraint(
            self.model.timesteps, rule=ts_power_phase_rule)

        # One operating phase active per time
        def ts_phase_per_time_rule(model, t):
            return sum(model.tsl_operation_phase_status[t, ph] for ph in model.tsl_phases) <= 1

        self.model.ts_power_per_time = Constraint(
            self.model.timesteps, rule=ts_phase_per_time_rule)

        # Each phase lasts one time interval, so it ends in the next interval.
        # ToDo this part is incorrect if each phase lasts more than 1 time interval.
        def ts_start_finish_rule(model, t, ph):
            return model.tsl_starting_phase_status[model.timesteps.prev(t), ph] == model.tsl_ending_phase_status[t, ph]

        self.model.ts_start_finish = Constraint(
            self.model.timesteps_not0, self.model.tsl_phases, rule=ts_start_finish_rule)

        # Change of phase duration status = change in start and end status
        def ts_start_end_phase_rule(model, t, ph):
            return (
                    model.tsl_starting_phase_status[t, ph] - model.tsl_ending_phase_status[t, ph] ==
                    model.tsl_operation_phase_status[t, ph]
                    - model.tsl_operation_phase_status[model.timesteps.prev(t), ph]
            )

        self.model.ts_start_end_phase = Constraint(
            self.model.timesteps_not0, self.model.tsl_phases, rule=ts_start_end_phase_rule)

        # End of one phase = start of the next one
        def ts_next_phase_rule(model, t, ph):
            return model.tsl_ending_phase_status[t, model.tsl_phases.prev(ph)] == model.tsl_starting_phase_status[t, ph]

        self.model.ts_next_phase = Constraint(
            self.model.timesteps, self.model.tsl_phases - {'phase1'}, rule=ts_next_phase_rule)

        # Number of times that device can operate during a day
        def ts_phase_activation_rule(model, ph):
            return sum(model.tsl_starting_phase_status[t, ph] for t in model.timesteps) == 1

        self.model.ts_phase_activation = Constraint(
            self.model.tsl_phases, rule=ts_phase_activation_rule)

        # No interruption in the operation, one phase after another is on
        def ts_phase_sequence_rule(model, t, ph):
            return (
                    model.tsl_operation_phase_status[model.timesteps.prev(t), model.tsl_phases.prev(ph)] ==
                    model.tsl_operation_phase_status[t, ph]
            )

        self.model.ts_phase_sequence = Constraint(
            self.model.timesteps_not0,
            self.model.tsl_phases - {'phase1'},
            rule=ts_phase_sequence_rule)

        def ts_operation_off_at_the_end_of_day_rule(model, ph):
            return model.tsl_operation_phase_status[model.timesteps.last(), ph] == 0

        self.model.ts_operation_off_at_the_end_of_day = Constraint(
            self.model.tsl_phases, rule=ts_operation_off_at_the_end_of_day_rule)

        def ts_operation_off_at_the_beginning_of_day_rule(model, ph):
            return model.tsl_operation_phase_status[model.timesteps.first(), ph] == 0

        self.model.ts_operation_off_at_the_beginning_of_day = Constraint(
            self.model.tsl_phases, rule=ts_operation_off_at_the_beginning_of_day_rule)

        self.house_model.domestic_electric_power_balance_element_list.append(
            self.model.tsl_power_consumption
        )


class ElectricVehicleCharger(AppModel):
    app_type = "electric_vehicle_charger"

    def __init__(self, household_data: HouseholdData, house_model: HouseModel):
        super().__init__(household_data, house_model)

        # EV parameters
        self.model.ev_rch = Param(
            within=Reals,
            initialize=-1 * self.household_data.electric_vehicle['charging_rate'])  # [kW] rated charging power
        self.model.ev_rdch = Param(
            within=Reals,
            initialize=self.household_data.electric_vehicle['discharging_rate'])  # [kW] rated discharging power
        self.model.ev_soe_min = Param(
            within=Reals, initialize=self.household_data.electric_vehicle['min_soe'])  # [kWh] min state-of-energy
        self.model.ev_soe_max = Param(
            within=Reals, initialize=self.household_data.electric_vehicle['max_soe'])  # [kWh] max state-of-energy
        self.model.ev_soe_ini = Param(
            within=Reals,
            initialize=self.household_data.electric_vehicle['soe_ini_mor'])  # [kWh] initial state-of-energy at t=0
        self.model.ev_soe_dep = Param(
            within=Reals, initialize=self.household_data.electric_vehicle[
                'soe_eom'])  # [kWh] desired state-of-energy end-of-morning (eom), i.e. at departure time
        self.model.ev_soe_arr = Param(
            within=Reals, initialize=self.household_data.electric_vehicle[
                'soe_ini_aft'])  # [kWh] initial state-of-energy arrival in the afternoon
        self.model.ev_soe_eod = Param(
            within=Reals, initialize=self.household_data.electric_vehicle[
                'soe_eod'])  # [kWh] desired state-of-energy end-of-day (eod), i.e. at time T
        self.model.ev_eta_ch = Param(
            within=Reals, initialize=self.household_data.electric_vehicle['ch_eta'])  # [/] charging efficiency
        self.model.ev_eta_dch = Param(
            within=Reals, initialize=self.household_data.electric_vehicle['dch_eta'])  # [/] discharging efficiency
        self.model.ev_t_dep = Param(
            within=Integers,
            initialize=self.household_data.electric_vehicle['t_dep'] / self.household_data.delta_t)  # departure time
        self.model.ev_t_arr = Param(
            within=Integers,
            initialize=self.household_data.electric_vehicle['t_arr'] / self.household_data.delta_t)  # arrival time
        self.model.Tev_nav = Set(
            ordered=True,
            initialize=self.household_data.timesteps[
                       int(self.model.ev_t_dep.value + 1):int(self.model.ev_t_arr.value)]
        )  # set of time intervals when EV is not available
        self.model.Tev_av = self.model.timesteps - self.model.Tev_nav  # set of time intervals when EV is available
        self.model.Tev_av_noedge = (
                self.model.Tev_av - [self.model.timesteps.first()]
                - [self.model.timesteps[self.model.ev_t_arr.value + 1]]
        )  # set of time intervals in the afternoon after arrival time
        # EV variables
        self.model.ev_power_charged = Var(
            self.model.timesteps, within=NonPositiveReals, initialize=0)  # EV charging power
        self.model.ev_power_discharged = Var(
            self.model.timesteps, within=NonNegativeReals, initialize=0)  # EV discharging power
        self.model.ev_soe = Var(
            self.model.timesteps, within=NonNegativeReals, initialize=0)  # EV state-of-energy
        self.model.ev_power_export = Var(
            self.model.timesteps, within=NonNegativeReals, initialize=0)  # EV power sold
        self.model.ev_domestic_power_used = Var(
            self.model.timesteps, within=NonNegativeReals, initialize=0)  # EV power used in the household
        self.model.ev_binary = Var(
            self.model.timesteps, within=Binary, initialize=0)  # binary variable: 1-EV charging, 0 otherwise

        # EV Constraints:
        # Update SOE over time intervals
        def ev_soe_balance_rule(model, t):
            return (
                    model.ev_soe[t] == model.ev_soe[model.timesteps.prev(t)] + (
                    -1 * model.ev_power_charged[model.timesteps.prev(t)] * model.ev_eta_ch -
                    model.ev_power_discharged[model.timesteps.prev(t)]) * model.delta_t
            )

        self.model.ev_soe_balance = Constraint(
            self.model.Tev_av_noedge, rule=ev_soe_balance_rule)

        # SOE for t=0
        def ev_soe_ini_mor_rule(model):
            return model.ev_soe[model.timesteps.first()] == model.ev_soe_ini

        self.model.ev_soe_mor = Constraint(rule=ev_soe_ini_mor_rule)

        # SOE at t=t_dep in the morning
        def ev_soe_end_mor_rule(model):
            return model.ev_soe[model.timesteps[model.ev_t_dep.value + 1]] >= model.ev_soe_dep

        self.model.ev_soe_ev_eom = Constraint(rule=ev_soe_end_mor_rule)

        # fix the EV SOE when the car is away to the SOE at departure time
        def ev_soe_balance_na_rule(model, t):
            return model.ev_soe[t] == 0

        self.model.ev_soe_balance_na = Constraint(
            self.model.Tev_nav, rule=ev_soe_balance_na_rule)

        # SOE for t=t_arrival
        def ev_soe_ini_aft_rule(model):
            return model.ev_soe[model.timesteps[model.ev_t_arr.value + 1]] == model.ev_soe_arr

        self.model.ev_soe_aft = Constraint(rule=ev_soe_ini_aft_rule)

        # SOE at t=T
        def ev_soe_end_aft_rule(model):
            return model.ev_soe[model.timesteps.last()] >= model.ev_soe_eod

        self.model.ev_soe_ev_eod = Constraint(rule=ev_soe_end_aft_rule)

        # State of charge minimum bound
        def ev_bounds_soe_rule_1(model, t):
            return model.ev_soe_min <= model.ev_soe[t]

        self.model.ev_soe_bound_1 = Constraint(
            self.model.Tev_av, rule=ev_bounds_soe_rule_1)

        # State of charge maximum bound
        def ev_bounds_soe_rule_2(model, t):
            return model.ev_soe[t] <= model.ev_soe_max

        self.model.ev_soe_bound_2 = Constraint(
            self.model.Tev_av, rule=ev_bounds_soe_rule_2)

        # Bounds rule for charging power for EV
        def ev_bounds_pb_rule(model, t):
            return model.ev_binary[t] * model.ev_rch <= model.ev_power_charged[t]

        self.model.ev_pb_bound = Constraint(
            self.model.Tev_av, rule=ev_bounds_pb_rule)

        # Fix charging power when EV is not available
        def ev_bounds_pb_nav_rule(model, t):
            return model.ev_power_charged[t] == 0

        self.model.ev_pb_nav_bound = Constraint(
            self.model.Tev_nav.union([
                [self.model.timesteps[self.model.ev_t_dep.value + 1]],
                self.model.timesteps.last()
            ]),
            rule=ev_bounds_pb_nav_rule
        )

        # Bounds rule for discharging for EV
        def ev_bounds_ps_rule(model, t):
            return model.ev_power_discharged[t] <= (1 - model.ev_binary[t]) * model.ev_rdch

        self.model.ev_ps_bound = Constraint(self.model.Tev_av, rule=ev_bounds_ps_rule)

        # Fix discharging power when is not available
        def ev_bounds_ps_nav_rule(model, t):
            return model.ev_power_discharged[t] == 0

        self.model.ev_ps_nav_bound = Constraint(
            self.model.Tev_nav.union([
                [self.model.timesteps[self.model.ev_t_dep.value + 1]],
                self.model.timesteps.last()
            ]),
            rule=ev_bounds_ps_nav_rule
        )

        # EV balance for discharged power
        def ev_balance_rule(model, t):
            return model.ev_domestic_power_used[t] + model.ev_power_export[t] == model.ev_power_discharged[
                t] * model.ev_eta_dch

        self.model.ev_balance = Constraint(self.model.timesteps, rule=ev_balance_rule)

        self.house_model.domestic_electric_power_balance_element_list.extend([
            self.model.ev_domestic_power_used,
            self.model.ev_power_charged
        ])

        self.house_model.electric_export_power_balance_element_list.append(
            self.model.ev_power_export
        )


class AppModelSet(object):

    def __init__(
            self,
            household_data: HouseholdData,
            model_flags: dict
    ):
        self.household_data = household_data
        self.model_flags = model_flags
        self.app_models = list(household_data.app_data.index.levels[0].drop(['carrier_limitation']))
        self.house_model_defined = HouseModel(household_data)
        self.house_model_defined.initialize_model()
        make_app_models(
            self.app_models,
            household_data=self.household_data,
            household_model=self.house_model_defined,
            model_flags=self.model_flags
        )
        self.house_model_defined.construct_energy_balance_constraint()

    def solve_model(self, multi_objective: bool = False):
        if multi_objective:
            self.solve_multi_objective_model()

        else:
            self.solve_single_objective_model()

    def solve_single_objective_model(self):

        self.house_model_defined.construct_single_objective_function()

        opt = SolverFactory('gurobi')
        opt.options.update({"MIPGap": 0})

        opt.solve(self.house_model_defined.model, tee=True)

    def solve_multi_objective_model(self):
        self.house_model_defined.construct_multi_objective_functions()
        # Deactivate the Objective functions
        n_obj = len(self.house_model_defined.model.obj_list)
        for obj in range(n_obj):
            self.house_model_defined.model.obj_list[obj + 1].deactivate()

        # AUGMECON related options
        opts = {
            'grid_points': 15,
            'cpu_count': 1,
            'redivide_work': False,
            'shared_flag': False
        }

        # Options passed to Gurobi
        solver_opts = {'MIPGap': 1e-4}

        self.moop = PyAugmecon(self.house_model_defined.model, opts, solver_opts)

        self.moop.solve()


def make_app_models(
        app_names: typing.List[str],
        household_data: HouseholdData,
        household_model: HouseModel,
        model_flags: dict
):

    for app_name in app_names:
        if household_data.app_data.loc[(app_name, 'operating'), 'value'] == 1 and model_flags[app_name]:
            make_app_model(app_name, household_data, household_model)


def make_app_model(app_name: str, household_data: HouseholdData, household_model: HouseModel):

    # Obtain App type.
    app_type = app_name

    # Obtain App model classes.
    app_model_classes = inspect.getmembers(
        sys.modules[__name__], lambda cls: inspect.isclass(cls) and issubclass(cls, AppModel)
    )

    # Obtain App model for given `app_type`.
    for app_model_class_name, app_model_class in app_model_classes:
        if app_type == app_model_class.app_type:
            app_model_class(household_data, household_model)
