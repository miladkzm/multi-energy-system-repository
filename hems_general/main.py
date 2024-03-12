from app_model_class import AppModelSet
from data_interface import HouseholdData
from plot import *


def set_model_flag():
    m_flag = {}
    m_flag['flex_load'] = False
    m_flag['pv_generator'] = True
    m_flag['electric_vehicle_charger'] = True
    m_flag['energy_storage_system'] = True
    m_flag['time_shiftable_load'] = True
    m_flag['heat_pump'] = True
    m_flag['house_thermal_system'] = True
    m_flag['gas_boiler'] = False
    m_flag['electric_boiler'] = False
    m_flag['solar_thermal'] = True
    m_flag['thermal_storage_system'] = True
    m_flag['electrolyzer'] = False
    m_flag['hydrogen_storage_system'] = False
    m_flag['fuel_cell'] = False
    m_flag['gas_carrier'] = True
    m_flag['heat_carrier'] = True
    m_flag['cooling_carrier'] = True
    m_flag['hydrogen_carrier'] = False

    return m_flag


if __name__ == '__main__':
    multi_objective = True

    model_flag = set_model_flag()

    data_path = {
        'time_series': '../data/sest_uc/household_time_series_data.csv',
        'appliance_data': '../data/sest_uc/house_hold_appliance_data.csv',
    }
    household_data = HouseholdData(data_path)
    # (model, model_variables, model_variables_with_phase) = det_HEMS_model_fun(household_data, model_flag=model_flag)
    app_model = AppModelSet(household_data, model_flags=model_flag)
    if multi_objective:
        app_model.solve_model(multi_objective=multi_objective)
        model = app_model.house_model_defined.model
        moop = app_model.moop

        # Multi-attribute decision-making (MADM) using TOPSIS:
        pareto_sol = np.array(list(moop.unique_pareto_sols.keys()))
        normalized_pareto_sol = pareto_sol / np.linalg.norm(pareto_sol, axis=0)
        cost_w = 0.65
        co2_w = 1 - cost_w
        weight = np.array([cost_w, co2_w])
        weighted_normalized_pareto_sol = normalized_pareto_sol * weight
        A_minus = weighted_normalized_pareto_sol.max(axis=0)
        A_plus = weighted_normalized_pareto_sol.min(axis=0)
        S_i_plus = np.linalg.norm(weighted_normalized_pareto_sol - A_plus, axis=1)
        S_i_minus = np.linalg.norm(weighted_normalized_pareto_sol - A_minus, axis=1)
        C_i_plus = S_i_minus / (S_i_plus + S_i_minus)
        preferred_solution_indices = np.flip(np.argsort(C_i_plus))
        most_preferred_solution_index = preferred_solution_indices[0]
        most_preferred_solution_key = tuple(pareto_sol[most_preferred_solution_index])
        # extract the results for the most preferred solution:
        most_preferred_solution_results = moop.unique_pareto_sols[most_preferred_solution_key]

        # print('Pay-off table', moop.model.payoff)  # this prints the payoff table
        # print('Unique pareto solutions',
        #       np.array(list(moop.unique_pareto_sols.keys())))  # this prints the unique Pareto optimal solutions

        fig, axs = plt.subplots(1, figsize=[4, 1.5])
        co2 = np.array(list(moop.unique_pareto_sols.keys()))[:, 1]
        cost = np.array(list(moop.unique_pareto_sols.keys()))[:, 0]
        most_preferred_solution_rounded = tuple(round(i, 2) for i in most_preferred_solution_key)
        axs.scatter(co2, cost, marker='*')
        for i in range(len(co2)):
            if i == most_preferred_solution_index:
                axs.annotate(chr(65 + i) + '=' + str(most_preferred_solution_rounded[::-1]),
                             (co2[i], cost[i] * 1.04), fontsize=7, fontweight='bold', color='red',
                             ha='left', va='center')
                # axs.annotate('=' + str(most_preferred_solution_rounded), (cost[i] + 0.04, co2[i] + 0.08), fontsize=8, color='red', ha='center', va='center')
            else:
                axs.annotate(chr(65 + i), (co2[i], cost[i] * 1.04), fontsize=7, fontweight='bold', color='black',
                             ha='center', va='center')

        axs.set_xlabel('CO2 [kg]')
        axs.set_ylabel(f'Cost [€]')
        axs.set_ylim([cost.min() * 0.98, cost.max() * 1.05])
        axs.set_xlim([co2.min() * 0.98, co2.max() * 1.02])
        axs.grid(True)
        # axs.set_aspect('equal')
        # axs.set_title('Pareto Optimal Solutions')
        fig.set_tight_layout(True)
        fig.show()
        # fig.savefig(f'hems_results/Pareto.pdf')
    else:
        app_model.solve_model(multi_objective=multi_objective)
        model = app_model.house_model_defined.model
        most_preferred_solution_results = dict()

    electicity_consumption_list = {
        'NFEL': 'inflexible_power_demand',
        # 'FEL': 'p_flex',
        'EV': 'ev_power_charged',
        'BESS': 'storage_charging_power',
        'TSEL': 'tsl_power_consumption',
        'HP': 'heat_pump_power_to_heating',
    }
    electricity_consumption = pd.DataFrame(
        {i: most_preferred_solution_results[v] if v in most_preferred_solution_results.keys()
            else getattr(model, v).extract_values() for i, v in electicity_consumption_list.items()}
    )
    electricity_consumption.index = electricity_consumption.index.strftime('%Y-%m-%d %H:%M:%S')

    # flexible_electric_load_comparison_list = {
    #     'NFEL': 'inflexible_power_demand',
    #     'FEL': 'p_flex',
    # }
    # flexible_electric_load_comparison = pd.DataFrame(
    #     {i: most_preferred_solution_results[v] if v in most_preferred_solution_results.keys()
    #         else getattr(model, v).extract_values() for i, v in flexible_electric_load_comparison_list.items()}
    # )

    electricity_generation_used_list = {
        'p_import': 'electric_power_imported',
        'PV': 'pv_power_domestic_use',
        'EV': 'ev_domestic_power_used',
        'BESS': 'storage_domestic_power_used',
        # 'mCHP': 'chp_power_used'
    }
    electricity_generation_used = pd.DataFrame(
        {i: most_preferred_solution_results[v] if v in most_preferred_solution_results.keys()
            else getattr(model, v).extract_values() for i, v in electricity_generation_used_list.items()}
    )
    electricity_generation_used.index = electricity_generation_used.index.strftime('%Y-%m-%d %H:%M:%S')

    electricity_sold_list = {
        # 'p_sell': 'power_sell_ext_grid',
        # 'mCHP': 'chp_power_sell',
        'EV': 'ev_power_export',
        'PV': 'pv_power_export',
        'BESS': 'storage_power_export'
    }
    electricity_sold = pd.DataFrame(
        {i: most_preferred_solution_results[v] if v in most_preferred_solution_results.keys()
            else getattr(model, v).extract_values() for i, v in electricity_sold_list.items()}
    )
    electricity_sold.index = electricity_sold.index.strftime('%Y-%m-%d %H:%M:%S')


    heating_consumption_list = {
        'HESS': 'thermal_storage_charging_power',
        'DHW': 'domestic_hot_water_heating_demand',
        'SHD': 'house_thermal_system_heating'
    }
    heating_consumption = pd.DataFrame(
        {i: most_preferred_solution_results[v] if v in most_preferred_solution_results.keys()
            else getattr(model, v).extract_values() for i, v in heating_consumption_list.items()}
    )
    heating_consumption.index = heating_consumption.index.strftime('%Y-%m-%d %H:%M:%S')

    heating_generation_used_list = {
        'h_import': 'heating_power_imported',
        # 'mCHP': 'chp_heat_used',
        'HP': 'heat_pump_heat_used',
        # 'HP_ng': 'heat_pump_space_heating_from_natural_gas',
        'ST': 'st_heating_power_domestic_use',
        'HESS': 'thermal_storage_domestic_power_used'
    }
    heating_generation_used = pd.DataFrame(
        {i: most_preferred_solution_results[v] if v in most_preferred_solution_results.keys()
        else getattr(model, v).extract_values() for i, v in heating_generation_used_list.items()}
    )
    heating_generation_used.index = heating_generation_used.index.strftime('%Y-%m-%d %H:%M:%S')


    heat_sold_list = {
        # 'h_sell': 'heat_sell_ext_grid',
        # 'mCHP': 'chp_heat_sell',
        'ST': 'st_heating_power_export',
        'HESS': 'thermal_storage_power_export',
        'HP': 'heat_pump_heat_sold'
    }
    heat_sold = 1 * pd.DataFrame(
        {i: most_preferred_solution_results[v] if v in most_preferred_solution_results.keys()
            else getattr(model, v).extract_values() for i, v in heat_sold_list.items()}
    )
    heat_sold.index = heat_sold.index.strftime('%Y-%m-%d %H:%M:%S')

    gas_generation_list = {
        'g_import': 'natural_gas_imported'
    }
    gas_generation = pd.DataFrame(
        {i: most_preferred_solution_results[v] if v in most_preferred_solution_results.keys()
            else getattr(model, v).extract_values() for i, v in gas_generation_list.items()}
    )
    gas_generation.index = gas_generation.index.strftime('%Y-%m-%d %H:%M:%S')

    gas_consumption_list = {
        # 'HP_sh': 'heat_pump_natural_gas_to_space_heating',
        # 'HP_dhw': 'heat_pump_natural_gas_to_dhw_heating',
        'NGD': 'inflexible_gas_demand'
    }
    gas_consumption = pd.DataFrame(
        {i: most_preferred_solution_results[v] if v in most_preferred_solution_results.keys()
            else getattr(model, v).extract_values() for i, v in gas_consumption_list.items()}
    )
    gas_consumption.index = gas_consumption.index.strftime('%Y-%m-%d %H:%M:%S')

    soe_list = {
        # 'FEL': 'e_flex',
        'BESS': 'storage_soe',
        'EV': 'ev_soe',
        'HESS': 'thermal_storage_soe'
    }
    soe = pd.DataFrame(
        {i: most_preferred_solution_results[v] if v in most_preferred_solution_results.keys()
            else getattr(model, v).extract_values() for i, v in soe_list.items()}
    )
    soe.index = soe.index.strftime('%Y-%m-%d %H:%M:%S')

    power_buy_sell_list = {
        'p_import': 'electric_power_imported',
        'p_export': 'electric_power_exported'
    }
    power_buy_sell = 1 * pd.DataFrame(
        {i: most_preferred_solution_results[v] if v in most_preferred_solution_results.keys()
            else getattr(model, v).extract_values() for i, v in power_buy_sell_list.items()}
    )
    power_buy_sell.index = power_buy_sell.index.strftime('%Y-%m-%d %H:%M:%S')

    heat_buy_sell_list= {
        'h_import': 'heating_power_imported',
        'h_export': 'heating_power_exported'
    }
    heat_buy_sell = 1 * pd.DataFrame(
        {i: most_preferred_solution_results[v] if v in most_preferred_solution_results.keys()
            else getattr(model, v).extract_values() for i, v in heat_buy_sell_list.items()}
    )

    if not multi_objective:
        most_preferred_solution_results['house_temperature'] = model.house_temperature.extract_values()
    input_parameters_list = {
        'Solar irradiance': household_data.solar_irradiance/1000,
        'Ambient temperature': household_data.ambient_temperature,
        'House temperature': most_preferred_solution_results['house_temperature'],
        'Electricity': household_data.electricity_price,
        'Heat': household_data.heat_price,
        'Gas': household_data.gas_price
    }
    input_parameters = pd.DataFrame(input_parameters_list)
    input_parameters.index = input_parameters.index.strftime('%Y-%m-%d %H:%M:%S')


    fig = step_plot(input_parameters[['Ambient temperature', 'House temperature']],
                    df2=input_parameters['Solar irradiance'],
                    left_y_label='Temperature [°C]', right_y_label='Irradiance [kW/m2]')
    # fig.savefig(f'hems_results/Irradiance and temperature.svg')

    fig = step_plot(input_parameters[['Electricity']], df2=input_parameters['Gas'],
                    left_y_label='Price [€/kWh]', right_y_label='Price [€/m3]')
    # fig.savefig(f'hems_results/Energy price.svg')

    # electricity_generation_used.index =
    fig = aggregate_stacked_bar(
        electricity_generation_used,
        electricity_consumption,
        electricity_sold,
        title1='Domestic electricity generation, import and consumption',
        title2="Export to external grid",
        y_label="Power [kW]"
    )
    # fig.savefig(f'hems_results/Electricity_profiles.svg')

    fig = aggregate_stacked_bar(
        heating_generation_used,
        heating_consumption,
        heat_sold,
        title1='Domestic heating generation, import and consumption',
        title2="Export to external network",
        y_label="Heat [kW]")
    # fig.savefig(f'hems_results/Space_heating_profiles.svg')

    title = 'Gas domestic consumption and import'
    fig = stacked_bar(gas_generation, gas_consumption, title, y_label='Gas [m3]')
    # fig.savefig(f'hems_results/Gas_profiles.svg')

    fig = step_plot(soe[['BESS', 'EV']], title='Energy storage systems SOE', left_y_label='SOE_B [kWh]')
    # fig.savefig(f'hems_results/SOE_profiles.svg')

    # fig = step_plot(soe['FEL'], title='Flexible load SOE', left_y_label='SOE [kWh]')
#     fig.savefig(f'hems_results/Flexible load SOE.svg')

    # EV = pd.DataFrame({key: most_preferred_solution_results[key] for key in ['P_ev_ch', 'P_ev_dch', 'ev_soe']})
    # fig = step_plot(EV[['P_ev_ch', 'P_ev_dch']], EV['ev_soe'], left_y_label='Power [kW]', right_y_label='SOE [kWh]')
#     fig.savefig(f'hems_results/EV_SOE_profiles.svg')

    # BESS = pd.DataFrame({key: most_preferred_solution_results[key] for key in ['P_ess_ch', 'P_ess_dch', 'ess_soe']})
    # fig = step_plot(BESS[['P_ess_ch', 'P_ess_dch']], BESS['ess_soe'], left_y_label='Power [kW]', right_y_label='SOE [kWh]')
#     fig.savefig(f'hems_results/BESS_SOE_profiles.svg')

    # HESS = pd.DataFrame({key: most_preferred_solution_results[key] for key in ['h_tss_ch', 'h_tss_dch', 'tss_soe']})
    # fig = step_plot(HESS[['h_tss_ch', 'h_tss_dch']], HESS['tss_soe'], left_y_label='Heat [kW]', right_y_label='SOH [kWh]')
#     fig.savefig(f'hems_results/HESS_SOH_profiles.svg')

    fig = step_plot(power_buy_sell, title='Power bought or sold from/to the external grid', left_y_label='Power [kW]')
#     fig.savefig(f'hems_results/Power_buy_sell_profiles.svg')

    fig = step_plot(heat_buy_sell, title='Heat bought or sold from/to the external network', left_y_label='Heat [kW]')
    # fig.savefig(f'hems_results/Heat_buy_sell_profiles.svg')

#     all_BESS = pd.DataFrame({(key, column): moop.unique_pareto_sols[key][column] for key in moop.unique_pareto_sols.keys() for column in ['P_ess_ch', 'P_ess_dch', 'ess_soe']})
#     all_BESS.to_csv('hems_results/BESS all results.csv')
    # for i in model_variables.columns:
    #     my_plot(model_variables, y_lable=i)



    print('done')