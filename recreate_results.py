#-----------------------------------------------------------------------------------------------------------------------
#External imports
from pathlib import Path
import sys, os
from functools import partial
#-----------------------------------------------------------------------------------------------------------------------
#Internal imports
from LIDBagging.Plotting.Plots.SpiderCharts import *
from LIDBagging.Plotting.Plots.Tables import *
from LIDBagging.Plotting.Plots.VariablePlot import *
from LIDBagging.Plotting.Plots.VariableInteraction import *
from LIDBagging.Plotting.Plots.MSEbars import *
from LIDBagging.RunningEstimators.Running2 import *
from LIDBagging.run_files.geom_prog import *
from LIDBagging.run_files.parameter_combinations import *
from LIDBagging.run_files.error_safe_running import *
#-----------------------------------------------------------------------------------------------------------------------
#Setup directories
#PROJECT_ROOT = Path(r"C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2")
PROJECT_ROOT = Path(r"C:\Users\User\PycharmProjects\LID\LIDBagging2")
os.chdir(PROJECT_ROOT)
p = str(PROJECT_ROOT)
if p not in sys.path:
    sys.path.insert(0, p)
#-----------------------------------------------------------------------------------------------------------------------
#Setup computation
multiprocess=True
load=False
load_data=True
worker_count=5
save_name='result'
#directory = r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\pkls'
directory = r"C:\pkls"
#directory=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2'
#-----------------------------------------------------------------------------------------------------------------------
#Setup parameter dictionaries
effectiveness_test_general_param_dict = param_dicts_general(effectiveness_test_base_param_dict,
                                                            effectiveness_variants_test_types,
                                                            effectiveness_estimator_names,
                                                            test_name='effectiveness_test')
Nbag_test_general_param_dict = param_dicts_general(Nbag_test_base_param_dict, variable_variants_test_types,
                                                   variable_estimator_names, test_name='Nbag_barchart_test')
sr_test_general_param_dict = param_dicts_general(sr_prog_test_base_param_dict, variable_variants_test_types,
                                                 variable_estimator_names, test_name='sr_barchart_test')
interaction_sr_Nbag_test_general_param_dict = param_dicts_general(interaction_sr_Nbag_test_base_param_dict,
                                                                  variable_variants_test_types,
                                                                  variable_estimator_names,
                                                                  test_name='interaction_sr_Nbag_heatmap_test')
interaction_sr_k_test_general_param_dict = param_dicts_general(interaction_sr_k_test_base_param_dict,
                                                               variable_variants_test_types,
                                                               variable_estimator_names,
                                                               test_name='interaction_sr_k_heatmap_test')
if __name__ == "__main__":
    setup_logging()
    # -----------------------------------------------------------------------------------------------------------------------
    #Generate datasets (to be reused in different experiments)
    results_data = new_result_generator(param_dicts_data, multiprocess=False, load=False, load_data=False,
                                        worker_count=None,
                                        save_name='data_generation',
                                        directory=directory)
    print('Data generation complete')
    # -----------------------------------------------------------------------------------------------------------------------
    #MAIN Paper results

    #Effectiveness of bagging (baseline, simple bagging, weighted bagging, neighborhood size adjustment)
    #Bagging and smoothing (baseline, baseline with smoothing, simple bagging, simple bagging with pre or/and post smoothing)

    #Number of bags test (mse bar charts)

    #Sampling rate test (mse bar charts)

    #Interaction of sampling rate and number of bags (mse difference heatmaps)

    #Interaction of k and sampling rate (mse difference heatmaps)

    tasks = [
        ("effectiveness_test",
         partial(general_result_generator, effectiveness_test_general_param_dict, multiprocess=multiprocess, load=load,
                                  load_data=load_data, worker_count=worker_count,
                                  save_name=save_name, directory=directory), (), {}),
        ("Number_of_bags_test",
         partial(general_result_generator,Nbag_test_general_param_dict, multiprocess=multiprocess, load=load, load_data=load_data, worker_count=worker_count,
                             save_name=save_name, directory=directory), (), {}),
        ("Sampling_rate_test",
         partial(general_result_generator,sr_test_general_param_dict, multiprocess=multiprocess, load=load, load_data=load_data, worker_count=worker_count,
                             save_name=save_name, directory=directory), (), {}),
        ("Interaction_of_sampling_rate_and_number_of_bags_test",
         partial(general_result_generator,interaction_sr_Nbag_test_general_param_dict, multiprocess=multiprocess, load=load, load_data=load_data, worker_count=worker_count,
                             save_name=save_name, directory=directory), (), {}),
        ("Interaction_of_k_and_sampling_rate_and_number_of_bags_test",
         partial(general_result_generator,interaction_sr_k_test_general_param_dict, multiprocess=multiprocess, load=load, load_data=load_data, worker_count=worker_count,
                             save_name=save_name, directory=directory), (), {}),
    ]
    results = {}
    failures = {}
    for key, func, args, kwargs in tasks:
        ok, value = run_task_safely(func, *args, **kwargs)
        if ok:
            results[key] = value
        else:
            failures[key] = True
    # -----------------------------------------------------------------------------------------------------------------------

    result_effectiveness_test = results["effectiveness_test"]

    result_Number_of_bags_test = results["Number_of_bags_test"]

    result_Sampling_rate_test = results["Sampling_rate_test"]

    result_Interaction_of_sampling_rate_and_number_of_bags_test = results["Interaction_of_sampling_rate_and_number_of_bags_test"]

    result_Interaction_of_k_and_sampling_rate_and_k_test = results["Interaction_of_k_and_sampling_rate_and_number_of_bags_test"]

    # -----------------------------------------------------------------------------------------------------------------------

    plotting_across_results_dict(result_effectiveness_test, plot_radar_best_of_sweep, sweep_params=['k', 'sr'],
                                 normalize_data=True, log=False, save=True, height_per_row=450, width_per_col=450,
                                 verbose=False, save_dir="./plots/radar")

    plotting_across_results_dict(result_effectiveness_test, plot_table_best_of_sweep, sweep_params=['k', 'sr'],
                                 mode="combined", normalize_data=False, log=False, metric_label_map=None, save_dir="./plots/table")

    plotting_across_results_dict(result_Number_of_bags_test, plot_experiment_mse_bars, vary_param='Nbag', grid=True,
    figsize=(8, 8), base_fontsize=4, label_every=3, save_dir="./plots/msebar")

    plotting_across_results_dict(result_Sampling_rate_test, plot_experiment_mse_bars, vary_param='sr', grid=True,
    figsize=(8, 8), base_fontsize=4, label_every=3, save_dir="./plots/msebar")

    plotting_across_results_dict(result_Interaction_of_sampling_rate_and_number_of_bags_test, plot_experiment_heatmaps,
                                 x_param='sr', y_param='Nbag', reverse_x=False, reverse_y=False,
                                 metrics=("mse", "bias2", "var"), label_every=1, grid=True, figsize=(10, 8),
                                 base_fontsize=9, cmap="RdBu", save_dir="./plots/interaction", log=True,
                                 type='difference', inlog=False)

    plotting_across_results_dict(result_Interaction_of_k_and_sampling_rate_and_k_test, plot_experiment_heatmaps,
                                 x_param='sr', y_param='k', reverse_x=False, reverse_y=False,
                                 metrics=("mse", "bias2", "var"), label_every=1, grid=True, figsize=(10, 8),
                                 base_fontsize=9, cmap="RdBu", save_dir="./plots/interaction", log=True,
                                 type='difference', inlog=False)

    #-----------------------------------------------------------------------------------------------------------------------
    #Supplements extra results
    # -----------------------------------------------------------------------------------------------------------------------
