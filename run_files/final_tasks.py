#-----------------------------------------------------------------------------------------------------------------------
#External imports
from functools import partial
#-----------------------------------------------------------------------------------------------------------------------
#Internal Imports
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
#Function to collect test runs
def setup_tasks(task_dict, multiprocess=False, load=False, load_data=False, worker_count=1, save_name='result', directory=r"C:\pkls"):
    tasks = []
    for key, value in task_dict.items():
        tasks.append((key, partial(general_result_generator, value, multiprocess=multiprocess, load=load, load_data=load_data, worker_count=worker_count, save_name=save_name, directory=directory), (), {}))
    return tasks

#-----------------------------------------------------------------------------------------------------------------------
#Setup parameter dictionaries for main tests
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
#-----------------------------------------------------------------------------------------------------------------------
#Setup plotting
plot_tasks = {
    "effectiveness_test": [
        (plot_radar_best_of_sweep, dict(sweep_params=['k', 'sr'], normalize_data=True, log=False,
                                        save=True, height_per_row=450, width_per_col=450,
                                        verbose=False, save_dir="./plots/radar")),
        (plot_table_best_of_sweep, dict(sweep_params=['k', 'sr'], mode="combined", normalize_data=False,
                                        log=False, metric_label_map=None, save_dir="./plots/table")),
    ],
    "Number_of_bags_test": [
        (plot_experiment_mse_bars, dict(vary_param='Nbag', grid=True, figsize=(12, 12),
                                        base_fontsize=6, label_every=1, save_dir="./plots/msebar")),
    ],
    "Sampling_rate_test": [
        (plot_experiment_mse_bars, dict(vary_param='sr', grid=True, figsize=(12, 12),
                                        base_fontsize=6, label_every=1, save_dir="./plots/msebar")),
    ],
    "Interaction_of_sampling_rate_and_number_of_bags_test": [
        (plot_experiment_heatmaps, dict(x_param='sr', y_param='Nbag', reverse_x=False, reverse_y=False,
                                        metrics=("mse", "bias2", "var"), label_every=1, grid=True,
                                        figsize=(25, 25), base_fontsize=10, cmap="RdBu",
                                        save_dir="./plots/interaction",
                                        log=True, type='difference', inlog=False, fig_title='Interaction of sampling rate and number of bags heatmap. \nBaseline Estimator: MLE')),
    ],
    "Interaction_of_k_and_sampling_rate_test": [
        (plot_experiment_heatmaps, dict(x_param='sr', y_param='k', reverse_x=False, reverse_y=False,
                                        metrics=("mse", "bias2", "var"), label_every=1, grid=True,
                                        figsize=(25, 25), base_fontsize=10, cmap="RdBu",
                                        save_dir="./plots/interaction",
                                        log=True, type='difference', inlog=False, fig_title='Interaction of sampling rate and k heatmap. \nBaseline Estimator: MLE')),
    ],
}