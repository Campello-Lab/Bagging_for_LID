from pathlib import Path
import sys, os

# Absolute path to your project root
#PROJECT_ROOT = Path(r"C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2")

PROJECT_ROOT = Path(r"C:\Users\User\PycharmProjects\LID\LIDBagging2")

# Optional: make it the current working directory (helps with relative file paths)
os.chdir(PROJECT_ROOT)

# Ensure it's first on the Python import search path
p = str(PROJECT_ROOT)
if p not in sys.path:
    sys.path.insert(0, p)

###################################################OWN IMPORT###################################################
from LIDBagging.Plotting.Plots.SpiderCharts import *
from LIDBagging.RunningEstimators.Running2 import *
######################################################################################################################################################################
import os, structlog
_devnull = open(os.devnull, "w")
structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=_devnull))

if __name__ == "__main__":
    #directory = r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\pkls'
    directory = r"C:\pkls"

    all = ['M1_Sphere', 'M2_Affine_3to5', 'M3_Nonlinear_4to6', 'M4_Nonlinear', 'M5b_Helix2d', 'M6_Nonlinear',
           'M7_Roll', 'M8_Nonlinear', 'M9_Affine', 'M10a_Cubic', 'M10b_Cubic', 'M10c_Cubic', 'M11_Moebius',
           'M12_Norm', 'M13a_Scurve', 'Mn1_Nonlinear', 'Mn2_Nonlinear', 'lollipop_', 'uniform']

    #all = all[0]

    sr_progression = [1, 0.8541315, 0.72954061, 0.62312362, 0.53222951, 0.45459399,
                      0.38828304, 0.33164478, 0.28326825, 0.24194833, 0.20665569,
                      0.17651114, 0.15076372, 0.12877204, 0.10998826, 0.09394443,
                      0.0802409, 0.06853628, 0.058539, 0.05]

    k_progression = [5, 6, 7, 8, 9, 11, 13, 15, 18, 21, 24, 28, 33, 39, 45, 53, 62, 73, 85, 100]

    n_progression = [300, 351, 411, 481, 564, 660, 773, 905, 1059, 1240, 1452, 1700, 1990, 2330, 2728, 3193, 3739,
                     4377,
                     5125, 6000]

    k_progression2 = [5, 6, 7, 8, 9, 11, 13, 15, 18, 21, 24, 28, 33, 39, 45, 53, 62, 73]

    n_progression2 = [300, 351, 411, 481, 564, 660, 773, 905, 1059, 1240, 1452, 1700, 1990, 2330, 2728, 3193, 3739,
                      4377]

    k_prog = [5, 7, 10, 14, 19, 26, 37, 51, 72, 100]

    lid_progression = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]

    Nbag_progression = [30, 40, 50, 60, 70, 80, 90, 100]

    Nbag_progression2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    sr_progression2 = [1.0, 0.5, 0.3333333333333333, 0.25, 0.2, 0.16666666666666666, 0.14285714285714285, 0.125,
                       0.1111111111111111, 0.1, 0.09090909090909091, 0.08333333333333333, 0.07692307692307693,
                       0.07142857142857142, 0.06666666666666667, 0.0625, 0.058823529411764705, 0.05555555555555555,
                       0.05263157894736842, 0.05]

    Nbag_progression3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    sr_progression3 = [1.0, 0.7071067811865475, 0.5773502691896258, 0.5, 0.4472135954999579, 0.4082482904638631,
                       0.3779644730092272,
                       0.35355339059327373, 0.3333333333333333, 0.31622776601683794, 0.30151134457776363,
                       0.2886751345948129,
                       0.2773500981126146, 0.2672612419124244, 0.2581988897471611, 0.25, 0.24253562503633297,
                       0.23570226039551587,
                       0.22941573387056174, 0.22360679774997896]
    Nbag_progression4 = [3, 4, 7, 10, 14, 21, 31, 46, 68, 100]
    sr_progression4 = [0.15811388300841897, 0.13571572071617163, 0.11516784446059201, 0.09687806066630292,
                       0.08096531670250254,
                       0.06734994988992811, 0.055838632982812095, 0.04618768723269577, 0.03814361187173469,
                       0.03146583877637763]

    Nbag5 = [3, 4, 5, 6, 8, 11, 14, 18, 24, 30, 39, 51, 66, 85, 110, 143, 185, 239, 309, 400]

    k_prog = [5, 7, 10, 14, 19, 26, 37, 51, 72]

    # k_prog = [5, 7, 10, 14]

    #k_prog = [5, 10, 19, 37, 72]

    #k_prog = [k_prog[0]]

    # sr_prog = [0.11538461538461539, 0.08108108108108109, 0.058823529411764705, 0.041666666666666664]

    sr_prog = [0.6, 0.42857142857142855, 0.3, 0.21428571428571427, 0.15789473684210525, 0.11538461538461539,
               0.08108108108108109, 0.058823529411764705, 0.041666666666666664]

    #sr_prog = [0.6, 0.3, 0.15789473684210525, 0.08108108108108109, 0.041666666666666664]
    #sr_prog = [sr_prog[0]]

    param_dicts7 = {'dataset_name': all,
                    'n': 2500,
                    'lid': None,
                    'dim': None,
                    'estimator_name': 'mle',
                    'bagging_method': ['bag'], #, 'approx_bagwth'
                    'submethod_0': '0',
                    'submethod_error': 'log_diff',
                    'k': k_prog,
                    'sr': sr_prog,
                    'Nbag': 10,
                    'pre_smooth': False,
                    'post_smooth': False,
                    't': 1}

    param_dicts_data = {'dataset_name': all,
                        'n': 2500,
                        'lid': None,
                        'dim': None,
                        'estimator_name': 'mle',
                        'bagging_method': None,
                        'submethod_0': '0',
                        'submethod_error': 'log_diff',
                        'k': 10,
                        'sr': 0.3,
                        'Nbag': 10,
                        'pre_smooth': False,
                        'post_smooth': False,
                        't': 1}

    param_dicts_mle1 = {'dataset_name': all,
                     'n': 2500,
                     'lid': None,
                     'dim': None,
                     'estimator_name': 'mle',
                     'bagging_method': None,
                     'submethod_0': '0',
                     'submethod_error': 'log_diff',
                     'k': k_prog,
                     'sr': sr_prog,
                     'Nbag': 10,
                     'pre_smooth': False,
                     'post_smooth': [True, False],
                     't': 1}

    param_dicts_mle2 = {'dataset_name': all,
                     'n': 2500,
                     'lid': None,
                     'dim': None,
                     'estimator_name': 'mle',
                     'bagging_method': 'bag',
                     'submethod_0': '0',
                     'submethod_error': 'log_diff',
                     'k': k_prog,
                     'sr': sr_prog,
                     'Nbag': 10,
                     'pre_smooth': [True, False],
                     'post_smooth': [True, False],
                     't': 1}

    param_dicts_mle = [param_dicts_mle1, param_dicts_mle2]

    param_dicts_tle1 = {'dataset_name': all,
                     'n': 2500,
                     'lid': None,
                     'dim': None,
                     'estimator_name': 'tle',
                     'bagging_method': None,
                     'submethod_0': '0',
                     'submethod_error': 'log_diff',
                     'k': k_prog,
                     'sr': sr_prog,
                     'Nbag': 10,
                     'pre_smooth': False,
                     'post_smooth': [True, False],
                     't': 1}

    param_dicts_tle2 = {'dataset_name': all,
                     'n': 2500,
                     'lid': None,
                     'dim': None,
                     'estimator_name': 'tle',
                     'bagging_method': 'bag',
                     'submethod_0': '0',
                     'submethod_error': 'log_diff',
                     'k': k_prog,
                     'sr': sr_prog,
                     'Nbag': 10,
                     'pre_smooth': [True, False],
                     'post_smooth': [True, False],
                     't': 1}

    param_dicts_tle = [param_dicts_tle1, param_dicts_tle2]

    param_dicts_mada1 = {'dataset_name': all,
                     'n': 2500,
                     'lid': None,
                     'dim': None,
                     'estimator_name': 'mada',
                     'bagging_method': None,
                     'submethod_0': '0',
                     'submethod_error': 'log_diff',
                     'k': k_prog,
                     'sr': sr_prog,
                     'Nbag': 10,
                     'pre_smooth': False,
                     'post_smooth': [True, False],
                     't': 1}

    param_dicts_mada2 = {'dataset_name': all,
                     'n': 2500,
                     'lid': None,
                     'dim': None,
                     'estimator_name': 'mada',
                     'bagging_method': 'bag',
                     'submethod_0': '0',
                     'submethod_error': 'log_diff',
                     'k': k_prog,
                     'sr': sr_prog,
                     'Nbag': 10,
                     'pre_smooth': [True, False],
                     'post_smooth': [True, False],
                     't': 1}

    param_dicts_mada = [param_dicts_mada1, param_dicts_mada2]

    ####################################################################################TrueExpStarts####################################################xx


    results_data = new_result_generator(param_dicts_data, multiprocess=False, load=True, load_data=True,
                                        worker_count=None,
                                        save_name='data_generation',
                                        directory=directory)
    print('data generation complete')

    # SMOOTHING
    #################################

    results_mle = new_result_generator(param_dicts_mle, multiprocess=True, load=True, load_data=True, worker_count=7,
                                     save_name='skdim_radar_chart_k_sr_mle_smooth',
                                     directory=directory)

    plot_radar_best_of_sweep(results_mle, sweep_params=['k', 'sr'], normalize_data=True, log=False, save=True,
                             save_prefix="skdim_radar_best_mle_smooth", height_per_row=450, width_per_col=450,
                             verbose=False)

    plot_table_best_of_sweep(results_mle, sweep_params=['k', 'sr'], mode="combined", normalize_data=False, log=False,
                             metric_label_map=None, save_prefix="skdim_table_best_mle_smooth")

    results_mada = new_result_generator(param_dicts_mada, multiprocess=True, load=True, load_data=True, worker_count=7,
                                     save_name='skdim_radar_chart_k_sr_mada_smooth',
                                     directory=directory)

    plot_radar_best_of_sweep(results_mada, sweep_params=['k', 'sr'], normalize_data=True, log=False, save=True,
                             save_prefix="skdim_radar_best_mada_smooth", height_per_row=450, width_per_col=450,
                             verbose=False)

    plot_table_best_of_sweep(results_mada, sweep_params=['k', 'sr'], mode="combined", normalize_data=False, log=False,
                             metric_label_map=None, save_prefix="skdim_table_best_mada_smooth")

    results_tle = new_result_generator(param_dicts_tle, multiprocess=True, load=True, load_data=True, worker_count=7,
                                       save_name='skdim_radar_chart_k_sr_tle_smooth',
                                       directory=directory)

    plot_radar_best_of_sweep(results_tle, sweep_params=['k', 'sr'], normalize_data=True, log=False, save=True,
                             save_prefix="skdim_radar_best_tle_smooth", height_per_row=450, width_per_col=450,
                              verbose=False)

    plot_table_best_of_sweep(results_tle, sweep_params=['k', 'sr'], mode="combined", normalize_data=False, log=False,
                             metric_label_map=None, save_prefix="skdim_table_best_tle_smooth")

    #Weighted Variants

    print('Smoothing Done, Starting Weighted Variants')

    results_mle = new_result_generator(param_dicts_mle, multiprocess=True, load=True, load_data=True, worker_count=7,
                                     save_name='skdim_radar_chart_k_sr_mle',
                                     directory=directory)

    plot_radar_best_of_sweep(results_mle, sweep_params=['k', 'sr'], normalize_data=True, log=False, save=True,
                             save_prefix="skdim_radar_best_mle", height_per_row=450, width_per_col=450,
                             verbose=False)

    plot_table_best_of_sweep(results_mle, sweep_params=['k', 'sr'], mode="combined", normalize_data=False, log=False,
                             metric_label_map=None, save_prefix="skdim_table_best_mle")

    results_tle = new_result_generator(param_dicts_tle, multiprocess=True, load=True, load_data=True, worker_count=7,
                                       save_name='skdim_radar_chart_k_sr_tle',
                                       directory=directory)

    plot_radar_best_of_sweep(results_tle, sweep_params=['k', 'sr'], normalize_data=True, log=False, save=True,
                             save_prefix="skdim_radar_best_tle", height_per_row=450, width_per_col=450,
                              verbose=False)

    plot_table_best_of_sweep(results_tle, sweep_params=['k', 'sr'], mode="combined", normalize_data=False, log=False,
                             metric_label_map=None, save_prefix="skdim_table_best_tle")

    results_mada = new_result_generator(param_dicts_mada, multiprocess=True, load=False, load_data=True, worker_count=5,
                                     save_name='skdim_radar_chart_k_sr_mada',
                                     directory=directory)

    plot_radar_best_of_sweep(results_mada, sweep_params=['k', 'sr'], normalize_data=True, log=False, save=True,
                             save_prefix="skdim_radar_best_mada", height_per_row=450, width_per_col=450,
                             verbose=False)

    plot_table_best_of_sweep(results_mada, sweep_params=['k', 'sr'], mode="combined", normalize_data=False, log=False,
                             metric_label_map=None, save_prefix="skdim_table_best_mada")

    # Plotting
    #################################

'''
    #plot_experiment_mse_bars(results, vary_param='Nbag', figsize=(30, 36), base_fontsize=13, label_every=1, save_name="new_bar_plot_NEW_Nbag")




    plot_experiment_heatmaps(experiments=results, x_param='sr', y_param='k', reverse_x=False, reverse_y=False,
                         metrics=("mse", "bias2", "var"), label_every=1, grid=True, figsize=(30, 30),
                         base_fontsize=9,
                         cmap="RdBu", save_name="sr_vs_Nbag", save_dir="./plots/interaction", log=True, type='difference',
                         inlog=False)

    plot_experiment_heatmaps(experiments=results, x_param='sr', y_param='Nbag', reverse_x=False, reverse_y=False,
                             metrics=("mse", "bias2", "var"), label_every=1, grid=True, figsize=(30, 30),
                             base_fontsize=9,
                             cmap="RdBu", save_name="sr_vs_Nbag", save_dir="./plots/interaction", log=False, type='difference',
                             inlog=False)

    plot_experiment_heatmaps(experiments=results, x_param='sr', y_param='Nbag', reverse_x=False, reverse_y=False,
                             metrics=("mse", "bias2", "var"), label_every=1, grid=True, figsize=(30, 30),
                             base_fontsize=9,
                             cmap="RdBu", save_name="sr_vs_Nbag", save_dir="./plots/interaction", log=True,
                             type='bagged',
                             inlog=False)

    plot_experiment_heatmaps(experiments=results, x_param='sr', y_param='Nbag', reverse_x=False, reverse_y=False,
                             metrics=("mse", "bias2", "var"), label_every=1, grid=True, figsize=(30, 30),
                             base_fontsize=9,
                             cmap="RdBu", save_name="sr_vs_Nbag", save_dir="./plots/interaction", log=False,
                             type='bagged',
                             inlog=False)

    plot_experiment_heatmaps(experiments=results, x_param='sr', y_param='Nbag', reverse_x=False, reverse_y=False,
                             metrics=("mse", "bias2", "var"), label_every=1, grid=True, figsize=(30, 30),
                             base_fontsize=9,
                             cmap="RdBu", save_name="sr_vs_Nbag", save_dir="./plots/interaction", log=True,
                             type='baseline',
                             inlog=False)

    plot_experiment_heatmaps(experiments=results, x_param='sr', y_param='Nbag', reverse_x=False, reverse_y=False,
                             metrics=("mse", "bias2", "var"), label_every=1, grid=True, figsize=(30, 30),
                             base_fontsize=9,
                             cmap="RdBu", save_name="sr_vs_Nbag", save_dir="./plots/interaction", log=False,
                             type='baseline',
                             inlog=False)
'''