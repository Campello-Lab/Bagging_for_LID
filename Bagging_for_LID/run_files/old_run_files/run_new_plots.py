###################################################OWN IMPORT###################################################
from LIDBagging.RunningEstimators.Running2 import *
from LIDBagging.Plotting.Plots.VariablePlot import *

######################################################################################################################################################################

if __name__ == "__main__":
    all = ['M1_Sphere', 'M2_Affine_3to5', 'M3_Nonlinear_4to6', 'M4_Nonlinear', 'M5b_Helix2d', 'M6_Nonlinear',
                'M7_Roll', 'M8_Nonlinear', 'M9_Affine', 'M10a_Cubic', 'M10b_Cubic', 'M10c_Cubic', 'M11_Moebius',
                'M12_Norm', 'M13a_Scurve', 'Mn1_Nonlinear', 'Mn2_Nonlinear', 'lollipop_', 'uniform']

    sr_progression = [1, 0.8541315, 0.72954061, 0.62312362, 0.53222951, 0.45459399,
                      0.38828304, 0.33164478, 0.28326825, 0.24194833, 0.20665569,
                      0.17651114, 0.15076372, 0.12877204, 0.10998826, 0.09394443,
                      0.0802409,  0.06853628, 0.058539, 0.05]

    k_progression = [5, 6, 7, 8, 9, 11, 13, 15, 18, 21, 24, 28, 33, 39, 45, 53, 62, 73, 85, 100]

    n_progression = [300, 351, 411, 481, 564, 660, 773, 905, 1059, 1240, 1452, 1700, 1990, 2330, 2728, 3193, 3739, 4377, 5125, 6000]

    k_progression2 = [5, 6, 7, 8, 9, 11, 13, 15, 18, 21, 24, 28, 33, 39, 45, 53, 62, 73]

    n_progression2 = [300, 351, 411, 481, 564, 660, 773, 905, 1059, 1240, 1452, 1700, 1990, 2330, 2728, 3193, 3739, 4377]

    lid_progression = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]

    param_dicts1 = {'dataset_name': all,
                    'n': 2500,
                    'lid': None,
                    'dim': None,
                    'estimator_name': 'mle',
                    'bagging_method': ['bagw', 'bagwth'],
                    'submethod_0': '0',
                    'submethod_error': ['log_diff'],
                    'k': [5 + i*2 for i in range(20)],
                    'sr': 0.3,
                    'Nbag': 10,
                    'pre_smooth': False,
                    'post_smooth': False,
                    't': 2}

    param_dicts2 = {'dataset_name': all,
                    'n': 2500,
                    'lid': None,
                    'dim': None,
                    'estimator_name': 'mle',
                    'bagging_method': [None, 'bag'],
                    'submethod_0': '0',
                    'submethod_error': 'log_diff',
                    'k': [5 + i*2 for i in range(20)],
                    'sr': 0.3,
                    'Nbag': 10,
                    'pre_smooth': False,
                    'post_smooth': False,
                    't': 2}

    param_dicts3 = {'dataset_name': all,
                    'n': 2500,
                    'lid': None,
                    'dim': None,
                    'estimator_name': 'mle',
                    'bagging_method': 'bag',
                    'submethod_0': '0',
                    'submethod_error': 'log_diff',
                    'k': k_progression,
                    'sr': sr_progression,
                    'Nbag': 10,
                    'pre_smooth': False,
                    'post_smooth': False,
                    't': 2}

    param_dicts4 = {'dataset_name': all,
                    'n': 2500,
                    'lid': None,
                    'dim': None,
                    'estimator_name': 'mle',
                    'bagging_method': None,
                    'submethod_0': '0',
                    'submethod_error': 'log_diff',
                    'k': k_progression,
                    'sr': 1,
                    'Nbag': 10,
                    'pre_smooth': False,
                    'post_smooth': False,
                    't': 2}

    param_dicts5 = {'dataset_name': all,
                    'n': n_progression2,
                    'lid': None,
                    'dim': None,
                    'estimator_name': 'mle',
                    'bagging_method': 'bag',
                    'submethod_0': '0',
                    'submethod_error': 'log_diff',
                    'k': k_progression2,
                    'sr': 0.3,
                    'Nbag': 10,
                    'pre_smooth': False,
                    'post_smooth': False,
                    't': 2}

    param_dicts6 = {'dataset_name': all,
                    'n': n_progression2,
                    'lid': None,
                    'dim': None,
                    'estimator_name': 'mle',
                    'bagging_method': None,
                    'submethod_0': '0',
                    'submethod_error': 'log_diff',
                    'k': k_progression2,
                    'sr': 1,
                    'Nbag': 10,
                    'pre_smooth': False,
                    'post_smooth': False,
                    't': 2}

    param_dicts = [param_dicts5, param_dicts6]

    #results3 = new_result_generator(param_dicts3, multiprocess=True, load=True, load_data=True, worker_count=None, save_name='res3', directory=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2')
    #results4 = new_result_generator(param_dicts4, multiprocess=True, load=True, load_data=True, worker_count=None, save_name='res4', directory=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2')
    #results5 = new_result_generator(param_dicts5, multiprocess=True, load=True, load_data=True, worker_count=None, save_name='res5', directory=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2')
    results_mada_lid = new_result_generator(param_dicts, multiprocess=True, load=True, load_data=True, worker_count=None, save_name='compare_mada_lid', directory=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2')
    results_mle_lid = new_result_generator(param_dicts, multiprocess=True, load=True, load_data=True, worker_count=None, save_name='compare_mle_lid', directory=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2')
    results_mle_lid_sm = new_result_generator(param_dicts, multiprocess=True, load=True, load_data=True, worker_count=None, save_name='compare_mle_lid_sm', directory=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2')
    results_mle_t = new_result_generator(param_dicts, multiprocess=True, load=True, load_data=True, worker_count=None, save_name='compare_mle_t', directory=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2')
    results_mada_t = new_result_generator(param_dicts, multiprocess=True, load=True, load_data=True, worker_count=None, save_name='compare_mada_t', directory=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2')
    results_mada_sr = new_result_generator(param_dicts, multiprocess=True, load=True, load_data=True, worker_count=None, save_name='compare_mada_sr', directory=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2')
    results_mle_sr_sm = new_result_generator(param_dicts, multiprocess=True, load=True, load_data=True, worker_count=None, save_name='compare_mle_sr_sm', directory=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2')
    results_mle_sr = new_result_generator(param_dicts, multiprocess=True, load=True, load_data=True, worker_count=None, save_name='compare_mle_sr', directory=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2')

    plot_experiment_metric_curves(
        experiments=results_mada_sr,
        vary_param="sr",
        log=True,
        label_every=1,
        figsize=(16, 15),
        save_name="compare_mada_sr",
        base_fontsize=5,
        save_dir="./plots/1var"
    )

    plot_experiment_metric_curves(
        experiments=results_mle_sr,
        vary_param="sr",
        log=True,
        label_every=1,
        figsize=(16, 15),
        save_name="compare_mle_sr",
        base_fontsize=5,
        save_dir="./plots/1var"
    )

    plot_experiment_metric_curves(
        experiments=results_mle_sr_sm,
        vary_param="sr",
        log=True,
        label_every=1,
        figsize=(16, 15),
        save_name="compare_mle_sr_sm",
        base_fontsize=5,
        save_dir="./plots/1var"
    )

    plot_experiment_metric_curves(
        experiments=results_mada_t,
        vary_param="t",
        log=True,
        label_every=1,
        figsize=(16, 15),
        save_name="compare_mada_t",
        base_fontsize=5,
        save_dir="./plots/1var"
    )

    plot_experiment_metric_curves(
        experiments=results_mle_t,
        vary_param="t",
        log=True,
        label_every=1,
        figsize=(16, 15),
        save_name="compare_mle_t",
        base_fontsize=5,
        save_dir="./plots/1var"
    )

    plot_experiment_metric_curves(
        experiments=results_mada_lid,
        vary_param="lid",
        log=True,
        label_every=1,
        figsize=(10, 8),
        save_name="compare_mada_lid",
        base_fontsize=9,
        save_dir="./plots/1var"
    )

    plot_experiment_metric_curves(
        experiments=results_mle_lid,
        vary_param="lid",
        log=True,
        label_every=1,
        figsize=(10, 8),
        save_name="compare_mle_lid",
        base_fontsize=9,
        save_dir="./plots/1var"
    )

    plot_experiment_metric_curves(
        experiments=results_mle_lid_sm,
        vary_param="lid",
        log=True,
        label_every=1,
        figsize=(10, 8),
        save_name="compare_mle_lid_sm",
        base_fontsize=9,
        save_dir="./plots/1var"
    )

    '''
        plot_experiment_heatmaps(experiments=results, x_param='lid', y_param='n', reverse_x=False, reverse_y=False,
    metrics = ("mse", "bias2", "var"), label_every=1, grid=True, figsize=(10, 8), base_fontsize=9,
    cmap="RdBu", save_name="n_vs_lid", save_dir="./plots/interaction", log=False, type='difference', inlog=False)

    plot_experiment_heatmaps(experiments=results, x_param='lid', y_param='n', reverse_x=False, reverse_y=False,
    metrics = ("mse", "bias2", "var"), label_every=1, grid=True, figsize=(10, 8), base_fontsize=9,
    cmap="RdBu", save_name="n_vs_lid", save_dir="./plots/interaction", log=True, type='difference', inlog=False)

    plot_experiment_heatmaps(experiments=results, x_param='lid', y_param='n', reverse_x=False, reverse_y=False,
    metrics = ("mse", "bias2", "var"), label_every=1, grid=True, figsize=(10, 8), base_fontsize=9,
    cmap="RdBu", save_name="n_vs_lid", save_dir="./plots/interaction", log=False, type='baseline', inlog=False)

    plot_experiment_heatmaps(experiments=results, x_param='lid', y_param='n', reverse_x=False, reverse_y=False,
    metrics = ("mse", "bias2", "var"), label_every=1, grid=True, figsize=(10, 8), base_fontsize=9,
    cmap="RdBu", save_name="n_vs_lid", save_dir="./plots/interaction", log=False, type='bagged', inlog=False)

    plot_experiment_heatmaps(experiments=results, x_param='lid', y_param='n', reverse_x=False, reverse_y=False,
                             metrics=("mse", "bias2", "var"), label_every=1, grid=True, figsize=(10, 8),
                             base_fontsize=9,
                             cmap="RdBu", save_name="n_vs_lid", save_dir="./plots/interaction", log=True,
                             type='baseline', inlog=False)

    plot_experiment_heatmaps(experiments=results, x_param='lid', y_param='n', reverse_x=False, reverse_y=False,
                             metrics=("mse", "bias2", "var"), label_every=1, grid=True, figsize=(10, 8),
                             base_fontsize=9,
                             cmap="RdBu", save_name="n_vs_lid", save_dir="./plots/interaction", log=True, type='bagged',
                             inlog=False)
                             
    '''

'''
    plot_experiment_mse_bars(
        experiments=results6[0],
        vary_param='bagging_method',
        figsize=(8, 8),
        base_fontsize=4,
        label_every=1,
        save_name="NEW_bagging_method_bar_plot")

    plot_experiment_mse_bars(
        experiments=results3[0],
        vary_param='lid',
        figsize=(8, 8),
        base_fontsize=4,
        label_every=3,
        save_name="NEW_lid_bar_plot")

    plot_experiment_mse_bars(
        experiments=results4[0],
        vary_param='k',
        figsize=(8, 8),
        base_fontsize=4,
        label_every=3,
        save_name="NEW_k_bar_plot")

    plot_experiment_mse_bars(
        experiments=results5[0],
        vary_param='Nbag',
        figsize=(8, 8),
        base_fontsize=4,
        label_every=3,
        save_name="NEW_Nbag_bar_plot")
'''