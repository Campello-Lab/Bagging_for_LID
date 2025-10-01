###################################################OWN IMPORT###################################################
from LIDBagging.Datasets.DatasetGeneration import *
from LIDBagging.Helper.Other import *
from LIDBagging.Helper.ComparrisonMeasures import *
from LIDBagging.RunningEstimators.BaseEstimators import *
from LIDBagging.RunningEstimators.Collecting import *
from LIDBagging.RunningEstimators.Running import *
from LIDBagging.Plotting.new_plots import *
from LIDBagging.Plotting.Plots.K_plots import *
from LIDBagging.Plotting.Plots.BarPlots import *
from LIDBagging.Plotting.Plots.KNN_Graph import *
from LIDBagging.Plotting.Plots.LocalPlot import *
from LIDBagging.Plotting.Plots.SpiderCharts import *
##############################################################################################################################################################################################################################################################

if __name__ == "__main__":
    interesting_low_dim = ['M7_Roll', 'M11_Moebius', 'M13a_Scurve', 'swiss_roll', 'torus_circle', 'M5b_Helix2d',
                           'lollipop_']
    medium_uniform_lid = ['M1_Sphere', 'M2_Affine_3to5', 'M3_Nonlinear_4to6', 'M4_Nonlinear', 'M6_Nonlinear',
                          'M8_Nonlinear', 'M9_Affine', 'M10a_Cubic', 'M10b_Cubic', 'M10c_Cubic', 'M12_Norm',
                          'Mn1_Nonlinear', 'Mn2_Nonlinear', 'affine_10D_5d_uniform', 'affine_10D_5d_laplace',
                          'affine_10D_5d_gaussian']
    medium_lid_unions = ['affine_10D_2d_4d_8d_gaussian', 'affine_10D_2d_4d_8d_laplace', 'affine_10D_2d_4d_8d_uniform',
                         'squiggly_05_freq_10D_2d_4d_8d_uniform', 'squiggly_10_freq_10D_2d_4d_8d_uniform',
                         'squiggly_1_freq_10D_2d_4d_8d_uniform', 'squiggly_5_freq_10D_2d_4d_8d_uniform']
    #param_list = [([i+2 for i in range(50)], [10], [0.1], ['mle'], [''])] #, 'smooth', 'smooth_geo', 'bag', 'bag_w_0', 'bag_w_bag'
    #param_list2 = [([i+2 for i in range(50)], [10], [0.5], ['mle'], ['bag_w_0'])]
    #param_list = [([i+2 for i in range(50)], [10], [0.3], ['mle'], ['', 'bag', 'bag_w_0'])] #, 'smooth', 'smooth_geo', 'bag', 'bag_w_0', 'bag_w_bag' #32
    #param_list2 = [([(i + 4)*2 for i in range(10)], [20], [0.1], ['mle'], [''])]
    #param_list2 = [([26], [10], [0.1], ['mle'], ['bag_w_0'])]
    #param_list = [([10], [10], [0.1], ['mle', 'mom', 'tle'], ['', 'smooth', 'bag', 'bag_w_0'])] #['mle', 'mom', 'tle', 'mada', 'ess']
    #param_combs2 = generate_param_combinations(param_list2)
    #param_combs = param_combs+param_combs2

    #param_list = [([2 * i + 5 for i in range(50)], [10], [0.3], ['mle'], ['', 'smooth', 'bag_f_f', 'bag_t_f', 'bag_f_t', 'bag_t_t'])]
    #param_list = [([10], [10], [0.3], ['ess'], ['', 'smooth', 'bag_f_f', 'bag_t_f', 'bag_f_t', 'bag_t_t'])]
    param_list = [([10], [10], [0.3], ['mle', 'tle', 'mada', '2nn', 'ess'], ['', 'bag_f_f', 'bag_w_1_n_f_f_0', 'bag_w_1_y_f_f_0'])] #'', 'bag_f_f', 'bag_w_1_n_f_f_0',
    #param_list = [([10], [10], [0.3], ['mle', 'tle', 'mada'], ['', 'smooth', 'bag_f_f', 'bag_t_f', 'bag_f_t', 'bag_t_t'])]
    #param_list = [([10], [10], [0.02+i*0.02 for i in range(50)], ['mle'], ['bag_f_f'])]
    param_combs = generate_param_combinations(param_list)

    #original1 = ['M10a_Cubic']
    all = ['M1_Sphere', 'M2_Affine_3to5', 'M3_Nonlinear_4to6', 'M4_Nonlinear', 'M5b_Helix2d', 'M6_Nonlinear',
                'M7_Roll', 'M8_Nonlinear', 'M9_Affine', 'M10a_Cubic', 'M10b_Cubic', 'M10c_Cubic', 'M11_Moebius',
                'M12_Norm', 'M13a_Scurve', 'Mn1_Nonlinear', 'Mn2_Nonlinear', 'lollipop_', 'uniform']
    save_name='new_radar_chart'
    dir = 'C:\\Users\\User\\PycharmProjects\\pythonProject3\\LIDstuff\\saved_results\\pkls2\\'
    load_path1 = f'{dir}{save_name}'
    data_load_path = r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2\n_2500_full'
    results, data_sets = result_generator(n=1000, param_list=param_combs, save=True, load=True, load_path=load_path1,
                                            data_load_path=data_load_path, save_name=save_name, usedata1=None,
                                            usedata2=all,
                                            test_types=None, bounddict=None, load_data=False, reduce_worker_count=1)
    result_dictionaries, dict_names = convert_results_for_plot(results)
    create_method_variant_radar_charts(data_sets, dictionaries=result_dictionaries, names=dict_names, normalize_data=True,
                                       save=True, save_prefix=save_name+'log', fill=True, height_per_row=450, width_per_col=450, log=True)
    #plot_k(result_dictionaries, dict_names, save_name=save_name, show=False, log=True, allowed_methods=None, partial_key=None, adjust_k=10)
    #plot_lid_results(datasets=data_sets, results=results, r=1.0, figsize=(8,8), show=False,
    #                 save_dir = r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots',
    #                 save_name='NEW_lollipop_tests1_2d_plot_100%_n10000_sr03', bounds=None, subset_key=None)
    #plot_lid_results(datasets=data_sets, results=results, r=1.0, figsize=(8,8), show=False,
    #                 save_dir = r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots',
    #                 save_name='NEW_lollipop_tests1_2d_plot_100%_local_n10000_sr03', bounds=bounds, subset_key=name)
    #print(name) 'NEW_lollipop_tests1_2d_plot_100%_n10000_sr03', 'NEW_lollipop_tests1_2d_plot_100%_local_n10000_sr03', 'NEW_lollipop_tests1_sr03_n10000'
    #plot_k(result_dictionaries, dict_names, save_name='all_Kplot_adjusted_local_n2500_sr0.25', show=False, log=True, allowed_methods=None, partial_key=name, adjust_k=4)
    #plot_k(result_dictionaries, dict_names, save_name='Kplot_figure_out_locally_optimal_k_2_n2500_sr0.3', show=False, log=True, allowed_methods=None, partial_key=name)
    #plot_graph_comparison_grid(results, data_sets, save_name='smoothing_plot_k', save_dir=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots', k=3, show=False, clip_threshold=1, position_method='lmds')
    #k_partial_results, k_full_results, processed_results = min_kstar_k2(results, data_sets, 'mle', '', 'bag_w_0')
    #dataset_dict = get_datasets(n=500)
    #plot_all_knn_graphs(dataset_dict, k=3, output_path='all_knn_graphs.pdf', position_method='lmds')
    #print({key: [np.unique(data_sets[key][1]), data_sets[key][2]] for key in data_sets})
    #print({key: {key2: results[key][1][key2] for key2 in results[key][1]} for key in results})
    #plot_graph_comparison_grid(results, data_sets, save_name='graph_grid_error_test_mle2_n500', save_dir=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots', k=3, show=False, clip_threshold=1, position_method='lmds')

    #result_dictionaries, dict_names = convert_results_for_plot(results)
    #print(dict_names)
    #print(result_dictionaries[3])
    #print(result_dictionaries[-1])
    #create_method_variant_spider_charts(data_sets, dictionaries=result_dictionaries, names=dict_names, normalize_data=True, save=True, save_prefix='spider_chart_new_n2500', fill=True)

    #bounds = [[1.1, 1.4], [1.1, 1.4]] # [[1.2, 1.35], [1.2, 1.35]]
    #name = 'spatial_subset_'
    #for i in range(len(bounds)):
    #    name = name + f'_x{i}_{bounds[i][0]}-{bounds[i][1]}_'

    '''
        param_list1 = [([5+i*2 for i in range(50)], [10], [0.1], ['mada'], ['', 'bag_f_f', 'bag_w_1_y_f_f_0', 'bag_w_1_n_f_f_0'])]
    param_combs1 = generate_param_combinations(param_list1)

    param_list2 = [([10], [10], [0.02+i*0.02 for i in range(50)], ['mada'], ['', 'bag_f_f'])]
    param_combs2 = generate_param_combinations(param_list2)

    param_list3 = [([5+i*2 for i in range(15)], [10], [0.1], ['tle'], ['', 'bag_f_f', 'bag_w_1_n_f_f_0'])]
    param_combs3 = generate_param_combinations(param_list3)

    param_list4 = [([10], [10], [0.02+i*0.02 for i in range(50)], ['tle'], ['', 'bag_f_f'])]
    param_combs4 = generate_param_combinations(param_list4)

    param_list5 = [([5+i*1 for i in range(15)], [10], [0.1], ['2nn'], ['', 'bag_f_f'])] #'bag_f_f', 'bag_w_1_n_f_f_0'
    param_combs5 = generate_param_combinations(param_list5)

    param_list6 = [([10], [10], [0.04+i*0.04 for i in range(20)], ['2nn'], ['', 'bag_f_f'])]
    param_combs6 = generate_param_combinations(param_list6)
        results1, data_sets1 = result_generator(n=2500, param_list=param_combs1, save=True, load=True, load_path=load_path1,
                                          data_load_path=data_load_path, save_name=save_name1, usedata1=None, usedata2=original,
                                          test_types=None, bounddict=None, load_data=False, reduce_worker_count=2)
    results2, data_sets2 = result_generator(n=2500, param_list=param_combs2, save=True, load=True, load_path=load_path2,
                                          data_load_path=data_load_path, save_name=save_name2, usedata1=None, usedata2=original,
                                          test_types=None, bounddict=None, load_data=False, reduce_worker_count=2)
    results3, data_sets3 = result_generator(n=2500, param_list=param_combs3, save=True, load=True, load_path=load_path3,
                                          data_load_path=data_load_path, save_name=save_name3, usedata1=None, usedata2=original,
                                          test_types=None, bounddict=None, load_data=False, reduce_worker_count=2)
    results4, data_sets4 = result_generator(n=2500, param_list=param_combs4, save=True, load=False, load_path=load_path4,
                                          data_load_path=data_load_path, save_name=save_name4, usedata1=None, usedata2=original,
                                          test_types=None, bounddict=None, load_data=False, reduce_worker_count=2)
    result_dictionaries1, dict_names1 = convert_results_for_plot(results1)
    result_dictionaries2, dict_names2 = convert_results_for_plot(results2)
    result_dictionaries3, dict_names3 = convert_results_for_plot(results3)
    result_dictionaries4, dict_names4 = convert_results_for_plot(results4)
    plot_k(results_list=result_dictionaries1, name_list=dict_names1, save_name="kplot_mada_full", log=True, grid=True, figsize=(8,8), base_fontsize=4)
    plot_bias_variance_bars(results_list=result_dictionaries2, name_list=dict_names2, fixed_k=10, grid=True, figsize=(8,8), base_fontsize=4, log=False, adjust_k=1, save_name="mada_sr_bar_plot", save_dir="./plots", label_every=3)
    plot_k(results_list=result_dictionaries3, name_list=dict_names3, save_name="kplot_tle_full", log=True, grid=True, figsize=(8,8), base_fontsize=4)
    plot_bias_variance_bars(results_list=result_dictionaries4, name_list=dict_names4, fixed_k=10, grid=True, figsize=(8,8), base_fontsize=4, log=False, adjust_k=1, save_name="tle_sr_bar_plot", save_dir="./plots", label_every=3)

1 = 'k_full_mada'
    save_name2 = 'sr_full_mada'
    save_name3 = 'k_full_tle'
    save_name4 = 'sr_full_tle'
    save_name5 = 'k_full_2nn'
    save_name6 = 'sr_full_2nn'
        load_path2 = f'{dir}{save_name2}'
    load_path3 = f'{dir}{save_name3}'
    load_path4 = f'{dir}{save_name4}'
    load_path5 = f'{dir}{save_name5}'
    load_path6 = f'{dir}{save_name6}'
    
    
        results5, data_sets5 = result_generator(n=500, param_list=param_combs5, save=True, load=False, load_path=load_path5,
                                          data_load_path=data_load_path, save_name=save_name5, usedata1=None, usedata2=original,
                                          test_types=None, bounddict=None, load_data=False, reduce_worker_count=2)
    results6, data_sets6 = result_generator(n=500, param_list=param_combs6, save=True, load=False, load_path=load_path6,
                                          data_load_path=data_load_path, save_name=save_name6, usedata1=None, usedata2=original,
                                          test_types=None, bounddict=None, load_data=False, reduce_worker_count=2)
    result_dictionaries5, dict_names5 = convert_results_for_plot(results5)
    result_dictionaries6, dict_names6 = convert_results_for_plot(results6)
    plot_k(results_list=result_dictionaries5, name_list=dict_names5, save_name="kplot_2nn_full", log=True, grid=True,
           figsize=(8, 8), base_fontsize=4)
    plot_bias_variance_bars(results_list=result_dictionaries6, name_list=dict_names6, fixed_k=10, grid=True,
                            figsize=(8, 8), base_fontsize=4, log=False, adjust_k=1, save_name="2nn_sr_bar_plot",
                            save_dir="./plots", label_every=3)


    '''

