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
from LIDBagging.Theory_testing.bag_covariance import *
################################################################################################################
if __name__ == "__main__":
    param_list1 = [0.04*(i+1) for i in range(22)]
    param_list = param_list1

    used_keys = ["M4_Nonlinear"]

    results = run_sr_cov_test(n=2500, size=100, param_list=param_list, k=35, lid_estimator=simple_MLE, query_amount=100, used_keys=used_keys, reduce_worker_count=2, load=False, save=False, load_data=True, save_data=False,
                        save_name='cov_test_full', load_path=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2\cov_test_full', save_name_data='cov_test_data_full',
                        load_path_data=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2\cov_test_data', sequential=True)
    #plot_results(results, save_name='sr_cov_plot_full', save_dir=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots', show=False)
    plot_sr_mean_std(results=results, figsize=(8,8), base_fontsize=4, save_name="sr_cov_plot_m4_k10")