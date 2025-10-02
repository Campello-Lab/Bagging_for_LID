#-----------------------------------------------------------------------------------------------------------------------
#External imports
from pathlib import Path
import sys, os
from functools import partial
#-----------------------------------------------------------------------------------------------------------------------
#Internal imports
from LIDBagging.Plotting.Plots.SpiderCharts import *
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
worker_count=7
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
    # -----------------------------------------------------------------------------------------------------------------------
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
    successes = 0
    for func, args, kwargs in tasks:
        ok, _ = run_task_safely(func, *args, **kwargs)
        if ok:
            successes += 1
    print(f"Completed {successes}/{len(tasks)} tasks (see run.log for details).")
    print('Data generation complete')
    #MAIN Paper results

    #Effectiveness of bagging (baseline, simple bagging, weighted bagging, neighborhood size adjustment)
    #Bagging and smoothing (baseline, baseline with smoothing, simple bagging, simple bagging with pre or/and post smoothing)

    #Number of bags test (mse bar charts)

    #Sampling rate test (mse bar charts)

    #Interaction of sampling rate and number of bags (mse difference heatmaps)

    #Interaction of k and sampling rate (mse difference heatmaps)

    #-----------------------------------------------------------------------------------------------------------------------
    #Supplements extra results

    #Interaction of k and sampling rate (mse difference heatmaps)