import gc
#-----------------------------------------------------------------------------------------------------------------------
#Internal imports
from Bagging_for_LID.run_files.final_tasks import *
#-----------------------------------------------------------------------------------------------------------------------
#Setup directories
PROJECT_ROOT = Path(r"C:\Users\krp\PycharmProjects\FinalFixLIDGit")
os.chdir(PROJECT_ROOT)
p = str(PROJECT_ROOT)
if p not in sys.path:
    sys.path.insert(0, p)
#-----------------------------------------------------------------------------------------------------------------------
#Setup computation
multiprocess=True
load=True
load_data=True
worker_count=7
save_name='mergedresult'
directory = r'C:\pkls'
#RESULTS ARE SAVED AT Bagging_for_LID.Output
#Modify output folders at Bagging_for_LID.run_files.final_tasks

if __name__ == "__main__":
    setup_logging()
    # -----------------------------------------------------------------------------------------------------------------------
    #Generate datasets (to be reused in different experiments)
    results_data = new_result_generator(param_dicts_data, multiprocess=False, load=True, load_data=True,
                                        worker_count=None,
                                        save_name='data_generation',
                                        directory=directory)
    #print('Data generation complete')
    # -----------------------------------------------------------------------------------------------------------------------
    #Setup result generation

    #Effectiveness of bagging (baseline, simple bagging, weighted bagging, neighborhood size adjustment)
    #Effectiveness of bagging t is changing additionally (baseline, simple bagging, weighted bagging, neighborhood size adjustment)
    #Bagging and smoothing (baseline, baseline with smoothing, simple bagging, simple bagging with pre or/and post smoothing)
    #Number of bags test (mse bar charts)
    #Sampling rate test (mse bar charts)
    #Interaction of sampling rate and number of bags (mse difference heatmaps)
    #Interaction of k and sampling rate (mse difference heatmaps)

    task_dict = {"effectiveness_test": effectiveness_test_general_param_dict,
                 'effectiveness_test_with_t': effectiveness_test_with_t_general_param_dict,
                 "Number_of_bags_test": Nbag_test_general_param_dict,
                 "Sampling_rate_test": sr_test_general_param_dict,
                 "Interaction_of_sampling_rate_and_number_of_bags_test": interaction_sr_Nbag_test_general_param_dict,
                 "Interaction_of_k_and_sampling_rate_test": interaction_sr_k_test_general_param_dict}

    tasks = setup_tasks(task_dict, multiprocess=multiprocess, load=load, load_data=load_data, worker_count=worker_count, save_name=save_name, directory=directory)

    # -----------------------------------------------------------------------------------------------------------------------
    #Run results and plotting at the same time to save RAM
    for key, func, args, kwargs in tasks:
        ok, result_dict = run_task_safely(func, *args, **kwargs)
        if not ok:
            continue
        try:
            consume_and_plot(key, result_dict, plot_tasks)
        finally:
            del result_dict
            gc.collect()
