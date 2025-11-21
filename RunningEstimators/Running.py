import pandas as pd
from tqdm import tqdm
import multiprocessing
import pickle
import os
###################################################OWN IMPORT###################################################
from Bagging_for_LID.RunningEstimators.Collecting import fast_skdim_estimators
from Bagging_for_LID.Helper.ComparrisonMeasures import get_lollipop_comparrison_measures
#from LIDBagging.Datasets.DatasetGeneration import get_datasets
###############################################################################################################################RUNNING ESTIMATORS###############################################################################################################################
def save_to_df(d, save_name):
    directory = r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\csvs'
    df = pd.DataFrame.from_dict(d, orient="index")
    df.to_csv(directory + '\\' + f'{save_name}.csv')

def run_method_fast(data_sets, estimator_names, method_type, k=10, n_bags=10, sampling_rate = 0.5, save=False, test_types = None, bounds = None):
    result_dictionary = {estimator_names[i]: {} for i in range(len(estimator_names))}
    for key in tqdm(data_sets):
        estimators_dict, avg_estimator_dict = fast_skdim_estimators(data_sets[key][0], estimator_names=estimator_names, method_type=method_type, n_bags=n_bags, sampling_rate=sampling_rate, k=k, progress_bar=False, correct=True)
        for name in estimator_names:
            result_dictionary[name][key] = (estimators_dict[name], avg_estimator_dict[name])
    estimators_results = {estimator_names[i]: get_lollipop_comparrison_measures(data_sets, result_dictionary[estimator_names[i]], test_types=test_types, bounds=bounds) for i in range(len(estimator_names))}
    if save:
        save_to_df(estimators_results, f"{'_'.join(str(x) for x in estimator_names)}_{method_type}_results")
    return result_dictionary, estimators_results

def run_task(args, data_sets, test_types, bounds):
    #test_types = ['dim', 'region']
    #bounds = {key: {0: (1, 1.5), 1: (1, 1.5)} for key in data_sets}
    """Helper function to run `run_method_fast` with the given arguments."""
    k, n_bags, sampling_rate, method_names, method_type = args
    results = {}
    if method_type.startswith('bag'):
        result_dictionary, estimators_results = run_method_fast(
            data_sets, method_names, method_type, k=k,
            n_bags=n_bags, sampling_rate=sampling_rate, save=False, test_types=test_types, bounds=bounds)
        for method_name in method_names:
            results[f'{method_name}_{method_type}_k_{k}_n_bags_{n_bags}_sampling_rate_{sampling_rate}'] = (
                result_dictionary[method_name], estimators_results[method_name]
            )
    else:
        result_dictionary, estimators_results = run_method_fast(
            data_sets, method_names, method_type, k=k, save=False, test_types=test_types, bounds=bounds)
        for method_name in method_names:
            results[f'{method_name}_{method_type}_k_{k}'] = (
                result_dictionary[method_name], estimators_results[method_name]
            )
    return results

def run_test_fast_multiprocess(data_sets, param_list, test_types = None, bounds = None, reduce_worker_count=1):
    """
    Run parallelized `run_method_fast` for a given list of parameter tuples.
    :param data_sets: Data used for computations.
    :param param_list: List of tuples, each containing (k, n_bags, sampling_rate, method_names, method_type)
    :return: Dictionary of results.
    """
    results = {}
    # Run in parallel using multiprocessing
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()//reduce_worker_count) as pool:
        results_list = pool.starmap(run_task, [(args, data_sets, test_types, bounds) for args in param_list])
    # Merge results from different processes
    for result in results_list:
        results.update(result)
    return results

def save_dict(data, directory, filename):
    os.makedirs(directory, exist_ok=True)  # Ensure directory exists
    filepath = os.path.join(directory, filename)  # Create full path
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

#results:   Dict["method_key"][0]["dataset_key"] = (lid_estimates, avg_lid_Estimate)
#           Dict["method_key"][1]["dataset_key"] = [avg_lid, mse, avg_bias2, est_var]