import pandas as pd
from tqdm import tqdm
import multiprocessing
import pickle
import os
###################################################OWN IMPORT###################################################
from LIDBagging.RunningEstimators.Collecting import fast_skdim_estimators
from LIDBagging.Helper.ComparrisonMeasures import get_lollipop_comparrison_measures
#from LIDBagging.Datasets.DatasetGeneration import get_datasets
from LIDBagging.experiment_class import *
###############################################################################################################################RUNNING ESTIMATORS###################################
import multiprocessing as mp
from itertools import repeat
from tqdm import tqdm
##############################################################################################################################################################
def save_results2(results, directory, save_name):
    os.makedirs(directory, exist_ok=True)  # Ensure directory exists
    filepath = os.path.join(directory, save_name)  # Create full path
    with open(filepath, "wb") as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)

def load_results2(directory, save_name):
    os.makedirs(directory, exist_ok=True)  # Ensure directory exists
    filepath = os.path.join(directory, save_name)  # Create full path
    with open(filepath, "rb") as fh:
        results = pickle.load(fh)
    return results

def run_experiment1(args, load=False, use_LIDkit=False, directory=r"C:\pkls"):
    params, load, use_LIDkit = args
    experiment = LID_experiment(params=params)
    experiment.generate_data(load=load, directory=directory)
    experiment.estimate(bounds=None, use_LIDkit=use_LIDkit)
    return experiment

def run_experiment2(params, load=False, use_LIDkit=False, directory=r"C:\pkls"):
    experiment = LID_experiment(params=params)
    experiment.generate_data(load=load, directory=directory)
    experiment.estimate(bounds=None, use_LIDkit=use_LIDkit)
    return experiment

def _run_star(args):
    return run_experiment2(*args)

def run_several_experiments_multiprocess(params_lists, worker_count=None, load=False, use_LIDkit=False, directory=r"C:\pkls"):
    if worker_count is None:
        worker_count = mp.cpu_count()

    tasks = zip(params_lists, repeat(load), repeat(use_LIDkit), repeat(directory))

    with mp.Pool(worker_count) as pool:
        results_iter = pool.imap_unordered(_run_star, tasks)

        results = list(tqdm(results_iter,
                            total=len(params_lists),
                            desc="running experiments",
                            unit="exp"))

    return results

def run_several_experiments_sequential(params_lists, load=False, use_LIDkit=False, directory=r"C:\pkls"):
    results = []
    for params in tqdm(params_lists):
        experiment = run_experiment1(args=(params, load, use_LIDkit, directory))
        results.append(experiment)
    return results

def new_result_generator(param_dicts, multiprocess=False, load=False, load_data=False, worker_count=None, save_name='res', use_LIDkit=False, directory=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2'):
    if not load:
        params_lists = expand_param_dicts(param_dicts)
        if multiprocess:
            results = run_several_experiments_multiprocess(params_lists, worker_count=worker_count, load=load_data, use_LIDkit=use_LIDkit, directory=directory)
        else:
            results = run_several_experiments_sequential(params_lists, load=load_data, use_LIDkit=use_LIDkit)
        save_results2(results, directory=directory, save_name=save_name)
    else:
        results = load_results2(directory=directory, save_name=save_name)
    return results





