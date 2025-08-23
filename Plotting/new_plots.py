import pickle
import os
###################################################OWN IMPORT###################################################
#from LIDBagging.Datasets.DatasetGeneration import get_datasets
from LIDBagging.RunningEstimators.Running import run_test_fast_multiprocess
from LIDBagging.Helper.Other import convert_results_for_plot
##############################################################################################################################################################################################################################################################
def save_dict(data, directory, filename):
    os.makedirs(directory, exist_ok=True)  # Ensure directory exists
    filepath = os.path.join(directory, filename)  # Create full path
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)