"""
def new_plots(n, param_list, save=True, load=False, load_path=None, save_name = '', usedata=None, test_types=None, bounddict=None):
    data_sets = get_datasets(n=n)
    if bounddict is not None:
        bounds = {key: bounddict for key in data_sets}
    else:
        bounds = None
    if usedata is not None:
        keys = list(data_sets.keys())
        data_sets = {keys[i]: data_sets[keys[i]] for i in usedata}
    if load:
        results = load_dict(load_path)
    else:
        results = run_test_fast_multiprocess(data_sets, param_list=param_list, test_types=test_types, bounds=bounds)
    if save and not load:
        directory = r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2'
        save_dict(data=results, directory=directory, filename=save_name)
    result_dictionaries, dict_names = convert_results_for_plot(results)
    return result_dictionaries, dict_names
"""