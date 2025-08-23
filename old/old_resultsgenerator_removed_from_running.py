"""
def result_generator(n, param_list, save=True, load=False, load_path=None, data_load_path=None, save_name = '', usedata1=None, usedata2=None, test_types=None, bounddict=None, load_data=True, reduce_worker_count=1):
    if load:
        data_sets = load_dict(f'{load_path}_data')
        results = load_dict(load_path)
    else:
        if not load_data:
            data_sets = get_datasets(n=n)
        else:
            data_sets = load_dict(f'{data_load_path}')
        if bounddict is not None:
            bounds = {key: bounddict for key in data_sets}
        else:
            bounds = None
        if usedata1 is not None:
            keys = list(data_sets.keys())
            data_sets = {keys[i]: data_sets[keys[i]] for i in usedata1}
        if usedata2 is not None:
            data_sets = {usedata2[i]: data_sets[usedata2[i]] for i in range(len(usedata2))}
        results = run_test_fast_multiprocess(data_sets, param_list=param_list, test_types=test_types, bounds=bounds, reduce_worker_count=reduce_worker_count)
    if save and not load:
        directory = r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2'
        save_dict(data=results, directory=directory, filename=save_name)
        save_dict(data=data_sets, directory=directory, filename=f'{save_name}_data')
    return results, data_sets
"""