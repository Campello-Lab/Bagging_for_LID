from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import numpy as np
###################################################OWN IMPORT###################################################
from LIDBagging.RunningEstimators.BaggingSmoothing.Smoothing import smoothing
from LIDKit.core._experimental.bagging.numpy import *
from LID_Bagging_and_Bayesian_Incomplete import *
from LIDBagging.RunningEstimators.BaseEstimators import *
##############################################################################################################################################################################################################################################################
def simple_bagging_skdim(estimator, Q, X, n_bags=10, k=10, sampling_rate=None,
                         progress_bar=False, estimators=None, estimator_names=None,
                         paralell_estimation=False, w=None, indexuse=None, pre_smooth=False,
                         geo=None, post_smooth=False, seed=42, smooth_style='code2'):
    rand_gen = np.random.RandomState(seed)
    def compute_distance_matrix(X):
        return squareform(pdist(X))

    def split_data_with_indices(X, n_bags=10):
        indices = np.random.permutation(len(X))
        if indexuse is not None:
            indices = np.intersect1d(indices, indexuse)
        split_indices = np.array_split(indices, n_bags)
        bags = [X[idx] for idx in split_indices]
        return bags, split_indices

    def sample_data_with_indices(X, n_bags=10, sampling_rate=0.8):
        n_samples = np.ceil(sampling_rate * len(X)).astype(int)
        indices = np.arange(len(X))
        bags = []
        selected_indices = []
        for _ in range(n_bags):
            chosen_idx = rand_gen.choice(indices, size=n_samples, replace=False)
            bags.append(X[chosen_idx])
            selected_indices.append(chosen_idx)
        return bags, selected_indices

    def k_smallest_nonzero_Q(distances, k=10, w=None):
        n = distances.shape[0]
        if w is not None:
            w = np.asarray(w)
            result_distances = []
            result_indices = []
            for q in range(n):
                nonzero_mask = distances[q] != 0
                valid_mask = nonzero_mask & (distances[q] <= w[q])
                dists = distances[q, valid_mask]
                inds = np.nonzero(valid_mask)[0]
                order = np.argsort(dists)
                result_distances.append(dists[order])
                result_indices.append(inds[order])
            return result_distances, result_indices
        else:
            result_distances = np.zeros((n, k))
            result_indices = np.zeros((n, k), dtype=int)
            for q in range(n):
                nonzero_mask = distances[q] != 0
                dists = distances[q, nonzero_mask]
                inds = np.nonzero(nonzero_mask)[0]
                if len(dists) >= k:
                    partition_indices = np.argpartition(dists, k)[:k]
                    sorted_indices = partition_indices[np.argsort(dists[partition_indices])]
                    result_distances[q, :] = dists[sorted_indices]
                    result_indices[q, :] = inds[sorted_indices]
                else:
                    raise ValueError("There are less nonzero distances than the given k")
            return result_distances, result_indices

    def k_smallest_distance_Q(distances, indices, k=10, w=None):
        considered_distances = distances[:, indices]
        smallest_distances, smallest_indices = k_smallest_nonzero_Q(considered_distances, k, w=w)
        if w is not None:
            smallest_indices = [indices[row_inds] for row_inds in smallest_indices]
        else:
            smallest_indices = indices[smallest_indices]
        return smallest_distances, smallest_indices

    n, m = Q.shape[0], Q.shape[1]
    distances = compute_distance_matrix(Q)
    #sorted_distances, sorted_indices = precompute_sorted_distances(distances)
    if sampling_rate is None:
        bags, split_indices = split_data_with_indices(X, n_bags=n_bags)
    else:
        bags, split_indices = sample_data_with_indices(X, n_bags=n_bags, sampling_rate=sampling_rate)
    if not paralell_estimation:
        estimates = np.zeros((n, n_bags))
        if progress_bar:
            for j in tqdm(range(n_bags)):
                smallest_distances, smallest_indices = k_smallest_distance_Q(distances=distances, indices=split_indices[j],
                                                                             k=k, w=w)
                #smallest_distances, smallest_indices = fast_k_smallest_in_bag(sorted_distances, sorted_indices, split_indices[j], k=k)
                estimates[:, j] = estimator(X=X, dists=smallest_distances, knnidx=smallest_indices, k=k, w=w)[0]
        else:
            for j in range(n_bags):
                smallest_distances, smallest_indices = k_smallest_distance_Q(distances=distances, indices=split_indices[j],
                                                                             k=k, w=w)
                #smallest_distances, smallest_indices = fast_k_smallest_in_bag(sorted_distances, sorted_indices,
                #                                                              split_indices[j], k=k)
                estimates[:, j] = estimator(X=X, dists=smallest_distances, knnidx=smallest_indices, k=k, w=w)[0]
        bagging_estimators = np.mean(estimates, axis=1)
        avg_bagging_estimator = np.mean(bagging_estimators)
        return bagging_estimators, avg_bagging_estimator
    else:
        estimator_dictionary = {estimator_names[i]: estimators[i] for i in range(len(estimator_names))}
        estimate_dictionary = {estimator_names[i]: np.zeros((n, n_bags)) for i in range(len(estimator_names))}
        if progress_bar:
            for j in tqdm(range(n_bags)):
                smallest_distances, smallest_indices = k_smallest_distance_Q(distances=distances, indices=split_indices[j],
                                                                             k=k, w=w)
                #smallest_distances, smallest_indices = fast_k_smallest_in_bag(sorted_distances, sorted_indices,
                #                                                              split_indices[j], k=k)
                for key in estimator_dictionary:
                    estimate_dictionary[key][:, j] = estimator_dictionary[key](X=X, dists=smallest_distances, knnidx=smallest_indices, k=k, w=w)[0]
        else:
            for j in range(n_bags):
                smallest_distances, smallest_indices = k_smallest_distance_Q(distances=distances, indices=split_indices[j],
                                                                             k=k, w=w)
                #print(n_bags)
                #print(smallest_distances.shape)
                #print(smallest_distances[0:5])
                #print(f'OG_smallest_distances_{smallest_distances.shape}')
                #print(f'OG_mean_smallest_distances_{smallest_distances.mean()}')
                #smallest_distances, smallest_indices = fast_k_smallest_in_bag(sorted_distances, sorted_indices,
                #                                                              split_indices[j], k=k)
                for key in estimator_dictionary:
                    if smooth_style != "code1":
                        estimate_dictionary[key][:, j] = estimator_dictionary[key](X=X, dists=smallest_distances, knnidx=smallest_indices, k=k, w=w, smooth=pre_smooth, geo=geo, smooth_style=smooth_style, bag_indices=split_indices[j])[0]
                    else:
                        estimate_dictionary[key][:, j] = \
                        estimator_dictionary[key](X=X, dists=smallest_distances, knnidx=smallest_indices, k=k, w=w, smooth=pre_smooth, geo=geo, smooth_style=smooth_style)[0]
                    #print(f'original_bag_estimate_for_bag_{j}: {np.mean(estimate_dictionary[key][:, j])}')
        bagging_estimator_dictionary = {estimator_names[i]: np.mean(estimate_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
        avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimator_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        if post_smooth:
            for i in range(len(estimator_names)):
                bagging_estimator_dictionary[estimator_names[i]], _ = smoothing(X, bagging_estimator_dictionary[estimator_names[i]], k=k, dists=None, knnidx=None, geo=geo)
            avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimator_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        return bagging_estimator_dictionary, avg_bagging_estimator_dictionary


def simple_bagging_LIDkit(estimator, Q, X, n_bags=10, k=10, sampling_rate=None, progress_bar=False,
                         estimators=None, estimator_names=None, paralell_estimation=False, w=None,
                         indexuse=None, pre_smooth=False, geo=None, post_smooth=False, log_level = "INFO", seed=42):
    n = Q.shape[0]
    LIDkit_estimator = NumpyBaggingLIDEstimator(k=k, num_bags = n_bags, samples_per_bag=np.ceil(sampling_rate*n).astype(int), log_level=log_level)
    if (len(estimator_names) == 1) and (estimator_names[0] == 'mle') and (estimator is None):
        result_components = {estimator_names[i]: LIDkit_estimator.estimate(query_points=Q, reference=X, pooling_method = 'linear', loss_function = 'quadratic', pre_smooth=pre_smooth) for i in range(len(estimator_names))}
        bagging_estimator_dictionary = {estimator_names[i]: result_components[estimator_names[i]].lids for i in range(len(estimator_names))}
        avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimator_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
    else:
        NotImplementedError('Wrong estimator type safeguard. Only mle works for now.')
    if post_smooth:
        for i in range(len(estimator_names)):
            smoothedi, _ = smoothing(X, bagging_estimator_dictionary[estimator_names[i]], k=k, dists=None, knnidx=None, geo=geo)
            bagging_estimator_dictionary[estimator_names[i]] = smoothedi
        avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimator_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
    return bagging_estimator_dictionary, avg_bagging_estimator_dictionary

def lid_Bagging_wrapper(data_array, *, ensemble_size=30, subsample_rate=0.3, rand_gen_seed=None, estimator_name="two_NN_LID_Estimator", **kwargs):
    if estimator_name == 'mle':
        estimator = MLE_LID_Estimator
    elif estimator_name == 'mom':
        estimator = MM_LID_Estimator
    elif estimator_name == 'tle':
        estimator = TLE_LID_Estimator
    elif estimator_name == 'mada':
        estimator = sk_MADA
    elif estimator_name == 'ess':
        estimator = sk_ESS
    elif estimator_name == '2nn':
        estimator = two_NN_LID_Estimator
    LID_estimates, _, _, _ = lid_Bagging(data_array=data_array, ensemble_size=ensemble_size, subsample_rate=subsample_rate, rand_gen_seed=rand_gen_seed, lid_Estimator=estimator, **kwargs)
    return LID_estimates

def simple_bagging_Ricardo(estimator, Q, X, n_bags=10, k=10, sampling_rate=None, progress_bar=False,
                         estimators=None, estimator_names=None, paralell_estimation=False, w=None,
                         indexuse=None, pre_smooth=False, geo=None, post_smooth=False, log_level = "INFO", seed = 42):
    n = Q.shape[0]
    result_components = {estimator_names[i]: lid_Bagging_wrapper(data_array=Q, ensemble_size=n_bags, subsample_rate=sampling_rate, rand_gen_seed=seed, estimator_name=estimator_names[i], neighbourhood_size=k, return_smoothed=pre_smooth, simple_smooth=False, geo=None) for i in range(len(estimator_names))}
    bagging_estimator_dictionary = {estimator_names[i]: result_components[estimator_names[i]] for i in range(len(estimator_names))}
    avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimator_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
    if post_smooth:
        for i in range(len(estimator_names)):
            smoothedi, _ = smoothing(X, bagging_estimator_dictionary[estimator_names[i]], k=k, dists=None, knnidx=None, geo=geo)
            bagging_estimator_dictionary[estimator_names[i]] = smoothedi
        avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimator_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
    return bagging_estimator_dictionary, avg_bagging_estimator_dictionary