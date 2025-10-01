###################################################OWN IMPORT###################################################
from LIDBagging.RunningEstimators.BaggingSmoothing.SimpleBagging import *
from LIDBagging.RunningEstimators.BaggingSmoothing.WeightedBagging import *
from LIDKit.core.estimators.numpy.faiss import LIDEstimatorFaissNumpy
from cleaning_bin.random_things.LID_Bagging_and_Bayesian_Incomplete import *
###############################################################################################################################COLLECTING ESTIMATORS###############################################################################################################################
def sk_estimators(X, k = 10, correct = True, estimator_names=None, smooth=False, geo=None):
    dists, knnidx = skdim._commonfuncs.get_nn(X, k=k, n_jobs=1)
    estimate_dict = {}
    avg_dict = {}
    for name in estimator_names:
        if name == 'mle':
            lid_estimates, mean_estimate = sk_MLE_full(X, k=k, correct=correct, dists=dists, knnidx=knnidx, smooth=smooth, geo=geo)
            estimate_dict[name] = lid_estimates
            avg_dict[name] = mean_estimate
        if name == 'mom':
            lid_estimates, mean_estimate = sk_MOM_full(X, k=k, correct=correct, dists=dists, knnidx=knnidx, smooth=smooth, geo=geo)
            estimate_dict[name] = lid_estimates
            avg_dict[name] = mean_estimate
        if name == 'tle':
            lid_estimates, mean_estimate = sk_TLE_full(X, k=k, correct=correct, dists=dists, knnidx=knnidx, smooth=smooth, geo=geo)
            estimate_dict[name] = lid_estimates
            avg_dict[name] = mean_estimate
        if name == 'mada':
            lid_estimates, mean_estimate = sk_MADA_full(X, k=k, correct=correct, dists=dists, knnidx=knnidx, smooth=smooth, geo=geo)
            estimate_dict[name] = lid_estimates
            avg_dict[name] = mean_estimate
        if name == 'ess':
            lid_estimates, mean_estimate = sk_ESS_full(X, k=k, correct=correct, dists=dists, knnidx=knnidx, smooth=smooth, geo=geo)
            estimate_dict[name] = lid_estimates
            avg_dict[name] = mean_estimate
        if name == '2nn':
            lid_estimates, mean_estimate = sk_2NN_full(X, k=k, correct=correct, dists=dists, knnidx=knnidx, smooth=smooth, geo=geo)
            estimate_dict[name] = lid_estimates
            avg_dict[name] = mean_estimate
        if name == 'lidl':
            lid_estimates, mean_estimate = LIDL_full(X, k=k, correct=correct, dists=dists, knnidx=knnidx, model_type="gm", smooth=smooth, geo=geo)
            estimate_dict[name] = lid_estimates
            avg_dict[name] = mean_estimate
    return estimate_dict, avg_dict

def LIDkit_estimators(X, k = 10, correct = True, estimator_names=None, smooth=False, geo=None):
    estimate_dict = {}
    avg_dict = {}
    for name in estimator_names:
        if name == 'mle':
            estimator = LIDEstimatorFaissNumpy(k=k)
            components = estimator.estimate(query_points=X, reference=X, metric='l2', loss='quadratic')
            lid_estimates = components.lids
            if smooth:
                lid_estimates, mean_estimate = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
                estimate_dict[name] = lid_estimates
                avg_dict[name] = mean_estimate
            else:
                estimate_dict[name] = lid_estimates
                avg_dict[name] = np.mean(lid_estimates)
        else:
            NotImplementedError('Wrong estimator type safeguard. Only mle works for now.')
    return estimate_dict, avg_dict

def Ricardo_estimators(X, k = 10, correct = True, estimator_names=None, smooth=False, geo=None):
    estimate_dict = {}
    avg_dict = {}
    for name in estimator_names:
        if name == 'mle':
            lid_estimates = MLE_LID_Estimator(data_array=X, subsample_indexes = None, neighbourhood_size = k, return_smoothed = smooth, simple_smooth=False, geo = geo)
            estimate_dict[name] = lid_estimates
            avg_dict[name] = np.mean(lid_estimates)
        if name == 'tle':
            lid_estimates = TLE_LID_Estimator(data_array=X, subsample_indexes=None, neighbourhood_size=k,
                                              return_smoothed=smooth, simple_smooth=False, geo=geo)
            estimate_dict[name] = lid_estimates
            avg_dict[name] = np.mean(lid_estimates)
        if name == 'mada':
            lid_estimates = MADA_LID_Estimator(data_array=X, subsample_indexes=None, neighbourhood_size=k,
                                              return_smoothed=smooth, simple_smooth=False, geo=geo)
            estimate_dict[name] = lid_estimates
            avg_dict[name] = np.mean(lid_estimates)
    return estimate_dict, avg_dict

def fast_skdim_estimators(data_set, estimator_names, method_type=None, n_bags=10, sampling_rate = 0.5, k=10, progress_bar=False, correct = True):
    if method_type == '':
        estimators_dict, avg_estimator_dict = sk_estimators(data_set, k=k, correct=correct, estimator_names=estimator_names)
    elif method_type == 'smooth':
        estimators_dict, avg_estimator_dict = sk_estimators(data_set, k=k, correct=correct, estimator_names=estimator_names, smooth=True, geo=None)
    elif method_type == 'smooth_geo':
        estimators_dict, avg_estimator_dict = sk_estimators(data_set, k=k, correct=correct, estimator_names=estimator_names, smooth=True, geo=3)
    elif method_type.startswith('bag') and not method_type.startswith('bag_w_'):
        typelist = method_type[len('bag_'):].split('_', 1)
        pre_smooth, post_smooth = typelist[0], typelist[1]
        if pre_smooth == "t":
            pre_smooth = True
        else:
            pre_smooth = False
        if post_smooth == "t":
            post_smooth = True
        else:
            post_smooth = False
        estimators = []
        used_estimator_names = []
        for i in range(len(estimator_names)):
            if estimator_names[i] == 'mle':
                estimators.append(sk_MLE)
                used_estimator_names.append('mle')
            elif estimator_names[i] == 'mom':
                estimators.append(sk_MOM)
                used_estimator_names.append('mom')
            elif estimator_names[i] == 'tle':
                estimators.append(sk_TLE)
                used_estimator_names.append('tle')
            elif estimator_names[i] == 'mada':
                estimators.append(sk_MADA)
                used_estimator_names.append('mada')
            elif estimator_names[i] == 'ess':
                estimators.append(sk_ESS)
                used_estimator_names.append('ess')
            elif estimator_names[i] == '2nn':
                estimators.append(sk_2NN)
                used_estimator_names.append('2nn')
        estimators_dict, avg_estimator_dict = simple_bagging_skdim(estimator=None, Q=data_set, X=data_set,
                                                                   n_bags=n_bags, k=k, sampling_rate=sampling_rate,
                                                                   progress_bar=progress_bar, estimators=estimators,
                                                                   estimator_names=used_estimator_names,
                                                                   paralell_estimation=True, geo=None, pre_smooth=pre_smooth,
                                                                   post_smooth=post_smooth)
    elif method_type.startswith('bag_w_'):
        if method_type.endswith('_bag'):
            weighing_type = method_type[len('bag_w_'):]
            splited = weighing_type.split('_', 3)
            t, pre_smooth, post_smooth, rest = float(splited[0]), splited[1], splited[2], splited[3]
        else:
            weighing_type = method_type[len('bag_w_'):]
            splited = weighing_type.split('_', 4)
            t, use_w, pre_smooth, post_smooth, rest = float(splited[0]), splited[1], splited[2], splited[3], splited[4]
            if pre_smooth == "t":
                pre_smooth = True
            else:
                pre_smooth = False
            if post_smooth == "t":
                post_smooth = True
            else:
                post_smooth = False
        estimators = []
        used_estimator_names = []
        for i in range(len(estimator_names)):
            if estimator_names[i] == 'mle':
                estimators.append(sk_MLE)
                used_estimator_names.append('mle')
            elif estimator_names[i] == 'mom':
                estimators.append(sk_MOM)
                used_estimator_names.append('mom')
            elif estimator_names[i] == 'tle':
                estimators.append(sk_TLE)
                used_estimator_names.append('tle')
            elif estimator_names[i] == 'mada':
                estimators.append(sk_MADA)
                used_estimator_names.append('mada')
            elif estimator_names[i] == 'ess':
                estimators.append(sk_ESS)
                used_estimator_names.append('ess')
            elif estimator_names[i] == '2nn':
                estimators.append(sk_2NN)
                used_estimator_names.append('2nn')
        if rest == 'bag':
            estimators_dict, avg_estimator_dict = outofbag_weighted_inside_bagging_skdim(estimator=None, Q=data_set,
                                                                                         X=data_set, n_bags=n_bags, k=k,
                                                                                         sampling_rate=sampling_rate,
                                                                                         progress_bar=progress_bar,
                                                                                         estimators=estimators,
                                                                                         estimator_names=used_estimator_names,
                                                                                         paralell_estimation=True, t=t,
                                                                                         geo=None, pre_smooth=pre_smooth,
                                                                                         post_smooth=post_smooth)
        else:
            estimators_dict, avg_estimator_dict = outofbag_weighted_bagging_skdim(estimator=None, Q=data_set, X=data_set,
                                                                                  n_bags=n_bags, k=k, sampling_rate=sampling_rate,
                                                                                  progress_bar=progress_bar, estimators=estimators,
                                                                                  estimator_names=used_estimator_names,
                                                                                  paralell_estimation=True,
                                                                                  weighing_type=rest, t=t, use_w=use_w,
                                                                                  geo=None, pre_smooth=pre_smooth,
                                                                                  post_smooth=post_smooth)
    return estimators_dict, avg_estimator_dict

#bagging_method: bag, bagw, bagwth, bagbag
#submethod_0: 0, inf
#submethod_error: diff, log_diff

def complete_estimators(dataset, k, sr, Nbag, pre_smooth, post_smooth, t, estimator, bagging_method, submethod_0, submethod_error, progress_bar=False, correct = True):
    estimators = []
    used_estimator_names = []
    if isinstance(estimator, list):
        estimator_names = estimator
    elif isinstance(estimator, str):
        estimator_names = [estimator]
    else:
        raise TypeError("Estimator name incorrect")
    for i in range(len(estimator_names)):
        if estimator_names[i] == 'mle':
            estimators.append(sk_MLE)
            used_estimator_names.append('mle')
        elif estimator_names[i] == 'mom':
            estimators.append(sk_MOM)
            used_estimator_names.append('mom')
        elif estimator_names[i] == 'tle':
            estimators.append(sk_TLE)
            used_estimator_names.append('tle')
        elif estimator_names[i] == 'mada':
            estimators.append(sk_MADA)
            used_estimator_names.append('mada')
        elif estimator_names[i] == 'ess':
            estimators.append(sk_ESS)
            used_estimator_names.append('ess')
        elif estimator_names[i] == '2nn':
            estimators.append(sk_2NN)
            used_estimator_names.append('2nn')
    if bagging_method is None:
        estimators_dict, avg_estimator_dict = sk_estimators(dataset, k=k, correct=correct, estimator_names=estimator_names, smooth=post_smooth, geo=None)
    elif bagging_method == 'bag':
        estimators_dict, avg_estimator_dict = simple_bagging_skdim(estimator=None, Q=dataset, X=dataset,
                                                                   n_bags=Nbag, k=k, sampling_rate=sr,
                                                                   progress_bar=progress_bar, estimators=estimators,
                                                                   estimator_names=used_estimator_names,
                                                                   paralell_estimation=True, geo=None, pre_smooth=pre_smooth,
                                                                   post_smooth=post_smooth)
    elif bagging_method == 'bagw':
        estimators_dict, avg_estimator_dict = outofbag_weighted_bagging_skdim(estimator=None, Q=dataset, X=dataset,
                                                                              n_bags=Nbag, k=k,
                                                                              sampling_rate=sr,
                                                                              progress_bar=progress_bar,
                                                                              estimators=estimators,
                                                                              estimator_names=used_estimator_names,
                                                                              paralell_estimation=True,
                                                                              weighing_type=submethod_0, t=t, use_w='n',
                                                                              geo=None, pre_smooth=pre_smooth,
                                                                              post_smooth=post_smooth, error_type=submethod_error)
    elif bagging_method == 'bagwth':
        estimators_dict, avg_estimator_dict = outofbag_weighted_bagging_skdim(estimator=None, Q=dataset, X=dataset,
                                                                              n_bags=Nbag, k=k,
                                                                              sampling_rate=sr,
                                                                              progress_bar=progress_bar,
                                                                              estimators=estimators,
                                                                              estimator_names=used_estimator_names,
                                                                              paralell_estimation=True,
                                                                              weighing_type=submethod_0, t=t, use_w='y',
                                                                              geo=None, pre_smooth=pre_smooth,
                                                                              post_smooth=post_smooth, error_type=submethod_error)
    elif bagging_method == 'approx_bagwth':
        estimators_dict, avg_estimator_dict = outofbag_weighted_bagging_skdim(estimator=None, Q=dataset, X=dataset,
                                                                              n_bags=Nbag, k=k,
                                                                              sampling_rate=sr,
                                                                              progress_bar=progress_bar,
                                                                              estimators=estimators,
                                                                              estimator_names=used_estimator_names,
                                                                              paralell_estimation=True,
                                                                              weighing_type=submethod_0, t=t, use_w='y_aprox',
                                                                              geo=None, pre_smooth=pre_smooth,
                                                                              post_smooth=post_smooth, error_type=submethod_error)
    elif bagging_method == 'bagbag':
        estimators_dict, avg_estimator_dict = outofbag_weighted_inside_bagging_skdim(estimator=None, Q=dataset,
                                                                                     X=dataset, n_bags=Nbag, k=k,
                                                                                     sampling_rate=sr,
                                                                                     progress_bar=progress_bar,
                                                                                     estimators=estimators,
                                                                                     estimator_names=used_estimator_names,
                                                                                     paralell_estimation=True, t=t,
                                                                                     geo=None, pre_smooth=pre_smooth,
                                                                                     post_smooth=post_smooth, error_type=submethod_error)
    return estimators_dict, avg_estimator_dict

def LIDkit_complete_estimators(dataset, k, sr, Nbag, pre_smooth, post_smooth, t, estimator, bagging_method, submethod_0, submethod_error, progress_bar=False, correct = True):
    estimators = []
    used_estimator_names = []
    if isinstance(estimator, list):
        estimator_names = estimator
    elif isinstance(estimator, str):
        estimator_names = [estimator]
    else:
        raise TypeError("Estimator name incorrect")
    for i in range(len(estimator_names)):
        if estimator_names[i] == 'mle':
            estimators.append(sk_MLE)
            used_estimator_names.append('mle')
        elif estimator_names[i] == 'mom':
            estimators.append(sk_MOM)
            used_estimator_names.append('mom')
        elif estimator_names[i] == 'tle':
            estimators.append(sk_TLE)
            used_estimator_names.append('tle')
        elif estimator_names[i] == 'mada':
            estimators.append(sk_MADA)
            used_estimator_names.append('mada')
        elif estimator_names[i] == 'ess':
            estimators.append(sk_ESS)
            used_estimator_names.append('ess')
        elif estimator_names[i] == '2nn':
            estimators.append(sk_2NN)
            used_estimator_names.append('2nn')
    if bagging_method is None:
        estimators_dict, avg_estimator_dict = LIDkit_estimators(dataset, k=k, correct=correct, estimator_names=estimator_names, smooth=post_smooth, geo=None)
    elif bagging_method == 'bag':
        estimators_dict, avg_estimator_dict = simple_bagging_LIDkit(estimator=None, Q=dataset, X=dataset,
                                                                   n_bags=Nbag, k=k, sampling_rate=sr,
                                                                   progress_bar=progress_bar, estimators=estimators,
                                                                   estimator_names=used_estimator_names,
                                                                   paralell_estimation=True, geo=None, pre_smooth=pre_smooth,
                                                                   post_smooth=post_smooth)
    elif bagging_method == 'bagw':
        estimators_dict, avg_estimator_dict = outofbag_weighted_bagging_LIDkit(estimator=None, Q=dataset, X=dataset,
                                                                              n_bags=Nbag, k=k,
                                                                              sampling_rate=sr,
                                                                              progress_bar=progress_bar,
                                                                              estimators=estimators,
                                                                              estimator_names=used_estimator_names,
                                                                              paralell_estimation=True,
                                                                              weighing_type=submethod_0, t=t, use_w='n',
                                                                              geo=None, pre_smooth=pre_smooth,
                                                                              post_smooth=post_smooth, error_type=submethod_error)
    elif bagging_method == 'approx_bagwth':
        estimators_dict, avg_estimator_dict = outofbag_weighted_bagging_LIDkit(estimator=None, Q=dataset, X=dataset,
                                                                              n_bags=Nbag, k=k,
                                                                              sampling_rate=sr,
                                                                              progress_bar=progress_bar,
                                                                              estimators=estimators,
                                                                              estimator_names=used_estimator_names,
                                                                              paralell_estimation=True,
                                                                              weighing_type=submethod_0, t=t, use_w='y_aprox',
                                                                              geo=None, pre_smooth=pre_smooth,
                                                                              post_smooth=post_smooth, error_type=submethod_error)

    elif bagging_method == 'bagbag':
        estimators_dict, avg_estimator_dict = outofbag_weighted_inside_bagging_skdim(estimator=None, Q=dataset,
                                                                                     X=dataset, n_bags=Nbag, k=k,
                                                                                     sampling_rate=sr,
                                                                                     progress_bar=progress_bar,
                                                                                     estimators=estimators,
                                                                                     estimator_names=used_estimator_names,
                                                                                     paralell_estimation=True, t=t,
                                                                                     geo=None, pre_smooth=pre_smooth,
                                                                                     post_smooth=post_smooth, error_type=submethod_error)
    elif bagging_method == 'bagwth':
        NotImplementedError('Differing k adjustments are not implemented for LIDkit yet, only the approximate version.')

    return estimators_dict, avg_estimator_dict

def Ricardo_complete_estimators(dataset, k, sr, Nbag, pre_smooth, post_smooth, t, estimator, bagging_method, submethod_0, submethod_error, progress_bar=False, correct = True):
    estimators = []
    used_estimator_names = []
    if isinstance(estimator, list):
        estimator_names = estimator
    elif isinstance(estimator, str):
        estimator_names = [estimator]
    else:
        raise TypeError("Estimator name incorrect")
    for i in range(len(estimator_names)):
        if estimator_names[i] == 'mle':
            estimators.append(Ricardo_MLE)
            used_estimator_names.append('mle')
        elif estimator_names[i] == 'mom':
            estimators.append(sk_MOM)
            used_estimator_names.append('mom')
        elif estimator_names[i] == 'tle':
            estimators.append(Ricardo_TLE)
            used_estimator_names.append('tle')
        elif estimator_names[i] == 'mada':
            estimators.append(Ricardo_MADA)
            used_estimator_names.append('mada')
        elif estimator_names[i] == 'ess':
            estimators.append(sk_ESS)
            used_estimator_names.append('ess')
        elif estimator_names[i] == '2nn':
            estimators.append(sk_2NN)
            used_estimator_names.append('2nn')
    if bagging_method is None:
        estimators_dict, avg_estimator_dict = Ricardo_estimators(dataset, k=k, correct=correct, estimator_names=estimator_names, smooth=post_smooth, geo=None)
    elif bagging_method == 'bag':
        estimators_dict, avg_estimator_dict = simple_bagging_Ricardo(estimator=None, Q=dataset, X=dataset,
                                                                   n_bags=Nbag, k=k, sampling_rate=sr,
                                                                   progress_bar=progress_bar, estimators=estimators,
                                                                   estimator_names=used_estimator_names,
                                                                   paralell_estimation=True, geo=None, pre_smooth=pre_smooth,
                                                                   post_smooth=post_smooth)
    elif bagging_method == 'bagw':
        estimators_dict, avg_estimator_dict = outofbag_weighted_bagging_Ricardo(estimator=None, Q=dataset, X=dataset,
                                                                              n_bags=Nbag, k=k,
                                                                              sampling_rate=sr,
                                                                              progress_bar=progress_bar,
                                                                              estimators=estimators,
                                                                              estimator_names=used_estimator_names,
                                                                              paralell_estimation=True,
                                                                              weighing_type=submethod_0, t=t, use_w='n',
                                                                              geo=None, pre_smooth=pre_smooth,
                                                                              post_smooth=post_smooth, error_type=submethod_error)
    elif bagging_method == 'approx_bagwth':
        estimators_dict, avg_estimator_dict = outofbag_weighted_bagging_Ricardo(estimator=None, Q=dataset, X=dataset,
                                                                              n_bags=Nbag, k=k,
                                                                              sampling_rate=sr,
                                                                              progress_bar=progress_bar,
                                                                              estimators=estimators,
                                                                              estimator_names=used_estimator_names,
                                                                              paralell_estimation=True,
                                                                              weighing_type=submethod_0, t=t, use_w='y_aprox',
                                                                              geo=None, pre_smooth=pre_smooth,
                                                                              post_smooth=post_smooth, error_type=submethod_error)

    elif bagging_method == 'bagbag':
        estimators_dict, avg_estimator_dict = outofbag_weighted_inside_bagging_skdim(estimator=None, Q=dataset,
                                                                                     X=dataset, n_bags=Nbag, k=k,
                                                                                     sampling_rate=sr,
                                                                                     progress_bar=progress_bar,
                                                                                     estimators=estimators,
                                                                                     estimator_names=used_estimator_names,
                                                                                     paralell_estimation=True, t=t,
                                                                                     geo=None, pre_smooth=pre_smooth,
                                                                                     post_smooth=post_smooth, error_type=submethod_error)
    elif bagging_method == 'bagwth':
        NotImplementedError('Differing k adjustments are not implemented for LIDkit yet, only the approximate version.')

    return estimators_dict, avg_estimator_dict


