import skdim
from tqdm import tqdm
import sys
cloned_folders = [
    r"C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\lidl",
]
for folder in cloned_folders:
    if folder not in sys.path:
        sys.path.append(folder)
import numpy as np
###################################################OWN IMPORT###################################################
from LIDBagging.RunningEstimators.BaggingSmoothing.Smoothing import smoothing
from LIDBagging.RunningEstimators.RewrittenRawEstimators.TLE import *
from LIDBagging.RunningEstimators.RewrittenRawEstimators.MADA import *
from LIDBagging.RunningEstimators.RewrittenRawEstimators.ESS import *
from LIDBagging.RunningEstimators.RewrittenRawEstimators.TwoNN import *
from LIDKit.core.estimators.numpy.tle import *
from LID_Bagging_and_Bayesian_Incomplete import *
###############################################################################################################################BASE ESTIMATORS###############################################################################################################################

def k_smallest_nonzero_0(x, k):
    x_nonzero = x[x != 0]
    if len(x_nonzero) > k:
        result = np.partition(x_nonzero, k)[:k]
    else:
        ValueError('There are less non-zero distances than the given k')
    return result

def k_smallest_distance_0(q, X, k=10, p=2):
    distances = np.linalg.norm(X - q, axis=1, ord=p)
    smallest_distances = k_smallest_nonzero_0(distances, k)
    max = np.max(smallest_distances)
    return smallest_distances, max

def LID_MLE_q(q, X, k=10, p=2, correction=0, w=None):
    smallest_distances, max = k_smallest_distance_0(q=q, X=X, k=k, p=p)
    mle = - (k-correction) / np.sum(np.log(smallest_distances / max))
    return mle

def LID_MOM_q(q, X, k=10, p=2, w=None):
    smallest_distances, max = k_smallest_distance_0(q=q, X=X, k=k, p=p)
    mu_hat = np.mean(smallest_distances)
    mom = - mu_hat/(mu_hat - max)
    return mom

def LID_MLE_Q(Q, X, k=10, p=2, progress_bar = False, w=None):
    m = Q.shape[0]
    estimates = np.empty(m)
    if progress_bar:
        for i in tqdm(range(m)):
            estimates[i] = LID_MLE_q(Q[i], X, k=k, p=p)
    else:
        for i in range(m):
            estimates[i] = LID_MLE_q(Q[i], X, k=k, p=p)
    return estimates, np.mean(estimates)

def LID_MOM_Q(Q, X, k=10, p=2, progress_bar = False, w=None):
    m = Q.shape[0]
    estimates = np.empty(m)
    if progress_bar:
        for i in tqdm(range(m)):
            estimates[i] = LID_MOM_q(Q[i], X, k=k, p=p)
    else:
        for i in range(m):
            estimates[i] = LID_MOM_q(Q[i], X, k=k, p=p)
    return estimates, np.mean(estimates)

def MLE(smallest_distances, correction=0, w=None):
    max = np.max(smallest_distances)
    mle = - (len(smallest_distances)-correction) / np.sum(np.log(smallest_distances / max))
    return mle

def MOM(smallest_distances, w=None):
    max = np.max(smallest_distances)
    mu_hat = np.mean(smallest_distances)
    mom = - mu_hat / (mu_hat-max)
    return mom

def sk_MOM(X, dists, knnidx, k = 10, w=None, return_ks = False, use_w = 'direct', smooth=False, geo=None):
    if w is None:
        mom = skdim.id.MOM()
        lid_estimates = mom._mom(dists)
        if return_ks:
            ks = np.repeat(k, X.shape[0])
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates), ks
            else:
                return lid_estimates, np.mean(lid_estimates), ks
        else:
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
            else:
                return lid_estimates, np.mean(lid_estimates)
    else:
        n = X.shape[0]
        lid_estimates = np.empty(n)
        ks = np.empty(n)
        for q in range(n):
            if use_w == 'direct':
                mu_hat = np.mean(dists[q])
                lid_estimates[q] = - mu_hat/(mu_hat-w[q])
                ks[q] = len(dists[q])
            elif use_w == 'indirect':
                if dists[q][-1] != np.max(dists[q]):
                    Warning('Distances are not ordered. Check failed.')
                mu_hat = np.mean(dists[q])
                lid_estimates[q] = - mu_hat / (mu_hat - np.max(dists[q]))
                ks[q] = len(dists[q])
        if return_ks:
            return lid_estimates, np.mean(lid_estimates), ks
        else:
            return lid_estimates, np.mean(lid_estimates)


def sk_TLE(X, dists, knnidx, k = 10, w=None, return_ks=False, use_w='indirect', smooth=False, geo=None, smooth_style="code1", bag_indices=None):
    if w is None:
        tle = skdim.id.TLE()
        tle._fit(X, dists=dists, knnidx=knnidx)
        lid_estimates = tle.dimension_pw_
        if return_ks:
            ks = np.repeat(k, X.shape[0])
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo, smooth_style=smooth_style)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates), ks
            else:
                return lid_estimates, np.mean(lid_estimates), ks
        else:
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo, smooth_style=smooth_style, bag_indices=bag_indices)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
            else:
                return lid_estimates, np.mean(lid_estimates)
    elif use_w == 'indirect':
        n = X.shape[0]
        lid_estimates = np.empty(n)
        ks = np.empty(n)
        tle = TLE()
        tle._fit(X, dists_list=dists, knnidx_list=knnidx)
        lid_estimates = tle.dimension_pw_
        if return_ks:
            return lid_estimates, np.mean(lid_estimates), ks
        else:
            return lid_estimates, np.mean(lid_estimates)
    else:
        NotImplemented(f"Not implemented use_w: {use_w}")

def sk_MLE(X, dists, knnidx, k = 10, correct = True, w = None, return_ks = False, use_w = 'direct', smooth=False, geo=None, smooth_style='code1', bag_indices=None):
    if w is None:
        mle = skdim.id.MLE()
        mle.fit(X, n_neighbors=k, comb='mean', precomputed_knn_arrays=(dists, knnidx))
        if correct:
            lid_estimates = k/(k-1)*mle.dimension_pw_
        else:
            lid_estimates = mle.dimension_pw_
        if return_ks:
            ks = np.repeat(k, X.shape[0])
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo, smooth_style=smooth_style)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates), ks
            else:
                return lid_estimates, np.mean(lid_estimates), ks
        else:
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo, smooth_style=smooth_style, bag_indices=bag_indices)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
            else:
                return lid_estimates, np.mean(lid_estimates)
    else:
        n = X.shape[0]
        lid_estimates = np.empty(n)
        ks = np.empty(n)
        for q in range(n):
            if use_w == 'direct':
                lid_estimates[q] = - len(dists[q])/np.sum(np.log(dists[q] / w[q]))
                ks[q] = len(dists[q])
            elif use_w == 'indirect':
                if dists[q][-1] != np.max(dists[q]):
                    Warning('Distances are not ordered. Check failed.')
                lid_estimates[q] = - len(dists[q])/np.sum(np.log(dists[q] / dists[q][-1]))
                ks[q] = len(dists[q])
        if return_ks:
            return lid_estimates, np.mean(lid_estimates), ks
        else:
            return lid_estimates, np.mean(lid_estimates)

def sk_2NN(X, dists, knnidx, k = 10, correct = True, w = None, return_ks = False, use_w = 'indirect', smooth=False, geo=None):
    if w is None:
        twonn = skdim.id.TwoNN()
        twonn.fit_pw(X, precomputed_knn=knnidx, smooth=False, n_neighbors=k, n_jobs=1)
        lid_estimates = twonn.dimension_pw_
        if return_ks:
            ks = np.repeat(k, X.shape[0])
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates), ks
            else:
                return lid_estimates, np.mean(lid_estimates), ks
        else:
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
            else:
                return lid_estimates, np.mean(lid_estimates)
    elif use_w == 'indirect':
        n = X.shape[0]
        lid_estimates = np.empty(n)
        ks = np.empty(n)
        twonn = skdim.id.TwoNN()
        twonn = fit_pw_with_list(twonn, X, knnidx, smooth=False)
        lid_estimates = twonn.dimension_pw_
        if return_ks:
            return lid_estimates, np.mean(lid_estimates), ks
        else:
            return lid_estimates, np.mean(lid_estimates)
    else:
        NotImplemented(f"Not implemented use_w: {use_w}")


def sk_ESS(X, dists, knnidx, k = 10, correct = True, w=None, return_ks = False, use_w = 'indirect', smooth=False, geo=None):
    if w is None:
        est_ess = skdim.id.ESS()
        est_ess._fit(X, dists, knnidx)
        lid_estimates = est_ess.dimension_pw_
        if return_ks:
            ks = np.repeat(k, X.shape[0])
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates), ks
            else:
                return lid_estimates, np.mean(lid_estimates), ks
        else:
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
            else:
                return lid_estimates, np.mean(lid_estimates)
    elif use_w == 'indirect':
        n = X.shape[0]
        lid_estimates = np.empty(n)
        ks = np.empty(n)
        est_ess = ESS()
        est_ess._fit(X, dists=dists, knnidx=knnidx)
        lid_estimates = est_ess.dimension_pw_
        if return_ks:
            return lid_estimates, np.mean(lid_estimates), ks
        else:
            return lid_estimates, np.mean(lid_estimates)
    else:
        NotImplemented(f"Not implemented use_w: {use_w}")

def sk_MADA(X, dists, knnidx, k = 10, correct = True, w=None, return_ks = False, use_w = 'indirect', smooth=False, geo=None, smooth_style='code2', bag_indices=None):
    if w is None:
        mada = MADA()
        mada._fit(X, dists=dists, knnidx=knnidx)
        lid_estimates = mada.dimension_pw_
        if return_ks:
            ks = np.repeat(k, X.shape[0])
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo, smooth_style=smooth_style)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates), ks
            else:
                return lid_estimates, np.mean(lid_estimates), ks
        else:
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo, smooth_style=smooth_style, bag_indices=bag_indices)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
            else:
                return lid_estimates, np.mean(lid_estimates)
    elif use_w == 'indirect':
        n = X.shape[0]
        lid_estimates = np.empty(n)
        ks = np.empty(n)
        mada = MADA()
        mada._fit(X, dists=dists, knnidx=knnidx)
        lid_estimates = mada.dimension_pw_
        if return_ks:
            return lid_estimates, np.mean(lid_estimates), ks
        else:
            return lid_estimates, np.mean(lid_estimates)
    else:
        NotImplemented(f"Not implemented use_w: {use_w}")

def LID_MLE(X, k=10, progress_bar = False, w=None):
    m = X.shape[0]
    estimates = np.empty(m)
    dists, knnidx = skdim._commonfuncs.get_nn(X, k=k, n_jobs=1)
    if progress_bar:
        for i in tqdm(range(m)):
            estimates[i] = MLE(dists[i])
    else:
        for i in range(m):
            estimates[i] = MLE(dists[i])
    return estimates, np.mean(estimates)

def LID_MOM(X, k=10, progress_bar = False, w=None):
    m = X.shape[0]
    estimates = np.empty(m)
    dists, knnidx = skdim._commonfuncs.get_nn(X, k=k, n_jobs=1)
    if progress_bar:
        for i in tqdm(range(m)):
            estimates[i] = MOM(dists[i])
    else:
        for i in range(m):
            estimates[i] = MOM(dists[i])
    return estimates, np.mean(estimates)

def sk_MOM_full(X, k = 10, correct = True, dists=None, knnidx=None, w=None, smooth=False, geo=None):
    if (dists is None) or (knnidx is None):
        dists, knnidx = skdim._commonfuncs.get_nn(X, k=k, n_jobs=1)
    mom = skdim.id.MOM()
    lid_estimates = mom._mom(dists)
    if smooth:
        lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
        return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
    else:
        return lid_estimates, np.mean(lid_estimates)

def sk_TLE_full(X, k = 10, correct = True, dists = None, knnidx= None, w=None, smooth=False, geo=None):
    if (dists is None) or (knnidx is None):
        dists, knnidx = skdim._commonfuncs.get_nn(X, k=k, n_jobs=1)
    tle = skdim.id.TLE()
    tle._fit(X, dists=dists, knnidx=knnidx)
    lid_estimates = tle.dimension_pw_
    if smooth:
        lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
        return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
    else:
        return lid_estimates, np.mean(lid_estimates)

def sk_MLE_full(X, k = 10, correct = True, dists = None, knnidx= None, w=None, smooth=False, geo=None):
    if (dists is None) or (knnidx is None):
        dists, knnidx = skdim._commonfuncs.get_nn(X, k=k, n_jobs=1)
    mle = skdim.id.MLE()
    mle.fit(X, n_neighbors=k, comb='mean', precomputed_knn_arrays=(dists, knnidx))
    if correct:
        lid_estimates = k / (k - 1) * mle.dimension_pw_
    else:
        lid_estimates = mle.dimension_pw_
    if smooth:
        lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
        return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
    else:
        return lid_estimates, np.mean(lid_estimates)

def sk_2NN_full(X, k = 10, correct = False, dists=None, knnidx=None, w=None, smooth=False, geo=None):
    if (dists is None) or (knnidx is None):
        dists, knnidx = skdim._commonfuncs.get_nn(X, k=k, n_jobs=1)
    twonn = skdim.id.TwoNN()
    twonn.fit_pw(X, precomputed_knn=knnidx, smooth=False, n_neighbors=k, n_jobs=1)
    lid_estimates = twonn.dimension_pw_
    if smooth:
        lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
        return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
    else:
        return lid_estimates, np.mean(lid_estimates)

def sk_ESS_full(X, k = 10, correct = True, dists = None, knnidx= None, w=None, smooth=False, geo=None):
    if (dists is None) or (knnidx is None):
        dists, knnidx = skdim._commonfuncs.get_nn(X, k=k, n_jobs=1)
    est_ess = skdim.id.ESS()
    est_ess._fit(X, dists, knnidx)
    lid_estimates = est_ess.dimension_pw_
    if smooth:
        lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
        return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
    else:
        return lid_estimates, np.mean(lid_estimates)

def sk_MADA_full(X, k = 10, correct = True, dists = None, knnidx= None, w=None, smooth=False, geo=None):
    if (dists is None) or (knnidx is None):
        dists, knnidx = skdim._commonfuncs.get_nn(X, k=k, n_jobs=1)
    mada = MADA()
    mada._fit(X, dists=dists, knnidx=knnidx)
    lid_estimates = mada.dimension_pw_
    if smooth:
        lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
        return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
    else:
        return lid_estimates, np.mean(lid_estimates)

def LIDL_full(X, k = 10, correct = True, dists = None, knnidx= None, model_type="gm", w=None, smooth=False, geo=None):
    gm = dim_estimators.LIDL(model_type=model_type)
    # the more deltas, the more accurate the estimate
    deltas = [0.025, 0.02835781, 0.03216662, 0.036487, 0.04138766,0.04694655,\
              0.05325205, 0.06040447, 0.06851755, 0.07772031, 0.08815913, 0.1]
    result = gm(deltas, X, X)
    lid_estimates = np.array(result)
    if smooth:
        lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
        return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
    else:
        return lid_estimates, np.mean(lid_estimates)

def LIDkit_TLE(data_array, subsample_indexes = None, k = 10, w=None, return_ks=False, use_w='indirect', smooth=False, geo=None):
    tle = LIDEstimatorTleNumpy(k=k)
    if subsample_indexes is not None:
        reference = data_array[subsample_indexes]
    else:
        reference = data_array
    if w is None:
        tle.estimate(query_points=data_array, reference=reference, k=k)
        lid_estimates = tle.lids
        if return_ks:
            ks = np.repeat(k, data_array.shape[0])
            if smooth:
                lid_smoothed_estimates, _ = smoothing(data_array, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
                return lid_smoothed_estimates
            else:
                return lid_estimates
        else:
            if smooth:
                lid_smoothed_estimates, _ = smoothing(data_array, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
                return lid_smoothed_estimates
            else:
                return lid_estimates
    else:
        NotImplemented(f"Not implemented use_w: {use_w}")

def Ricardo_TLE(X, dists, knnidx, k = 10, w=None, return_ks=False, use_w='indirect', smooth=False, geo=None, **kwargs):
    if w is None:
        lid_estimates = TLE_LID_Estimator(data_array=X, neighbourhood_size=k, return_smoothed=False, simple_smooth=False, geo=False, nn_dist=dists, KNN_indices=knnidx, **kwargs)
        if return_ks:
            ks = np.repeat(k, X.shape[0])
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates), ks
            else:
                return lid_estimates, np.mean(lid_estimates), ks
        else:
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
            else:
                return lid_estimates, np.mean(lid_estimates)
    else:
        NotImplemented(f"Not implemented use_w: {use_w}")

def Ricardo_MLE(X, dists, knnidx, k = 10, w=None, return_ks=False, use_w='indirect', smooth=False, geo=None, **kwargs):
    if w is None:
        lid_estimates = MLE_LID_Estimator(data_array=X, neighbourhood_size=k, return_smoothed=False, simple_smooth=False, geo=False, nn_dist=dists, KNN_indices=knnidx, **kwargs)
        if return_ks:
            ks = np.repeat(k, X.shape[0])
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates), ks
            else:
                return lid_estimates, np.mean(lid_estimates), ks
        else:
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
            else:
                return lid_estimates, np.mean(lid_estimates)
    else:
        NotImplemented(f"Not implemented use_w: {use_w}")

def Ricardo_MADA(X, dists, knnidx, k = 10, w=None, return_ks=False, use_w='indirect', smooth=False, geo=None, **kwargs):
    if w is None:
        lid_estimates = MADA_LID_Estimator(data_array=X, neighbourhood_size=k, return_smoothed=False, simple_smooth=False, geo=False, nn_dist=dists, KNN_indices=knnidx, **kwargs)
        if return_ks:
            ks = np.repeat(k, X.shape[0])
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates), ks
            else:
                return lid_estimates, np.mean(lid_estimates), ks
        else:
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
            else:
                return lid_estimates, np.mean(lid_estimates)
    else:
        NotImplemented(f"Not implemented use_w: {use_w}")