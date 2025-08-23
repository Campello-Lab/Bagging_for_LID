from copy import deepcopy
import numpy as np

def fit_pw_with_list(estimator, X, knnidx_list, smooth=False):
    """
    Wrapper to call fit_pw() on an skdim estimator using a list of variable-length neighbor indices.

    Parameters
    ----------
    estimator : skdim GlobalEstimator instance (e.g., skdim.id.TwoNN())
        The estimator you want to apply locally.
    X : np.ndarray, shape (n_samples, n_features)
        The full dataset.
    knnidx_list : list of lists/arrays of int
        Each element is a list of neighbor indices for query i.
    smooth : bool, optional
        Whether to smooth the estimates as in original fit_pw.

    Returns
    -------
    estimator : fitted estimator
        With estimator.dimension_pw_ (and optionally estimator.dimension_pw_smooth_) populated.
    """
    n = len(knnidx_list)
    dimension_pw = np.empty(n)
    dimension_pw[:] = np.nan  # prefill with NaNs

    for i in range(n):
        idx = [i] + list(knnidx_list[i])  # include query point
        X_local = X[idx]
        est_i = deepcopy(estimator)
        try:
            est_i.fit(X_local)
            dimension_pw[i] = est_i.dimension_
        except Exception as e:
            print(f"Query {i}: Estimation failed ({e})")
            dimension_pw[i] = np.inf

    estimator.dimension_pw_ = dimension_pw

    if smooth:
        dimension_pw_smooth = np.zeros(n)
        for i, neighbors in enumerate(knnidx_list):
            values = [dimension_pw[j] for j in neighbors if j < n and not np.isnan(dimension_pw[j])]
            values.append(dimension_pw[i])
            dimension_pw_smooth[i] = np.mean(values)
        estimator.dimension_pw_smooth_ = dimension_pw_smooth

    return estimator