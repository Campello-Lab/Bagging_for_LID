import numpy as np
from skdim._commonfuncs import LocalEstimator


class MADA(LocalEstimator):
    """
    Intrinsic dimension estimation using the Manifold-Adaptive Dimension Estimation algorithm. [Farahmand2007]_, [IDHino]_

    MADA uses a variant of fractal dimension called the local information dimension.
    It estimates the local ID by comparing distances to the k-th and (k/2)-th neighbors.

    Parameters
    ----------
    DM: bool
        Whether input is a full precomputed distance matrix (ignored here in favor of local lists)
    """

    _N_NEIGHBORS = 20

    def __init__(self, DM=False):
        self.DM = DM

    def _fit(self, X, dists=None, knnidx=None, n_jobs=1):  # <-- CHANGED: now expects dists as list of arrays
        self.dimension_pw_ = self._mada(dists)  # <-- CHANGED: we now pass dists_list directly

    def _mada(self, dists_list):  # <-- CHANGED: accepts dists_list
        """
        Estimate LID per point using variable-length kNN distances from dists_list.
        """

        n = len(dists_list)
        ests = np.zeros(n)

        for i in range(n):
            dists1 = np.sort(np.array(dists_list[i]))  # sort just in case  <-- ADDED
            k_q = len(dists1)  # <-- CHANGED: allow different k per query

            if k_q < 2:
                ests[i] = np.nan  # <-- HANDLE EDGE CASE
                continue

            k_half = int(np.floor(k_q / 2))

            rk = dists1[k_q - 1]  # k-th nearest distance
            rk2 = dists1[k_half - 1]  # k/2-th nearest distance

            if rk2 == 0 or rk == rk2:
                ests[i] = np.nan  # avoid division by zero / log(1)
            else:
                ests[i] = np.log(2) / np.log(rk / rk2)

        return ests