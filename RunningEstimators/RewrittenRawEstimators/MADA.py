import numpy as np
from skdim._commonfuncs import LocalEstimator

class MADA(LocalEstimator):

    _N_NEIGHBORS = 20

    def __init__(self, DM=False):
        self.DM = DM

    def _fit(self, X, dists=None, knnidx=None, n_jobs=1):
        self.dimension_pw_ = self._mada(dists)

    def _mada(self, dists_list):
        n = len(dists_list)
        ests = np.zeros(n)
        for i in range(n):
            dists1 = np.sort(np.array(dists_list[i]))
            k_q = len(dists1)
            if k_q < 2:
                ests[i] = np.nan
                continue
            k_half = int(np.floor(k_q / 2))
            rk = dists1[k_q - 1]
            rk2 = dists1[k_half - 1]
            if rk2 == 0 or rk == rk2:
                ests[i] = np.nan
            else:
                ests[i] = np.log(2) / np.log(rk / rk2)
        return ests