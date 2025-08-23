import numpy as np
from skdim._commonfuncs import LocalEstimator
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed

class TLE(LocalEstimator):
    """Intrinsic dimension estimation using the Tight Local intrinsic dimensionality Estimator algorithm. [Amsaleg2019]_ [IDRadovanović]_

    Parameters
    ----------
    epsilon: float
    """

    _N_NEIGHBORS = 20  # Default fallback if needed

    def __init__(self, epsilon=1e-4):
        self.epsilon = epsilon

    def _fit(self, X, dists_list, knnidx_list, n_jobs=1):  # <-- CHANGED: dists and knnidx now lists
        self.dimension_pw_ = np.zeros(len(X))
        if n_jobs > 1:
            with Parallel(n_jobs=n_jobs) as parallel:
                results = parallel(
                    delayed(self._idtle)(
                        X[knnidx_list[i]], np.array(dists_list[i])  # <-- CHANGED
                    ) for i in range(len(X))
                )
            self.dimension_pw_ = np.array(results)
        else:
            for i in range(len(X)):
                nn = X[knnidx_list[i]]  # <-- CHANGED
                dists = np.array(dists_list[i])  # <-- CHANGED
                if len(dists) < 2:
                    self.dimension_pw_[i] = np.inf
                else:
                    self.dimension_pw_[i] = self._idtle(nn, dists)

    def _idtle(self, nn, dists):  # <-- CHANGED: dists is now 1D array (not 2D slice)
        dists = dists.reshape(1, -1)  # <-- ADDED: ensure shape (1, k)
        r = dists[0, -1]  # distance to farthest neighbor (still valid for fixed-k variant)

        if r == 0:
            raise ValueError("All k-NN distances are zero!")

        n_neighbors = dists.shape[1]
        V = squareform(pdist(nn))  # pairwise distances between neighbors

        Di = np.tile(dists.T, (1, n_neighbors))
        Dj = Di.T
        Z2 = 2 * Di ** 2 + 2 * Dj ** 2 - V ** 2

        S = r * (
            ((Di ** 2 + V ** 2 - Dj ** 2) ** 2 + 4 * V ** 2 * (r ** 2 - Di ** 2)) ** 0.5
            - (Di ** 2 + V ** 2 - Dj ** 2)
        ) / (2 * (r ** 2 - Di ** 2))

        T = r * (
            ((Di ** 2 + Z2 - Dj ** 2) ** 2 + 4 * Z2 * (r ** 2 - Di ** 2)) ** 0.5
            - (Di ** 2 + Z2 - Dj ** 2)
        ) / (2 * (r ** 2 - Di ** 2))

        Dr = (dists == r).squeeze()
        S[Dr, :] = r * V[Dr, :] ** 2 / (r ** 2 + V[Dr, :] ** 2 - Dj[Dr, :] ** 2)
        T[Dr, :] = r * Z2[Dr, :] / (r ** 2 + Z2[Dr, :] - Dj[Dr, :] ** 2)

        Di0 = (Di == 0).squeeze()
        T[Di0] = Dj[Di0]
        S[Di0] = Dj[Di0]

        Dj0 = (Dj == 0).squeeze()
        T[Dj0] = r * V[Dj0] / (r + V[Dj0])
        S[Dj0] = r * V[Dj0] / (r + V[Dj0])

        V0 = (V == 0).squeeze()
        np.fill_diagonal(V0, False)
        T[V0] = r
        S[V0] = r
        nV0 = np.sum(V0)

        TSeps = (T < self.epsilon) | (S < self.epsilon)
        np.fill_diagonal(TSeps, 0)
        nTSeps = np.sum(TSeps)
        T[TSeps] = r
        T = np.log(T / r)
        S[TSeps] = r
        S = np.log(S / r)
        np.fill_diagonal(T, 0)
        np.fill_diagonal(S, 0)

        s1t = np.sum(T)
        s1s = np.sum(S)

        Deps = dists < self.epsilon
        nDeps = np.sum(Deps, dtype=int)
        dists = dists[nDeps:]
        s2 = np.sum(np.log(dists / r))

        ID = -2 * (n_neighbors ** 2 - nTSeps - nDeps - nV0) / (s1t + s1s + 2 * s2)
        return ID