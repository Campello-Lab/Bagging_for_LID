import numpy as np
import bisect
import warnings
from scipy.special import gamma
from functools import lru_cache
from joblib import Parallel, delayed
from skdim._commonfuncs import (
    lens,
    indComb,
    indnComb,
    efficient_indnComb,
    check_random_generator,
)
from sklearn.utils.validation import check_array
from skdim._commonfuncs import LocalEstimator


class ESS(LocalEstimator):
    """
    Intrinsic dimension estimation using the Expected Simplex Skewness algorithm. [Johnsson2015]_

    Parameters
    ----------
    ver: str, 'a' or 'b'
       See Johnsson et al. (2015).
    d: int, default=1
        For ver='a', any value of d is possible; for ver='b', only d=1 is supported.
    """

    def __init__(self, ver="a", d=1, random_state=None):
        self.ver = ver
        self.d = d
        self.random_state = random_state

    def _fit(self, X, dists=None, knnidx=None, n_jobs=1):  # <-- dists unused, knnidx now can be a list
        self.random_state_ = check_random_generator(self.random_state)

        self.dimension_pw_, self.essval_ = np.zeros(len(X)), np.zeros(len(X))

        if n_jobs > 1:
            with Parallel(n_jobs=n_jobs) as parallel:
                results = parallel(
                    delayed(self._ess_wrapper)(
                        X[knnidx[i]], i  # <-- changed for list-based access
                    ) for i in range(len(X))
                )
        else:
            results = [self._ess_wrapper(X[knnidx[i]], i) for i in range(len(X))]

        self.dimension_pw_[:], self.essval_[:] = zip(*results)

    def fit_once(self, X, y=None):
        """ Fit ESS on a single neighborhood. /!\ Not meant to be used on a complete dataset - X should be a local patch of a dataset, otherwise call .fit()
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples. /!\ Should be a local patch of a dataset
        y : dummy parameter to respect the sklearn API

        Returns
        -------
        self : object
            Returns self.
        """
        self.random_state_ = check_random_generator(self.random_state)
        X = check_array(X, ensure_min_samples=2, ensure_min_features=2)

        self.dimension_, self.essval_ = self._essLocalDimEst(X)

        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _ess_wrapper(self, neighborhood, idx):
        if len(neighborhood) < self.d + 1:
            warnings.warn(
                f"[ESS Warning] Query point {idx} has only {len(neighborhood)} neighbors — "
                f"not enough to build {self.d}-dim simplices (requires at least {self.d + 1}). "
                "Returning NaN for this point."
            )
            return (np.nan, np.nan)
        else:
            return self._essLocalDimEst(neighborhood)

    def fit_once(self, X, y=None):
        self.random_state_ = check_random_generator(self.random_state)
        X = check_array(X, ensure_min_samples=2, ensure_min_features=2)

        self.dimension_, self.essval_ = self._essLocalDimEst(X)
        self.is_fitted_ = True
        return self

    def _essLocalDimEst(self, X):
        essval = self._computeEss(X, verbose=False)
        if np.isnan(essval):
            return (np.nan, essval)

        mindim = 1
        maxdim = 20
        dimvals = self._essReference(maxdim, mindim)
        while (self.ver == "a" and essval > dimvals[maxdim - 1]) or (
            self.ver == "b" and essval < dimvals[maxdim - 1]
        ):
            mindim = maxdim + 1
            maxdim = 2 * (maxdim - 1)
            dimvals = np.append(dimvals, self._essReference(maxdim, mindim))

        if self.ver == "a":
            i = np.searchsorted(dimvals[mindim - 1 : maxdim], essval)
        else:
            i = len(range(mindim, maxdim + 1)) - np.searchsorted(
                dimvals[mindim - 1 : maxdim][::-1], essval
            )

        de_integer = mindim + i - 1
        de_fractional = (essval - dimvals[de_integer - 1]) / (
            dimvals[de_integer] - dimvals[de_integer - 1]
        )
        return (de_integer + de_fractional, essval)

    def _computeEss(self, X, verbose=False):
        p = self.d + 1
        n = X.shape[1]
        if p > n:
            return 0 if self.ver == "a" else 1

        vectors = self._vecToCs_onedir(X, 1)
        groups = efficient_indnComb(len(vectors), p, self.random_state_)
        Alist = [vectors[group] for group in groups]
        weight = np.prod([lens(l) for l in Alist], axis=1)

        if self.ver == "a":
            vol = np.array([np.linalg.det(vecgr.dot(vecgr.T)) for vecgr in Alist])
            if np.any(vol < 0) and not hasattr(self, "_warned"):
                self._warned = True
                print(
                    "Warning: data might contain duplicate rows, affecting results."
                )
            vol = np.sqrt(np.abs(vol))
            return np.sum(vol) / np.sum(weight)

        elif self.ver == "b":
            if self.d == 1:
                proj = [np.abs(np.sum(vecgr[0, :] * vecgr[1, :])) for vecgr in Alist]
                return np.sum(proj) / np.sum(weight)
            else:
                raise ValueError('For ver == "b", d > 1 is not supported.')

        raise ValueError("Not a valid version")

    @staticmethod
    def _vecToC_onedir(
        points, add_mids=False, weight_mids=1, mids_maxdist=float("inf")
    ):

        # Mean center data
        center = np.mean(points, axis=0)
        vecOneDir = points - center

        if add_mids:  # Add midpoints
            pt1, pt2, ic = indComb(len(vecOneDir))
            mids = (vecOneDir[ic[pt1],] + vecOneDir[ic[pt2],]) / 2
            dist = lens(vecOneDir[ic[pt1],] - vecOneDir[ic[pt2],])
            # Remove midpoints for very distant
            mids = mids[
                dist <= mids_maxdist,
            ]
            # points
            vecOneDir = np.vstack((vecOneDir, weight_mids * mids))

        return vecOneDir
    def _vecToCs_onedir(self, points, n_group):
        if n_group == 1:
            return self._vecToC_onedir(points)

        NN = len(points)
        ind_groups = indnComb(NN, n_group)
        reshape_ind_groups = ind_groups.reshape((n_group, -1))
        point_groups = points[reshape_ind_groups, :].reshape((-1, n_group))
        group_centers = np.array(
            [points[ind_group, :].mean(axis=0) for ind_group in ind_groups]
        )
        centers = group_centers[np.repeat(np.arange(len(group_centers)), n_group), :]
        return point_groups - centers

    @lru_cache()
    def _essReference(self, maxdim, mindim=1):

        if maxdim <= self.d + 2:
            raise ValueError(
                "maxdim (", maxdim, ") must be larger than d + 2 (", self.d + 2, ")",
            )

        if self.ver == "a":
            # ID(n) = factor1(n)**d * factor2(n)
            # factor1(n) = gamma(n/2)/gamma((n+1)/2)
            # factor2(n) = gamma(n/2)/gamma((n-d)/2)

            # compute factor1
            # factor1(n) = gamma(n/2)/gamma((n+1)/2)
            # [using the rule gamma(n+1) = n * gamma(n)] repeatedly
            # = gamma(1/2)/gamma(2/2) * prod{j \in J1} j/(j+1) if n is odd
            # = gamma(2/2)/gamma(3/2) * prod(j \in J2) j/(j+1) if n is even
            # where J1 = np.arange(1, n-2, 2), J2 = np.arange(2, n-2, 2)
            J1 = np.array([1 + i for i in range(0, maxdim + 2, 2) if 1 + i <= maxdim])
            J2 = np.array([2 + i for i in range(0, maxdim + 2, 2) if 2 + i <= maxdim])
            factor1_J1 = (
                    gamma(1 / 2)
                    / gamma(2 / 2)
                    * np.concatenate((np.array([1]), np.cumprod(J1 / (J1 + 1))[:-1]))
            )
            factor1_J2 = (
                    gamma(2 / 2)
                    / gamma(3 / 2)
                    * np.concatenate((np.array([1]), np.cumprod(J2 / (J2 + 1))[:-1]))
            )
            factor1 = np.repeat(np.nan, maxdim)
            factor1[J1 - 1] = factor1_J1
            factor1[J2 - 1] = factor1_J2

            # compute factor2
            # factor2(n) = gamma(n/2)/gamma((n-d)/2)
            # = gamma((d+1)/2)/gamma(1/2) * prod{k \in K1} k/(k-d) if n-d is odd
            # = gamma((d+2)/2)/gamma(2/2) * prod(k \in K2) k/(k-d) if n-d is even
            # where K1 = np.arange(d+1, n-2, 2), K2 = np.arange(d+2, n-2, 2)
            # if n > d+2, otherwise 0.
            K1 = np.array(
                [
                    self.d + 1 + i
                    for i in range(0, maxdim + 2, 2)
                    if self.d + 1 + i <= maxdim
                ]
            )
            K2 = np.array(
                [
                    self.d + 2 + i
                    for i in range(0, maxdim + 2, 2)
                    if self.d + 2 + i <= maxdim
                ]
            )
            factor2_K1 = (
                    gamma((self.d + 1) / 2)
                    / gamma(1 / 2)
                    * np.concatenate((np.array([1]), np.cumprod(K1 / (K1 - self.d))[:-1]))
            )
            factor2_K2 = (
                    gamma((self.d + 2) / 2)
                    / gamma(2 / 2)
                    * np.concatenate((np.array([1]), np.cumprod(K2 / (K2 - self.d))[:-1]))
            )
            factor2 = np.zeros(maxdim)
            factor2[K1 - 1] = factor2_K1
            factor2[K2 - 1] = factor2_K2
            # compute ID
            ID = factor1 ** self.d * factor2
            ID = ID[mindim - 1: maxdim]
            return ID

    # Other methods (_vecToC_onedir, _vecToCs_onedir, _essReference) remain unchanged