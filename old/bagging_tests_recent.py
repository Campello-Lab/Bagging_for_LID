import skdim
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import matplotlib
#from matplotlib.backends.backend_pgf import FigureCanvasPgf
#matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
import ast
import multiprocessing
import random
import itertools
import pickle
import re
from collections import defaultdict
import sys
import os
from scipy.stats import chi2
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import math
import yaml
import torch
from data.datasets.generated import LIDSyntheticDataset
from data.distributions.manifold_mixture import ManifoldMixture, AffineManifoldMixture
import pathlib
from sklearn.manifold import MDS
from matplotlib.patches import Patch
from pathlib import Path
from data.distributions import (
    Lollipop, SwissRoll, Torus, Mondrian, MultiscaleGaussian, VonMisesEuclidean,
    ManifoldMixture, AffineManifoldMixture, SquigglyManifoldMixture
)

cloned_folders = [
    r"C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\lidl",
]
# Add each cloned folder to sys.path
for folder in cloned_folders:
    if folder not in sys.path:
        sys.path.append(folder)

from pathlib import Path
import os
import yaml
import numpy as np
from data.datasets.generated import LIDSyntheticDataset
from data.distributions.lollipop import Lollipop
from data.distributions.swiss_roll import SwissRoll
from data.distributions.torus import Torus
from data.distributions.mondrian import Mondrian
from data.distributions.multiscale_gaussian import MultiscaleGaussian
from data.distributions.von_mises import VonMisesEuclidean
from data.distributions.manifold_mixture import (
    ManifoldMixture,
    AffineManifoldMixture,
    SquigglyManifoldMixture,
)
##############################################################################################################################################################################################################################################################

###############################################################################################################################DATASET GENERATION###############################################################################################################################

def load_all_datasets(n=10000):
    DIST_CLASS_MAP = {
        "Lollipop": Lollipop,
        "SwissRoll": SwissRoll,
        "Torus": Torus,
        "Mondrian": Mondrian,
        "MultiscaleGaussian": MultiscaleGaussian,
        "VonMisesEuclidean": VonMisesEuclidean,
        "ManifoldMixture": ManifoldMixture,
        "AffineManifoldMixture": AffineManifoldMixture,
        "SquigglyManifoldMixture": SquigglyManifoldMixture,
    }
    PROJECT_ROOT = Path(r"C:\Users\User\PycharmProjects\pythonProject3")
    yaml_dirs = [PROJECT_ROOT / "dgm_geometry" / "conf" / "dataset"] + [
        PROJECT_ROOT / "dgm_geometry" / "conf" / "dataset" / "manifolds" / subdir
        for subdir in ["large", "medium", "small", "toy"]]
    def is_flow_based(dist_cfg):
        target = dist_cfg.get("_target_", "")
        if "RQNSF" in target or "Flow" in target:
            return True
        diff = dist_cfg.get("diffeomorphism_instantiator")
        if isinstance(diff, list):
            return any("RQNSF" in str(d.get("_target_", "")) or "Flow" in str(d.get("_target_", "")) for d in diff if isinstance(d, dict))
        if isinstance(diff, dict):
            return "RQNSF" in str(diff.get("_target_", "")) or "Flow" in str(diff.get("_target_", ""))
        return False
    dataset_dict = {}
    for base_dir in yaml_dirs:
        if not base_dir.exists():
            continue
        for fname in os.listdir(base_dir):
            if not fname.endswith(".yaml"):
                continue
            full_path = base_dir / fname
            name = fname.replace(".yaml", "")
            with open(full_path, "r") as f:
                cfg = yaml.safe_load(f)
            val_cfg = cfg.get("val")
            if not val_cfg:
                continue
            dist_cfg = val_cfg.get("distribution") if isinstance(val_cfg.get("distribution"), dict) else cfg.get("train", {}).get("distribution")
            if not dist_cfg:
                print(f"[Skipped] {name} missing 'distribution' config.")
                continue
            if isinstance(dist_cfg.get("device"), dict):
                print(f"[Skipped] {name} uses unresolved device Hydra reference.")
                continue
            if is_flow_based(dist_cfg):
                print(f"[Skipped] {name} uses flow-based models (e.g., RQNSF).")
                continue
            target = dist_cfg.get("_target_", "")
            cls_name = target.split(".")[-1]
            DistClass = DIST_CLASS_MAP.get(cls_name)
            if not DistClass:
                print(f"[Skipped] Unknown dist type in {name} ({cls_name})")
                continue
            dist_cfg = {k: v for k, v in dist_cfg.items() if k != "_target_"}
            standardize = val_cfg.get("standardize", False)
            try:
                for key in ["frequency", "amplitude", "scale", "kappa_control"]:
                    if isinstance(dist_cfg.get(key), str):
                        val = float(dist_cfg[key])
                        dist_cfg[key] = int(val) if val.is_integer() else val
                if isinstance(dist_cfg.get("manifold_dims"), list):
                    dist_cfg["manifold_dims"] = [
                        int(x) if isinstance(x, str) and x.isdigit() else x
                        for x in dist_cfg["manifold_dims"]
                    ]
                dist = DistClass(**dist_cfg)
                dataset = LIDSyntheticDataset(size=n, distribution=dist, standardize=standardize)
                dataset_dict[name] = {
                    "x": dataset.x.numpy(),
                    "lid": dataset.lid.numpy(),
                    "idx": dataset.idx.numpy(),
                }
                print(f"[Loaded] {name}: {dataset.x.shape[0]} points in {dataset.x.shape[1]}D")
            except Exception as e:
                print(f"[Error] Failed to load {name}: {e}")
    return dataset_dict

def lollipop_dataset(bs, seed=0):
    np.random.seed(seed)
    cs = int(0.95 * bs)
    x = np.zeros((bs, 2))
    intrinsic_dims = np.zeros(bs, dtype=int)
    r = np.random.uniform(size=cs)
    fi = np.random.uniform(0, 2 * np.pi, size=cs)
    x[:cs, 0] = r ** 0.5 * np.sin(fi)
    x[:cs, 1] = r ** 0.5 * np.cos(fi)
    x[:cs] += 2
    intrinsic_dims[:cs] = 2
    stick = np.random.uniform(high=2 - 1 / np.sqrt(2), size=(bs - cs))
    x[cs:, 0] = stick
    x[cs:, 1] = stick
    intrinsic_dims[cs:] = 1
    return x, intrinsic_dims

def lollipop_dataset_0(bs, seed=0):
    np.random.seed(seed)
    cs = int(0.94 * bs)
    cp = int(0.99 * bs)
    x = np.zeros((bs, 2))
    intrinsic_dims = np.zeros(bs, dtype=int)
    r = np.random.uniform(size=cs)
    fi = np.random.uniform(0, 2 * np.pi, size=cs)
    x[:cs, 0] = r ** 0.5 * np.sin(fi)
    x[:cs, 1] = r ** 0.5 * np.cos(fi)
    x[:cs] += 2
    intrinsic_dims[:cs] = 2
    stick = np.random.uniform(high=2 - 1 / np.sqrt(2), size=(cp - cs))
    x[cs:cp, 0] = stick
    x[cs:cp, 1] = stick
    intrinsic_dims[cs:cp] = 1
    x[cp:] = np.random.normal(loc=(-.5, -.5), scale=1e-3, size=(bs - cp, 2))
    intrinsic_dims[cp:] = 2
    x = np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1)
    return x, intrinsic_dims


def lollipop_dataset_0_dense_head(bs, seed=0):
    np.random.seed(seed)
    cs = int(0.5 * bs)
    cp = int(0.7 * bs)
    x = np.zeros((bs, 2))
    intrinsic_dims = np.zeros(bs, dtype=int)
    r = np.random.uniform(size=cs)
    fi = np.random.uniform(0, 2 * np.pi, size=cs)
    x[:cs, 0] = r ** 0.5 * np.sin(fi)
    x[:cs, 1] = r ** 0.5 * np.cos(fi)
    x[:cs] += 2
    intrinsic_dims[:cs] = 2
    stick = np.random.uniform(high=2 - 1 / np.sqrt(2), size=(cp - cs))
    x[cs:cp, 0] = stick
    x[cs:cp, 1] = stick
    intrinsic_dims[cs:cp] = 1
    x[cp:] = np.random.normal(loc=(-.5, -.5), scale=1e-3, size=(bs - cp, 2))
    intrinsic_dims[cp:] = 2
    x = np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1)
    return x, intrinsic_dims

def get_datasets(used_params=None, n=2500):
    data_gen = skdim.datasets.BenchmarkManifolds()
    all_keys = [key for key in data_gen.dict_gen]
    keys = all_keys[0:4] + all_keys[5:13] + all_keys[14:17] + all_keys[19:21]
    # True (d, m) pairs: intrinsic and ambient dimensions
    d_vals = [10, 3, 4, 4, 2, 6, 2, 12, 20, 10, 17, 24, 2, 20, 2, 18, 24]
    m_vals = [11, 5, 6, 8, 3, 36, 3, 72, 20, 11, 18, 25, 3, 20, 3, 72, 96]
    params = [(keys[i], [d_vals[i], m_vals[i]]) for i in range(len(keys))]
    if used_params is None:
        used_params = dict(params)
    # Generate standard datasets
    pairs = [(key,
              [data_gen.dict_gen[key](n=n, d=used_params[key][0], dim=used_params[key][1]),
               used_params[key][0],  # true intrinsic dim
               used_params[key][1]])  # ambient dim
             for key in used_params]
    result = dict(pairs)
    def add_lollipop(name, func, m_val):
        data, intrinsic_dim_array = func(n)
        result[name] = [data, intrinsic_dim_array, m_val]
    add_lollipop("lollipop_", lollipop_dataset, 2)
    add_lollipop("lollipop_0", lollipop_dataset_0, 3)
    add_lollipop("lollipop_0_dense_head", lollipop_dataset_0_dense_head, 3)
    #add_lollipop("plane_line_linear_half_spaces", twod_generate_touching_halfspaces, 4)
    #add_lollipop("plane_line_cube_linear_half_spaces", threed_generate_touching_halfspaces, 4)
    #add_lollipop("plane_line_plane_linear_half_spaces", twoplanes_generate_touching_halfspaces, 4)
    new_manifolds = load_all_datasets(n=n)
    new_dict = {key: [new_manifolds[key]['x'],np.array(new_manifolds[key]['lid']),new_manifolds[key]['x'].shape[1],new_manifolds[key]['idx']] for key in new_manifolds}
    result = result | new_dict
    return result

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

def sk_MOM(X, dists, knnidx, k = 10, w=None, return_ks = False):
    if w is None:
        mom = skdim.id.MOM()
        lid_estimates = mom._mom(dists)
        return lid_estimates, np.mean(lid_estimates)

def sk_TLE(X, dists, knnidx, k = 10, w=None, return_ks = False):
    if w is None:
        tle = skdim.id.TLE()
        tle._fit(X, dists=dists, knnidx=knnidx)
        lid_estimates = tle.dimension_pw_
        return lid_estimates, np.mean(lid_estimates)
    else:
        n = X.shape[0]
        lid_estimates = np.empty(n)
        ks = np.empty(n)
        for q in range(n):
            lid_estimates[q] = - len(dists[q])/np.sum(np.log(dists[q] / w[q])) #!!!!
            ks[q] = len(dists[q])
        if return_ks:
            return lid_estimates, np.mean(lid_estimates), ks
        else:
            return lid_estimates, np.mean(lid_estimates)
def sk_MLE(X, dists, knnidx, k = 10, correct = True, w = None, return_ks = False):
    if w is None:
        mle = skdim.id.MLE()
        mle.fit(X, n_neighbors=k, comb='mean', precomputed_knn_arrays=(dists, knnidx))
        if correct:
            lid_estimates = k/(k-1)*mle.dimension_pw_
        else:
            lid_estimates = mle.dimension_pw_
        return lid_estimates, np.mean(lid_estimates)
    else:
        n = X.shape[0]
        lid_estimates = np.empty(n)
        ks = np.empty(n)
        for q in range(n):
            lid_estimates[q] = - len(dists[q])/np.sum(np.log(dists[q] / w[q]))
            ks[q] = len(dists[q])
        if return_ks:
            return lid_estimates, np.mean(lid_estimates), ks
        else:
            return lid_estimates, np.mean(lid_estimates)
def sk_ESS(X, dists, knnidx, k = 10, correct = True, w=None, return_ks = False):
    if w is None:
        est_ess = skdim.id.ESS()
        est_ess._fit(X, dists, knnidx)
        lid_estimates = est_ess.dimension_pw_
        return lid_estimates, np.mean(lid_estimates)

def sk_MADA(X, dists, knnidx, k = 10, correct = True, w=None, return_ks = False):
    if w is None:
        mada = skdim.id.MADA()
        mada.fit(X, precomputed_knn_arrays=(dists, knnidx))
        lid_estimates = mada.dimension_pw_
        return lid_estimates, np.mean(lid_estimates)

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
        lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=10, dists=None, knnidx=None, geo=geo)
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
        lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=10, dists=None, knnidx=None, geo=geo)
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
        lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=10, dists=None, knnidx=None, geo=geo)
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
        lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=10, dists=None, knnidx=None, geo=geo)
        return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
    else:
        return lid_estimates, np.mean(lid_estimates)

def sk_MADA_full(X, k = 10, correct = True, dists = None, knnidx= None, w=None, smooth=False, geo=None):
    if (dists is None) or (knnidx is None):
        dists, knnidx = skdim._commonfuncs.get_nn(X, k=k, n_jobs=1)
    mada = skdim.id.MADA()
    mada.fit(X, precomputed_knn_arrays=(dists, knnidx))
    lid_estimates = mada.dimension_pw_
    if smooth:
        lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=10, dists=None, knnidx=None, geo=geo)
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
        lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=10, dists=None, knnidx=None, geo=geo)
        return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
    else:
        return lid_estimates, np.mean(lid_estimates)

###############################################################################################################################BAGGING+SMOOTHING###############################################################################################################################
def geodesic_knn(X, k_euc, n_geo):
    n_samples = X.shape[0]
    dists, knnidx = skdim._commonfuncs.get_nn(X, k=k_euc, n_jobs=1)
    row_idx = np.repeat(np.arange(n_samples), k_euc)
    col_idx = knnidx.flatten()
    data = dists.flatten()
    A = csr_matrix((data, (row_idx, col_idx)), shape=(n_samples, n_samples))
    A = A.maximum(A.T)
    dist_matrix = dijkstra(csgraph=A, directed=False, return_predecessors=False)
    geodesic_indices = []
    geodesic_dists = []
    for i in range(n_samples):
        dists_i = dist_matrix[i]
        dists_i[i] = np.inf  # ignore self
        finite_mask = np.isfinite(dists_i)
        reachable = np.where(finite_mask)[0]
        sorted_idx = np.argsort(dists_i[reachable])[:n_geo]
        selected = reachable[sorted_idx]
        selected_dists = dists_i[selected]
        geodesic_indices.append(selected.tolist())
        geodesic_dists.append(selected_dists.tolist())
    return geodesic_dists, geodesic_indices

def smoothing(X, lid_estimates, k = 10, dists = None, knnidx= None, geo=None):
    if (dists is None) or (knnidx is None):
        if geo is None:
            dists, knnidx = skdim._commonfuncs.get_nn(X, k=k, n_jobs=1)
        else:
            dists, knnidx = geodesic_knn(X, k_euc=geo, n_geo=k)
    smoothed_estimates = np.empty(len(lid_estimates))
    for i in range(len(lid_estimates)):
        smoothed_estimates[i] = (np.sum(lid_estimates[knnidx[i]]) + lid_estimates[i])/(k+1)
    return smoothed_estimates, np.mean(smoothed_estimates)

def simple_bagging_skdim(estimator, Q, X, n_bags=10, k=10, sampling_rate=None, progress_bar=False, estimators=None, estimator_names=None, paralell_estimation=False, w=None, indexuse=None):
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
        n_samples = int(sampling_rate * len(X))  # Number of samples per bag
        indices = np.arange(len(X))  # All possible indices
        bags = []
        selected_indices = []
        for _ in range(n_bags):
            chosen_idx = np.random.choice(indices, size=n_samples, replace=False)  # Sample without replacement per bag
            bags.append(X[chosen_idx])  # Extract the corresponding data
            selected_indices.append(chosen_idx)  # Store selected indices
        return bags, selected_indices

    def k_smallest_nonzero_Q(distances, k=10, w=None):
        n = distances.shape[0]
        if w is not None:
            # Ensure w is a numpy array
            w = np.asarray(w)
            result_distances = []
            result_indices = []
            for q in range(n):
                # Get nonzero distances and their indices for point q.
                nonzero_mask = distances[q] != 0
                # Select only those distances that are less than or equal to w[q]
                valid_mask = nonzero_mask & (distances[q] <= w[q])
                dists = distances[q, valid_mask]
                inds = np.nonzero(valid_mask)[0]
                # Sort the valid distances
                order = np.argsort(dists)
                result_distances.append(dists[order])
                result_indices.append(inds[order])
            return result_distances, result_indices
        else:
            # Original fixed-k behavior
            result_distances = np.zeros((n, k))
            result_indices = np.zeros((n, k), dtype=int)
            for q in range(n):
                nonzero_mask = distances[q] != 0
                dists = distances[q, nonzero_mask]
                inds = np.nonzero(nonzero_mask)[0]
                if len(dists) >= k:
                    # Use argpartition to efficiently find the k smallest distances,
                    # then sort them.
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
            # smallest_indices is a list of arrays relative to the 'indices' array.
            # Map them back to the original indices.
            smallest_indices = [indices[row_inds] for row_inds in smallest_indices]
        else:
            # In the fixed-k case, smallest_indices is a 2D array of positions in 'indices'
            smallest_indices = indices[smallest_indices]
        return smallest_distances, smallest_indices

    def precompute_sorted_distances(distances):
        sorted_indices = np.argsort(distances, axis=1)
        sorted_distances = np.take_along_axis(distances, sorted_indices, axis=1)
        return sorted_distances, sorted_indices

    def fast_k_smallest_in_bag(sorted_distances, sorted_indices, bag_indices, k):
        n_points = sorted_indices.shape[0]
        bag_set = set(bag_indices)  # O(1) lookup time
        nearest_distances = np.zeros((n_points, k))
        nearest_indices = np.zeros((n_points, k), dtype=int)
        for q in range(n_points):
            # Find k nearest neighbors that are in the bag
            valid_neighbors = [idx for idx, d in zip(sorted_indices[q], sorted_distances[q]) if
                               idx in bag_set and d > 0]
            if len(valid_neighbors) >= k:
                nearest_indices[q] = valid_neighbors[:k]
                nearest_distances[q] = sorted_distances[
                    q, [np.where(sorted_indices[q] == idx)[0][0] for idx in nearest_indices[q]]]
            else:
                raise ValueError(f"Point {q} has fewer than {k} neighbors in the bag.")
        return nearest_distances, nearest_indices

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
                #smallest_distances, smallest_indices = fast_k_smallest_in_bag(sorted_distances, sorted_indices,
                #                                                              split_indices[j], k=k)
                for key in estimator_dictionary:
                    estimate_dictionary[key][:, j] = estimator_dictionary[key](X=X, dists=smallest_distances, knnidx=smallest_indices, k=k, w=w)[0]
        bagging_estimator_dictionary = {estimator_names[i]: np.mean(estimate_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
        avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimator_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        return bagging_estimator_dictionary, avg_bagging_estimator_dictionary

def outofbag_weighted_bagging_skdim(estimator, Q, X, n_bags=10, k=10, sampling_rate=None, progress_bar=False, estimators=None,
                         estimator_names=None, paralell_estimation=False, weighing_type='0'):
    def compute_distance_matrix(X):
        return squareform(pdist(X))

    def sample_or_split_data_with_indices(X, n_bags=10, sampling_rate=None):
        indices = np.arange(len(X))  # Full index set

        if sampling_rate is None:
            # Split method (Original Behavior)
            split_indices = np.array_split(indices, n_bags)  # Evenly split indices
            bags = [X[idx] for idx in split_indices]  # Split X accordingly
            out_of_bag_indices = [
                np.concatenate([split_indices[k] for k in range(n_bags) if k != j])  # OOB: all other splits
                for j in range(n_bags)
            ]
        else:
            # Sampling method (New Behavior)
            n_samples = int(sampling_rate * len(X))  # Number of samples per bag
            bags = []
            split_indices = []  # Store selected indices per bag
            out_of_bag_indices = []

            for _ in range(n_bags):
                chosen_idx = np.random.choice(indices, size=n_samples, replace=False)  # Sample without replacement
                bags.append(X[chosen_idx])  # Store sampled data
                split_indices.append(chosen_idx)  # Store sampled indices

                # Compute OOB indices (elements not in the sampled indices for that bag)
                oob_idx = np.setdiff1d(indices, chosen_idx)
                out_of_bag_indices.append(oob_idx)

        return bags, split_indices, out_of_bag_indices

    def k_smallest_nonzero_Q(distances, k=10, w=None):
        n = distances.shape[0]
        if w is not None:
            # Ensure w is a numpy array
            w = np.asarray(w)
            result_distances = []
            result_indices = []
            for q in range(n):
                # Get nonzero distances and their indices for point q.
                nonzero_mask = distances[q] != 0
                # Select only those distances that are less than or equal to w[q]
                valid_mask = nonzero_mask & (distances[q] <= w[q])
                dists = distances[q, valid_mask]
                inds = np.nonzero(valid_mask)[0]
                # Sort the valid distances
                order = np.argsort(dists)
                result_distances.append(dists[order])
                result_indices.append(inds[order])
            return result_distances, result_indices
        else:
            # Original fixed-k behavior
            result_distances = np.zeros((n, k))
            result_indices = np.zeros((n, k), dtype=int)
            for q in range(n):
                nonzero_mask = distances[q] != 0
                dists = distances[q, nonzero_mask]
                inds = np.nonzero(nonzero_mask)[0]
                if len(dists) >= k:
                    # Use argpartition to efficiently find the k smallest distances,
                    # then sort them.
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
            # smallest_indices is a list of arrays relative to the 'indices' array.
            # Map them back to the original indices.
            smallest_indices = [indices[row_inds] for row_inds in smallest_indices]
        else:
            # In the fixed-k case, smallest_indices is a 2D array of positions in 'indices'
            smallest_indices = indices[smallest_indices]
        return smallest_distances, smallest_indices

    def precompute_sorted_distances(distances):
        sorted_indices = np.argsort(distances, axis=1)
        sorted_distances = np.take_along_axis(distances, sorted_indices, axis=1)
        return sorted_distances, sorted_indices

    def fast_k_smallest_in_bag(sorted_distances, sorted_indices, bag_indices, k):
        n_points = sorted_indices.shape[0]
        in_bag = np.isin(sorted_indices, bag_indices)
        nearest_distances = np.full((n_points, k), np.inf)
        nearest_indices = np.full((n_points, k), -1)
        for q in range(n_points):
            valid_mask = in_bag[q] & (sorted_distances[q] > 0)
            valid_distances = sorted_distances[q][valid_mask]
            valid_indices = sorted_indices[q][valid_mask]
            if len(valid_distances) >= k:
                nearest_distances[q] = valid_distances[:k]
                nearest_indices[q] = valid_indices[:k]
            else:
                raise ValueError(f"Point {q} has fewer than {k} neighbors in the bag.")
        return nearest_distances, nearest_indices

    def discrepancy_pvalues(X1, X2, k1, k2, weighing_type = 'p_val_mean'):
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        k1 = np.asarray(k1)
        k2 = np.asarray(k2)
        # Ensure broadcastable shapes
        assert X1.shape == X2.shape, "X1 and X2 must have the same shape"
        shape = X1.shape
        k1 = np.broadcast_to(k1, shape)
        k2 = np.broadcast_to(k2, shape)
        # Initialize p-value arrays with NaNs (to be filled)
        pval_weighted = np.full(shape, np.nan)
        pval_symmetric = np.full(shape, np.nan)
        pval = np.full(shape, np.nan)
        # Where k2 == 0 → force p-value = 0
        mask_k2_zero = (k2 == 0)
        # Where k2 > 0 → compute normally
        mask_valid = ~mask_k2_zero
        # Common parts
        diff_sq = (X1 - X2) ** 2
        inv_k_sum = np.zeros_like(X1)
        inv_k_sum[mask_valid] = (1.0 / k1[mask_valid]) + (1.0 / k2[mask_valid])
        if weighing_type == 'p_val_mean':
            # Method 1: Weighted mean
            mu_hat_weighted = np.zeros_like(X1)
            mu_hat_weighted[mask_valid] = (
                    (k1[mask_valid] * X1[mask_valid] + k2[mask_valid] * X2[mask_valid])
                    / (k1[mask_valid] + k2[mask_valid])
            )
            tau_sq_weighted = np.zeros_like(X1)
            tau_sq_weighted[mask_valid] = mu_hat_weighted[mask_valid] ** 2 * inv_k_sum[mask_valid]
            T_weighted = np.zeros_like(X1)
            T_weighted[mask_valid] = diff_sq[mask_valid] / tau_sq_weighted[mask_valid]
            pval_weighted[mask_valid] = 1 - chi2.cdf(T_weighted[mask_valid], df=1)
            pval_weighted[mask_k2_zero] = 0.0
            pval = pval_weighted
        elif weighing_type == 'p_val_symmetric':
            # Method 2: Symmetric estimator
            mu_hat_sym = (X1 ** 2 + X2 ** 2) / 2
            tau_sq_sym = mu_hat_sym * inv_k_sum
            T_sym = np.zeros_like(X1)
            T_sym[mask_valid] = diff_sq[mask_valid] / tau_sq_sym[mask_valid]
            pval_symmetric[mask_valid] = 1 - chi2.cdf(T_sym[mask_valid], df=1)
            pval_symmetric[mask_k2_zero] = 0.0
            pval = pval_symmetric
        return pval

    def results_aggregating(estimate_dictionary, estimator_names, out_of_bag_estimate_dictionary, k_1_dict, k_2_dict, weighing_type='0'):
        if weighing_type == '0':
            test_errors_dictionary = {estimator_names[i]: np.abs(estimate_dictionary[estimator_names[i]] - out_of_bag_estimate_dictionary[estimator_names[i]])**2 for i in range(len(estimator_names))}
            weights_dictionary = {estimator_names[i]: 1 / (test_errors_dictionary[estimator_names[i]] * np.sum(1 / test_errors_dictionary[estimator_names[i]], axis=1, keepdims=True)) for i in range(len(estimator_names))}
            bagging_estimators_dictionary = {estimator_names[i]: np.sum(estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
            avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        elif weighing_type == 'inf':
            test_errors_dictionary = {estimator_names[i]: np.abs(estimate_dictionary[estimator_names[i]] - out_of_bag_estimate_dictionary[estimator_names[i]]) ** 2 for i in range(len(estimator_names))}
            weights_dictionary = {}
            for i in range(len(estimator_names)):
                mask = np.isfinite(test_errors_dictionary[estimator_names[i]])
                X_safe = np.where(mask, test_errors_dictionary[estimator_names[i]], np.nan)
                row_sums = np.nansum(1 / X_safe, axis=1, keepdims=True)
                Y = np.where(mask, 1 / (test_errors_dictionary[estimator_names[i]] * row_sums), 0)
                all_bad_rows = ~np.any(mask, axis=1, keepdims=True)
                Y[all_bad_rows.repeat(Y.shape[1], axis=1)] = 1.0 / Y.shape[1]
                weights_dictionary[estimator_names[i]] = Y
            bagging_estimators_dictionary = {estimator_names[i]: np.sum(estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
            avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        elif weighing_type == 'equalizing':
            test_errors_dictionary = {estimator_names[i]: 1/(1/k_1_dict[estimator_names[i]] + 1/k_2_dict[estimator_names[i]])*np.abs(estimate_dictionary[estimator_names[i]] - out_of_bag_estimate_dictionary[estimator_names[i]]) ** 2 for i in range(len(estimator_names))}
            weights_dictionary = {}
            for i in range(len(estimator_names)):
                mask = np.isfinite(test_errors_dictionary[estimator_names[i]])
                X_safe = np.where(mask, test_errors_dictionary[estimator_names[i]], np.nan)
                row_sums = np.nansum(1 / X_safe, axis=1, keepdims=True)
                Y = np.where(mask, 1 / (test_errors_dictionary[estimator_names[i]] * row_sums), 0)
                all_bad_rows = ~np.any(mask, axis=1, keepdims=True)
                Y[all_bad_rows.repeat(Y.shape[1], axis=1)] = 1.0 / Y.shape[1]
                weights_dictionary[estimator_names[i]] = Y
            bagging_estimators_dictionary = {estimator_names[i]: np.sum(estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
            avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        elif weighing_type == 'penalizing':
            test_errors_dictionary = {estimator_names[i]: (1 / k_1_dict[estimator_names[i]] + 1 / k_2_dict[estimator_names[i]]) * np.abs(estimate_dictionary[estimator_names[i]] - out_of_bag_estimate_dictionary[estimator_names[i]]) ** 2 for i in range(len(estimator_names))}
            weights_dictionary = {}
            for i in range(len(estimator_names)):
                mask = np.isfinite(test_errors_dictionary[estimator_names[i]])
                X_safe = np.where(mask, test_errors_dictionary[estimator_names[i]], np.nan)
                row_sums = np.nansum(1 / X_safe, axis=1, keepdims=True)
                Y = np.where(mask, 1 / (test_errors_dictionary[estimator_names[i]] * row_sums), 0)
                all_bad_rows = ~np.any(mask, axis=1, keepdims=True)
                Y[all_bad_rows.repeat(Y.shape[1], axis=1)] = 1.0 / Y.shape[1]
                weights_dictionary[estimator_names[i]] = Y
            bagging_estimators_dictionary = {estimator_names[i]: np.sum(estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
            avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        elif weighing_type == 'p_val_mean':
            probs_dictionary = {estimator_names[i]: -1/np.log(discrepancy_pvalues(estimate_dictionary[estimator_names[i]], out_of_bag_estimate_dictionary[estimator_names[i]], k_1_dict[estimator_names[i]], k_2_dict[estimator_names[i]], weighing_type='p_val_mean')) for i in range(len(estimator_names))}
            clipped_probs_dictionary = {estimator_names[i]: np.maximum(probs_dictionary[estimator_names[i]], np.max(probs_dictionary[estimator_names[i]], axis=1, keepdims=True)/100) for i in range(len(estimator_names))}
            weights_dictionary = {estimator_names[i]: clipped_probs_dictionary[estimator_names[i]]/np.sum(clipped_probs_dictionary[estimator_names[i]], axis=1, keepdims=True) for i in range(len(estimator_names))}
            bagging_estimators_dictionary = {estimator_names[i]: np.sum(estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
            avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        elif weighing_type == 'p_val_raw':
            clipped_probs_dictionary = {estimator_names[i]: discrepancy_pvalues(estimate_dictionary[estimator_names[i]], out_of_bag_estimate_dictionary[estimator_names[i]], k_1_dict[estimator_names[i]], k_2_dict[estimator_names[i]], weighing_type='p_val_mean') for i in range(len(estimator_names))}
            #clipped_probs_dictionary = {estimator_names[i]: np.maximum(probs_dictionary[estimator_names[i]], np.max(probs_dictionary[estimator_names[i]], axis=1, keepdims=True)/100) for i in range(len(estimator_names))}
            weights_dictionary = {estimator_names[i]: clipped_probs_dictionary[estimator_names[i]]/np.sum(clipped_probs_dictionary[estimator_names[i]], axis=1, keepdims=True) for i in range(len(estimator_names))}
            bagging_estimators_dictionary = {estimator_names[i]: np.sum(estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
            avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        elif weighing_type == 'p_val_mean2':
            clipped_probs_dictionary = {estimator_names[i]: -1/np.log(discrepancy_pvalues(estimate_dictionary[estimator_names[i]], out_of_bag_estimate_dictionary[estimator_names[i]], k_1_dict[estimator_names[i]], k_2_dict[estimator_names[i]], weighing_type='p_val_mean')) for i in range(len(estimator_names))}
            #clipped_probs_dictionary = {estimator_names[i]: probs_dictionary[estimator_names[i]], np.max(probs_dictionary[estimator_names[i]], axis=1, keepdims=True)/10) for i in range(len(estimator_names))}
            weights_dictionary = {estimator_names[i]: clipped_probs_dictionary[estimator_names[i]]/np.sum(clipped_probs_dictionary[estimator_names[i]], axis=1, keepdims=True) for i in range(len(estimator_names))}
            bagging_estimators_dictionary = {estimator_names[i]: np.sum(estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
            avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        elif weighing_type == 'p_val_symmetric':
            probs_dictionary = {estimator_names[i]: -1/np.log(discrepancy_pvalues(estimate_dictionary[estimator_names[i]], out_of_bag_estimate_dictionary[estimator_names[i]], k_1_dict[estimator_names[i]], k_2_dict[estimator_names[i]], weighing_type='p_val_symmetric')) for i in range(len(estimator_names))}
            clipped_probs_dictionary = {estimator_names[i]: np.maximum(probs_dictionary[estimator_names[i]], np.max(probs_dictionary[estimator_names[i]], axis=1, keepdims=True)/100) for i in range(len(estimator_names))}
            weights_dictionary = {estimator_names[i]: clipped_probs_dictionary[estimator_names[i]]/np.sum(clipped_probs_dictionary[estimator_names[i]], axis=1, keepdims=True) for i in range(len(estimator_names))}
            bagging_estimators_dictionary = {estimator_names[i]: np.sum(estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
            avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        elif weighing_type == 'p_val_symmetric_raw':
            clipped_probs_dictionary = {estimator_names[i]: discrepancy_pvalues(estimate_dictionary[estimator_names[i]], out_of_bag_estimate_dictionary[estimator_names[i]], k_1_dict[estimator_names[i]], k_2_dict[estimator_names[i]], weighing_type='p_val_symmetric') for i in range(len(estimator_names))}
            #clipped_probs_dictionary = {estimator_names[i]: np.maximum(probs_dictionary[estimator_names[i]], np.max(probs_dictionary[estimator_names[i]], axis=1, keepdims=True)/100) for i in range(len(estimator_names))}
            weights_dictionary = {estimator_names[i]: clipped_probs_dictionary[estimator_names[i]]/np.sum(clipped_probs_dictionary[estimator_names[i]], axis=1, keepdims=True) for i in range(len(estimator_names))}
            bagging_estimators_dictionary = {estimator_names[i]: np.sum(estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
            avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        return bagging_estimators_dictionary, avg_bagging_estimator_dictionary

    n, m = Q.shape[0], Q.shape[1]
    distances = compute_distance_matrix(X)
    #sorted_distances, sorted_indices = precompute_sorted_distances(distances)
    bags, split_indices, out_of_bag_indices = sample_or_split_data_with_indices(X, n_bags=n_bags, sampling_rate=sampling_rate)
    if not paralell_estimation:
        estimates = np.zeros((n, n_bags))
        out_of_bag_estimates = np.zeros((n, n_bags))
        kth_dists = np.zeros((n, n_bags))
        if progress_bar:
            for j in tqdm(range(n_bags)):
                smallest_distances, smallest_indices = k_smallest_distance_Q(distances=distances,
                                                                             indices=split_indices[j],
                                                                             k=k)
                estimates[:, j] = estimator(X=X, dists=smallest_distances, knnidx=smallest_indices, k=k)[0]
                out_of_bag_indices = np.concatenate([split_indices[k] for k in range(len(split_indices)) if k != j])
                out_of_bag_smallest_distances, out_of_bag_smallest_indices = k_smallest_distance_Q(distances=distances,
                                                                                                   indices=out_of_bag_indices,
                                                                                                   k=k)
                out_of_bag_estimates[:, j] = \
                estimator(X=X, dists=out_of_bag_smallest_distances, knnidx=out_of_bag_smallest_indices, k=k)[0]
        else:
            for j in range(n_bags):
                smallest_distances, smallest_indices = k_smallest_distance_Q(distances=distances,
                                                                             indices=split_indices[j],
                                                                             k=k)
                kth_dists[:, j] = np.max(smallest_distances, axis = 1)
                estimates[:, j] = estimator(X=X, dists=smallest_distances, knnidx=smallest_indices, k=k)[0]
                out_of_bag_indices = np.concatenate([split_indices[k] for k in range(len(split_indices)) if k != j])
                out_of_bag_smallest_distances, out_of_bag_smallest_indices = k_smallest_distance_Q(distances=distances,
                                                                                                   indices=out_of_bag_indices,
                                                                                                   k=k)
                out_of_bag_estimates[:, j] = \
                estimator(X=X, dists=out_of_bag_smallest_distances, knnidx=out_of_bag_smallest_indices, k=k)[0]
        test_errors = np.abs(estimates - out_of_bag_estimates)
        weights = 1 / (test_errors * np.sum(1 / test_errors, axis=1, keepdims=True))
        bagging_estimators = np.sum(estimates * weights, axis=1)
        avg_bagging_estimator = np.mean(bagging_estimators)
        return bagging_estimators, avg_bagging_estimator
    else:
        estimator_dictionary = {estimator_names[i]: estimators[i] for i in range(len(estimator_names))}
        estimate_dictionary = {estimator_names[i]: np.zeros((n, n_bags)) for i in range(len(estimator_names))}
        k_1_dict = {estimator_names[i]: k*np.ones((n, n_bags)) for i in range(len(estimator_names))}
        k_2_dict = {estimator_names[i]: np.zeros((n, n_bags)) for i in range(len(estimator_names))}
        out_of_bag_estimate_dictionary = {estimator_names[i]: np.zeros((n, n_bags)) for i in range(len(estimator_names))}
        if progress_bar:
            for j in tqdm(range(n_bags)):
                smallest_distances, smallest_indices = k_smallest_distance_Q(distances=distances, indices=split_indices[j],
                                                                             k=k)
                bag_ws = smallest_distances[:, -1]
                out_of_bag_smallest_distances, out_of_bag_smallest_indices = k_smallest_distance_Q(distances=distances, indices=out_of_bag_indices[j],
                                                                             k=k, w=bag_ws)
                for key in estimator_dictionary:
                    estimate_dictionary[key][:, j] = \
                    estimator_dictionary[key](X=X, dists=smallest_distances, knnidx=smallest_indices, k=k)[0]
                    out_of_bag_e = estimator_dictionary[key](X=X, dists=out_of_bag_smallest_distances, knnidx=out_of_bag_smallest_indices, k=k, w=bag_ws)[0]
                    masks = np.isnan(out_of_bag_e)
                    out_of_bag_e[masks] = 0
                    out_of_bag_estimate_dictionary[key][:, j] = out_of_bag_e
        else:
            for j in range(n_bags):
                smallest_distances, smallest_indices = k_smallest_distance_Q(distances=distances, indices=split_indices[j],
                                                                             k=k)
                bag_ws = smallest_distances[:, -1]
                out_of_bag_smallest_distances, out_of_bag_smallest_indices = k_smallest_distance_Q(distances=distances, indices=out_of_bag_indices[j],
                                                                             k=k, w=bag_ws)
                for key in estimator_dictionary:
                    estimate_dictionary[key][:, j] = estimator_dictionary[key](X=X, dists=smallest_distances, knnidx=smallest_indices, k=k)[0]
                    out_of_bag_e, _, ks = estimator_dictionary[key](X=X, dists=out_of_bag_smallest_distances, knnidx=out_of_bag_smallest_indices, k=k, w=bag_ws, return_ks=True)
                    k_2_dict[key][:, j] = ks
                    masks = ~np.isfinite(out_of_bag_e)
                    if weighing_type == '0':
                        out_of_bag_e[masks] = 0
                        out_of_bag_estimate_dictionary[key][:, j] = out_of_bag_e
                    else:
                        out_of_bag_estimate_dictionary[key][:, j] = out_of_bag_e
        bagging_estimators_dictionary, avg_bagging_estimator_dictionary = results_aggregating(estimate_dictionary=estimate_dictionary, estimator_names = estimator_names, out_of_bag_estimate_dictionary=out_of_bag_estimate_dictionary, k_1_dict=k_1_dict, k_2_dict=k_2_dict, weighing_type=weighing_type)
        return bagging_estimators_dictionary, avg_bagging_estimator_dictionary

def outofbag_weighted_inside_bagging_skdim(estimator, Q, X, n_bags=10, k=10, sampling_rate=None, progress_bar=False, estimators=None,
                         estimator_names=None, paralell_estimation=False):
    def compute_distance_matrix(X):
        return squareform(pdist(X))

    def sample_or_split_data_with_indices(X, n_bags=10, sampling_rate=None):
        indices = np.arange(len(X))  # Full index set

        if sampling_rate is None:
            # Split method (Original Behavior)
            split_indices = np.array_split(indices, n_bags)  # Evenly split indices
            bags = [X[idx] for idx in split_indices]  # Split X accordingly
            out_of_bag_indices = [
                np.concatenate([split_indices[k] for k in range(n_bags) if k != j])  # OOB: all other splits
                for j in range(n_bags)
            ]
        else:
            # Sampling method (New Behavior)
            n_samples = int(sampling_rate * len(X))  # Number of samples per bag
            bags = []
            split_indices = []  # Store selected indices per bag
            out_of_bag_indices = []

            for _ in range(n_bags):
                chosen_idx = np.random.choice(indices, size=n_samples, replace=False)  # Sample without replacement
                bags.append(X[chosen_idx])  # Store sampled data
                split_indices.append(chosen_idx)  # Store sampled indices

                # Compute OOB indices (elements not in the sampled indices for that bag)
                oob_idx = np.setdiff1d(indices, chosen_idx)
                out_of_bag_indices.append(oob_idx)

        return bags, split_indices, out_of_bag_indices

    def k_smallest_nonzero_Q(distances, k=10, w=None):
        n = distances.shape[0]
        if w is not None:
            # Ensure w is a numpy array
            w = np.asarray(w)
            result_distances = []
            result_indices = []
            for q in range(n):
                # Get nonzero distances and their indices for point q.
                nonzero_mask = distances[q] != 0
                # Select only those distances that are less than or equal to w[q]
                valid_mask = nonzero_mask & (distances[q] <= w[q])
                dists = distances[q, valid_mask]
                inds = np.nonzero(valid_mask)[0]
                # Sort the valid distances
                order = np.argsort(dists)
                result_distances.append(dists[order])
                result_indices.append(inds[order])
            return result_distances, result_indices
        else:
            # Original fixed-k behavior
            result_distances = np.zeros((n, k))
            result_indices = np.zeros((n, k), dtype=int)
            for q in range(n):
                nonzero_mask = distances[q] != 0
                dists = distances[q, nonzero_mask]
                inds = np.nonzero(nonzero_mask)[0]
                if len(dists) >= k:
                    # Use argpartition to efficiently find the k smallest distances,
                    # then sort them.
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
            # smallest_indices is a list of arrays relative to the 'indices' array.
            # Map them back to the original indices.
            smallest_indices = [indices[row_inds] for row_inds in smallest_indices]
        else:
            # In the fixed-k case, smallest_indices is a 2D array of positions in 'indices'
            smallest_indices = indices[smallest_indices]
        return smallest_distances, smallest_indices

    def precompute_sorted_distances(distances):
        sorted_indices = np.argsort(distances, axis=1)
        sorted_distances = np.take_along_axis(distances, sorted_indices, axis=1)
        return sorted_distances, sorted_indices

    def fast_k_smallest_in_bag(sorted_distances, sorted_indices, bag_indices, k):
        n_points = sorted_indices.shape[0]
        in_bag = np.isin(sorted_indices, bag_indices)
        nearest_distances = np.full((n_points, k), np.inf)
        nearest_indices = np.full((n_points, k), -1)
        for q in range(n_points):
            valid_mask = in_bag[q] & (sorted_distances[q] > 0)
            valid_distances = sorted_distances[q][valid_mask]
            valid_indices = sorted_indices[q][valid_mask]
            if len(valid_distances) >= k:
                nearest_distances[q] = valid_distances[:k]
                nearest_indices[q] = valid_indices[:k]
            else:
                raise ValueError(f"Point {q} has fewer than {k} neighbors in the bag.")
        return nearest_distances, nearest_indices

    n, m = Q.shape[0], Q.shape[1]
    distances = compute_distance_matrix(X)
    #sorted_distances, sorted_indices = precompute_sorted_distances(distances)
    bags, split_indices, out_of_bag_indices = sample_or_split_data_with_indices(X, n_bags=n_bags, sampling_rate=sampling_rate)
    estimator_dictionary = {estimator_names[i]: estimators[i] for i in range(len(estimator_names))}
    estimate_dictionary = {estimator_names[i]: np.zeros((n, n_bags)) for i in range(len(estimator_names))}
    out_of_bag_estimate_dictionary = {estimator_names[i]: np.zeros((n, n_bags)) for i in range(len(estimator_names))}
    for j in range(n_bags):
        smallest_distances, smallest_indices = k_smallest_distance_Q(distances=distances, indices=split_indices[j],
                                                                     k=k)
        bag_ws = smallest_distances[:, -1]
        #out_of_bag_smallest_distances, out_of_bag_smallest_indices = k_smallest_distance_Q(distances=distances, indices=out_of_bag_indices[j],
        #                                                             k=k, w=bag_ws)
        for key in estimator_dictionary:
            estimate_dictionary[key][:, j] = \
            estimator_dictionary[key](X=X, dists=smallest_distances, knnidx=smallest_indices, k=k)[0]
            out_of_bag_e = simple_bagging_skdim(estimator=estimator_dictionary[key], Q=X, X=X, n_bags=n_bags, k=k, sampling_rate=sampling_rate/(1-sampling_rate), progress_bar=False, estimators=None, estimator_names=None, paralell_estimation=False, w=None, indexuse=out_of_bag_indices[j])[0]
            masks = ~np.isfinite(out_of_bag_e)
            out_of_bag_e[masks] = 0
            out_of_bag_estimate_dictionary[key][:, j] = out_of_bag_e
    test_errors_dictionary = {estimator_names[i]: np.abs(estimate_dictionary[estimator_names[i]] - out_of_bag_estimate_dictionary[estimator_names[i]])**2 for i in range(len(estimator_names))}
    weights_dictionary = {estimator_names[i]: 1 / (test_errors_dictionary[estimator_names[i]] * np.sum(1 / test_errors_dictionary[estimator_names[i]], axis=1, keepdims=True)) for i in range(len(estimator_names))}
    bagging_estimators_dictionary = {estimator_names[i]: np.sum(estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
    avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
    return bagging_estimators_dictionary, avg_bagging_estimator_dictionary

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
        if name == 'lidl':
            lid_estimates, mean_estimate = LIDL_full(X, k=k, correct=correct, dists=dists, knnidx=knnidx, model_type="gm", smooth=smooth, geo=geo)
            estimate_dict[name] = lid_estimates
            avg_dict[name] = mean_estimate
    return estimate_dict, avg_dict

def fast_skdim_estimators(data_set, estimator_names, method_type=None, n_bags=10, sampling_rate = 0.5, k=10, progress_bar=False, correct = True):
    if method_type == '':
        estimators_dict, avg_estimator_dict = sk_estimators(data_set, k=k, correct=correct, estimator_names=estimator_names)
    elif method_type == 'smooth':
        estimators_dict, avg_estimator_dict = sk_estimators(data_set, k=k, correct=correct, estimator_names=estimator_names, smooth=True, geo=None)
    elif method_type == 'smooth_geo':
        estimators_dict, avg_estimator_dict = sk_estimators(data_set, k=k, correct=correct, estimator_names=estimator_names, smooth=True, geo=3)
    elif method_type == 'bag':
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
        estimators_dict, avg_estimator_dict = simple_bagging_skdim(estimator=None, Q = data_set, X = data_set, n_bags=n_bags, k=k, sampling_rate=sampling_rate, progress_bar=progress_bar, estimators=estimators, estimator_names=used_estimator_names, paralell_estimation=True)
    elif method_type.startswith('bag_w_'):
        weighing_type = method_type[len('bag_w_'):]
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
        if weighing_type == 'bag':
            estimators_dict, avg_estimator_dict = outofbag_weighted_inside_bagging_skdim(estimator=None, Q=data_set,
                                                                                         X=data_set, n_bags=n_bags, k=k,
                                                                                         sampling_rate=sampling_rate,
                                                                                         progress_bar=progress_bar,
                                                                                         estimators=estimators,
                                                                                         estimator_names=used_estimator_names,
                                                                                         paralell_estimation=True)
        else:
            estimators_dict, avg_estimator_dict = outofbag_weighted_bagging_skdim(estimator=None, Q=data_set, X=data_set, n_bags=n_bags, k=k, sampling_rate=sampling_rate, progress_bar=progress_bar, estimators=estimators, estimator_names=used_estimator_names, paralell_estimation=True, weighing_type=weighing_type)
    return estimators_dict, avg_estimator_dict
###############################################################################################################################RUNNING ESTIMATORS###############################################################################################################################
def run_method_fast(data_sets, estimator_names, method_type, k=10, n_bags=10, sampling_rate = 0.5, save=True, test_types = None, bounds = None):
    result_dictionary = {estimator_names[i]: {} for i in range(len(estimator_names))}
    for key in tqdm(data_sets):
        estimators_dict, avg_estimator_dict = fast_skdim_estimators(data_sets[key][0], estimator_names=estimator_names, method_type=method_type, n_bags=n_bags, sampling_rate=sampling_rate, k=k, progress_bar=False, correct=True)
        for name in estimator_names:
            result_dictionary[name][key] = (estimators_dict[name], avg_estimator_dict[name])
    estimators_results = {estimator_names[i]: get_lollipop_comparrison_measures(data_sets, result_dictionary[estimator_names[i]], test_types=test_types, bounds=bounds) for i in range(len(estimator_names))}
    if save:
        save_to_df(estimators_results, f"{'_'.join(str(x) for x in estimator_names)}_{method_type}_results")
    return result_dictionary, estimators_results

def run_task(args, data_sets, test_types, bounds):
    #test_types = ['dim', 'region']
    #bounds = {key: {0: (1, 1.5), 1: (1, 1.5)} for key in data_sets}
    """Helper function to run `run_method_fast` with the given arguments."""
    k, n_bags, sampling_rate, method_names, method_type = args
    results = {}
    if method_type.startswith('bag'):
        result_dictionary, estimators_results = run_method_fast(
            data_sets, method_names, method_type, k=k,
            n_bags=n_bags, sampling_rate=sampling_rate, save=False, test_types=test_types, bounds=bounds)
        for method_name in method_names:
            results[f'{method_name}_{method_type}_k_{k}_n_bags_{n_bags}_sampling_rate_{sampling_rate}'] = (
                result_dictionary[method_name], estimators_results[method_name]
            )
    else:
        result_dictionary, estimators_results = run_method_fast(
            data_sets, method_names, method_type, k=k, save=False, test_types=test_types, bounds=bounds)
        for method_name in method_names:
            results[f'{method_name}_{method_type}_k_{k}'] = (
                result_dictionary[method_name], estimators_results[method_name]
            )
    return results

def run_test_fast_multiprocess(data_sets, param_list, test_types = None, bounds = None):
    """
    Run parallelized `run_method_fast` for a given list of parameter tuples.
    :param data_sets: Data used for computations.
    :param param_list: List of tuples, each containing (k, n_bags, sampling_rate, method_names, method_type)
    :return: Dictionary of results.
    """
    results = {}
    # Run in parallel using multiprocessing
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results_list = pool.starmap(run_task, [(args, data_sets, test_types, bounds) for args in param_list])
    # Merge results from different processes
    for result in results_list:
        results.update(result)
    return results

def save_dict(data, directory, filename):
    """Saves a dictionary of dictionaries to a specified directory using pickle."""
    os.makedirs(directory, exist_ok=True)  # Ensure directory exists
    filepath = os.path.join(directory, filename)  # Create full path
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def result_generator(n, param_list, save=True, load=False, load_path=None, save_name = '', usedata=None, test_types=None, bounddict=None):
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
    return results, data_sets
###############################################################################################################################PLOTTING HELPER###############################################################################################################################

###############################################################################################################################MEASURES###############################################################################################################################
def get_comparrison_measures(data_sets, estimators):
    result = dict([(key, [np.mean(estimators[key][0]), np.mean((estimators[key][0] - data_sets[key][1]) ** 2),
                          np.mean(estimators[key][0] - data_sets[key][1]) ** 2, np.var(estimators[key][0])]) for key in
                   data_sets])
    return result

def subset_estimates(data_set, data_set_name, estimators, test_types=['dim'], bounds = None):
    existing_dims = np.unique(data_set[1])
    subset_data = {data_set_name: [[data_set[0], data_set[1], data_set[2]], estimators[0]]}
    X = data_set[0]
    for test in test_types:
        if test == 'dim':
            subset = {f'{data_set_name}_dim_{dim}' : [[data_set[0][data_set[1] == dim], data_set[1][data_set[1] == dim], data_set[2]], estimators[0][data_set[1] == dim]] for dim in existing_dims}
            subset_data = subset_data | subset
        if test == 'region':
            mask = np.ones(X.shape[0], dtype=bool)
            for j in range(len(bounds)):
                for coord, (low, high) in bounds[j].items():
                    mask &= (X[:, coord] >= low) & (X[:, coord] <= high)
                inbound_indices = np.where(mask)[0]
                outbound_indices = np.where(~mask)[0]
                subset = {f'{data_set_name}_inregion_{j}': [[data_set[0][inbound_indices], data_set[1][inbound_indices], data_set[2]], estimators[0][inbound_indices]]}
                #subset[f'{data_set_name}_outregion_{j}'] = [[data_set[0][outbound_indices], data_set[1][outbound_indices], data_set[2]], estimators[0][outbound_indices]]
                subset_data = subset_data | subset
    return subset_data

def get_lollipop_comparrison_measures(data_sets, estimators, test_types = None, bounds = None):
    if test_types is None:
        result = get_comparrison_measures(data_sets, estimators)
    else:
        subset_datas = {}
        for key in data_sets:
            subset_data = subset_estimates(data_sets[key], key, estimators[key], test_types=test_types, bounds=bounds[key])
            subset_datas = subset_datas | subset_data
        result = dict([(key, [np.mean(subset_datas[key][1]), np.mean((subset_datas[key][1] - subset_datas[key][0][1]) ** 2),
                          np.mean(subset_datas[key][1] - subset_datas[key][0][1]) ** 2, np.var(subset_datas[key][1])]) for key in
                   subset_datas])
    return result
###############################################################################################################################OTHER###############################################################################################################################
def split_long_name(name, max_length=20):
    parts = name.split('_')
    lines = []
    current_line = ""
    for part in parts:
        if current_line:  # If current_line is not empty, check length before adding
            if len(current_line) + len(part) + 1 <= max_length:  # +1 for underscore or space
                current_line += "_" + part
            else:
                lines.append(current_line)
                current_line = part
        else:
            current_line = part
    if current_line:  # Append the last line
        lines.append(current_line)
    return "\n".join(lines)

def normalize(values):
    min_val = np.min(values)
    max_val = np.max(values)
    return [(1 - (x - min_val) / (max_val - min_val)) if max_val != min_val else 0 for x in values]

def save_to_df(d, save_name):
    directory = r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\csvs'
    df = pd.DataFrame.from_dict(d, orient="index")
    df.to_csv(directory + '\\' + f'{save_name}.csv')

def load_from_df(load_path):
    df = pd.read_csv(load_path, index_col=0)
    d = df.to_dict("split")
    d = dict(zip(d["index"], d["data"]))
    for key in d:
        value = d[key]
        if isinstance(value, str):
            sanitized_string = value.replace("\n", "").replace(" ", "")
            try:
                d[key] = ast.literal_eval(sanitized_string)
            except ValueError as e:
                continue
        elif isinstance(value, list):
            for i in range(len(value)):
                if isinstance(value[i], str):
                    try:
                        value[i] = ast.literal_eval(value[i])
                    except ValueError as e:
                        continue
            d[key] = value
    return d

def convert_results_for_plot(res):
    result_dictionaries = []
    dict_names = []
    for key in res.keys():
        result_dictionaries.append(res[key][1])
        dict_names.append(f'{key}')
    return result_dictionaries, dict_names

def generate_param_combinations(param_list):
    expanded_combinations = []

    for param_tuple in param_list:
        k, n_bags, sampling_rate, method_names, method_type = param_tuple

        # Ensure each parameter (except method_names) is treated as a list
        k = k if isinstance(k, list) else [k]
        n_bags = n_bags if isinstance(n_bags, list) else [n_bags]
        sampling_rate = sampling_rate if isinstance(sampling_rate, list) else [sampling_rate]
        method_type = method_type if isinstance(method_type, list) else [method_type]

        # Generate all valid parameter combinations, keeping method_names as-is
        for k_val, method_type_val in itertools.product(k, method_type):
            if method_type_val.startswith('bag'):
                for n_bags_val, sampling_rate_val in itertools.product(n_bags, sampling_rate):
                    expanded_combinations.append((k_val, n_bags_val, sampling_rate_val, method_names, method_type_val))
            else:
                # If method_type is '', ignore n_bags and sampling_rate
                expanded_combinations.append((k_val, None, None, method_names, method_type_val))

    return expanded_combinations
###############################################################################################################################PLOTTS###############################################################################################################################
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
###############################################################################################################################SPIDER CHARTS###############################################################################################################################
def create_spider_chart(data_sets, dictionaries, names, normalize_data=False, metric='MSE', save=True, save_name='spider_chart', fill=True):
    if metric == 'MSE':
        metric_val = 1
    elif metric == 'Bias2':
        metric_val = 2
    elif metric == 'Var':
        metric_val = 3
    methods = list(data_sets.keys())
    num_methods = len(methods)
    values = []
    for d in dictionaries:
        chosen_values = [d[method][metric_val] for method in methods]
        values.append(chosen_values)
    values_array = np.array(values)
    if normalize_data:
        values_array = np.array([normalize(values_array[:, i]) for i in range(num_methods)]).T
    angles = np.linspace(0, 2 * np.pi, num_methods, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(15, 15), dpi=120, subplot_kw=dict(polar=True))
    for idx, value in enumerate(values_array):
        value = list(value) + [value[0]]
        ax.plot(angles, value, linewidth=2, linestyle='solid', label=names[idx])
        if fill:
            ax.fill(angles, value, alpha=0.05)
    ax.set_yticks(np.arange(-0.1, 1.1, 0.1))
    ax.set_yticklabels([f'{x:.2f}' for x in np.arange(-0.1, 1.1, 0.1)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    if normalize_data:
        plt.title(f"Spider Chart ({metric}) normalized", size=20, color='blue', y=1.1)
    else:
        plt.title(f"Spider Chart ({metric})", size=20, color='blue', y=1.1)
    if save:
        directory = r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots'
        plt.savefig(directory + '\\' + f'{save_name}.pdf')

def create_stacked_spider_charts(data_sets, dictionaries, names, normalize_data=False, save=True,
                                 save_name='stacked_spider_charts', fill=True):
    metrics = ['MSE', 'Bias2', 'Var']
    metric_indices = {'MSE': 1, 'Bias2': 2, 'Var': 3}
    methods = list(data_sets.keys())
    num_methods = len(methods)
    angles = np.linspace(0, 2 * np.pi, num_methods, endpoint=False).tolist()
    angles += angles[:1]
    fig, axs = plt.subplots(nrows=3, figsize=(12, 18), dpi=120, subplot_kw=dict(polar=True))
    for ax, metric in zip(axs, metrics):
        metric_val = metric_indices[metric]
        values = []
        for d in dictionaries:
            chosen_values = [d[method][metric_val] for method in methods]
            values.append(chosen_values)
        values_array = np.array(values)
        if normalize_data:
            values_array = np.array([normalize(values_array[:, i]) for i in range(num_methods)]).T
        for idx, value in enumerate(values_array):
            value = list(value) + [value[0]]  # close the loop
            ax.plot(angles, value, linewidth=2, linestyle='solid', label=names[idx])
            if fill:
                ax.fill(angles, value, alpha=0.05)
        ax.set_yticks(np.arange(-0.1, 1.1, 0.1))
        ytick_labels = [f'{x:.2f}' for x in np.arange(-0.1, 1.1, 0.1)]
        tick_objs = ax.set_yticklabels(ytick_labels)
        for label in tick_objs:
            label.set_fontsize(8)
            label.set_color('gray')
        ax.set_xticks(angles[:-1])
        labels = methods
        for angle, label in zip(angles[:-1], labels):
            angle_deg = np.degrees(angle)
            if angle_deg >= 0 and angle_deg <= 90 or angle_deg >= 270:
                ha = 'left'
            elif 90 < angle_deg < 270:
                ha = 'right'
            else:
                ha = 'center'
            ax.text(angle, 1.15, label, size=10, horizontalalignment=ha, verticalalignment='center')
        ax.set_xticklabels([])  # Clear default labels
        ax.set_title(f"{metric} (normalized)" if normalize_data else metric, size=16, y=1.1)
    axs[0].legend(
        loc='center left',
        bbox_to_anchor=(1, 1),  # further to the right
        title='Methods'
    )
    plt.tight_layout()
    plt.subplots_adjust(right=0.85, hspace=0.4)
    if save:
        directory = r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots'
        plt.savefig(directory + '\\' + f'{save_name}.pdf')

def create_method_variant_spider_charts(data_sets, dictionaries, names, normalize_data=False, save=True,
                                        save_prefix='spider_chart_by_method', fill=True):
    metrics = ['MSE', 'Bias2', 'Var']
    metric_indices = {'MSE': 1, 'Bias2': 2, 'Var': 3}

    methods = list(data_sets.keys())
    num_methods = len(methods)
    angles = np.linspace(0, 2 * np.pi, num_methods, endpoint=False).tolist()
    angles += angles[:1]

    # Step 1: Group variant names by their base method (before '_')
    method_groups = defaultdict(list)
    for idx, name in enumerate(names):
        base_method = name.split('_')[0]
        method_groups[base_method].append((idx, name))

    for metric in metrics:
        metric_val = metric_indices[metric]
        num_main_methods = len(method_groups)
        num_cols = 3
        num_rows = math.ceil(num_main_methods / num_cols)

        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 5, num_rows * 5), dpi=120,
                                subplot_kw=dict(polar=True))
        axs = axs.flatten()

        for i, (main_method, variant_list) in enumerate(method_groups.items()):
            ax = axs[i]
            for idx, variant_name in variant_list:
                values = [d[method][metric_val] for method in methods for d in [dictionaries[idx]]]
                values_array = np.array(values)
                if normalize_data:
                    values_array = normalize(values_array)
                value = list(values_array) + [values_array[0]]
                ax.plot(angles, value, linewidth=2, linestyle='solid', label=variant_name)
                if fill:
                    ax.fill(angles, value, alpha=0.05)

            # Draw radial grid and labels
            ax.set_yticks(np.arange(-0.1, 1.1, 0.1))
            ytick_labels = [f'{x:.2f}' for x in np.arange(-0.1, 1.1, 0.1)]
            tick_objs = ax.set_yticklabels(ytick_labels)
            for label in tick_objs:
                label.set_fontsize(8)
                label.set_color('gray')

            # Add method axis labels
            for angle, label in zip(angles[:-1], methods):
                angle_deg = np.degrees(angle)
                if angle_deg >= 0 and angle_deg <= 90 or angle_deg >= 270:
                    ha = 'left'
                elif 90 < angle_deg < 270:
                    ha = 'right'
                else:
                    ha = 'center'
                ax.text(angle, 1.15, label, size=9, ha=ha, va='center')

            ax.set_xticklabels([])  # Remove default
            ax.set_title(main_method, size=14, y=1.08)

            if i == 0:
                ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.6), title='Variants')

        # Hide unused subplots (in case of 5 plots, the 6th is unused)
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        fig.suptitle(f"{metric} (normalized)" if normalize_data else metric, size=18, y=1.02)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, right=0.85)

        if save:
            directory = r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots'
            file_name = f'{save_prefix}_{metric}.pdf'
            plt.savefig(directory + '\\' + file_name, bbox_inches='tight')
        plt.show()

###############################################################################################################################KNN GRAPH###############################################################################################################################
def plot_all_knn_graphs(dataset_dict, k=5, output_path='all_knn_graphs.pdf'):
    num_datasets = len(dataset_dict)
    fig_height = num_datasets * 3
    fig, axes = plt.subplots(num_datasets, 1, figsize=(8, fig_height))
    if num_datasets == 1:
        axes = [axes]
    for ax, (name, data) in zip(axes, dataset_dict.items()):
        x = np.asarray(data[0])
        if not isinstance(data[1], np.ndarray):
            lid_values = np.full(len(x), data[1])
        else:
            lid_values = np.asarray(data[1])
        labels = np.asarray(data[3]) if len(data) >= 4 else np.zeros(len(x), dtype=int)
        n = len(x)
        # Compute kNN
        dists, neighbors = skdim._commonfuncs.get_nn(x, k=k, n_jobs=1)
        # Embed in 2D with MDS
        pairwise_dist = np.linalg.norm(x[:, None] - x[None, :], axis=2)
        pos = MDS(n_components=2, dissimilarity="precomputed", random_state=42).fit_transform(pairwise_dist)
        pos_dict = {i: pos[i] for i in range(n)}
        # Build the graph
        G = nx.Graph()
        for i in range(n):
            G.add_node(i, label=labels[i])
            for j in neighbors[i]:
                color = 'lightgray' if labels[i] == labels[j] else 'dimgray'
                G.add_edge(i, j, color=color)
        edge_colors = [G[u][v]['color'] for u, v in G.edges()]
        node_labels = [G.nodes[i]['label'] for i in G.nodes]
        node_colors = plt.cm.tab10(np.array(node_labels) % 10)
        # Draw the graph
        nx.draw_networkx_edges(G, pos_dict, edge_color=edge_colors, width=0.5, alpha=0.6, ax=ax)
        nx.draw_networkx_nodes(G, pos_dict, node_color=node_colors, node_size=10, alpha=0.9, ax=ax)
        # Add legend info with LID
        unique_labels = np.unique(labels)
        legend_entries = []
        for label in unique_labels:
            mask = labels == label
            avg_lid = lid_values[mask].mean() if lid_values is not None else None
            if avg_lid is None:
                lid_info = ""
            elif avg_lid.is_integer():
                lid_info = f"LID = {int(avg_lid)}"
            else:
                lid_str = f"{avg_lid:.5f}".rstrip("0").rstrip(".")
                lid_info = f"LID = {lid_str}"
            legend_entries.append(f"Label {label}: {lid_info}")
        ax.set_title(f"{name}\n" + " | ".join(legend_entries), fontsize=10)
        ax.axis("off")
        print(f"{name} is done. Remaining: {num_datasets - (axes.tolist().index(ax) + 1)}")
    plt.tight_layout()
    save_path = Path(r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots') / output_path
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved all plots to {save_path}")

###############################################################################################################################K-PLOT###############################################################################################################################
def plot_k(results_list, name_list, save_name='kplot', show = False, log = False, allowed_methods=None):
    data_by_dataset = defaultdict(lambda: defaultdict(list))

    for name, result in zip(name_list, results_list):
        if not name.startswith("mle_"):
            continue

        try:
            # Remove 'mle_' prefix and split at first '_k_'
            raw = name[len("mle_"):]  # e.g. 'bag_w_equalizing_k_2_n_bags_10_sampling_rate_0.1'
            method_part, rest = raw.split('_k_', 1)
            method_type = method_part if method_part else 'mle' # e.g. 'bag_w_equalizing'
            k = int(rest.split('_')[0])  # e.g. 2
        except Exception as e:
            print(f"[ERROR] Failed to parse: {name} — {e}")
            continue

        # Extract sampling rate (if present)
        sr_match = re.search(r'sampling_rate_([0-9.]+)', name)
        sampling_rate = sr_match.group(1) if sr_match else None

        if allowed_methods is not None:
            if method_type not in allowed_methods:
                continue
            allowed_rates = allowed_methods[method_type]
            if isinstance(allowed_rates, list):
                if sampling_rate is None or sampling_rate not in allowed_rates:
                    print(f"[SKIPPED] {method_type} with rate {sampling_rate}")
                    continue
            elif allowed_rates is False:
                continue  # explicitly blocked

        # Generate human-readable label
        if method_type == 'mle':
            method_label = 'MLE'
        else:
            readable = method_type.replace('_', '-').title()
            method_label = f"{readable} (rate={sampling_rate})" if sampling_rate else readable

        # Store the result per dataset
        for dataset, values in result.items():
            mse = values[1]
            bias2 = values[2]
            var = values[3]
            data_by_dataset[dataset][method_label].append({
                'k': k,
                'mse': mse,
                'bias2': bias2,
                'var': var
            })

    # Sort entries by k
    for dataset in data_by_dataset:
        for method in data_by_dataset[dataset]:
            data_by_dataset[dataset][method].sort(key=lambda x: x['k'])

    # Generic plot function for one metric
    def plot_metric(metric_name, ylabel):
        dataset_names = sorted(data_by_dataset.keys())
        fig, axes = plt.subplots(len(dataset_names), 1, figsize=(10, 4 * len(dataset_names)), sharex=True)
        if len(dataset_names) == 1:
            axes = [axes]

        for ax, dataset in zip(axes, dataset_names):
            for method_label, entries in sorted(data_by_dataset[dataset].items()):
                ks = [e['k'] for e in entries]
                ys = [e[metric_name] for e in entries]
                if not log:
                    ax.plot(ks, ys, label=method_label, marker='o', markersize=3)
                else:
                    ax.plot(ks, np.log10(ys), label=method_label, marker='o', markersize=3)

            ax.set_title(f'{metric_name.upper()} - {dataset}')
            if log:
                ax.set_ylabel(f'log\u2081\u2080{ylabel})')
            else:
                ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True)
            ax.set_xlabel('k')
            ax.tick_params(labelbottom=True)

        plt.tight_layout()
        directory = r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots'
        plt.savefig(directory + '\\' + f'{save_name}_{metric_name}.pdf', bbox_inches="tight")
        if show:
            plt.show()

    # Create all 3 plots
    plot_metric('mse', 'Mean Squared Error')
    plot_metric('bias2', 'Bias²')
    plot_metric('var', 'Variance')

##############################################################################################################################LOCAL PLOT###############################################################################################################################
def plot_dimension_comparison_grid_local_with_mse_bars(results, data_sets, save_name='grid_plot',
                                                       save_dir=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots',
                                                       show=False):
    dataset_names = list(data_sets.keys())
    method_keys = list(results.keys())
    green_red_cmap = mcolors.LinearSegmentedColormap.from_list("GreenRed", ["green", "red"])
    num_methods = len(method_keys)
    num_datasets = len(dataset_names)
    # Create a wider figure to fit the spatial plot + MSE bar
    fig, axs = plt.subplots(num_methods, num_datasets,
                            figsize=(10 * num_datasets, 7 * num_methods),
                            dpi=400)
    if num_methods == 1:
        axs = np.expand_dims(axs, axis=0)
    if num_datasets == 1:
        axs = np.expand_dims(axs, axis=1)
    # --- Collect bias²/variance from results using advanced parsing ---
    mse_data = defaultdict(dict)
    global_mse_max = 0

    for method_key in method_keys:
        try:
            raw = method_key[len("mle_"):] if method_key.startswith("mle_") else method_key
            method_part, rest = raw.split('_k_', 1)
            method_type = method_part if method_part else 'mle'
            k = int(rest.split('_')[0])
        except Exception as e:
            print(f"[ERROR] Could not parse method_key: {method_key}")
            continue

        sr_match = re.search(r'sampling_rate_([0-9.]+)', method_key)
        sampling_rate = sr_match.group(1) if sr_match else None

        if method_type == 'mle':
            label = 'MLE'
        else:
            readable = method_type.replace('_', '-').title()
            label = f"{readable} (rate={sampling_rate})" if sampling_rate else readable

        _, result_metrics = results[method_key]
        for dataset, vals in result_metrics.items():
            if len(vals) < 4:
                continue
            bias2, var = vals[2], vals[3]
            mse = bias2 + var
            mse_data[dataset][label] = {'bias2': bias2, 'var': var, 'mse': mse}
            global_mse_max = max(global_mse_max, mse)

    global_mse_max = 0
    for dataset_dict in mse_data.values():
        for method_dict in dataset_dict.values():
            mse = method_dict['bias2'] + method_dict['var']
            global_mse_max = max(global_mse_max, mse)

    # --- Main Plot Grid ---
    for row_idx, method_key in enumerate(method_keys):
        est_dict, _ = results[method_key]

        raw = method_key[len("mle_"):] if method_key.startswith("mle_") else method_key
        method_part, rest = raw.split('_k_', 1)
        method_type = method_part if method_part else 'mle'
        k = int(rest.split('_')[0])
        sr_match = re.search(r'sampling_rate_([0-9.]+)', method_key)
        sampling_rate = sr_match.group(1) if sr_match else None
        label = 'MLE' if method_type == 'mle' else f"{method_type.replace('_', '-').title()} (rate={sampling_rate})" if sampling_rate else method_type.title()

        for col_idx, dataset_name in enumerate(dataset_names):
            ax = axs[row_idx, col_idx]
            if dataset_name not in est_dict:
                ax.axis("off")
                continue

            X, true_dims, _ = data_sets[dataset_name]
            estimated_dims = est_dict[dataset_name][0]
            x_2d = X[:, :2]

            #mask = (x_2d[:, 0] > 1) & (x_2d[:, 0] < 1.5) & (x_2d[:, 1] > 1) & (x_2d[:, 1] < 1.5)
            #if np.sum(mask) == 0:
            #    ax.axis("off")
            #    continue

            x_2d_filtered = x_2d
            true_filtered = true_dims
            est_filtered = estimated_dims

            abs_error = np.abs(est_filtered - true_filtered)
            normalized_error = np.where(true_filtered > 0, abs_error / true_filtered, abs_error)
            normalized_error_clipped = np.clip(normalized_error, 0, 1)

            sc = ax.scatter(x_2d_filtered[:, 0], x_2d_filtered[:, 1],
                            s=5,
                            c=normalized_error_clipped,
                            cmap=green_red_cmap,
                            vmin=0,
                            vmax=1,
                            alpha=0.8,
                            rasterized=True)

            if row_idx == 0:
                ax.set_title(f"{dataset_name}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

            # --- Dynamic MSE bar as scaled axis height ---
            if dataset_name in mse_data and label in mse_data[dataset_name]:
                bias2 = mse_data[dataset_name][label]['bias2']
                var = mse_data[dataset_name][label]['var']
                mse = bias2 + var

                # Normalize height relative to global max
                norm_height = mse / global_mse_max
                height = 0.7 * norm_height
                bar_ax = ax.inset_axes([1.02, 0.15, 0.05, height])  # scale bar height dynamically
                bar_ax.axis("off")

                bar_ax.bar(0, bias2 / mse, width=1, color='green')
                bar_ax.bar(0, var / mse, width=1, bottom=bias2 / mse, color='red')

    # Colorbar
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=green_red_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Normalized Error (0 = green, ≥1 = red)")

    plt.tight_layout(rect=[0, 0, 0.93, 1])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{save_name}.pdf")
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    if show:
        plt.show()

##############################################################################################################################BAR PLOT###############################################################################################################################

def plot_bias_variance_bars_sr(results_list, name_list, fixed_k, show = False, save_name='sr_bar_plot'):
    # Step 1: Collect data by dataset and method for the fixed k
    data_by_dataset = defaultdict(list)
    for name, result in zip(name_list, results_list):
        k_match = re.search(r'_k_(\d+)(?:_|$)', name)
        if not k_match:
            continue
        k = int(k_match.group(1))
        if k != fixed_k:
            continue
        if 'mle_bag' in name:
            sr_match = re.search(r'sampling_rate_([0-9.]+)', name)
            sampling_rate = sr_match.group(1) if sr_match else 'unknown'
            method_label = f'{sampling_rate}'
        else:
            method_label = 'MLE'

        for dataset, values in result.items():
            mse = values[1]
            bias2 = values[2]
            var = values[3]
            data_by_dataset[dataset].append({
                'method': method_label,
                'mse': mse,
                'bias2': bias2,
                'var': var
            })
    # Step 2: Plot
    dataset_names = sorted(data_by_dataset.keys())
    fig, axes = plt.subplots(len(dataset_names), 1, figsize=(10, 3.5 * len(dataset_names)), sharex=True)
    if len(dataset_names) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, dataset_names):
        entries = data_by_dataset[dataset]
        methods = [e['method'] for e in entries]
        mse_vals = [e['mse'] for e in entries]
        bias_vals = [e['bias2'] for e in entries]
        var_vals = [e['var'] for e in entries]
        x = list(range(len(methods)))
        bar_width = 0.6
        # Plot bias² (bottom part in green)
        ax.bar(x, bias_vals, width=bar_width, color='green', label='Bias²')
        # Plot variance (stacked on top in red)
        ax.bar(x, var_vals, width=bar_width, bottom=bias_vals, color='red', label='Variance')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_title(f'MSE Decomposition at k={fixed_k} - {dataset}')
        ax.set_ylabel('MSE')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.legend()
    plt.tight_layout()
    directory = r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots'
    plt.savefig(directory + '\\' + f'{save_name}.pdf', bbox_inches="tight")
    if show:
        plt.show()

def plot_bias_variance_bars_varying_n_bags(results_list, name_list, fixed_k, fixed_sampling_rate, show = False, save_name='n_bags_bar_plot'):
    # Step 1: Collect data by dataset and n_bags for given k and sampling_rate
    data_by_dataset = defaultdict(list)

    for name, result in zip(name_list, results_list):
        # Only consider mle_bag
        if 'mle_bag' not in name:
            continue

        # Extract k, sampling_rate, n_bags
        k_match = re.search(r'_k_(\d+)', name)
        sr_match = re.search(r'sampling_rate_([0-9.]+)', name)
        nb_match = re.search(r'n_bags_(\d+)', name)

        if not (k_match and sr_match and nb_match):
            continue

        k = int(k_match.group(1))
        sr = sr_match.group(1)
        n_bags = int(nb_match.group(1))

        if k != fixed_k or sr != str(fixed_sampling_rate):
            continue

        method_label = f'{n_bags}'

        for dataset, values in result.items():
            mse = values[1]
            bias2 = values[2]
            var = values[3]
            data_by_dataset[dataset].append({
                'method': method_label,
                'mse': mse,
                'bias2': bias2,
                'var': var,
                'n_bags': n_bags
            })

    # Step 2: Sort entries by n_bags
    for dataset in data_by_dataset:
        data_by_dataset[dataset].sort(key=lambda e: e['n_bags'])

    # Step 3: Plot
    dataset_names = sorted(data_by_dataset.keys())
    fig, axes = plt.subplots(len(dataset_names), 1, figsize=(10, 3.5 * len(dataset_names)), sharex=True)
    if len(dataset_names) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, dataset_names):
        entries = data_by_dataset[dataset]
        if not entries:
            continue

        methods = [e['method'] for e in entries]
        mse_vals = [e['mse'] for e in entries]
        bias_vals = [e['bias2'] for e in entries]
        var_vals = [e['var'] for e in entries]

        x = list(range(len(methods)))
        bar_width = 0.6

        ax.bar(x, bias_vals, width=bar_width, color='green', label='Bias²')
        ax.bar(x, var_vals, width=bar_width, bottom=bias_vals, color='red', label='Variance')

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_title(f'MSE Decomposition at k={fixed_k}, rate={fixed_sampling_rate} - {dataset}')
        ax.set_ylabel('MSE')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.legend()

    plt.tight_layout()
    directory = r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots'
    plt.savefig(directory + '\\' + f'{save_name}.pdf', bbox_inches="tight")
    if show:
        plt.show()