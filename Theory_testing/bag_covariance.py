###################################################Imports###################################################
import numpy as np
from scipy.spatial.distance import cdist
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union, Mapping, Any

import matplotlib.pyplot as plt
import numpy as np
###################################################OWN IMPORT###################################################
from LIDBagging.Datasets.DatasetGeneration import *
from LIDBagging.Helper.Other import *
from LIDBagging.Helper.ComparrisonMeasures import *
from LIDBagging.RunningEstimators.BaseEstimators import *
from LIDBagging.RunningEstimators.Collecting import *
from LIDBagging.RunningEstimators.Running import *
from LIDBagging.Plotting.new_plots import *
from LIDBagging.Plotting.Plots.K_plots import *
from LIDBagging.Plotting.Plots.BarPlots import *
from LIDBagging.Plotting.Plots.KNN_Graph import *
from LIDBagging.Plotting.Plots.LocalPlot import *
from LIDBagging.Plotting.Plots.SpiderCharts import *

def generate_data(n, size, generator, d, dim):
    D1 = [generator(n=n, d=d, dim=dim) for i in range(size)]
    D2 = [generator(n=n, d=d, dim=dim) for i in range(size)]
    Do = [generator(n=n, d=d, dim=dim) for i in range(size)]
    Dq = generator(n=n, d=d, dim=dim)
    data_dict = {'D1': D1, 'D2': D2, 'Do': Do, 'Dq': Dq, 'params': (n, size)}
    return data_dict

def generate_all_data(n, size, used_keys=None, load=False, save=True, save_name='cov_test_data', load_path='cov_test_data', small_data=True):
    if load:
        result = load_dict(load_path)
    else:
        data_gen = skdim.datasets.BenchmarkManifolds()
        all_keys = [key for key in data_gen.dict_gen]
        keys = all_keys[0:4] + all_keys[5:13] + all_keys[14:17] + all_keys[19:21]
        # True (d, m) pairs: intrinsic and ambient dimensions
        d_vals = [10, 3, 4, 4, 2, 6, 2, 12, 20, 10, 17, 24, 2, 20, 2, 18, 24]
        m_vals = [11, 5, 6, 8, 3, 36, 3, 72, 20, 11, 18, 25, 3, 20, 3, 72, 96]
        params = [(keys[i], [d_vals[i], m_vals[i]]) for i in range(len(keys))]
        used_params = dict(params)
        if small_data:
            used_params = {key: used_params[key] for key in used_keys}
        print('Generating Data')
        pairs = [(key, generate_data(n=n, size=size, generator=data_gen.dict_gen[key], d=used_params[key][0], dim=used_params[key][1])) for key in tqdm(used_params)]
        result = dict(pairs)
        if save and not load:
            directory = r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2'
            save_dict(data=result, directory=directory, filename=save_name)
    if used_keys is not None and not small_data:
        result = {key: result[key] for key in used_keys}
    return result

def smallest_nonzero_dists(D, k):
    D_nz = D.copy()
    D_nz[D_nz == 0] = np.inf
    idx = np.argpartition(D_nz, kth=k-1, axis=1)[:, :k]
    topk = np.take_along_axis(D_nz, idx, axis=1)
    topk.sort(axis=1)
    return topk

def simple_MLE(smallest_distances, k):
    if k != smallest_distances.shape[1]:
        Warning('k mismatch detected')
    else:
        k = smallest_distances.shape[1]
    row_max = smallest_distances[:, -1][:, None]
    ratios = smallest_distances / row_max
    mle = -k / np.sum(np.log(ratios), axis=1)
    return mle

def estimate_covariance(sr, data_dict, k, Dq_, lid_estimator=None, query_amount=None):
    n, size = data_dict['params']
    if lid_estimator is None:
        lid_estimator = simple_MLE
    if query_amount is None:
        query_amount = n
    m = int(n*sr)
    m_overlap = int(n*(sr**2))
    rng = np.random.default_rng()
    m_indep = m - m_overlap
    D1_ = [rng.choice(data_dict['D1'][i], size=m_indep, replace=False, axis=0) for i in range(size)]
    D2_ = [rng.choice(data_dict['D2'][i], size=m_indep, replace=False, axis=0) for i in range(size)]
    Do_ = [rng.choice(data_dict['Do'][i], size=m_overlap, replace=False, axis=0) for i in range(size)]
    #data_dict_ = {'D1': D1_, 'D2': D2_, 'Do': Do_, 'Dq': Dq_, 'params': (n, sr)}
    D1_X = [cdist(Dq_, D1_[i], metric='euclidean') for i in range(size)]
    D2_X = [cdist(Dq_, D2_[i], metric='euclidean') for i in range(size)]
    Do_X = [cdist(Dq_, Do_[i], metric='euclidean') for i in range(size)]

    D1_merged = [np.hstack((D1_X[i], Do_X[i])) for i in range(size)]
    D2_merged = [np.hstack((D2_X[i], Do_X[i])) for i in range(size)]

    D1_smallest_k = [smallest_nonzero_dists(D1_merged[i], k) for i in range(size)]
    D2_smallest_k = [smallest_nonzero_dists(D2_merged[i], k) for i in range(size)]

    D1_estimates = [lid_estimator(D1_smallest_k[i], k) for i in range(size)]
    D2_estimates = [lid_estimator(D2_smallest_k[i], k) for i in range(size)]

    Query_estimates_1 = [[D1_estimates[i][j] for i in range(size)] for j in range(query_amount)]
    Query_estimates_2 = [[D2_estimates[i][j] for i in range(size)] for j in range(query_amount)]

    sample_cov = [np.cov(Query_estimates_1[j], Query_estimates_2[j], ddof=1)[0, 1] for j in range(query_amount)]

    return sample_cov

def run_covariance_task(args, data_dict, Dq_, lid_estimator=None, k=10, query_amount=None):
    sr = args
    resultss = {}
    resultss[sr] = estimate_covariance(sr, data_dict, k=k, Dq_=Dq_, lid_estimator=lid_estimator, query_amount=query_amount)
    return resultss

def run_sr_test_multiprocess(param_list, data_dict, lid_estimator=None, k=10, query_amount=None, reduce_worker_count=1):
    resultss = {}
    rng = np.random.default_rng()
    Dq_ = rng.choice(data_dict['Dq'], size=query_amount, replace=False, axis=0)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()//reduce_worker_count) as pool:
        results_list = pool.starmap(run_covariance_task, [(args, data_dict, Dq_, lid_estimator, k, query_amount) for args in param_list])
    for result in results_list:
        resultss.update(result)
    return resultss

def run_sr_test_sequential(param_list, data_dict, lid_estimator=None, k=10, query_amount=None):
    results = {}
    keys = data_dict.keys()
    for key in tqdm(keys):
        data_result = {}
        rng = np.random.default_rng()
        Dq_ = rng.choice(data_dict[key]['Dq'], size=query_amount, replace=False, axis=0)
        for sr in tqdm(param_list):
            result = run_covariance_task(sr, data_dict[key], Dq_, lid_estimator, k, query_amount)
            data_result.update(result)
        results[key] = data_result
    return results

def run_sr_cov_test(n, size, param_list, k, lid_estimator=None, query_amount=None, used_keys=None, reduce_worker_count=1, load=False, save=True, load_data=False, save_data=True,
                    save_name='cov_test', load_path=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2\cov_test', save_name_data='cov_test_data',
                    load_path_data=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2\cov_test_data', sequential=True):
    if load:
        results = load_dict(load_path)
    else:
        data_dict = generate_all_data(n=n, size=size, used_keys=used_keys, load=load_data, save=save_data, save_name=save_name_data, load_path=load_path_data)
        if sequential:
            results = run_sr_test_sequential(param_list=param_list, data_dict=data_dict, lid_estimator=lid_estimator, k=k, query_amount=query_amount)
        else:
            results = run_sr_test_multiprocess(param_list=param_list, data_dict=data_dict, lid_estimator=lid_estimator, k=k, query_amount=query_amount, reduce_worker_count=reduce_worker_count)
        if save and not load:
            directory = r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\pkls2'
            save_dict(data=results, directory=directory, filename=save_name)
    return results

def plot_results(results, save_name='sr_cov_plot', save_dir=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots', show = False):
    srs = []
    covs = []
    avg_covs = []
    sdt_covs = []
    for key, value in results.items():
        srs.append(key)
        covs.append(value)
        avg_covs.append(np.mean(value))
        sdt_covs.append(np.std(value))
    idx = np.argsort(srs)
    X = np.array(srs)[idx]
    mu = np.array(avg_covs)[idx]
    sigma = np.array(sdt_covs)[idx]
    fig, ax = plt.subplots(figsize=(12, 8))  # optional size
    ax.plot(X, mu, linewidth=2, label='mean')
    ax.fill_between(X, mu - sigma, mu + sigma,
                    alpha=0.3, label='± 1 std-dev')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Mean with ±1 SD')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend(loc='best')
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{save_name}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    covs = np.array(covs)[idx]
    return covs

def _auto_grid(n_plots: int) -> Tuple[int, int]:
    """Return (rows, cols) for a near-square grid with rows ≥ cols."""
    if n_plots <= 0:
        raise ValueError("Number of sub-plots must be positive")
    n_cols = max(1, int(math.floor(math.sqrt(n_plots))))
    n_rows = int(math.ceil(n_plots / n_cols))
    while n_rows < n_cols:          # force rows ≥ cols
        n_cols -= 1
        n_rows = int(math.ceil(n_plots / n_cols))
    return n_rows, n_cols


def _auto_fontsize(figsize: Tuple[float, float], base: Union[int, float, None]):
    return base if base is not None else max(6, 0.9 * min(figsize) + 2)


def plot_sr_mean_std(
    results: Mapping[Any, Any],
    *,
    # ── layout ───────────────────────────────────────────────────────────
    grid: bool = True,
    figsize: Tuple[float, float] | None = None,
    base_fontsize: int | float | None = None,
    # ── style ────────────────────────────────────────────────────────────
    colors: Tuple[str, str] = ("tab:blue", "tab:blue"),   # (line, fill)
    label_every: int = 1,
    # ── output ───────────────────────────────────────────────────────────
    save_name: str = "sr_cov_plot",
    save_dir: str | Path = "./plots",
    formats: Tuple[str, ...] = ("pdf",),
    show: bool = False,
):
    """
    Plot **mean ± 1 SD** of coverage vs. sampling-rate.

    Parameters
    ----------
    results
        Either ``{rate: list[values]}`` for a *single* dataset
        **or** ``{dataset: {rate: list[values]}}`` for several datasets.
    label_every
        Show only every *n*-th x-tick label (helps with long rate lists).
    """

    # ── detect if we got a nested dict (multi-dataset) ───────────────────
    sample_val = next(iter(results.values()))
    if isinstance(sample_val, Mapping):
        ds_dict = results                             # multi-dataset
    else:
        ds_dict = {"data": results}                   # wrap single set

    ds_names = sorted(ds_dict)
    n_rows, n_cols = (
        _auto_grid(len(ds_names)) if grid and len(ds_names) > 1 else (len(ds_names), 1)
    )

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)            # ~4×3″ per panel
    bfs = _auto_fontsize(figsize, base_fontsize)

    rc = {
        "axes.titlesize": bfs * 1.2,
        "axes.labelsize": bfs,
        "xtick.labelsize": bfs * 0.9,
        "ytick.labelsize": bfs * 0.9,
        "legend.fontsize": bfs * 0.9,
    }

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── plotting ─────────────────────────────────────────────────────────
    with plt.rc_context(rc):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=False)
        axes = np.asarray(axes).reshape(-1)

        # hide unused cells
        for ax in axes[len(ds_names):]:
            ax.axis("off")

        for ax, ds in zip(axes, ds_names):
            rates, means, stds = [], [], []
            for r, vals in ds_dict[ds].items():
                rates.append(float(r))
                means.append(np.mean(vals))
                stds.append(np.std(vals))

            idx = np.argsort(rates)
            x = np.array(rates)[idx]
            mu = np.array(means)[idx]
            sigma = np.array(stds)[idx]

            ax.plot(x, mu, lw=2, color=colors[0], label="mean")
            ax.fill_between(x, mu - sigma, mu + sigma,
                            color=colors[1], alpha=0.25, label="±1 SD")

            # x-tick labels (sparsified + rotated)
            ax.set_xticks(x)
            lbls = [f"{v:g}" if i % label_every == 0 else ""
                    for i, v in enumerate(x)]
            ax.set_xticklabels(lbls, rotation=45, ha="right")

            ax.set_xlabel("Sampling rate")
            ax.set_ylabel("Coverage")
            ax.set_title(ds)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            ax.legend()

        fig.tight_layout()

        # ── save ────────────────────────────────────────────────────────
        for fmt in formats:
            outfile = save_dir / f"{save_name}.{fmt}"
            fig.savefig(outfile, bbox_inches="tight")
            print(f"[SAVED] {outfile}")

        if show:
            plt.show()
        else:
            plt.close(fig)










