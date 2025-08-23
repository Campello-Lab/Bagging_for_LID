import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
###############################################################################################################################K-PLOT###############################################################################################################################
def plot_k2(
    results_list,
    name_list,
    save_name='kplot',
    show=False,
    log=False,
    allowed_methods=None,
    partial_key=None,
    adjust_k=4,
    directory="C:\pkls",
):
    data_by_dataset = defaultdict(lambda: defaultdict(list))
    for name, result in zip(name_list, results_list):
        try:
            prefix, remainder = name.split('_', 1)
            method_part, rest = remainder.split('_k_', 1)
            k = int(rest.split('_')[0])
        except Exception as e:
            print(f"[ERROR] Failed to parse: {name} — {e}")
            continue
        is_baseline = (method_part == '')
        method_type = method_part if not is_baseline else prefix
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
                continue
        if is_baseline:
            method_label = prefix.upper()
        else:
            readable = method_type.replace('_', '-').title()
            method_label = f"{readable} (rate={sampling_rate})" if sampling_rate else readable
        for dataset, values in result.items():
            if partial_key is None:
                mse, bias2, var = values[1:4]
            else:
                mse, bias2, var = values[4][partial_key][1:4]
            entry = {
                'k': int(k / adjust_k) if is_baseline else k,
                'mse': mse,
                'bias2': bias2,
                'var': var,
            }
            data_by_dataset[dataset][method_label].append(entry)
    for dataset in data_by_dataset:
        for method in data_by_dataset[dataset]:
            data_by_dataset[dataset][method].sort(key=lambda x: x['k'])
    def plot_metric(metric_name, ylabel, directory):
        dataset_names = sorted(data_by_dataset.keys())
        fig, axes = plt.subplots(
            len(dataset_names), 1,
            figsize=(10, 4 * len(dataset_names)),
        )
        if len(dataset_names) == 1:
            axes = [axes]
        for ax, dataset in zip(axes, dataset_names):
            for method_label, entries in sorted(data_by_dataset[dataset].items()):
                ks = [e['k'] for e in entries]
                ys = [e[metric_name] for e in entries]
                ax.plot(
                    ks,
                    np.log10(ys) if log else ys,
                    label=method_label,
                    marker='o',
                    markersize=3,
                )
                ax.set_xticks(ks[::3])
            ax.set_title(f'{metric_name.upper()} - {dataset}')
            ax.set_ylabel(f'log\u2081\u2080({ylabel})' if log else ylabel)
            ax.set_xlabel('k')
            ax.grid(True)
            ax.legend()
        plt.tight_layout()
        directory = (
            directory
        )
        plt.savefig(f"{directory}\\{save_name}_{metric_name}.pdf", bbox_inches="tight")
        if show:
            plt.show()
    plot_metric('mse', 'MSE', directory=directory)
    plot_metric('bias2', 'Bias²', directory=directory)
    plot_metric('var', 'Variance', directory=directory)

def find_parameter(string, substring):
    index = string.find(substring)
    if index != -1:
        after_substring = string[index + len(substring):]
        underscore_index = after_substring.find("_")
        number_str = after_substring[:underscore_index] if underscore_index != -1 else after_substring
        number = int(number_str)
        return number
    else:
        print("Substring not found")

def simplified_transposed_measurement_results(results, datasets):
    new_results = {key: {} for key in datasets}
    dataset_list = []
    for key, value in results.items():
        for key2 in value[1].keys():
            new_results[key2] = new_results[key2] | {key: value[1][key2]}
    return new_results

def split_simplified_results(new_results, method_name, method_type1, method_type2):
    prefix1 = f'{method_name}_{method_type1}'
    prefix2 = f'{method_name}_{method_type2}'
    method_1_results = {key: {} for key in new_results}
    method_2_results = {key: {} for key in new_results}
    for key, value in new_results.items():
        for key2 in value.keys():
            if key2.startswith(prefix1) and not key2.startswith(prefix2):
                method_1_results[key][key2] = value[key2]
            elif key2.startswith(prefix2):
                method_2_results[key][key2] = value[key2]
            else:
                continue
    return method_1_results, method_2_results

def extract_k_in_results(new_results):
    for key, value in new_results.items():
        for key2 in value.keys():
            k = find_parameter(key2, "_k_")
            new_results[key][key2] = (new_results[key][key2], k)
    return new_results

#results:   Dict["method_key"][0]["dataset_key"] = (lid_estimates, avg_lid_Estimate)
#           Dict["method_key"][1]["dataset_key"] = [avg_lid, mse, avg_bias2, est_var]

def sorted_function_results(new_result):
    k_list = []
    measurement_list = []
    for key, value in new_result.items():
        measurement_list.append(value[0])
        k_list.append(value[1])
    paired = list(zip(k_list, measurement_list))
    paired_sorted = sorted(paired, key=lambda x: x[0])
    sorted_inputs, sorted_outputs = zip(*paired_sorted)
    sorted_inputs = list(sorted_inputs)
    sorted_outputs = list(sorted_outputs)
    return sorted_inputs, sorted_outputs

def final_process_results(results, datasets, method_name, method_type1, method_type2):
    prefix1 = f'{method_name}_{method_type1}'
    prefix2 = f'{method_name}_{method_type2}'
    new_results = simplified_transposed_measurement_results(results, datasets)
    method_1_results, method_2_results = split_simplified_results(new_results, method_name, method_type1, method_type2)
    method_1_results = extract_k_in_results(method_1_results)
    method_2_results = extract_k_in_results(method_2_results)
    processed_results = {key: {} for key in new_results}
    for key in new_results.keys():
        sorted_inputs1, sorted_outputs1 = sorted_function_results(method_1_results[key])
        sorted_inputs2, sorted_outputs2 = sorted_function_results(method_2_results[key])
        processed_results[key][prefix1] = (sorted_inputs1, sorted_outputs1)
        processed_results[key][prefix2] = (sorted_inputs2, sorted_outputs2)
    return processed_results

#measurement_index, 1: mse, 2: bias2, 3: var
def single_calc_k(processed_result, measurement_index=1):
    method_1_ks, method_1_measurements = processed_result[list(processed_result.keys())[0]]
    method_2_ks, method_2_measurements = processed_result[list(processed_result.keys())[1]]
    method_1_ks = np.array(method_1_ks)
    method_2_ks = np.array(method_2_ks)
    method_1_measurements = np.array(method_1_measurements)
    method_2_measurements = np.array(method_2_measurements)
    used_method_1_measurements = method_1_measurements[:, measurement_index]
    used_method_2_measurements = method_2_measurements[:, measurement_index]
    if (method_1_ks != method_2_ks).any():
        Warning("Different set of ks were used for the two methods.")
    kstar = method_1_ks[np.argmin(used_method_1_measurements)]
    fstar = np.min(used_method_1_measurements)
    kstar2 = method_2_ks[np.argmin(used_method_2_measurements)]
    S_k2 = method_2_ks[used_method_2_measurements <= fstar]
    if len(S_k2) > 0:
        k2 = np.min(S_k2)
    else:
        k2 = kstar2
    f2 = np.min(used_method_2_measurements)
    return kstar, k2, fstar, f2, kstar2, S_k2

def calc_kstar_k2(processed_results):
    k_full_results = {key: {} for key in processed_results}
    k_partial_results = {key: {} for key in processed_results}
    for key, value in processed_results.items():
        for i, measurement in enumerate(['mse', 'bias2', 'var']):
            kstar, k2, fstar, f2, kstar2, S_k2 = single_calc_k(value, measurement_index=i+1)
            k_full_results[key][measurement] = (kstar, k2, fstar, f2, kstar2, S_k2)
            k_partial_results[key][measurement] = (kstar, k2, fstar, f2)
    return k_full_results, k_partial_results

def min_kstar_k2(results, datasets, method_name, method_type1, method_type2):
    processed_results = final_process_results(results, datasets, method_name, method_type1, method_type2)
    k_full_results, k_partial_results = calc_kstar_k2(processed_results)
    print(k_partial_results)
    print('\n\n')
    print(k_full_results)
    return k_partial_results, k_full_results, processed_results

def min_kstar_k22(results, datasets, method_name, method_type1, method_type2):
    processed_results = final_process_results(results, datasets, method_name, method_type1, method_type2)
    k_full_results, k_partial_results = calc_kstar_k2(processed_results)
    print(k_partial_results)
    print('\n\n')
    print(k_full_results)
    return k_partial_results, k_full_results, processed_results

__all__ = [
    "parse_experiment_name",
    "plot_bias_variance_bars_new",
    "plot_k",
]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def parse_experiment_name(name: str, *, adjust_k: int = 4):
    """Parse the experiment *file / run* name to recover meta‑information.

    Naming convention supported::

        <prefix>_<method>_k_<k>[...]sampling_rate_<rate>[...]

    Returns a dictionary with keys::
        k, method_label, method_type, sampling_rate, is_baseline
    """
    try:
        prefix, remainder = name.split("_", 1)
        method_part, rest = remainder.split("_k_", 1)
        k_raw = int(rest.split("_")[0])
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Unrecognised experiment name: {name}") from exc

    is_baseline = method_part == ""
    method_type = method_part if not is_baseline else prefix

    sr_match = re.search(r"sampling_rate_([0-9.]+)", name)
    sampling_rate = sr_match.group(1) if sr_match else None

    k_value = int(k_raw / adjust_k) if is_baseline else k_raw

    if is_baseline:
        method_label = prefix.upper()
    else:
        readable = method_type.replace("_", "-").title()
        method_label = (
            f"{readable} (rate={sampling_rate})" if sampling_rate else readable
        )

    return {
        "k": k_value,
        "method_label": method_label,
        "method_type": method_type,
        "sampling_rate": sampling_rate,
        "is_baseline": is_baseline,
    }


def _extract_metrics(values, *, partial_key=None):
    """Return (mse, bias², var) from the raw *values* container."""
    if partial_key is None:
        mse, bias2, var = values[1:4]
    else:
        mse, bias2, var = values[4][partial_key][1:4]
    return float(mse), float(bias2), float(var)


def _auto_grid(n_plots: int) -> Tuple[int, int]:
    """Return *(n_rows, n_cols)* so rows ≥ cols and layout ≈ square."""
    if n_plots <= 0:
        raise ValueError("Number of sub-plots must be positive")

    n_cols = max(1, int(math.floor(math.sqrt(n_plots))))
    n_rows = int(math.ceil(n_plots / n_cols))

    while n_rows < n_cols:  # enforce rows ≥ cols
        n_cols -= 1
        n_rows = int(math.ceil(n_plots / n_cols))
    return n_rows, n_cols

# ---------------------------------------------------------------------------
# 1. Bias–Variance stacked bar plot
# ---------------------------------------------------------------------------

def plot_bias_variance_bars_new(
    results_list: Iterable,
    name_list: Iterable[str],
    fixed_k: int,
    *,
    grid: bool = False,
    figsize: Union[Tuple[float, float], None] = None,
    base_fontsize: Union[int, float, None] = None,
    colors: Tuple[str, str] = ("tab:green", "tab:red"),
    log: bool = False,
    allowed_methods: dict | None = None,
    partial_key=None,
    adjust_k: int = 4,
    show: bool = False,
    save_name: str = "sr_bar_plot",
    save_dir: Union[str, Path] = "./plots",
    formats: Tuple[str, ...] = ("pdf",),
):
    """Stacked‑bar visualisation of Bias² + Variance (their sum is MSE)."""
    data_by_dataset: dict[str, list[dict]] = defaultdict(list)
    for name, result in zip(name_list, results_list):
        try:
            parsed = parse_experiment_name(name, adjust_k=adjust_k)
        except ValueError:
            print(f"[WARN] Skipping unparsable run: {name}")
            continue
        if parsed["k"] != fixed_k:
            continue
        if allowed_methods is not None:
            mtype = parsed["method_type"]
            if mtype not in allowed_methods:
                continue
            allowed_rates = allowed_methods[mtype]
            if isinstance(allowed_rates, list):
                if parsed["sampling_rate"] not in allowed_rates:
                    continue
            elif allowed_rates is False:
                continue
        for dataset, values in result.items():
            mse, bias2, var = _extract_metrics(values, partial_key=partial_key)
            if log:
                bias2, var = np.log10([bias2, var])
                mse = np.log10(mse)
            data_by_dataset[dataset].append(
                {
                    "method": parsed["method_label"],
                    "bias2": bias2,
                    "var": var,
                    "mse": mse,
                }
            )

    datasets = sorted(data_by_dataset)
    n_plots = len(datasets)
    if n_plots == 0:
        raise ValueError("No data matched the criteria – nothing to plot.")

    if grid and n_plots > 1:
        n_rows, n_cols = _auto_grid(n_plots)
    else:
        n_rows, n_cols = n_plots, 1

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    if base_fontsize is None:
        base_fontsize = max(6, 0.9 * min(figsize) + 2)

    rc = {
        "axes.titlesize": base_fontsize * 1.2,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize * 0.9,
        "ytick.labelsize": base_fontsize * 0.9,
        "legend.fontsize": base_fontsize * 0.9,
    }

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            sharex=False,
        )
        axes = np.array(axes).reshape(-1)
        for ax in axes[n_plots:]:
            ax.axis("off")

        for ax, dataset in zip(axes, datasets):
            entries = sorted(data_by_dataset[dataset], key=lambda d: d["method"])
            methods = [e["method"] for e in entries]
            bias_vals = [e["bias2"] for e in entries]
            var_vals = [e["var"] for e in entries]
            x = np.arange(len(methods))
            bar_width = 0.6
            ax.bar(x, bias_vals, width=bar_width, color=colors[0], label="Bias²")
            ax.bar(x, var_vals, width=bar_width, bottom=bias_vals, color=colors[1], label="Variance")
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha="right")
            ax.set_ylabel("log₁₀(MSE comp.)" if log else "MSE")
            ax.set_title(f"MSE decomposition at k={fixed_k} – {dataset}")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.legend()

        fig.tight_layout()
        for fmt in formats:
            outfile = save_dir / f"{save_name}.{fmt}"
            fig.savefig(outfile, bbox_inches="tight")
            try:
                rel = outfile.relative_to(Path.cwd())
            except ValueError:
                rel = outfile
            print(f"[SAVED] {rel}")
        if show:
            plt.show()
        else:
            plt.close(fig)
    return fig

# ---------------------------------------------------------------------------
# 2. k‑vs‑metric line plots (one figure per metric)
# ---------------------------------------------------------------------------

def plot_k(
    results_list: Sequence,
    name_list: Sequence[str],
    save_name: str = "kplot",
    show: bool = False,
    log: bool = False,
    allowed_methods: dict | None = None,
    partial_key=None,
    adjust_k: int = 1,
    *,
    # new visual options
    grid: bool = False,
    figsize: Union[Tuple[float, float], None] = None,
    base_fontsize: Union[int, float, None] = None,
    # output control
    save_dir: Union[str, Path] = "./plots",
    formats: Tuple[str, ...] = ("pdf",),
):
    """Plot MSE, Bias², Variance vs. *k* for every dataset & method.

    *If* ``grid=True`` the dataset panels are laid out in a roughly‑square grid;
    otherwise they are stacked vertically (the original behaviour).
    """

    # ------------------------------------------------------------------
    # Ingest raw data → structureᴅ as dataset → method → list[metrics]
    # ------------------------------------------------------------------
    data_by_dataset: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for name, result in zip(name_list, results_list):
        try:
            parsed = parse_experiment_name(name, adjust_k=adjust_k)
        except ValueError:
            print(f"[WARN] Skipping unparsable run: {name}")
            continue

        # filter methods / rates
        if allowed_methods is not None:
            mtype = parsed["method_type"]
            if mtype not in allowed_methods:
                continue
            allowed_rates = allowed_methods[mtype]
            if isinstance(allowed_rates, list):
                if parsed["sampling_rate"] not in allowed_rates:
                    continue
            elif allowed_rates is False:
                continue

        for dataset, values in result.items():
            mse, bias2, var = _extract_metrics(values, partial_key=partial_key)
            entry = {
                "k": parsed["k"],
                "mse": mse,
                "bias2": bias2,
                "var": var,
                "sr": parsed["sampling_rate"]
            }
            data_by_dataset[dataset][parsed["method_label"]].append(entry)

    # sort by k
    for dataset in data_by_dataset:
        for method in data_by_dataset[dataset]:
            data_by_dataset[dataset][method].sort(key=lambda x: x["k"])

    def _plot_one_metric(metric: str, ylabel: str):
        datasets = sorted(data_by_dataset)
        n_plots = len(datasets)
        if grid and n_plots > 1:
            n_rows, n_cols = _auto_grid(n_plots)
        else:
            n_rows, n_cols = n_plots, 1

        if figsize is None:
            fig_size = (4 * n_cols, 3 * n_rows)
        else:
            fig_size = figsize

        if base_fontsize is None:
            bfs = max(6, 0.9 * min(fig_size) + 2)
        else:
            bfs = base_fontsize

        rc = {
            "axes.titlesize": bfs * 1.2,
            "axes.labelsize": bfs,
            "xtick.labelsize": bfs * 0.9,
            "ytick.labelsize": bfs * 0.9,
            "legend.fontsize": bfs * 0.9,
        }

        with plt.rc_context(rc):
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=fig_size,
                sharex=False,
            )
            axes = np.array(axes).reshape(-1)
            for ax in axes[n_plots:]:
                ax.axis("off")

            for ax, dataset in zip(axes, datasets):
                srs = []
                for method_label, entries in sorted(data_by_dataset[dataset].items()):
                    sr = [float(e["sr"]) if e["sr"] is not None else 1 for e in entries]
                    srs = srs + sr
                    ks = [e["k"] for e in entries]
                    ys = [e[metric] for e in entries]
                    ys_plot = np.log10(ys) if log else ys
                    ax.plot(ks, ys_plot, marker="o", markersize=0.5, label=method_label, linewidth=0.75)
                    ax.set_xticks(ks[:: max(1, len(ks) // 10)])
                #– at sr={np.min(srs):.2f}
                ax.set_title(f"{metric.upper()} – {dataset}")
                ylab = f"log₁₀({ylabel})" if log else ylabel
                ax.set_ylabel(ylab)
                ax.set_xlabel("k")
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.legend()

            fig.tight_layout()
            sd = Path(save_dir)
            sd.mkdir(parents=True, exist_ok=True)
            for fmt in formats:
                outfile = sd / f"{save_name}_{metric}.{fmt}"
                fig.savefig(outfile, bbox_inches="tight")
                try:
                    rel = outfile.relative_to(Path.cwd())
                except ValueError:
                    rel = outfile
                print(f"[SAVED] {rel}")
            if show:
                plt.show()
            else:
                plt.close(fig)

    _plot_one_metric("mse", "MSE")
    _plot_one_metric("bias2", "Bias²")
    _plot_one_metric("var", "Variance")
