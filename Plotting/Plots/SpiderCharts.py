from __future__ import annotations
import matplotlib.pyplot as plt
from pathlib import Path
import importlib.metadata as _imeta
import re
import pandas as pd
from typing import Any, Mapping
###################################################OWN IMPORT###################################################
from LIDBagging.Helper.Other import normalize
###############################################################################################################################SPIDER CHARTS###############################################################################################################################
def _normalize(arr):
    a = np.asarray(arr, dtype=float)
    return 1 - (a - a.min()) / (a.max() - a.min()) if a.ptp() else np.zeros_like(a)

def _check_versions():
    pv = tuple(map(int, _imeta.version("plotly").split(".")[:2]))
    kv = tuple(map(int, _imeta.version("kaleido").split(".")[:2]))
    if pv >= (6, 0) and kv < (0, 2):          # Plotly ≥ 6 wants Kaleido ≥ 0.2
        raise RuntimeError(
            f"Plotly {pv} requires Kaleido ≥ 0.2.*, "
            f"but Kaleido {kv} is installed. "
            "Run  pip install -U 'kaleido>=0.2.1,<1'."
        )

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

    # Group variant indices by base method
    method_groups = defaultdict(list)
    for idx, name in enumerate(names):
        base_method = name.split('_')[0]
        method_groups[base_method].append((idx, name))

    for metric in metrics:
        metric_val = metric_indices[metric]
        main_methods = list(method_groups.keys())
        num_main_methods = len(main_methods)
        num_cols = 3
        num_rows = math.ceil(num_main_methods / num_cols)

        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols,
                                figsize=(num_cols * 4.5, num_rows * 4.5), dpi=120,
                                subplot_kw=dict(polar=True))
        axs = axs.flatten()

        for i, (main_method, variant_list) in enumerate(method_groups.items()):
            ax = axs[i]

            # Step 1: Build values array like in your working example
            values = []
            variant_names = []
            for idx, variant_name in variant_list:
                vals = [dictionaries[idx][method][metric_val] for method in methods]
                values.append(vals)
                variant_names.append(variant_name)
            values_array = np.array(values)  # shape [num_variants, num_methods]

            # Step 2: Normalize per-method across variants
            if normalize_data:
                values_array = np.array([normalize(values_array[:, i]) for i in range(num_methods)]).T

            # Step 3: Plot each variant
            for idx, value in enumerate(values_array):
                val = list(value) + [value[0]]
                ax.plot(angles, val, linewidth=1.5, linestyle='solid', label=variant_names[idx])
                if fill:
                    ax.fill(angles, val, alpha=0.04)

            ax.set_ylim(-0.1, 1)
            ax.set_yticks(np.arange(-0.1, 1.1, 0.2))
            ytick_labels = [f'{x:.1f}' for x in np.arange(-0.1, 1.1, 0.2)]
            tick_objs = ax.set_yticklabels(ytick_labels)
            for label in tick_objs:
                label.set_fontsize(7)
                label.set_color('gray')

            for angle, label in zip(angles[:-1], methods):
                angle_deg = np.degrees(angle)
                ha = 'left' if (0 <= angle_deg <= 90 or angle_deg >= 270) else 'right'
                ax.text(angle, 1.08, label, size=8, ha=ha, va='center')

            ax.set_xticklabels([])
            ax.set_title(main_method, size=11, y=1.12)

            if len(variant_list) > 1:
                ax.legend(fontsize=6, loc='lower left', bbox_to_anchor=(0.0, -0.3), frameon=False, title='Variants')

        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        fig.suptitle(f"{metric} (normalized)" if normalize_data else metric, size=14, y=0.98)
        plt.subplots_adjust(hspace=0.5, wspace=0.4, top=0.90, bottom=0.05, left=0.05, right=0.95)

        if save:
            directory = r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots'
            file_name = f'{save_prefix}_{metric}.pdf'
            plt.savefig(directory + '\\' + file_name, bbox_inches='tight')

# ---------- internal writer ------------------------------------------------- #
def _write_figure(fig, path, fmt, **size_kw):
    """Render with Kaleido, verify bytes, then write to disk."""
    img_bytes = fig.to_image(format=fmt, engine="kaleido", **size_kw)
    if not img_bytes:
        raise RuntimeError("Kaleido returned zero bytes – export aborted.")
    with open(path, "wb") as fh:
        fh.write(img_bytes)

def _pretty_variant(raw: str) -> str:
    """
    Turn cryptic file names into legend labels.

    Patterns handled
    ----------------
    baseline_k_10
    smooth_k_10
    bag_f_f_k_10_n_bags_10_sampling_rate_0.3
    bag_t_f_k_10_n_bags_10_sampling_rate_0.3
    bag_w_2_n_f_t_k_10_n_bags_10_sampling_rate_0.3
    bag_w_2_y_t_t_k_10_n_bags_10_sampling_rate_0.3
    """

    # ------------ Baseline -------------------------------------------------
    if raw.startswith("_"):
        k = re.search(r"_k_(\d+)", raw)
        return f"Baseline (k={k.group(1)})" if k else "Baseline"

    if raw.startswith("smooth"):                       # baseline + post-smoothing
        k = re.search(r"_k_(\d+)", raw)
        return f"Baseline (k={k.group(1)}, post-smoothing)" if k else "Baseline"

    # helper: extract smoothing flags --------------------------------------
    def _smoothing_flags(s: str):
        m = re.search(r'_(t|f)_(t|f)_', s)
        if not m:
            return []
        pre, post = m.groups()
        out = []
        if pre  == "t": out.append("pre-smoothing")
        if post == "t": out.append("post-smoothing")
        return out

    # ------------ Bagging-Weights -----------------------------------------
    if raw.startswith("bag_w"):
        t        = re.search(r"bag_w_(\d+)", raw).group(1)
        adjust   = "_y_" in raw
        k        = re.search(r"_k_(\d+)", raw).group(1)
        n        = re.search(r"_n_bags_(\d+)", raw).group(1)
        sr       = re.search(r"_sampling_rate_([0-9.]+)", raw).group(1)

        parts = [f"t={t}", "adjust" if adjust else None]
        parts += _smoothing_flags(raw)                    # ← NEW
        parts += [f"k={k}", f"N={n}", f"sr={sr}"]

        return "Bagging-Weights (" + ", ".join(p for p in parts if p) + ")"

    # ------------ plain Bagging -------------------------------------------
    if raw.startswith("bag"):
        k  = re.search(r"_k_(\d+)", raw).group(1)
        n  = re.search(r"_n_bags_(\d+)", raw).group(1)
        sr = re.search(r"_sampling_rate_([0-9.]+)", raw).group(1)

        parts = _smoothing_flags(raw)                     # ← NEW
        parts += [f"k={k}", f"N={n}", f"sr={sr}"]

        return "Bagging (" + ", ".join(parts) + ")"

    # ------------ fallback -------------------------------------------------
    return raw

def create_method_variant_radar_charts(
        data_sets,
        dictionaries,
        names,
        *,
        normalize_data=False,
        save=True,
        export_format="pdf",
        save_prefix="spider_chart_by_method",
        fill=True,
        outdir=Path(
            r"C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots"),
        height_per_row=450,
        width_per_col=450,
        verbose=False,
        log = False
):
    """
    Adds:
      • brighter colour cycle (Plotly qualitative palette)
      • nicer subplot titles (Capitalised, larger)
      • adaptive layout for a single panel
    """
    _check_versions()
    LEFT_MARGIN = 90  # px  (room for left labels)
    RIGHT_MARGIN = 90  # px  (room for right labels)      # ← NEW
    TOP_MARGIN = 90
    BOTTOM_MARGIN = 90
    metrics = ["MSE", "Bias2", "Var"]
    metric_idx = {"MSE": 1, "Bias2": 2, "Var": 3}
    methods = list(data_sets.keys())
    n_methods = len(methods)

    # use Plotly's qualitative colours → vivid lines
    colour_cycle = px.colors.qualitative.Plotly               # <- NEW

    # group variant indices by base method
    groups = defaultdict(list)
    for i, n in enumerate(names):
        groups[n.split("_")[0]].append((i, n))

    # ------------------------------------------------------------------ loop
    for metric in metrics:
        midx = metric_idx[metric]
        main_methods = list(groups.keys())
        n_panels = len(main_methods)

        # adaptive grid: 1×1 if you have only one panel               # <- NEW
        if n_panels == 1:
            n_cols = n_rows = 1
        else:
            n_cols, n_rows = 3, math.ceil(n_panels / 3)

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            specs=[[{"type": "polar"}] * n_cols] * n_rows,
            subplot_titles=[m.upper() for m in main_methods],
            horizontal_spacing=0.08 if n_panels > 1 else 0.10,   # <- NEW
            vertical_spacing=0.10,
        )

        # -- populate --------------------------------------------------------
        legend_seen = set()
        for k, (base_meth, variants) in enumerate(groups.items()):
            row, col = divmod(k, n_cols); row += 1; col += 1

            vals = [[dictionaries[idx][m][midx] for m in methods]
                    for idx, _ in variants]
            vals = np.asarray(vals)
            if log:
                vals = np.vstack([np.log10(vals[:, j]) for j in range(n_methods)]).T
            if normalize_data:
                vals = np.vstack([_normalize(vals[:, j]) for j in range(n_methods)]).T

            for v_i, (r_vals, (_, raw_name)) in enumerate(zip(vals, variants)):
                trimmed = raw_name.split("_", 1)[1]
                clean = _pretty_variant(trimmed)  # ← changes
                colour = colour_cycle[v_i % len(colour_cycle)]

                r, g, b = [int(c) for c in px.colors.hex_to_rgb(colour)]
                fig.add_trace(
                    go.Scatterpolar(
                        r=list(r_vals) + [r_vals[0]],
                        theta=list(methods) + [methods[0]],
                        name=clean,
                        mode="lines+markers",
                        line=dict(width=2.5, color=colour),
                        fill="toself" if fill else None,
                        fillcolor=f"rgba({r},{g},{b},0.15)" if fill else None,
                        showlegend=clean not in legend_seen,
                        legendgroup=clean,
                    ),
                    row=row, col=col,
                )
                legend_seen.add(clean)

            # tidy axis
            polar_key = f"polar{'' if (row == col == 1) else k + 1}"
            fig.update_layout(**{
                f"{polar_key}_radialaxis_range": [-0.1, 1] if normalize_data else None,
                f"{polar_key}_radialaxis_tickfont_size": 8,
                f"{polar_key}_angularaxis_tickfont_size": 8
            })


        # ----------- global look -------------------------------------------
        base_h = height_per_row * n_rows
        base_w = width_per_col  if n_panels == 1 else width_per_col * n_cols

        title = metric
        if normalize_data: title += " (log10-normalised)" if log else " (normalised)"

        fig.update_layout(
            title=dict(text=title, x=0.5, y=0.99),                      # ← NEW centred
            height=base_h, width=base_w,
            template="plotly_white",
            legend=dict(orientation="h", x=0.5, y=-0.13,        # ← NEW bottom-centre
                        xanchor="center", yanchor="top",
                        bgcolor="rgba(0,0,0,0)"),
            margin=dict(
                t=TOP_MARGIN,
                l=LEFT_MARGIN if n_panels > 1 else 40,
                r=RIGHT_MARGIN if n_panels > 1 else 20,  # ← NEW
                b=BOTTOM_MARGIN
            ),       # ← NEW extra left/bottom
            font=dict(size=20))
        # enlarge subplot-title font
        for ann in fig.layout.annotations:
            ann.y += 0.04  # was 0.03   ← NEW (a bit higher)
            ann.font.size = 16  # was 14     ← NEW (larger)
            ann.font.family = "Arial Black"

        # ------------------------------------------------------------------
        if save:
            outdir.mkdir(parents=True, exist_ok=True)
            path = (outdir / f"{save_prefix}_{metric}.{export_format}").resolve()
            img = fig.to_image(format=export_format, engine="kaleido",
                               width=base_w, height=base_h)
            with open(path, "wb") as fh:
                fh.write(img)
            if verbose:
                print(f"✔ saved {path}")


"""Radar‑chart of *best* metric per dataset for each variant.

A **variant** is defined by parameters that are *not* listed in
`sweep_params`, and excludes dataset‑specific parameters (`dataset_name`,
`n`, `lid`, `dim`).  For each variant and dataset we pick the experiment that
achieves the **lowest** metric over the sweep of `sweep_params`.  One trace
per variant, spokes = datasets.
"""

import math
from pathlib import Path
from collections import defaultdict
from typing import Any, Sequence, Tuple, Iterable

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ────────────────────────────────────────────────────────────────────────
_NUMERIC_PARAMS = {"n", "k", "sr", "Nbag", "lid", "dim", "t"}
_DATASET_SPECIFIC = {"dataset_name", "n", "lid", "dim"}
_KNOWN_PARAMS = {
    "estimator_name", "bagging_method", "submethod_0", "submethod_error",
    "k", "sr", "Nbag", "pre_smooth", "post_smooth", "t",
}

def _fmt_val(p: str, v: Any) -> str:
    if v is None:
        return "None"
    if p in {"sr", "t"}:
        return f"{float(v):.3f}"
    if p in {"n", "k", "Nbag", "lid", "dim"}:
        return str(int(v))
    return str(v)

def unordered_lookup(query, original_map = None, sep= '|'):
    if original_map is None:
        original_map  =  {
    'bagging_method:bag | pre_smooth:False | post_smooth:False': 'Simple bagging',
    'bagging_method:bag | pre_smooth:False | post_smooth:True': 'Simple bagging with post-smoothing',
    'bagging_method:bag | pre_smooth:True | post_smooth:False': 'Simple bagging with pre-smoothing',
    'bagging_method:bag | pre_smooth:True | post_smooth:True': 'Simple bagging with pre-smoothing and post-smoothing',
    'bagging_method:None | pre_smooth:False | post_smooth:False': 'Baseline',
    'bagging_method:None | pre_smooth:False | post_smooth:True': 'Baseline with smoothing'}
    def build_canonical_map(original: dict[str, str], sep: str = '|') -> dict[tuple[str, ...], str]:
        return {
            tuple(sorted(part.strip() for part in key.split(sep))): value
            for key, value in original.items()
        }
    canonical_map = build_canonical_map(original_map)
    signature = tuple(sorted(part.strip() for part in query.split(sep)))
    return canonical_map.get(signature)

def modify_label(label):
    if label == 'bagging_method:bag':
        label = 'Simple bagging'
    elif label == 'bagging_method:bagw':
        label = 'Bagging with out-of-bag weights'
    elif label == 'bagging_method:bagwth':
        label = 'Bagging with out-of-bag weights (adjust)'
    elif label == 'bagging_method:approx_bagwth':
        label = 'Bagging with out-of-bag weights (approximate adjust)'
    elif label == 'bagging_method:None':
        label = 'Baseline'
    else:
        label = unordered_lookup(label)
    return label

def plot_radar_best_of_sweep_old(
    experiments: Sequence[Any],
    *,
    sweep_params: Iterable[str] | None = None,
    normalize_data: bool = False,
    log: bool = False,
    inner_radius: float = 0.1,
    save: bool = True,
    export_format: str = "pdf",
    save_prefix: str = "radar_best",
    outdir: Path | str = "./plots",
    fill: bool = True,
    height_per_row: int = 450,
    width_per_col: int = 450,
    verbose: bool = False,
):
    """Plot radar charts of best metric values per dataset for each variant."""

    if not experiments:
        raise ValueError("experiments list is empty")

    if sweep_params is None:
        sweep_params = _NUMERIC_PARAMS
    sweep_params = set(sweep_params)

    # spokes on radar = datasets
    datasets = sorted({e.dataset_name for e in experiments})

    # params that define a variant signature
    sig_params = _KNOWN_PARAMS - sweep_params

    # bucket experiments by variant signature then dataset
    variant_map: dict[tuple, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for e in experiments:
        sig = tuple((p, getattr(e, p)) for p in sig_params)
        variant_map[sig][e.dataset_name].append(e)

    variants_sorted = sorted(variant_map.keys(), key=lambda s: str(s))

    # params that actually differ across variants → legend label
    diff_params = set()
    if len(variants_sorted) > 1:
        for idx in range(len(variants_sorted[0])):
            p_name = variants_sorted[0][idx][0]
            if {sig[idx][1] for sig in variants_sorted}.__len__() > 1:
                diff_params.add(p_name)

    metric_keys = [("mse", "MSE"), ("bias2", "Bias²"), ("var", "Variance")]
    colour_cycle = px.colors.qualitative.Plotly

    for met_key, met_label in metric_keys:
        # best metric per variant/dataset
        val_matrix = []  # shape (n_variants, n_datasets)
        chosen_info = {}
        for sig in variants_sorted:
            row = []
            ds_info = {}
            for ds in datasets:
                runs = variant_map[sig].get(ds, [])
                if not runs:
                    row.append(np.nan)
                    ds_info[ds] = None
                    continue
                best = min(runs, key=lambda r: getattr(r, f"total_{met_key}"))
                val = getattr(best, f"total_{met_key}")
                if log:
                    val = np.log10(val)
                    print(val)
                row.append(val)
                ds_info[ds] = (val, {p: getattr(best, p) for p in sweep_params})
            val_matrix.append(row)
            chosen_info[sig] = ds_info

        vals = np.asarray(val_matrix, dtype=float)
        if normalize_data:
            for j in range(vals.shape[1]):
                vals[:, j] = _normalize(vals[:, j])

        # plotly figure
        fig = go.Figure()
        radial_range = [-inner_radius, 1] if normalize_data else None
        for idx, sig in enumerate(variants_sorted):
            label = " | ".join(f"{k}:{_fmt_val(k,v)}" for k, v in sig if k in diff_params) or "default"
            label = modify_label(label)
            print(label)
            colour = colour_cycle[idx % len(colour_cycle)]
            r, g, b = [int(c) for c in px.colors.hex_to_rgb(colour)]
            r_vals = vals[idx]
            fig.add_trace(go.Scatterpolar(
                r=list(r_vals) + [r_vals[0]],
                theta=datasets + [datasets[0]],
                name=label,
                mode="lines+markers",
                line=dict(width=2.5, color=colour),
                fill="toself" if fill else None,
                fillcolor=f"rgba({r},{g},{b},0.15)" if fill else None,
            ))

            if verbose:
                print(f"▶ {label}")
                for ds in datasets:
                    info = chosen_info[sig][ds]
                    if info is None:
                        print(f"   {ds}: MISSING")
                    else:
                        val_, params_ = info
                        print(f"   {ds}: metric={val_:.4g}, params={params_}")

        title_txt = f"{met_label} | Estimator:{_fmt_val('estimator_name', getattr(experiments[0], 'estimator_name', None).upper())}"
        fig.update_layout(
            title=dict(text=title_txt, x=0.5, y=0.95),
            template="plotly_white",
            height=height_per_row,
            width=width_per_col,
            polar=dict(
                radialaxis=dict(range=radial_range, tickfont_size=9, showline=True, linewidth=1),
                angularaxis=dict(tickfont_size=9),
            ),
            legend=dict(orientation="h", x=0.5, y=-0.15, xanchor="center"),
            margin=dict(t=80, b=100, l=80, r=80),
        )

        if save:
            Path(outdir).mkdir(parents=True, exist_ok=True)
            out_path = Path(outdir) / f"{save_prefix}_{met_key}.{export_format}"
            img = fig.to_image(format=export_format, engine="kaleido",
                               width=width_per_col, height=height_per_row)
            out_path.write_bytes(img)
            if verbose:
                print("✔ saved", out_path)
        else:
            fig.show()


def sorted_experiments(experiments, sweep_params):
    _NUMERIC_PARAMS = {"n", "k", "sr", "Nbag", "lid", "dim", "t"}
    _DATASET_SPECIFIC = {"dataset_name", "n", "lid", "dim"}
    _KNOWN_PARAMS = {"estimator_name", "bagging_method", "submethod_0", "submethod_error",
                     "k", "sr", "Nbag", "pre_smooth", "post_smooth", "t"}

    if sweep_params is None:
        sweep_params = _NUMERIC_PARAMS
    sweep_params = set(sweep_params)
    method_params = _KNOWN_PARAMS - sweep_params
    datasets = sorted({getattr(e, "dataset_name") for e in experiments})

    def method_sig(e):
        return tuple((p, getattr(e, p)) for p in sorted(method_params))

    methods = sorted({method_sig(e) for e in experiments}, key=str)

    df = pd.DataFrame(index=datasets, columns=methods, dtype=object)
    for ds in datasets:
        for m in methods:
            df.at[ds, m] = []

    for e in experiments:
        ds = getattr(e, "dataset_name")
        sig = method_sig(e)
        df.at[ds, sig].append(e)

    fixed_ds_specific = (_DATASET_SPECIFIC - sweep_params) - {"dataset_name"}
    for ds in datasets:
        for m in methods:
            runs = df.at[ds, m]
            if not runs:
                continue
            for p in fixed_ds_specific:
                vals = {getattr(r, p) for r in runs}
                if len(vals) > 1:
                    raise ValueError(
                        f"Inconsistent '{p}' within cell dataset={ds}, method={m}. "
                        f"Found values: {sorted(vals)}"
                    )
    return df


def extract_params(df, params):
    datasets = df.index
    methods = df.columns
    dummydf = pd.DataFrame(index=datasets, columns=methods, dtype=object)
    for ds in datasets:
        for m in methods:
            if m[1][0] != 'bagging_method':
                print('Problem dedected with method searching code in extract_params')
            if m[1][1] is None:
                used_params = [p for p in params if p !='sr']
            else:
                used_params = params
            if isinstance(df.at[ds, m], list):
                dummydf.at[ds, m] = [{p: getattr(e, p) for p in used_params} for e in df.at[ds, m]]
            else:
                dummydf.at[ds, m] = {p: getattr(df.at[ds, m], p) for p in used_params}
    return dummydf

def extract_optimal(df, metric_key, return_values=False):
    from operator import attrgetter
    datasets = df.index
    methods = df.columns
    best_df  = pd.DataFrame(index=datasets, columns=methods, dtype=object)
    vals_df  = pd.DataFrame(index=datasets, columns=methods, dtype=float)
    for ds in datasets:
        for m in methods:
            runs = df.at[ds, m]
            best = min(runs, key=attrgetter(metric_key))
            best_df.at[ds, m] = best
            vals_df.at[ds, m] = float(getattr(best, metric_key))
    return (best_df, vals_df) if return_values else best_df

def extract_optimal_results(df, params, metric_key):
    best_df, vals_df = extract_optimal(df, metric_key, return_values=True)
    optimal_params = extract_params(best_df, params)
    return optimal_params, vals_df

def extract_metric_results(df, params, metric_keys=None):
    if metric_keys is None:
        metric_keys = ['total_mse', 'total_var', 'total_bias2']
    out = {}
    for key in metric_keys:
        out[key] = extract_optimal_results(df, params, key)
    return out

def result_extraction(experiments, sweep_params, metric_keys=None):
    df = sorted_experiments(experiments, sweep_params)
    metric_results = extract_metric_results(df, sweep_params, metric_keys=metric_keys)
    return metric_results

def plot_radar_from_results(
    results: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    normalize_data: bool = False,
    log: bool = False,
    inner_radius: float = 0.1,
    save: bool = True,
    export_format: str = "pdf",
    save_prefix: str = "radar_best",
    outdir: Path | str = "./plots",
    fill: bool = True,
    height_per_row: int = 450,
    width_per_col: int = 450,
    verbose: bool = False,
    estimator_name: str | None = None,
    metric_label_map: Mapping[str, str] | None = None,
):
    """
    Plot radar charts using the output of `result_extraction`.

    Parameters
    ----------
    results : dict[str, (params_df, values_df)]
        From `result_extraction(...)`. Each pair shares the same index/columns:
        - index  = datasets (row order defines theta order)
        - columns= method signatures (tuples of (param, value))
    """

    def pretty_metric_name(key: str) -> str:
        if metric_label_map and key in metric_label_map:
            return metric_label_map[key]
        base = key[6:] if key.startswith("total_") else key
        return {"mse": "MSE", "bias2": "Bias²", "var": "Variance"}.get(base, base.upper())

    colour_cycle = px.colors.qualitative.Plotly

    # We'll render one figure per metric key
    for met_key, (params_df, values_df) in results.items():
        if not isinstance(values_df, pd.DataFrame) or not isinstance(params_df, pd.DataFrame):
            raise TypeError(f"results['{met_key}'] must be a (params_df, values_df) pair of DataFrames.")

        # Align shapes (defensive)
        if not values_df.index.equals(params_df.index) or not values_df.columns.equals(params_df.columns):
            raise ValueError(f"Index/columns mismatch for '{met_key}' between params_df and values_df.")

        datasets = list(values_df.index)             # theta order
        method_sigs = list(values_df.columns)        # one trace per method

        # Figure out which params actually differ across methods (for labels)
        diff_params: set[str] = set()
        if len(method_sigs) > 1:
            # all columns are tuples like ((p1,v1),(p2,v2),...)
            # assume ordering is consistent (as produced by sorted_experiments)
            for idx in range(len(method_sigs[0])):
                pname = method_sigs[0][idx][0]
                uniq_vals = {sig[idx][1] for sig in method_sigs}
                if len(uniq_vals) > 1:
                    diff_params.add(pname)

        # Prepare numeric values
        vals = values_df.astype(float).copy()

        # Optional log10 (apply before normalization, like the original)
        if log:
            with np.errstate(divide="ignore", invalid="ignore"):
                vals = np.log10(vals.astype(float))
        # Optional per-dataset normalization (row-wise)
        if normalize_data:
            for ds in vals.index:
                vals.loc[ds] = _normalize(vals.loc[ds].to_numpy(dtype=float))

        # Build Plotly figure
        fig = go.Figure()
        radial_range = [-inner_radius, 1] if normalize_data else None

        for idx, sig in enumerate(method_sigs):
            # Legend label from differing params only
            label = " | ".join(f"{k}:{_fmt_val(k, v)}" for k, v in sig if k in diff_params) or "default"
            label = modify_label(label)

            colour = colour_cycle[idx % len(colour_cycle)]
            r, g, b = [int(c) for c in px.colors.hex_to_rgb(colour)]

            # Each trace is the method’s values across datasets
            r_vals = vals.iloc[:, idx].to_list()
            # Close the polar loop
            fig.add_trace(go.Scatterpolar(
                r=r_vals + [r_vals[0] if len(r_vals) else None],
                theta=datasets + [datasets[0]] if datasets else [],
                name=label,
                mode="lines+markers",
                line=dict(width=2.5, color=colour),
                fill="toself" if fill else None,
                fillcolor=f"rgba({r},{g},{b},0.15)" if fill else None,
            ))

            if verbose:
                print(f"▶ {label}")
                for ds in datasets:
                    v = vals.at[ds, sig]
                    p = params_df.at[ds, sig]
                    if p is None or (isinstance(v, float) and not np.isfinite(v)):
                        print(f"   {ds}: MISSING")
                    else:
                        print(f"   {ds}: metric={v:.4g}, params={p}")

        # Title
        est_txt = _fmt_val("estimator_name", estimator_name.upper()) if (estimator_name and hasattr(estimator_name, "upper")) else (estimator_name or "")
        met_label = pretty_metric_name(met_key)
        title_txt = f"{met_label}" + (f" | Estimator:{est_txt}" if est_txt else "")

        fig.update_layout(
            title=dict(text=title_txt, x=0.5, y=0.95),
            template="plotly_white",
            height=height_per_row,
            width=width_per_col,
            polar=dict(
                radialaxis=dict(range=radial_range, tickfont_size=9, showline=True, linewidth=1),
                angularaxis=dict(tickfont_size=9),
            ),
            legend=dict(orientation="h", x=0.5, y=-0.15, xanchor="center"),
            margin=dict(t=80, b=100, l=80, r=80),
        )

        if save:
            Path(outdir).mkdir(parents=True, exist_ok=True)
            out_path = Path(outdir) / f"{save_prefix}_{met_key}.{export_format}"
            img = fig.to_image(format=export_format, engine="kaleido",
                               width=width_per_col, height=height_per_row)
            out_path.write_bytes(img)
        else:
            fig.show()

def plot_radar_best_of_sweep(
    experiments: Sequence[Any],
    *,
    sweep_params: Iterable[str] | None = None,
    normalize_data: bool = False,
    log: bool = False,
    inner_radius: float = 0.1,
    save: bool = True,
    export_format: str = "pdf",
    save_prefix: str = "radar_best",
    outdir: Path | str = "./plots",
    fill: bool = True,
    height_per_row: int = 450,
    width_per_col: int = 450,
    verbose: bool = False,
):
    results = result_extraction(experiments, sweep_params, metric_keys=None)
    estimator_name = experiments[0].estimator_name
    plot_radar_from_results(results=results, normalize_data=normalize_data,
    log=log,
    inner_radius=inner_radius,
    save=save,
    export_format=export_format,
    save_prefix=save_prefix,
    outdir=outdir,
    fill=fill,
    height_per_row=height_per_row,
    width_per_col=width_per_col,
    verbose=verbose,
    estimator_name=estimator_name,
    metric_label_map=None)

import plotly.graph_objects as go
import pathlib
def plot_tables_from_results(
    results: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]] ,
    *,
    mode: str = "combined",            # "combined" | "values" | "params"
    normalize_data: bool = False,
    log: bool = False,
    metric_label_map: Mapping[str, str] | None = None,
    float_fmt: str = "{:.4g}",
    title_prefix: str | None = None,
    save: bool = True,
    export_format: str = "pdf",
    save_prefix: str = "table_best",
    outdir: str | pathlib.Path = "./plots",
) -> dict[str, go.Figure]:
    """Render tables with Plotly (one figure per metric) and export to PDF (Kaleido)."""
    if mode not in {"combined", "values", "params"}:
        raise ValueError("mode must be one of {'combined','values','params'}")

    _fmt_val   = globals().get("_fmt_val", lambda k, v: f"{v}")
    _normalize = globals().get("_normalize", lambda x: x)
    modify_label = globals().get("modify_label", lambda s: s)

    def pretty_metric_name(key: str) -> str:
        if metric_label_map and key in metric_label_map:
            return metric_label_map[key]
        base = key[6:] if key.startswith("total_") else key
        return {"mse": "MSE", "bias2": "Bias²", "var": "Variance"}.get(base, base.upper())

    figs: dict[str, go.Figure] = {}

    for met_key, (params_df, values_df) in results.items():
        if not isinstance(values_df, pd.DataFrame) or not isinstance(params_df, pd.DataFrame):
            raise TypeError(f"results['{met_key}'] must be a (params_df, values_df) pair of DataFrames.")
        if not values_df.index.equals(params_df.index) or not values_df.columns.equals(params_df.columns):
            raise ValueError(f"Index/columns mismatch for '{met_key}' between params_df and values_df.")

        datasets = list(values_df.index)
        method_sigs = list(values_df.columns)

        # concise headers: only differing params
        diff_params: set[str] = set()
        if method_sigs and isinstance(method_sigs[0], tuple):
            for idx in range(len(method_sigs[0])):
                pname = method_sigs[0][idx][0]
                if len({sig[idx][1] for sig in method_sigs}) > 1:
                    diff_params.add(pname)

        col_headers = ["Dataset"]
        for sig in method_sigs:
            if isinstance(sig, tuple):
                lbl = " | ".join(f"{k}:{_fmt_val(k, v)}" for k, v in sig if k in diff_params) or "default"
                if modify_label(lbl) is not None:
                    col_headers.append(modify_label(lbl).replace(" | ", "<br>"))  # allow line breaks
                else:
                    col_headers.append(str(sig))  # allow line breaks
            else:
                col_headers.append(str(sig))

        # values (log then per-dataset normalize, like radar)
        vals = values_df.astype(float).copy()
        if log:
            with np.errstate(divide="ignore", invalid="ignore"):
                vals = np.log10(vals.to_numpy(dtype=float))
            vals = pd.DataFrame(vals, index=values_df.index, columns=values_df.columns)
        if normalize_data:
            arr = vals.to_numpy(dtype=float)
            for i in range(arr.shape[0]):
                arr[i, :] = _normalize(arr[i, :])
            vals = pd.DataFrame(arr, index=values_df.index, columns=values_df.columns)

        metric_label = pretty_metric_name(met_key)

        # Build cell matrix: datasets × (dataset_col + methods)
        cell_matrix: list[list[str]] = []
        for ds in datasets:
            row = [str(ds)]
            for sig in method_sigs:
                v = vals.at[ds, sig]
                p = params_df.at[ds, sig]
                if mode == "values":
                    txt = "—" if (v is None or not np.isfinite(float(v))) else float_fmt.format(float(v))
                elif mode == "params":
                    if not isinstance(p, dict) or not p:
                        txt = "—"
                    else:
                        txt = ", ".join(f"{k}:{_fmt_val(k, p[k])}" for k in sorted(p))
                else:  # combined
                    top = ", ".join(f"{k}:{_fmt_val(k, p[k])}" for k in sorted(p)) if isinstance(p, dict) and p else ""
                    bot = f"{metric_label}:{float_fmt.format(float(v))}" if (v is not None and np.isfinite(float(v))) else ""
                    txt = f"{top}<br>{bot}" if (top and bot) else (top or bot or "—")
                row.append(txt)
            cell_matrix.append(row)

        # Plotly Table uses columns → transpose
        columns = list(map(list, zip(*cell_matrix))) if cell_matrix else [[] for _ in col_headers]

        fig = go.Figure(
            data=[go.Table(
                header=dict(values=col_headers, align="left", font=dict(size=12)),
                cells=dict(values=columns, align="left", font=dict(size=11)),
            )]
        )

        # ------- Dynamic size so nothing gets cropped -------
        def _lines(s: str) -> int:
            return 1 + str(s).count("<br>") + str(s).count("\n") if s is not None else 1

        header_lines = max((_lines(h) for h in col_headers), default=1)
        row_lines = [max((_lines(c) for c in row), default=1) for row in cell_matrix]

        hmax = max((len(h) for h in col_headers), default=1)/10

        HEADER_LINE_PX = 28
        ROW_LINE_PX    = 26
        TOP_PX, BOT_PX = 50, 14
        SIDE_PX        = 10
        hmax_PX = 10

        h = TOP_PX + header_lines * HEADER_LINE_PX + sum(max(1, n) * ROW_LINE_PX for n in row_lines) + BOT_PX + hmax_PX * hmax
        n_methods = max(0, len(col_headers) - 1)
        w = 260 + 160 * n_methods  # simple, roomy width

        title_txt = (", ".join([p for p in (title_prefix, f'Optimized for: {metric_label}') if p])) if title_prefix else f'Optimized for: {metric_label}'
        fig.update_layout(title=title_txt, margin=dict(l=SIDE_PX, r=SIDE_PX, t=TOP_PX, b=BOT_PX),
                          width=int(w), height=int(h))
        # ----------------------------------------------------

        if save:
            outdir = pathlib.Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            out_path = outdir / f"{save_prefix}_{met_key}_{mode}.{export_format}"
            img = fig.to_image(format=export_format, engine="kaleido", width=int(w), height=int(h), scale=1)
            out_path.write_bytes(img)

        figs[met_key] = fig

    return figs

def plot_table_best_of_sweep(
    experiments,
    sweep_params,
    *,
    mode: str = "combined",
    normalize_data: bool = False,
    log: bool = False,
    metric_label_map: Mapping[str, str] | None = None,
    float_fmt: str = "{:.4g}",
    title_prefix: str | None = None,
    save: bool = True,
    export_format: str = "pdf",
    save_prefix: str = "table_best",
    outdir: str | Path = "./plots",
    font_size: int = 10,
    cell_pad: float = 0.4,
    header_pad: float = 0.6,
    max_chars = 16
):
    import dataframe_image as dfi
    results = result_extraction(experiments, sweep_params, metric_keys=None)
    estimator_name = experiments[0].estimator_name
    plot_tables_from_results(
    results = results,
    mode = mode,
    normalize_data = normalize_data,
    log = log,
    metric_label_map = metric_label_map,
    float_fmt = float_fmt,
    title_prefix = f'Estimatior: {estimator_name.upper()}',
    save = save,
    export_format = export_format,
    save_prefix = save_prefix,
    outdir = outdir,
    )
    #font_size = font_size,
    #cell_pad= cell_pad,
    #header_pad = header_pad,
    #max_chars=max_chars

'''
    plot_tables_from_results(
    results = results,
    mode = mode,
    normalize_data = normalize_data,
    log = log,
    metric_label_map = metric_label_map,
    float_fmt = float_fmt,
    title_prefix = title_prefix,
    save = save,
    export_format = export_format,
    save_prefix = save_prefix,
    outdir = outdir,
    )
'''