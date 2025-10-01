from __future__ import annotations
import matplotlib.pyplot as plt
from pathlib import Path
import importlib.metadata as _imeta
import re
import pandas as pd
from typing import Any, Mapping
from plotly.colors import get_colorscale, sample_colorscale

import plotly.graph_objects as go
import pathlib
from typing import Mapping, Literal
import numpy as np
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


def reorder_sorted_experiments(df, order=None, keep_rest=True):
    order_mle = [(('Nbag', 10),
                  ('bagging_method', None),
                  ('estimator_name', 'mle'),
                  ('post_smooth', False),
                  ('pre_smooth', False),
                  ('submethod_0', '0'),
                  ('submethod_error', 'log_diff'),
                  ('t', 1)), (('Nbag', 10),
                              ('bagging_method', None),
                              ('estimator_name', 'mle'),
                              ('post_smooth', True),
                              ('pre_smooth', False),
                              ('submethod_0', '0'),
                              ('submethod_error', 'log_diff'),
                              ('t', 1)), (('Nbag', 10),
                                          ('bagging_method', 'bag'),
                                          ('estimator_name', 'mle'),
                                          ('post_smooth', False),
                                          ('pre_smooth', False),
                                          ('submethod_0', '0'),
                                          ('submethod_error', 'log_diff'),
                                          ('t', 1)), (('Nbag', 10),
                                                      ('bagging_method', 'bag'),
                                                      ('estimator_name', 'mle'),
                                                      ('post_smooth', True),
                                                      ('pre_smooth', False),
                                                      ('submethod_0', '0'),
                                                      ('submethod_error', 'log_diff'),
                                                      ('t', 1)), (('Nbag', 10),
                                                                  ('bagging_method', 'bag'),
                                                                  ('estimator_name', 'mle'),
                                                                  ('post_smooth', False),
                                                                  ('pre_smooth', True),
                                                                  ('submethod_0', '0'),
                                                                  ('submethod_error', 'log_diff'),
                                                                  ('t', 1)), (('Nbag', 10),
                                                                              ('bagging_method', 'bag'),
                                                                              ('estimator_name', 'mle'),
                                                                              ('post_smooth', True),
                                                                              ('pre_smooth', True),
                                                                              ('submethod_0', '0'),
                                                                              ('submethod_error', 'log_diff'),
                                                                              ('t', 1)), (('Nbag', 10),
                                                                                          ('bagging_method', 'bagw'),
                                                                                          ('estimator_name', 'mle'),
                                                                                          ('post_smooth', False),
                                                                                          ('pre_smooth', False),
                                                                                          ('submethod_0', '0'),
                                                                                          ('submethod_error',
                                                                                           'log_diff'),
                                                                                          ('t', 1)), (('Nbag', 10),
                                                                                                      ('bagging_method',
                                                                                                       'approx_bagwth'),
                                                                                                      ('estimator_name',
                                                                                                       'mle'),
                                                                                                      ('post_smooth',
                                                                                                       False),
                                                                                                      ('pre_smooth',
                                                                                                       False),
                                                                                                      ('submethod_0',
                                                                                                       '0'),
                                                                                                      (
                                                                                                      'submethod_error',
                                                                                                      'log_diff'),
                                                                                                      ('t', 1))]

    order_tle = [(('Nbag', 10),
                  ('bagging_method', None),
                  ('estimator_name', 'tle'),
                  ('post_smooth', False),
                  ('pre_smooth', False),
                  ('submethod_0', '0'),
                  ('submethod_error', 'log_diff'),
                  ('t', 1)), (('Nbag', 10),
                              ('bagging_method', None),
                              ('estimator_name', 'tle'),
                              ('post_smooth', True),
                              ('pre_smooth', False),
                              ('submethod_0', '0'),
                              ('submethod_error', 'log_diff'),
                              ('t', 1)), (('Nbag', 10),
                                          ('bagging_method', 'bag'),
                                          ('estimator_name', 'tle'),
                                          ('post_smooth', False),
                                          ('pre_smooth', False),
                                          ('submethod_0', '0'),
                                          ('submethod_error', 'log_diff'),
                                          ('t', 1)), (('Nbag', 10),
                                                      ('bagging_method', 'bag'),
                                                      ('estimator_name', 'tle'),
                                                      ('post_smooth', True),
                                                      ('pre_smooth', False),
                                                      ('submethod_0', '0'),
                                                      ('submethod_error', 'log_diff'),
                                                      ('t', 1)), (('Nbag', 10),
                                                                  ('bagging_method', 'bag'),
                                                                  ('estimator_name', 'tle'),
                                                                  ('post_smooth', False),
                                                                  ('pre_smooth', True),
                                                                  ('submethod_0', '0'),
                                                                  ('submethod_error', 'log_diff'),
                                                                  ('t', 1)), (('Nbag', 10),
                                                                              ('bagging_method', 'bag'),
                                                                              ('estimator_name', 'tle'),
                                                                              ('post_smooth', True),
                                                                              ('pre_smooth', True),
                                                                              ('submethod_0', '0'),
                                                                              ('submethod_error', 'log_diff'),
                                                                              ('t', 1)), (('Nbag', 10),
                                                                                          ('bagging_method', 'bagw'),
                                                                                          ('estimator_name', 'tle'),
                                                                                          ('post_smooth', False),
                                                                                          ('pre_smooth', False),
                                                                                          ('submethod_0', '0'),
                                                                                          ('submethod_error',
                                                                                           'log_diff'),
                                                                                          ('t', 1)), (('Nbag', 10),
                                                                                                      ('bagging_method',
                                                                                                       'approx_bagwth'),
                                                                                                      ('estimator_name',
                                                                                                       'tle'),
                                                                                                      ('post_smooth',
                                                                                                       False),
                                                                                                      ('pre_smooth',
                                                                                                       False),
                                                                                                      ('submethod_0',
                                                                                                       '0'),
                                                                                                      (
                                                                                                      'submethod_error',
                                                                                                      'log_diff'),
                                                                                                      ('t', 1))]

    order_mada = [(('Nbag', 10),
                   ('bagging_method', None),
                   ('estimator_name', 'mada'),
                   ('post_smooth', False),
                   ('pre_smooth', False),
                   ('submethod_0', '0'),
                   ('submethod_error', 'log_diff'),
                   ('t', 1)), (('Nbag', 10),
                               ('bagging_method', None),
                               ('estimator_name', 'mada'),
                               ('post_smooth', True),
                               ('pre_smooth', False),
                               ('submethod_0', '0'),
                               ('submethod_error', 'log_diff'),
                               ('t', 1)), (('Nbag', 10),
                                           ('bagging_method', 'bag'),
                                           ('estimator_name', 'mada'),
                                           ('post_smooth', False),
                                           ('pre_smooth', False),
                                           ('submethod_0', '0'),
                                           ('submethod_error', 'log_diff'),
                                           ('t', 1)), (('Nbag', 10),
                                                       ('bagging_method', 'bag'),
                                                       ('estimator_name', 'mada'),
                                                       ('post_smooth', True),
                                                       ('pre_smooth', False),
                                                       ('submethod_0', '0'),
                                                       ('submethod_error', 'log_diff'),
                                                       ('t', 1)), (('Nbag', 10),
                                                                   ('bagging_method', 'bag'),
                                                                   ('estimator_name', 'mada'),
                                                                   ('post_smooth', False),
                                                                   ('pre_smooth', True),
                                                                   ('submethod_0', '0'),
                                                                   ('submethod_error', 'log_diff'),
                                                                   ('t', 1)), (('Nbag', 10),
                                                                               ('bagging_method', 'bag'),
                                                                               ('estimator_name', 'mada'),
                                                                               ('post_smooth', True),
                                                                               ('pre_smooth', True),
                                                                               ('submethod_0', '0'),
                                                                               ('submethod_error', 'log_diff'),
                                                                               ('t', 1)), (('Nbag', 10),
                                                                                           ('bagging_method', 'bagw'),
                                                                                           ('estimator_name', 'mada'),
                                                                                           ('post_smooth', False),
                                                                                           ('pre_smooth', False),
                                                                                           ('submethod_0', '0'),
                                                                                           ('submethod_error',
                                                                                            'log_diff'),
                                                                                           ('t', 1)), (('Nbag', 10),
                                                                                                       (
                                                                                                       'bagging_method',
                                                                                                       'approx_bagwth'),
                                                                                                       (
                                                                                                       'estimator_name',
                                                                                                       'mada'),
                                                                                                       ('post_smooth',
                                                                                                        False),
                                                                                                       ('pre_smooth',
                                                                                                        False),
                                                                                                       ('submethod_0',
                                                                                                        '0'),
                                                                                                       (
                                                                                                       'submethod_error',
                                                                                                       'log_diff'),
                                                                                                       ('t', 1))]
    default_order = order_mle + order_tle + order_mada
    order = default_order if order is None else order

    ordered = [c for c in order if c in df.columns]
    the_rest = [c for c in df.columns if c not in order] if keep_rest else []

    # 👇 prevent pandas from interpreting the key as a 3D array
    key = pd.Index(ordered + the_rest, dtype=object)
    return df.loc[:, key]

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

def save_results2(results, directory, save_name):
    import os
    import pickle
    os.makedirs(directory, exist_ok=True)  # Ensure directory exists
    filepath = os.path.join(directory, save_name)  # Create full path
    with open(filepath, "wb") as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)

######################################################################
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

###########################################################################

def result_extraction(experiments, sweep_params, metric_keys=None, save=False, directory=None):
    df = sorted_experiments(experiments, sweep_params)
    df = reorder_sorted_experiments(df)
    metric_results = extract_metric_results(df, sweep_params, metric_keys=metric_keys)
    if save:
        save_results2(metric_results, directory=directory)
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

from plotly.colors import get_colorscale, sample_colorscale

def _parse_rgba(col: str):
    col = str(col).strip()
    if col.startswith("#"):
        h = col[1:]
        if len(h) == 3:
            r, g, b = [int(ch*2, 16) for ch in h]
        else:
            r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return (r, g, b, 1.0)
    if col.lower().startswith("rgba"):
        a = col[col.find("(")+1:col.find(")")].split(",")
        return (int(float(a[0])), int(float(a[1])), int(float(a[2])), float(a[3]))
    if col.lower().startswith("rgb"):
        a = col[col.find("(")+1:col.find(")")].split(",")
        return (int(float(a[0])), int(float(a[1])), int(float(a[2])), 1.0)
    # named colors → fallback via small scale
    return _parse_rgba(get_colorscale([(0.0, col), (1.0, col)])[-1][1])

def _lerp(a, b, t): return a + (b - a) * t

def _color_at(stops, p):
    """linear interpolate color on stops at position p ∈ [0,1]"""
    p = max(0.0, min(1.0, float(p)))
    for i in range(len(stops)-1):
        p0, c0 = stops[i]
        p1, c1 = stops[i+1]
        if p0 <= p <= p1:
            t = 0.0 if p1 == p0 else (p - p0) / (p1 - p0)
            r0,g0,b0,a0 = _parse_rgba(c0); r1,g1,b1,a1 = _parse_rgba(c1)
            r = int(round(_lerp(r0, r1, t))); g = int(round(_lerp(g0, g1, t)))
            b = int(round(_lerp(b0, b1, t))); a = _lerp(a0, a1, t)
            return f"rgba({r},{g},{b},{a:.3f})"
    r,g,b,a = _parse_rgba(stops[-1][1])
    return f"rgba({r},{g},{b},{a:.3f})"

def truncate_and_stretch(cs_like="Reds", cut_top=0.20):
    """
    Make a colorscale like Plotly 'Reds' but with the darkest tail removed.
    cut_top = 0.20 drops the top 20% (dark maroon) and stretches the rest to 1.0.
    Returns: list of (pos, color) stops usable anywhere Plotly expects a colorscale.
    """
    # base stops
    base = get_colorscale(cs_like) if isinstance(cs_like, str) else list(cs_like)
    base = sorted([(float(p), str(c)) for p,c in base], key=lambda x: x[0])

    alpha = max(1e-6, 1.0 - float(cut_top))   # keep [0, alpha] of the original
    # remap existing stops up to alpha
    kept = [(p/alpha, c) for (p,c) in base if p <= alpha]
    # ensure we end exactly at 1.0 with the color that was at alpha
    end_col = _color_at(base, alpha)
    if not kept or kept[-1][0] < 1.0 - 1e-9:
        kept.append((1.0, end_col))
    else:
        kept[-1] = (1.0, end_col)
    # ensure we start at pure white (optional but matches your request)
    start_col = _color_at(base, 0.0)
    kept[0] = (0.0, "rgba(255,255,255,1.0)")  # force true white start
    return kept

def plot_tables_from_results(
    results: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    mode: str = "combined",                   # "combined" | "values" | "params"
    normalize_data: bool = False,
    log: bool = False,
    # NEW: styling & heatmap options
    best_mark: bool = True,
    best_font_color: str = "green",
    best_by: Literal["min","max","auto"] = "auto",
    heatmap_cells: bool = False,
    heatmap_colorscale: str | list = "RdBu_r",   # large -> red, small -> blue by default
    color_scale_mode: Literal["linear","log"] = "linear",
    nan_fill_color: str = "rgba(240,240,240,0.8)",
    show_row_colorbars: bool = True,
    colorbar_len_px: int = 80,
    colorbar_gap_px: int = 10,
    colorbar_thickness_px: int = 8,
    # existing
    metric_label_map: Mapping[str, str] | None = None,
    float_fmt: str = "{:.4g}",
    title_prefix: str | None = None,
    save: bool = True,
    export_format: str = "pdf",
    save_prefix: str = "table_best",
    outdir: str | pathlib.Path = "./plots",
) -> dict[str, go.Figure]:
    """Render tables with Plotly (one figure per metric) and export to PDF (Kaleido).

    Adds optional per-row heatmap cell coloring + row colorbars and best-cell text coloring.
    """
    if mode not in {"combined", "values", "params"}:
        raise ValueError("mode must be one of {'combined','values','params'}")

    _fmt_val     = globals().get("_fmt_val", lambda k, v: f"{v}")
    _normalize   = globals().get("_normalize", lambda x: x)
    modify_label = globals().get("modify_label", lambda s: s)

    def pretty_metric_name(key: str) -> str:
        if metric_label_map and key in metric_label_map:
            return metric_label_map[key]
        base = key[6:] if key.startswith("total_") else key
        return {"mse": "MSE", "bias2": "Bias²", "var": "Variance"}.get(base, base.upper())

    def _lines(s: str) -> int:
        return 1 + str(s).count("<br>") + str(s).count("\n") if s is not None else 1

    figs: dict[str, go.Figure] = {}

    for met_key, (params_df, values_df) in results.items():
        if not isinstance(values_df, pd.DataFrame) or not isinstance(params_df, pd.DataFrame):
            raise TypeError(f"results['{met_key}'] must be a (params_df, values_df) pair of DataFrames.")
        if not values_df.index.equals(params_df.index) or not values_df.columns.equals(params_df.columns):
            raise ValueError(f"Index/columns mismatch for '{met_key}' between params_df and values_df.")

        datasets    = list(values_df.index)
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
                mod = modify_label(lbl)
                col_headers.append((mod if mod is not None else str(sig)).replace(" | ", "<br>"))
            else:
                col_headers.append(str(sig))

        # values used for text (apply log + normalize like original)
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

        # ----------------- Best-cell text colors -----------------
        n_rows = len(datasets)
        font_colors_per_column: list[list[str]] = []
        # first column (Dataset names)
        font_colors_per_column.append(["black"] * n_rows)

        metric_base = met_key[6:] if met_key.startswith("total_") else met_key
        if best_mark:
            # pick rule
            if best_by == "auto":
                criterion = "min" if metric_base in {"mse", "bias2", "var"} else "min"
            else:
                criterion = best_by

            data = vals.to_numpy(dtype=float)  # (n_rows, n_methods)
            # init all method columns as black
            for _ in method_sigs:
                font_colors_per_column.append(["black"] * n_rows)

            for r in range(n_rows):
                row_vals = data[r, :]
                finite = np.isfinite(row_vals)
                if not np.any(finite):
                    continue
                target = np.nanmin(row_vals[finite]) if criterion == "min" else np.nanmax(row_vals[finite])
                is_best = np.isclose(row_vals, target, equal_nan=False)
                for c, ok in enumerate(is_best):
                    if ok and finite[c]:
                        font_colors_per_column[c + 1][r] = best_font_color
        else:
            for _ in method_sigs:
                font_colors_per_column.append(["black"] * n_rows)

        # ----------------- Background fill colors (per-row heatmap) -----------------
        fill_colors_per_column: list[list[str]] | str = "white"
        per_row_scales: list[tuple[float, float]] = []  # for colorbars

        def _to_color(v: float, vmin: float, vmax: float) -> float:
            # normalize to [0,1]
            if not np.isfinite(v):
                return np.nan
            if vmax <= vmin:
                return 0.5
            return (v - vmin) / (vmax - vmin)

        if heatmap_cells:
            tbl_vals = values_df.astype(float).to_numpy(copy=True)  # use raw metric for color scaling
            if color_scale_mode == "log":
                with np.errstate(divide="ignore", invalid="ignore"):
                    tbl_vals = np.log10(tbl_vals)

            # per-row min/max (ignoring non-finite)
            vmins = np.nanmin(np.where(np.isfinite(tbl_vals), tbl_vals, np.nan), axis=1)
            vmaxs = np.nanmax(np.where(np.isfinite(tbl_vals), tbl_vals, np.nan), axis=1)
            per_row_scales = [(vmins[i], vmaxs[i]) for i in range(tbl_vals.shape[0])]

            # prepare fill color matrix: one list per column
            fill_colors_per_column = []
            # first column (Dataset names) stays white
            fill_colors_per_column.append(["white"] * n_rows)

            # Map normalized value → rgba via Plotly colorscale
            from plotly.colors import get_colorscale

            def _resolve_colorscale(cs_like):
                """
                Accept:
                  • Plotly built-in name (str)
                  • list of (pos, color) pairs
                  • list of color strings
                Return: sorted list of (pos∈[0,1], color_str)
                """
                # 1) String name → built-in
                if isinstance(cs_like, str):
                    return get_colorscale(cs_like)

                # 2) numpy array → list
                try:
                    import numpy as np
                    if isinstance(cs_like, np.ndarray):
                        cs_like = cs_like.tolist()
                except Exception:
                    pass

                # 3) List/tuple
                if isinstance(cs_like, (list, tuple)) and cs_like:
                    first = cs_like[0]
                    # Already (pos, color) pairs?
                    if isinstance(first, (list, tuple)) and len(first) == 2 and isinstance(first[0], (int, float)):
                        out = []
                        for p, c in cs_like:
                            p = float(p)
                            p = 0.0 if p < 0 else (1.0 if p > 1 else p)
                            out.append((p, str(c)))
                        out.sort(key=lambda pc: pc[0])
                        return out
                    # Plain list of colors → make evenly spaced stops
                    else:
                        n = len(cs_like)
                        if n == 1:
                            return [(0.0, str(cs_like[0])), (1.0, str(cs_like[0]))]
                        return [(i / (n - 1), str(c)) for i, c in enumerate(cs_like)]

                # Fallback
                return get_colorscale("Reds")

            def _parse_color(col: str):
                """Return (r,g,b,a) with r,g,b in 0..255, a in 0..1."""
                col = str(col).strip()
                if col.startswith("#"):
                    # hex #RGB, #RRGGBB
                    h = col[1:]
                    if len(h) == 3:
                        r, g, b = [int(ch * 2, 16) for ch in h]
                    else:
                        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                    return (r, g, b, 1.0)
                if col.lower().startswith("rgba"):
                    nums = col[col.find("(") + 1:col.find(")")].split(",")
                    r, g, b = int(float(nums[0])), int(float(nums[1])), int(float(nums[2]))
                    a = float(nums[3])
                    return (r, g, b, a)
                if col.lower().startswith("rgb"):
                    nums = col[col.find("(") + 1:col.find(")")].split(",")
                    r, g, b = int(float(nums[0])), int(float(nums[1])), int(float(nums[2]))
                    return (r, g, b, 1.0)
                # Last resort: ask Plotly to name-resolve via a tiny colorscale sample
                # (works for 'red', 'orange', etc.) – build a 2-stop scale and take end
                try:
                    cs_tmp = get_colorscale([(0.0, col), (1.0, col)])
                except Exception:
                    cs_tmp = [(0.0, "#000000"), (1.0, "#000000")]
                # recurse on the normalized color string
                return _parse_color(cs_tmp[-1][1])

            def _lerp(a, b, t):
                return a + (b - a) * t

            def _interp_color_from_stops(stops, x: float) -> str:
                """stops: list[(pos, color_str)], x in [0,1] → 'rgba(r,g,b,a)'"""
                import math
                if not math.isfinite(x):
                    return nan_fill_color
                x = float(max(0.0, min(1.0, x)))
                # find segment
                for i in range(len(stops) - 1):
                    p0, c0 = stops[i]
                    p1, c1 = stops[i + 1]
                    if x >= p0 and x <= p1:
                        t = 0.0 if p1 == p0 else (x - p0) / (p1 - p0)
                        r0, g0, b0, a0 = _parse_color(c0)
                        r1, g1, b1, a1 = _parse_color(c1)
                        r = int(round(_lerp(r0, r1, t)))
                        g = int(round(_lerp(g0, g1, t)))
                        b = int(round(_lerp(b0, b1, t)))
                        a = _lerp(a0, a1, t)
                        return f"rgba({r},{g},{b},{a:.3f})"
                # x beyond last stop
                r, g, b, a = _parse_color(stops[-1][1])
                return f"rgba({r},{g},{b},{a:.3f})"

            cs = _resolve_colorscale(heatmap_colorscale)

            def _interp_color(x: float) -> str:
                return _interp_color_from_stops(cs, x)

            for c in range(tbl_vals.shape[1]):
                col_fill = []
                for r in range(tbl_vals.shape[0]):
                    v = tbl_vals[r, c]
                    vmin, vmax = per_row_scales[r]
                    if not np.isfinite(v):
                        col_fill.append(nan_fill_color)
                    else:
                        x = _to_color(v, vmin, vmax)
                        col_fill.append(_interp_color(x))
                fill_colors_per_column.append(col_fill)

        # ------- Dynamic size so nothing gets cropped -------
        header_lines = max((_lines(h) for h in col_headers), default=1)
        row_lines = [max((_lines(c) for c in row), default=1) for row in cell_matrix]
        hmax = max((len(h) for h in col_headers), default=1)/10

        HEADER_LINE_PX = 28
        ROW_LINE_PX    = 26
        TOP_PX, BOT_PX = 30, 0
        SIDE_PX        = 10
        hmax_PX        = 10

        # compute base table size
        table_h = TOP_PX + header_lines * HEADER_LINE_PX + sum(max(1, n) * ROW_LINE_PX for n in row_lines) + BOT_PX + hmax_PX * hmax
        n_methods = max(0, len(col_headers) - 1)
        table_w = 260 + 160 * n_methods

        # extra width for row colorbars (if enabled)
        extra_w = 0
        if heatmap_cells and show_row_colorbars:
            extra_w = colorbar_gap_px + colorbar_len_px

        fig_w = int(table_w + extra_w + SIDE_PX*2)
        fig_h = int(table_h)

        # Reserve a right strip for the bars (inside the plotting area, not margins)
        plot_area_w_px = fig_w - SIDE_PX - max(SIDE_PX, colorbar_gap_px)  # left + right margins
        reserve_px = (colorbar_gap_px + colorbar_len_px) if (heatmap_cells and show_row_colorbars) else 0
        reserve_frac = min(0.5, max(0.0, reserve_px / max(1, plot_area_w_px))) if (heatmap_cells and show_row_colorbars) else 0  # fraction of plot area
        table_domain_right = 1.0 - reserve_frac

        HEADER_BLUE = "paleturquoise"  # column headers
        ROW_HEADER_BLUE = "aliceblue"

        header_height_px = HEADER_LINE_PX * header_lines

        fig = go.Figure(
            data=[go.Table(
                header=dict(values=col_headers, align="left", font=dict(size=12), height=header_height_px, line=dict(color="darkgrey", width=1)), #,fill_color=HEADER_BLUE
                cells=dict(
                    values=columns,
                    align="left",
                    font=dict(size=11, color=font_colors_per_column),
                    line=dict(color="darkgrey", width=1),
                    fill_color=([ [ROW_HEADER_BLUE] * n_rows ] + fill_colors_per_column[1:]) if isinstance(fill_colors_per_column, list)
                   else [ [ROW_HEADER_BLUE] * n_rows ] + [fill_colors_per_column]*(len(col_headers)-1),
                    height=ROW_LINE_PX,  # <-- uniform row height; critical for alignment
                ),
                domain=dict(x=[0.0, table_domain_right], y=[0.0, 1.0]),  # <-- leaves a strip on the right
            )]
        )

        title_txt = (", ".join([p for p in (title_prefix, f'Optimized for: {metric_label}') if p])) if title_prefix else f'Optimized for: {metric_label}'
        fig.update_layout(
            title=title_txt,
            margin=dict(l=SIDE_PX, r=max(SIDE_PX, colorbar_gap_px), t=TOP_PX, b=BOT_PX),
            width=fig_w,
            height=fig_h,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(visible=False, showgrid=True, zeroline=False, showticklabels=False, fixedrange=True, gridcolor="darkgrey", gridwidth=1)
        fig.update_yaxes(visible=False, showgrid=True, zeroline=False, showticklabels=False, fixedrange=True, gridcolor="darkgrey", gridwidth=1)

        # ----------------- Per-row horizontal colorbars (right side) -----------------
        # ----------------- Per-row horizontal colorbars (right side) -----------------
        # ----------------- Per-row horizontal colorbars (right side) -----------------
        if heatmap_cells and show_row_colorbars and per_row_scales:
            # Row midpoints in paper coords using UNIFORM row height
            row_mid_y_papers = []
            for r in range(len(datasets)):
                # absolute pixels from top of full figure to the row r midpoint
                mid_global_px = TOP_PX + header_height_px + r * ROW_LINE_PX + (ROW_LINE_PX / 2.0)
                y_paper = 1.0 - (mid_global_px / fig_h)  # paper uses 0..1 from bottom
                row_mid_y_papers.append(float(np.clip(y_paper, 0.0, 1.0)))

            # Left edge of the table domain in paper coords
            left_pad_frac = SIDE_PX / fig_w
            right_pad_frac = max(SIDE_PX, colorbar_gap_px) / fig_w
            plot_area_frac = 1.0 - left_pad_frac - right_pad_frac

            table_right_paper = left_pad_frac + table_domain_right * plot_area_frac

            # Bar placement: start immediately after the table with a pixel gap
            x_left_paper = table_right_paper + (colorbar_gap_px / fig_w)

            # Bar length in pixels (exact), thickness in px
            bar_len_px = max(20, colorbar_len_px)  # a small minimum looks better

            # --- Better ticks (denser), log-aware
            def _bar_ticks(vmin_z: float, vmax_z: float, mode: str, max_ticks: int = 9):
                if mode == "log":
                    lo, hi = float(vmin_z), float(vmax_z)  # z is already log10 for log mode
                    if not np.isfinite(lo) or not np.isfinite(hi):
                        return None, None, None, None
                    if hi <= lo:
                        hi = lo + 1e-12
                    e0, e1 = np.floor(lo), np.ceil(hi)
                    span = e1 - e0
                    # try decade ticks; if narrow, include 2 and 5 within each decade
                    if span <= 3:
                        # decade + minor (2,5) ticks
                        exps = np.arange(e0, e1 + 1e-9, 1.0)
                        tickvals = []
                        ticktext = []
                        for e in exps:
                            for m in [1, 2, 5]:
                                v = np.log10(m) + e
                                if v >= lo - 1e-9 and v <= hi + 1e-9:
                                    tickvals.append(float(v))
                                    ticktext.append(f"{m * 10 ** e:.3g}")
                        return tickvals, ticktext, None, "array"
                    else:
                        step = max(1.0, np.ceil(span / (max(3, max_ticks - 1))))
                        exps = np.arange(e0, e1 + 1e-9, step)
                        tickvals = list(exps.astype(float))
                        ticktext = [f"{np.power(10.0, e):.3g}" for e in exps]
                        return tickvals, ticktext, None, "array"
                else:
                    # linear: let Plotly choose but nudge it for density & compact labels
                    return None, None, "g", "auto"

            for r, (vmin_col, vmax_col) in enumerate(per_row_scales):
                if not (np.isfinite(vmin_col) and np.isfinite(vmax_col)):
                    continue
                if vmax_col <= vmin_col:
                    vmax_col = vmin_col + 1e-12

                # z-domain matches how you colored cells: log uses log10, linear uses raw
                if color_scale_mode == "log":
                    vmin_z, vmax_z = vmin_col, vmax_col  # already log10
                else:
                    vmin_z, vmax_z = vmin_col, vmax_col

                tickvals, ticktext, tickformat, tickmode = _bar_ticks(vmin_z, vmax_z, color_scale_mode, max_ticks=9)

                fig.add_trace(go.Heatmap(
                    z=[[vmin_z, vmax_z]],
                    zmin=vmin_z, zmax=vmax_z,
                    colorscale=heatmap_colorscale,
                    showscale=True,
                    opacity=1e-6,  # keep colorbar in static export
                    hoverinfo="skip",
                    colorbar=dict(
                        orientation="h",
                        x=x_left_paper,  # right after the table
                        xanchor="left",
                        y=row_mid_y_papers[r],
                        yanchor="middle",
                        lenmode="pixels",  # <-- exact pixel length
                        len=bar_len_px,
                        thickness=colorbar_thickness_px,
                        outlinewidth=0,
                        tickmode=tickmode,
                        tickvals=tickvals,
                        ticktext=ticktext,
                        tickformat=tickformat,
                        tickfont=dict(size=9),  # smaller labels to avoid overlap
                        ticks="outside",
                        title=dict(text="", side="top"),
                    ),
                ))

        # ----------------- Save -----------------
        if save:
            outdir = pathlib.Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            out_path = outdir / f"{save_prefix}_{met_key}_{mode}.{export_format}"
            img = fig.to_image(format=export_format, engine="kaleido", width=fig_w, height=fig_h, scale=1)
            out_path.write_bytes(img)

        figs[met_key] = fig

    return figs

reds_bright = truncate_and_stretch("Reds", cut_top=0.33)

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
    max_chars = 16,
    best_mark: bool = True,
    best_font_color: str = "green",
    best_by: Literal["min","max","auto"] = "min",
    heatmap_cells: bool = True,
    heatmap_colorscale: str | list = reds_bright,   # large -> red, small -> blue by default
    color_scale_mode: Literal["linear","log"] = "log",
    nan_fill_color: str = "rgba(240,240,240,0.8)",
    show_row_colorbars: bool = False,
    colorbar_len_px: int = 70,
    colorbar_gap_px: int = 7,
    colorbar_thickness_px: int = 8,
    savedf=False
):
    import dataframe_image as dfi
    results = result_extraction(experiments, sweep_params, metric_keys=None, directory=outdir, save=savedf)
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
    best_mark = best_mark,
    best_font_color = best_font_color,
    best_by = best_by,
    heatmap_cells = heatmap_cells,
    heatmap_colorscale = heatmap_colorscale,  # large -> red, small -> blue by default
    color_scale_mode = color_scale_mode,
    nan_fill_color = nan_fill_color,
    show_row_colorbars = show_row_colorbars,
    colorbar_len_px = colorbar_len_px,
    colorbar_gap_px = colorbar_gap_px,
    colorbar_thickness_px = colorbar_thickness_px,
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