from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence, Tuple, Union, Mapping
import matplotlib.pyplot as plt
import numpy as np
from LIDBagging.Plotting.plotting_helpers import *

_NUMERIC_PARAMS = {"n", "k", "sr", "Nbag", "lid", "dim", "t"} # class parameters that can change for bagged estimators
_BASELINE_PARAMS = {"n", "k", "lid", "dim"}            # class parameters that can change for baseline estimator
_BOOL_STR_PARAMS = { #these are not changable class parameters for this interaction plot
    "pre_smooth",
    "post_smooth",
    "estimator_name",
    "bagging_method",
    "submethod_0",
    "submethod_error",
}
_ALL_PARAMS = _NUMERIC_PARAMS | _BOOL_STR_PARAMS

def plot_experiment_heatmaps(
    experiments: Sequence[Any],
    *,
    x_param: str,
    y_param: str,
    reverse_x: bool = False,
    reverse_y: bool = False,
    metrics: Tuple[str, ...] = ("mse", "bias2", "var"),
    label_every: int = 1,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    base_fontsize: int | float | None = None,
    cmap: str = "bwr",
    save_prefix: str = "heat",
    save_dir: str | Path = "./plots",
    formats: Tuple[str, ...] = ("pdf",),
    show: bool = False,
    log=False,
    type='difference',
    inlog = False,
    fig_title = None
):
    """Draw baseline‑vs‑bagged metric differences as 2‑D heat‑maps, where the two axes represent varying parameters."""

    #sanity check ---------------------------------------------------------
    if x_param == y_param:
        raise ValueError("x_param and y_param must differ")
    for p in (x_param, y_param):
        if p not in _NUMERIC_PARAMS:
            raise ValueError(
                f"{p} must be numeric param in {sorted(_NUMERIC_PARAMS)}")
    if not experiments:
        raise ValueError("experiments list is empty")

    #separate experiments based on dataset ---------------------------------------------------------
    ds_runs: dict[str, list[Any]] = defaultdict(list)
    for e in experiments:
        ds_runs[e.dataset_name].append(e)

    #Figure out the global title that explains all the used parameters, unless a title is provided --------------------------------------------
    if fig_title is None:
        fixed_global = {}
        for p in _ALL_PARAMS - {x_param, y_param, "bagging_method", "dataset_name"}:
            vals = {getattr(e, p) for e in experiments}
            if len(vals) == 1:
                fixed_global[p] = vals.pop()
        fig_title = " | ".join(f"{k}:{fmt_val(k,v)}" for k, v in fixed_global.items())

    #automatically set up the layout, fonts -------------------------------------------------
    rows, cols = auto_grid(len(ds_runs)) if grid and len(ds_runs) > 1 else (len(ds_runs), 1)
    if figsize is None:
        figsize = (4 * cols, 3.5 * rows)
    bfs = auto_fontsize(figsize, base_fontsize)
    rc = {
        "axes.titlesize": bfs * 1.8,
        "axes.labelsize": bfs * 1.6,
        "xtick.labelsize": bfs * 0.8,
        "ytick.labelsize": bfs * 1.2,
    }

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    key_to_label = {"mse": "MSE", "bias2": "Bias²", "var": "Variance"} #Mapping from metric to their label

    # we do this plot separately for the 3 metrics: MSE, Bias² and Variance
    for met_key in metrics:
        if met_key not in key_to_label:
            raise ValueError(f"Unknown metric '{met_key}'")
        met_label = key_to_label[met_key]

        with plt.rc_context(rc):
            fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
            axes = axes.flatten()
            for ax in axes[len(ds_runs):]:
                ax.axis("off")

            for ax, (ds_name, runs) in zip(axes, sorted(ds_runs.items())):

                # we separate baseline and bagged experiments, because later we will need to compare them
                baseline_lookup: dict[tuple, Any] = {}
                bagged_list = []
                for r in runs:
                    base_key = tuple(getattr(r, p) for p in _BASELINE_PARAMS) #extract individual experiment params
                    if r.bagging_method is None: # if it's an experiment with the baseline estimator
                        baseline_lookup[base_key] = r #we save the experiment here, together with the relevant params in this dictionary
                    else:
                        bagged_list.append(r) #othrwise we save it in the bagged experiment list

                xs_sorted = sorted({getattr(r, x_param) for r in bagged_list}) #extract the relevant bagged experiment params x axis, sort them using the natural ordering of these params (usually increasing numbers)
                ys_sorted = sorted({getattr(r, y_param) for r in bagged_list}) #extract the relevant bagged experiment params y axis, sort them using the natural ordering of these params (usually increasing numbers)
                #Sometimes we want to have them in decreasing or increasing order
                if reverse_x:
                    xs_sorted = xs_sorted[::-1]
                if reverse_y:
                    ys_sorted = ys_sorted[::-1]
                xs_map = {v: i for i, v in enumerate(xs_sorted)}
                ys_map = {v: i for i, v in enumerate(ys_sorted)}
                data = np.full((len(ys_sorted), len(xs_sorted)), np.nan) #Would be function values for the map of 2d parameter value combinations

                #now we're ready to figure out the 2 function's values to build the color map
                for r in bagged_list:
                    base_key = tuple(getattr(r, p) for p in _BASELINE_PARAMS)
                    base_run = baseline_lookup.get(base_key)
                    if base_run is None:
                        continue  # no baseline → leave NaN
                    #A lot of decisions based on function settings on how to measure difference, if we even want to measure difference. We can also just build a heatmap from the bagged or baseline results individually.
                    if inlog:
                        if log:
                            if type == 'difference':
                                diff = np.log2(getattr(base_run, f"log_total_{met_key}")) - np.log2(getattr(r, f"log_total_{met_key}"))
                            elif type == 'baseline':
                                diff = -np.log2(getattr(base_run, f"log_total_{met_key}"))
                            elif type == 'bagged':
                                diff = -np.log2(getattr(r, f"log_total_{met_key}"))
                        else:
                            if type == 'difference':
                                diff = getattr(base_run, f"log_total_{met_key}") - getattr(r, f"log_total_{met_key}")
                            elif type == 'baseline':
                                diff = -getattr(base_run, f"log_total_{met_key}")
                            elif type == 'bagged':
                                diff = -getattr(r, f"log_total_{met_key}")
                    else:
                        if log:
                            if type == 'difference':
                                diff = np.log2(getattr(base_run, f"total_{met_key}")) - np.log2(getattr(r, f"total_{met_key}"))
                            elif type == 'baseline':
                                diff = -np.log2(getattr(base_run, f"total_{met_key}"))
                            elif type == 'bagged':
                                diff = -np.log2(getattr(r, f"total_{met_key}"))
                        else:
                            if type == 'difference':
                                diff = getattr(base_run, f"total_{met_key}") - getattr(r, f"total_{met_key}")
                            elif type == 'baseline':
                                diff = getattr(base_run, f"total_{met_key}")
                            elif type == 'bagged':
                                diff = getattr(r, f"total_{met_key}")
                    xi = xs_map[getattr(r, x_param)]
                    yi = ys_map[getattr(r, y_param)]
                    data[yi, xi] = diff #fill up the function values

                #setup colormap style
                vmax = np.nanmax(np.abs(data)) or 1.0
                if type == 'difference' or log:
                    im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower")
                else:
                    im = ax.imshow(data, cmap="Reds", vmin=0, vmax=vmax, origin="lower")

                # ticks --------------------------------------------------
                ax.set_xticks(range(len(xs_sorted)))
                ax.set_xticklabels([
                    fmt_val(x_param, v) if (i % label_every == 0 or i in {0, len(xs_sorted)-1}) else ""
                    for i, v in enumerate(xs_sorted)
                ], rotation=45, ha="right")
                ax.set_yticks(range(len(ys_sorted)))
                ax.set_yticklabels([
                    fmt_val(y_param, v) if (i % label_every == 0 or i in {0, len(ys_sorted)-1}) else ""
                    for i, v in enumerate(ys_sorted)
                ])

                ax.set_xlabel(x_param)
                ax.set_ylabel(y_param)
                ax.set_title(ds_name)
                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                if inlog:
                    if log:
                        if type == 'difference':
                            cbar.ax.set_ylabel(f"{met_label}\nlog₂(log_baseline) – log₂(log_bagged)")
                        elif type == 'baseline':
                            cbar.ax.set_ylabel(f"{met_label}\n-log₂(log_baseline)")
                        elif type == 'bagged':
                            cbar.ax.set_ylabel(f"{met_label}\n-log₂(log_bagged)")
                    else:
                        if type == 'difference':
                            cbar.ax.set_ylabel(f"{met_label}\nlog_baseline – log_bagged")
                        elif type == 'baseline':
                            cbar.ax.set_ylabel(f"{met_label}\n-log_baseline")
                        elif type == 'bagged':
                            cbar.ax.set_ylabel(f"{met_label}\n-log_bagged")
                else:
                    if log:
                        if type == 'difference':
                            cbar.ax.set_ylabel(f"{met_label}\nlog₂(baseline) – log₂(bagged)")
                        elif type == 'baseline':
                            cbar.ax.set_ylabel(f"{met_label}\n-log₂(baseline)")
                        elif type == 'bagged':
                            cbar.ax.set_ylabel(f"{met_label}\n-log₂(bagged)")
                    else:
                        if type == 'difference':
                            cbar.ax.set_ylabel(f"{met_label}\nbaseline – bagged")
                        elif type == 'baseline':
                            cbar.ax.set_ylabel(f"{met_label}\nbaseline")
                        elif type == 'bagged':
                            cbar.ax.set_ylabel(f"{met_label}\nbagged")

            if fig_title:
                fig.suptitle(fig_title, y=1.02, fontsize=bfs * 3)
            fig.tight_layout()
            if log:
                logsavename = '_log'
            else:
                logsavename = ''
            if inlog:
                inlogsavename = '_inlog'
            else:
                inlogsavename = ''
            for fmt in formats:
                out = save_dir / f"{save_prefix}_{met_key}_{type}{logsavename}{inlogsavename}.{fmt}"
                fig.savefig(out, bbox_inches="tight")
                print(f"[SAVED] {out}")
            if show:
                plt.show()
            else:
                plt.close(fig)


def _get_vec_value(
    e: Any, *,
    value_attr: str,
    value_index: int,
) -> float:
    vec = getattr(e, value_attr)
    if vec is None:
        return np.nan
    # accept list/np array/etc.
    try:
        return float(vec[value_index])
    except Exception:
        return np.nan


def _default_value_label(value_attr: str, value_index: int) -> str:
    # For point_bag_avg_knn_dists, index maps to neighbor rank (index 0 -> 1-NN).
    if value_attr == "point_bag_avg_knn_dists":
        if value_index == -1:
            return "avg k-NN distance"
        return f"avg {(value_index + 1)}-NN distance"
    return f"{value_attr}[{value_index}]"


def plot_experiment_attr(
    experiments: Sequence[Any],
    *,
    # axes
    x_param: str = "k",
    y_param: str = "sr",
    reverse_x: bool = False,
    reverse_y: bool = False,

    # value
    value_attr: str = "point_bag_avg_knn_dists",
    value_index: int = -1,
    value_label: str | None = None,

    # comparison
    mode: str = "value",  # {"value","difference"}
    compare_param: str = "bagging_method",
    compare_values: Tuple[Any, Any] = (None, "bag"),  # (A,B)
    select_value: Any | None = None,  # only for mode="value"
    diff_kind: str = "difference",  # {"difference","log2"}

    # plot kind
    plot_kind: str = "heatmap",  # {"heatmap","slice1d"}
    slice_param: str | None = None,   # e.g. "k"
    slice_value: Any | None = None,   # e.g. 50

    # visuals
    label_every: int = 1,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    base_fontsize: int | float | None = None,
    cmap: str = "bwr",

    # saving
    save_prefix: str = "plot_attr",
    save_dir: str | Path = "./plots",
    formats: Tuple[str, ...] = ("pdf",),
    show: bool = False,

    fig_title: str | None = None,
    strict_other_params: bool = True,
):
    # ---- checks ----
    if not experiments:
        raise ValueError("experiments list is empty")
    if mode not in {"value", "difference"}:
        raise ValueError("mode must be 'value' or 'difference'")
    if diff_kind not in {"difference", "log2"}:
        raise ValueError("diff_kind must be 'difference' or 'log2'")
    if plot_kind not in {"heatmap", "slice1d"}:
        raise ValueError("plot_kind must be 'heatmap' or 'slice1d'")

    if value_label is None:
        value_label = _default_value_label(value_attr, value_index)

    # for slice1d, determine which parameter is the varying axis
    if plot_kind == "slice1d":
        if slice_param is None or slice_value is None:
            raise ValueError("slice1d requires slice_param and slice_value")
        if slice_param not in {x_param, y_param}:
            raise ValueError("slice_param must be either x_param or y_param")

        varying_param = y_param if slice_param == x_param else x_param

    # ---- group by dataset ----
    ds_runs: dict[str, list[Any]] = defaultdict(list)
    for e in experiments:
        ds_runs[e.dataset_name].append(e)

    # ---- title ----
    if fig_title is None:
        exclude = {x_param, y_param, "dataset_name"}
        if mode == "difference":
            exclude.add(compare_param)
        else:
            if select_value is not None:
                exclude.add(compare_param)
        if plot_kind == "slice1d":
            exclude.add(slice_param)

        fixed_global = {}
        for p in (_ALL_PARAMS - exclude):
            vals = {getattr(e, p) for e in experiments}
            if len(vals) == 1:
                fixed_global[p] = vals.pop()
        fig_title = " | ".join(f"{k}:{fmt_val(k, v)}" for k, v in fixed_global.items())

    # ---- layout/fonts (same spirit) ----
    rows, cols = auto_grid(len(ds_runs)) if grid and len(ds_runs) > 1 else (len(ds_runs), 1)
    if figsize is None:
        figsize = (4 * cols, 3.5 * rows)
    bfs = auto_fontsize(figsize, base_fontsize)
    rc = {
        "axes.titlesize": bfs * 1.8,
        "axes.labelsize": bfs * 1.6,
        "xtick.labelsize": bfs * 0.8,
        "ytick.labelsize": bfs * 1.2,
    }

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def safe_vals_repr(vals: set[Any]) -> str:
        return ", ".join(sorted(repr(v) for v in vals))

    with plt.rc_context(rc):
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        for ax in axes[len(ds_runs):]:
            ax.axis("off")

        for ax, (ds_name, runs) in zip(axes, sorted(ds_runs.items())):

            # ---- optional strict check ----
            if strict_other_params:
                exclude = {x_param, y_param, "dataset_name"}
                if mode == "difference":
                    exclude.add(compare_param)
                else:
                    if select_value is not None:
                        exclude.add(compare_param)
                if plot_kind == "slice1d":
                    exclude.add(slice_param)

                for p in (_ALL_PARAMS - exclude):
                    vals = {getattr(r, p) for r in runs}
                    if len(vals) > 1:
                        raise ValueError(
                            f"[{ds_name}] parameter '{p}' varies ({safe_vals_repr(vals)}). "
                            f"Assumes all params except '{x_param}', '{y_param}'"
                            f"{' and compare_param' if mode=='difference' else ''}"
                            f"{' and slice_param' if plot_kind=='slice1d' else ''} are fixed. "
                            "Set strict_other_params=False to disable."
                        )

            # ---- filter to a fixed slice, if requested ----
            if plot_kind == "slice1d":
                runs = [r for r in runs if getattr(r, slice_param) == slice_value]

            if not runs:
                ax.set_title(ds_name)
                ax.text(0.5, 0.5, "No runs", ha="center", va="center")
                continue

            # ---- compute data depending on mode ----
            if mode == "difference":
                a_val, b_val = compare_values
                runs_a = [r for r in runs if getattr(r, compare_param, None) == a_val]
                runs_b = [r for r in runs if getattr(r, compare_param, None) == b_val]
                relevant = runs_a + runs_b
            else:
                if select_value is None:
                    relevant = runs
                else:
                    relevant = [r for r in runs if getattr(r, compare_param, None) == select_value]

            if not relevant:
                ax.set_title(ds_name)
                ax.text(0.5, 0.5, "No runs selected", ha="center", va="center")
                continue

            # ---- HEATMAP ----
            if plot_kind == "heatmap":
                xs_sorted = sorted({getattr(r, x_param) for r in relevant})
                ys_sorted = sorted({getattr(r, y_param) for r in relevant})
                if reverse_x:
                    xs_sorted = xs_sorted[::-1]
                if reverse_y:
                    ys_sorted = ys_sorted[::-1]
                xs_map = {v: i for i, v in enumerate(xs_sorted)}
                ys_map = {v: i for i, v in enumerate(ys_sorted)}

                if mode == "difference":
                    data_a = np.full((len(ys_sorted), len(xs_sorted)), np.nan)
                    data_b = np.full((len(ys_sorted), len(xs_sorted)), np.nan)

                    for r in runs_a:
                        xi = xs_map[getattr(r, x_param)]
                        yi = ys_map[getattr(r, y_param)]
                        data_a[yi, xi] = _get_vec_value(r, value_attr=value_attr, value_index=value_index)
                    for r in runs_b:
                        xi = xs_map[getattr(r, x_param)]
                        yi = ys_map[getattr(r, y_param)]
                        data_b[yi, xi] = _get_vec_value(r, value_attr=value_attr, value_index=value_index)

                    if diff_kind == "difference":
                        data = data_a - data_b
                        cbar_label = f"{value_label}\n{compare_param}={a_val} – {compare_param}={b_val}"
                    else:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            data = np.log2(data_a) - np.log2(data_b)
                        cbar_label = f"{value_label}\nlog₂({compare_param}={a_val}) – log₂({compare_param}={b_val})"

                    vmax = np.nanmax(np.abs(data)) or 1.0
                    im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower")

                else:
                    data = np.full((len(ys_sorted), len(xs_sorted)), np.nan)
                    for r in relevant:
                        xi = xs_map[getattr(r, x_param)]
                        yi = ys_map[getattr(r, y_param)]
                        data[yi, xi] = _get_vec_value(r, value_attr=value_attr, value_index=value_index)

                    vmax = np.nanmax(np.abs(data)) or 1.0
                    im = ax.imshow(data, cmap="Reds", vmin=0, vmax=vmax, origin="lower")
                    cbar_label = value_label if select_value is None else f"{value_label}\n{compare_param}={select_value}"

                # ticks like before
                ax.set_xticks(range(len(xs_sorted)))
                ax.set_xticklabels(
                    [
                        fmt_val(x_param, v) if (i % label_every == 0 or i in {0, len(xs_sorted)-1}) else ""
                        for i, v in enumerate(xs_sorted)
                    ],
                    rotation=45, ha="right"
                )
                ax.set_yticks(range(len(ys_sorted)))
                ax.set_yticklabels(
                    [
                        fmt_val(y_param, v) if (i % label_every == 0 or i in {0, len(ys_sorted)-1}) else ""
                        for i, v in enumerate(ys_sorted)
                    ]
                )
                ax.set_xlabel(x_param)
                ax.set_ylabel(y_param)
                ax.set_title(ds_name)

                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                cbar.ax.set_ylabel(cbar_label)

            # ---- SLICE1D ----
            else:
                # plot across varying_param
                zs = sorted({getattr(r, varying_param) for r in relevant})
                if (varying_param == x_param and reverse_x) or (varying_param == y_param and reverse_y):
                    zs = zs[::-1]
                z_map = {v: i for i, v in enumerate(zs)}

                if mode == "difference":
                    a_val, b_val = compare_values
                    y_a = np.full(len(zs), np.nan)
                    y_b = np.full(len(zs), np.nan)

                    for r in runs_a:
                        zi = z_map.get(getattr(r, varying_param))
                        if zi is not None:
                            y_a[zi] = _get_vec_value(r, value_attr=value_attr, value_index=value_index)
                    for r in runs_b:
                        zi = z_map.get(getattr(r, varying_param))
                        if zi is not None:
                            y_b[zi] = _get_vec_value(r, value_attr=value_attr, value_index=value_index)

                    if diff_kind == "difference":
                        y = y_a - y_b
                        ylab = f"{value_label} ({compare_param}={a_val} – {compare_param}={b_val})"
                    else:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            y = np.log2(y_a) - np.log2(y_b)
                        ylab = f"{value_label} (log₂ {compare_param}={a_val} – log₂ {compare_param}={b_val})"
                else:
                    y = np.full(len(zs), np.nan)
                    for r in relevant:
                        zi = z_map.get(getattr(r, varying_param))
                        if zi is not None:
                            y[zi] = _get_vec_value(r, value_attr=value_attr, value_index=value_index)
                    ylab = value_label if select_value is None else f"{value_label} ({compare_param}={select_value})"

                x = np.arange(len(zs))
                ax.plot(x, y)  # default matplotlib styling
                ax.set_title(ds_name)
                ax.set_xlabel(varying_param)
                ax.set_ylabel(ylab)

                ax.set_xticks(x)
                ax.set_xticklabels(
                    [
                        fmt_val(varying_param, v) if (i % label_every == 0 or i in {0, len(zs)-1}) else ""
                        for i, v in enumerate(zs)
                    ],
                    rotation=45, ha="right"
                )
                ax.grid(True, alpha=0.3)

                # annotate the slice choice
                ax.text(
                    0.01, 0.98,
                    f"{slice_param}={fmt_val(slice_param, slice_value)}",
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=bfs * 1.0
                )

        if fig_title:
            fig.suptitle(fig_title, y=1.02, fontsize=bfs * 3)
        fig.tight_layout()

        safe_idx = str(value_index).replace("-", "m")
        slice_tag = ""
        if plot_kind == "slice1d":
            slice_tag = f"_slice_{slice_param}_{fmt_val(slice_param, slice_value)}"

        for fmt in formats:
            out = save_dir / f"{save_prefix}_{value_attr}_{safe_idx}_{plot_kind}_{mode}{('_'+diff_kind) if mode=='difference' else ''}{slice_tag}.{fmt}"
            fig.savefig(out, bbox_inches="tight")
            print(f"[SAVED] {out}")

        if show:
            plt.show()
        else:
            plt.close(fig)