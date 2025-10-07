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
        "xtick.labelsize": bfs * 1,
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
