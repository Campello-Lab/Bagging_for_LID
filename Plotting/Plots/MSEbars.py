import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union, Mapping, Any
import matplotlib.pyplot as plt
import numpy as np
from LIDBagging.Plotting.plotting_helpers import *
##############################################################################################################################MSE BAR PLOT#############################

#allow the testing of MSE changes when one of these are changing
ALLOWED_PARAMS = {
    "n", "k", "sr", "Nbag", "lid", "dim", "pre_smooth", "post_smooth",
    "t", "estimator_name", "bagging_method", "submethod_0", "submethod_error",
}

def plot_experiment_mse_bars(
    experiments: Sequence[Any],  # LID_experiments
    *,
    vary_param: str | None = None,
    grid: bool = True,
    figsize: Tuple[float, float] | None = None,
    base_fontsize: int | float | None = None,
    colors: Tuple[str, str] = ("tab:green", "tab:red"),
    label_every: int = 1,
    save_prefix: str = "exp_mse_bar_plot",
    save_dir: str | Path = "./plots",
    formats: Tuple[str, ...] = ("pdf",),
    show: bool = False,
    xlabel: str | None = None,
    title: str | None = None,
):
    """Stacked-bar MSE decomposition for a *list* of ``LID_experiment`` objects.
    """

    # Sometimes we want to have the baseline before or after the bagged variants for reference.
    # But it's a bit annoying to handle this automatically, so we just redefine it as a bagged experiment, with 1 bag that has a sr of 1, as that would be identical.
    for experiment in experiments:
        if experiment.bagging_method == None:
            experiment.sr = 1
            experiment.bagging_method = 'bag'
            experiment.Nbag = int(1)

    #This is just for different labeling of different varying params (where to cut the decimals)
    if vary_param == 'Nbag':
        deci = 0
    else:
        deci = 3

    #Selects a class attribute
    def _get(exp, attr):
        return getattr(exp, attr, None)

    #separate experiments based on dataset
    by_ds: dict[str, list[Any]] = defaultdict(list)
    for exp in experiments:
        by_ds[exp.dataset_name].append(exp)

    #This here automatically tries to figure out which class parameter is changing (e.g., sampling rate, number of bags, but can be something else from ALLOWED_PARAMS), in case it is not prespecified.
    if vary_param is None:
        diffs = []
        for p in ALLOWED_PARAMS:
            if any(len({_get(e, p) for e in exps}) > 1 for exps in by_ds.values()):
                diffs.append(p)
        if not diffs:
            raise ValueError("All experiments share identical parameters – nothing varies.")
        if len(diffs) > 1:
            raise ValueError(
                "More than one parameter varies across experiments. "
                "Specify which one with `vary_param=`.  Varying params: " + ", ".join(diffs)
            )
        vary_param = diffs[0]
    elif vary_param not in ALLOWED_PARAMS:
        raise ValueError(f"'{vary_param}' not in allowed parameters: {ALLOWED_PARAMS}")

    #across all the experiments, we to figure out if other than the (possibly prespecified varying parameter) are the other ones changing (which would invalidate the experimet, as this signals that the input data was wrong)
    #then we extract a more focused data dictionary, which only cares about the numbers necessary for plotting
    for ds_name, exps in by_ds.items():
        ref = exps[0]
        for e in exps[1:]:
            for p in ALLOWED_PARAMS - {vary_param}:
                if _get(e, p) != _get(ref, p):
                    raise ValueError(f"Dataset '{ds_name}': parameter '{p}' differs while varying '{vary_param}'.")

            data_by_ds: dict[str, list[dict]] = defaultdict(list)
            for ds_name, exps in by_ds.items():
                for exp in exps:
                    x_val = _get(exp, vary_param)
                    sort_key = float(x_val) if isinstance(x_val, (int, float)) else str(x_val)
                    data_by_ds[ds_name].append({
                        "x_val": x_val,
                        "sort_key": sort_key,
                        "bias2": exp.total_bias2,
                        "var": exp.total_var,
                    })

            ds_names = sorted(data_by_ds)
            n_rows, n_cols = (_auto_grid(len(ds_names))
                              if grid and len(ds_names) > 1 else (len(ds_names), 1))

    #default figsize, fontsize, global title, label for varying parameter we try to automatically set these up well enough
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)
    if title is None:
        title = "MSE decompositions for estimator: " + str(experiments[0].estimator_name).upper()
    if xlabel is None:
        xlabel = f'{vary_param}' # This parameter's name will be put on the x-axis, but only if some other description is not prespecificed
    bfs = _auto_fontsize(figsize, base_fontsize)
    rc = {
        "axes.titlesize": bfs * 1.2,
        "axes.labelsize": bfs,
        "xtick.labelsize": bfs * 0.9,
        "ytick.labelsize": bfs * 0.9,
        "legend.fontsize": bfs * 0.9,
    }

    #we will save the plot here
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)


    with plt.rc_context(rc):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=False)
        axes = np.asarray(axes).reshape(-1)
        for ax in axes[len(ds_names):]:
            ax.axis("off")

        for ax, ds in zip(axes, ds_names):
            entries = sorted(data_by_ds[ds], key=lambda d: d["sort_key"]) #We sort the results along the x-axis in increasing order of the varying parameter, (e.g.,like sr, number of bags).
            labels = [f'{e["x_val"]:.{deci}f}' if isfloat(e["x_val"]) else e["x_val"] for e in entries]
            b_vals = [e["bias2"] for e in entries]
            v_vals = [e["var"] for e in entries]
            x = np.arange(len(entries))
            ax.bar(x, b_vals, width=0.6, color=colors[0], label="Bias²")
            ax.bar(x, v_vals, width=0.6, bottom=b_vals, color=colors[1], label="Variance")
            ax.set_xticks(x)
            disp_lbl = [lbl if i % label_every == 0 else "" for i, lbl in enumerate(labels)]
            disp_lbl = [lbls if lbls is not None else experiments[0].estimator_name for lbls in disp_lbl] #The none case corresponds to something that generally shouldn't happen, we will just plot the estimator name in this case
            ax.set_xticklabels(disp_lbl, rotation=45, ha="right")
            ax.set_ylabel("MSE") #This plot is fixed to the mse
            ax.set_xlabel(f"{xlabel}") #
            ax.set_title(f"Data set: {ds}")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.legend()

        fig.suptitle(f'{title}')
        fig.tight_layout()
        for fmt in formats:
            out = save_dir / f"{save_prefix}.{fmt}"
            fig.savefig(out, bbox_inches="tight")
            print(f"[SAVED] {out}")
        if show:
            plt.show()
        else:
            plt.close(fig)
    return fig