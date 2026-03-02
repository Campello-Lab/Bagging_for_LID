import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union, Mapping, Any
import matplotlib.pyplot as plt
import numpy as np
import warnings
from Bagging_for_LID.Plotting.plotting_helpers import *
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
    save_dir: str | Path = "./Output",
    formats: Tuple[str, ...] = ("pdf",),
    show: bool = False,
    xlabel: str | None = None,
    title: str | None = None,
    fig_title: bool = False,
    n_rows: int | None = None,
    n_cols: int | None = None,
):
    """Stacked-bar MSE decomposition for a *list* of ``LID_experiment`` objects.
    """

    # Sometimes we want to have the baseline before or after the bagged variants for reference.
    # But it's a bit annoying to handle this automatically, so we just redefine it as a bagged experiment, with 1 bag that has a sr of 1, as that would be identical.
    #for experiment in experiments:
    #    if experiment.bagging_method == None:
    #        experiment.sr = 1
    #        experiment.bagging_method = 'bag'
    #        experiment.Nbag = int(1)

    #This is just for different labeling of different varying params (where to cut the decimals)
    if vary_param == 'Nbag':
        deci = 0
    else:
        deci = 3

    #Selects a class attribute
    def _get(exp, attr, default=None):
        return getattr(exp, attr, default)

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

    #These functions handle the annoying case where the input had multiple baseline experiments (with maybe diffrerent sr or Nbag which are also irrelevant)
    def _params_consistent(ref, e, ignore: set[str]):
        for p in (ALLOWED_PARAMS - ignore):
            ref_val = _get(ref, p)
            e_val = _get(e, p)
            if p in {"sr", "Nbag"}:
                if _get(ref, "bagging_method") is None or _get(e, "bagging_method") is None:
                    continue
            if ref_val != e_val:
                return False, p
        return True, None

    def _pick_single_baseline(baselines: list[Any]):
        if not baselines:
            return None
        sr1 = [b for b in baselines if _get(b, "sr") == 1]
        if sr1:
            return sr1[0]
        nb1 = [b for b in baselines if _get(b, "Nbag") == 1]
        if nb1:
            return nb1[0]
        return baselines[0]


    #across all the experiments, we to figure out if other than the (possibly prespecified varying parameter) are the other ones changing (which would invalidate the experimet, as this signals that the input data was wrong)
    #then we extract a more focused data dictionary, which only cares about the numbers necessary for plotting
    data_by_ds: dict[str, list[dict]] = defaultdict(list)

    for ds_name, exps in by_ds.items():
        if not exps:
            continue

        # pick a non-baseline as reference if possible, else any
        non_base = [e for e in exps if _get(e, "bagging_method") is not None]
        ref = non_base[0] if non_base else exps[0]

        # validate outside of variable consistency vs reference
        for e in exps:
            ok, p_bad = _params_consistent(ref, e, ignore={vary_param})
            if not ok:
                warnings.warn(
                    f"Dataset '{ds_name}': parameter '{p_bad}' differs while varying '{vary_param}'. "
                    "Proceeding anyway.",
                    category=UserWarning,
                    stacklevel=2,
                )
                print(f"Warning: Dataset '{ds_name}': parameter '{p_bad}' differs while varying '{vary_param}'. {_get(e, p_bad)} differs from {_get(ref, p_bad)}"
                    "Proceeding anyway.")

        baselines = [e for e in exps if _get(e, "bagging_method") is None]
        variants = [e for e in exps if _get(e, "bagging_method") is not None]

        # If varying Nbag or sr: include exactly one baseline (placed leftmost/rightmost)
        if vary_param in {"Nbag", "sr"}:
            chosen_base = _pick_single_baseline(baselines)
            if chosen_base is not None:
                label = "Baseline"
                # Force sort position via sentinel sort_key
                if vary_param == "Nbag":
                    sort_key = (float("-inf"), "baseline")  # leftmost
                    x_val_for_label = label
                else:  # vary_param == "sr"
                    sort_key = (float("+inf"), "baseline")  # rightmost
                    x_val_for_label = label
                if np.array([_get(e, 'sr')!=1 for e in variants]).all():
                    data_by_ds[ds_name].append({
                        "x_val": x_val_for_label,
                        "sort_key": sort_key,
                        "bias2": _get(chosen_base, "total_bias2"),
                        "var": _get(chosen_base, "total_var"),
                    })

        else:
            # Not varying Nbag/sr: include ALL baselines, ignore their sr/Nbag for labeling
            for i, b in enumerate(baselines, start=1):
                label = "Baseline" if i == 1 else f"Baseline({i})"
                # Put baselines first to the left in stable order
                data_by_ds[ds_name].append({
                    "x_val": label,
                    "sort_key": (-1, f"baseline{i:03d}"),
                    "bias2": _get(b, "total_bias2"),
                    "var": _get(b, "total_var"),
                })

        # Add the variant (bagged) experiments; sort by actual varying param
        for e in variants:
            x_val = _get(e, vary_param)
            if isinstance(x_val, (int, float)):
                sort_key = (0, float(x_val))
                x_lab = f"{x_val:.{deci}f}"
            else:
                sort_key = (0, str(x_val))
                x_lab = str(x_val)
            data_by_ds[ds_name].append({
                "x_val": x_lab,
                "sort_key": sort_key,
                "bias2": _get(e, "total_bias2"),
                "var": _get(e, "total_var"),
            })
    # Figure layout
    ds_names = sorted(data_by_ds)
    if n_rows is None and n_cols is None:
        n_rows, n_cols = (auto_grid(len(ds_names))
                          if grid and len(ds_names) > 1 else (len(ds_names), 1))
    #default figsize, fontsize, global title, label for varying parameter we try to automatically set these up well enough
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)
    if title is None:
        if vary_param == 'sr':
            paranname = 'sampling rate'
        elif vary_param == 'Nbag':
            paranname = 'number of bags'
        else:
            paranname = vary_param
        title = f"MSE decompositions for changing {paranname}. \nBaseline Estimator: {_get(experiments[0],'estimator_name').upper()}"
    if xlabel is None:
        if vary_param == 'Nbag':
            xlabel = 'Number of Bags (B)'
        elif vary_param == 'sr':
            xlabel = 'Sampling rate (r)'
        else:
            xlabel = f"{vary_param}"
    bfs = auto_fontsize(figsize, base_fontsize)
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
            labels = [e["x_val"] for e in entries]
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
            ax.set_xlabel(f"{xlabel}", labelpad=0) #
            ax.set_title(f"Data set: {ds}")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            if ds == 'uniform':
                ax.legend(loc="lower right")
            else:
                ax.legend(loc="upper right")

        if fig_title:
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