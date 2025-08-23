import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union, Mapping, Any

import matplotlib.pyplot as plt
import numpy as np
##############################################################################################################################MSE BAR PLOT#############################

ALLOWED_PARAMS = {
    "n", "k", "sr", "Nbag", "lid", "dim", "pre_smooth", "post_smooth",
    "t", "estimator_name", "bagging_method", "submethod_0", "submethod_error",
}

def isfloat(num):
    if num is not None:
        try:
            float(num)
            return True
        except ValueError:
            return False
    else:
        return False

def _auto_grid(n_plots: int) -> Tuple[int, int]:
    """Return rows, cols for a near‑square grid with *rows ≥ cols*."""
    if n_plots <= 0:
        raise ValueError("Number of sub‑plots must be positive")
    n_cols = max(1, int(math.floor(math.sqrt(n_plots))))
    n_rows = int(math.ceil(n_plots / n_cols))
    while n_rows < n_cols:
        n_cols -= 1
        n_rows = int(math.ceil(n_plots / n_cols))
    return n_rows, n_cols

def _auto_fontsize(figsize: Tuple[float, float], base: Union[int, float, None]):
    return base if base is not None else max(6, 0.9 * min(figsize) + 2)

def plot_experiment_mse_bars(
    experiments: Sequence[Any],  # sequence of LID_experiment
    *,
    vary_param: str | None = None,
    grid: bool = True,
    figsize: Tuple[float, float] | None = None,
    base_fontsize: int | float | None = None,
    colors: Tuple[str, str] = ("tab:green", "tab:red"),
    label_every: int = 1,
    save_name: str = "exp_mse_bar_plot",
    save_dir: str | Path = "./plots",
    formats: Tuple[str, ...] = ("pdf",),
    show: bool = False,
):
    """Stacked-bar MSE decomposition for a *list* of ``LID_experiment`` objects.

    Parameters
    ----------
    experiments : Sequence[LID_experiment]
        Collection of experiment runs *all belonging to the same study*.
    vary_param : str, optional
        Which attribute to place on the x‑axis. If *None*, the function
        auto‑detects the single parameter that differs across the list.
    label_every : int, default=1
        Show only every *nth* x‑tick label for dense grids.
    All other keyword arguments mirror the styling / saving options of our
    earlier helpers.
    """
    if not experiments:
        raise ValueError("The experiments list is empty.")

    for experiment in experiments:
        if experiment.bagging_method == None:
            experiment.sr = 0.05
            experiment.bagging_method = 'bag'
            experiment.Nbag = int(0)

    if vary_param == 'Nbag':
        deci = 0
    else:
        deci = 3
    # ─────────────────────────────────────────────────────────────────┐
    # 1. Decide which parameter varies (auto-detect if None)          │
    #    …and verify all *other* parameters are constant *per dataset*│
    # ─────────────────────────────────────────────────────────────────┘
    def _get(exp, attr):
        return getattr(exp, attr, None)  # silent fallback to None

    # Bucket the experiments by dataset up-front
    by_ds: dict[str, list[Any]] = defaultdict(list)
    for exp in experiments:
        by_ds[exp.dataset_name].append(exp)

    # ── which parameter varies? ──────────────────────────────────────
    if vary_param is None:
        diffs = []
        for p in ALLOWED_PARAMS:
            # p varies if, in *any* dataset group, we see >1 distinct value
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
        raise ValueError(
            f"'{vary_param}' not in allowed parameters: {ALLOWED_PARAMS}"
        )

    # ── sanity-check: within each dataset only `vary_param` may differ ──────
    for ds_name, exps in by_ds.items():
        ref = exps[0]
        for e in exps[1:]:
            for p in ALLOWED_PARAMS - {vary_param}:
                if _get(e, p) != _get(ref, p):
                    raise ValueError(f"Dataset '{ds_name}': parameter '{p}' differs while varying '{vary_param}'.")

            # ─────────────────────────────────────────────────────────────────┐
            # 2. Aggregate metrics per dataset for plotting                   │
            # ─────────────────────────────────────────────────────────────────┘
            data_by_ds: dict[str, list[dict]] = defaultdict(list)
            for ds_name, exps in by_ds.items():
                for exp in exps:
                    x_val = _get(exp, vary_param)
                    sort_key = float(x_val) if isinstance(x_val, (int, float)) else str(x_val)
                    if x_val == 0:
                        x_val = None
                    data_by_ds[ds_name].append({
                        "x_val": x_val,
                        "sort_key": sort_key,
                        "bias2": exp.total_bias2,
                        "var": exp.total_var,
                    })

            ds_names = sorted(data_by_ds)
            n_rows, n_cols = (_auto_grid(len(ds_names))
                              if grid and len(ds_names) > 1 else (len(ds_names), 1))

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)
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

    with plt.rc_context(rc):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=False)
        axes = np.asarray(axes).reshape(-1)
        for ax in axes[len(ds_names):]:
            ax.axis("off")

        for ax, ds in zip(axes, ds_names):
            entries = sorted(data_by_ds[ds], key=lambda d: d["sort_key"])
            labels = [f'{e["x_val"]:.{deci}f}' if isfloat(e["x_val"]) else e["x_val"] for e in entries]
            b_vals = [e["bias2"] for e in entries]
            v_vals = [e["var"] for e in entries]
            x = np.arange(len(entries))
            ax.bar(x, b_vals, width=0.6, color=colors[0], label="Bias²")
            ax.bar(x, v_vals, width=0.6, bottom=b_vals, color=colors[1], label="Variance")
            ax.set_xticks(x)
            disp_lbl = [lbl if i % label_every == 0 else "" for i, lbl in enumerate(labels)]
            disp_lbl = [lbls if lbls is not None else experiments[0].estimator_name for lbls in disp_lbl]
            ax.set_xticklabels(disp_lbl, rotation=45, ha="right")
            ax.set_ylabel("MSE")
            ax.set_xlabel(f"Number of bags (B)")
            ax.set_title(f"Data set: {ds}")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.legend()

        fig.tight_layout()
        for fmt in formats:
            out = save_dir / f"{save_name}.{fmt}"
            fig.savefig(out, bbox_inches="tight")
            print(f"[SAVED] {out}")
        if show:
            plt.show()
        else:
            plt.close(fig)
    return fig