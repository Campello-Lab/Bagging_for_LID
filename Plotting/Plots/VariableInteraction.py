from __future__ import annotations

"""Heat‑map visualisation of baseline–vs‑bagging differences for LID experiments.

* Revised 2025‑06‑03 to **reuse a single baseline** whenever the varying
  parameters are `sr`, `Nbag`, or `t`, which do **not** influence the
  baseline estimator.  Baseline lookup is keyed only by parameters that *do*
  matter to the baseline (`n`, `k`, `lid`, `dim`).

* Adds `reverse_x`/`reverse_y` to flip either axis.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence, Tuple, Union, Mapping

import matplotlib.pyplot as plt
import numpy as np

# ────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────
_NUMERIC_PARAMS = {"n", "k", "sr", "Nbag", "lid", "dim", "t"}
_BASELINE_PARAMS = {"n", "k", "lid", "dim"}            # affect baseline
_BOOL_STR_PARAMS = {
    "pre_smooth",
    "post_smooth",
    "estimator_name",
    "bagging_method",
    "submethod_0",
    "submethod_error",
}
_ALL_PARAMS = _NUMERIC_PARAMS | _BOOL_STR_PARAMS


def _auto_grid(n: int) -> tuple[int, int]:
    if n <= 0:
        raise ValueError("n must be positive")
    cols = int(np.floor(np.sqrt(n))) or 1
    rows = int(np.ceil(n / cols))
    while rows < cols:
        cols -= 1
        rows = int(np.ceil(n / cols))
    return rows, cols


def _auto_fontsize(figsize: tuple[float, float], base: int | float | None) -> float:
    return float(base) if base is not None else max(6.0, 0.9 * min(figsize) + 2)


def _fmt_val(p: str, v: Any) -> str:
    if v is None:
        return "None"
    if p in {"sr", "t"}:
        return f"{float(v):.3f}"
    if p in {"n", "k", "Nbag", "lid", "dim"}:
        return str(int(v))
    return str(v)

# ────────────────────────────────────────────────────────────────────────
# main plotting routine
# ────────────────────────────────────────────────────────────────────────

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
    save_name: str = "heat",
    save_dir: str | Path = "./plots",
    formats: Tuple[str, ...] = ("pdf",),
    show: bool = False,
    log=False,
    type='difference',
    inlog = False,
):
    """Draw baseline‑vs‑bagged metric differences as 2‑D heat‑maps."""

    # sanity ---------------------------------------------------------
    if x_param == y_param:
        raise ValueError("x_param and y_param must differ")
    for p in (x_param, y_param):
        if p not in _NUMERIC_PARAMS:
            raise ValueError(
                f"{p} must be numeric param in {sorted(_NUMERIC_PARAMS)}")
    if not experiments:
        raise ValueError("experiments list is empty")

    # bucket by dataset ---------------------------------------------
    ds_runs: dict[str, list[Any]] = defaultdict(list)
    for e in experiments:
        ds_runs[e.dataset_name].append(e)

    # figure‑level title --------------------------------------------
    fixed_global = {}
    for p in _ALL_PARAMS - {x_param, y_param, "bagging_method", "dataset_name"}:
        vals = {getattr(e, p) for e in experiments}
        if len(vals) == 1:
            fixed_global[p] = vals.pop()
    fig_title = " | ".join(f"{k}:{_fmt_val(k,v)}" for k, v in fixed_global.items())

    # layout & fonts -------------------------------------------------
    rows, cols = _auto_grid(len(ds_runs)) if grid and len(ds_runs) > 1 else (len(ds_runs), 1)
    if figsize is None:
        figsize = (4 * cols, 3.5 * rows)
    bfs = _auto_fontsize(figsize, base_fontsize)
    rc = {
        "axes.titlesize": bfs * 1.4,
        "axes.labelsize": bfs * 1.3,
        "xtick.labelsize": bfs * 1.1,
        "ytick.labelsize": bfs * 1.2,
    }

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    key_to_label = {"mse": "MSE", "bias2": "Bias²", "var": "Variance"}

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

                # baseline lookup keyed by params that affect baseline
                baseline_lookup: dict[tuple, Any] = {}
                bagged_list = []
                for r in runs:
                    base_key = tuple(getattr(r, p) for p in _BASELINE_PARAMS)
                    if r.bagging_method is None:
                        baseline_lookup[base_key] = r
                    else:
                        bagged_list.append(r)

                xs_sorted = sorted({getattr(r, x_param) for r in bagged_list})
                ys_sorted = sorted({getattr(r, y_param) for r in bagged_list})
                if reverse_x:
                    xs_sorted = xs_sorted[::-1]
                if reverse_y:
                    ys_sorted = ys_sorted[::-1]
                xs_map = {v: i for i, v in enumerate(xs_sorted)}
                ys_map = {v: i for i, v in enumerate(ys_sorted)}
                data = np.full((len(ys_sorted), len(xs_sorted)), np.nan)

                for r in bagged_list:
                    base_key = tuple(getattr(r, p) for p in _BASELINE_PARAMS)
                    base_run = baseline_lookup.get(base_key)
                    if base_run is None:
                        continue  # no baseline → leave NaN
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
                    data[yi, xi] = diff

                vmax = np.nanmax(np.abs(data)) or 1.0
                if type == 'difference' or log:
                    im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower")
                else:
                    im = ax.imshow(data, cmap="Reds", vmin=0, vmax=vmax, origin="lower")

                # ticks --------------------------------------------------
                ax.set_xticks(range(len(xs_sorted)))
                ax.set_xticklabels([
                    _fmt_val(x_param, v) if (i % label_every == 0 or i in {0, len(xs_sorted)-1}) else ""
                    for i, v in enumerate(xs_sorted)
                ], rotation=45, ha="right")
                ax.set_yticks(range(len(ys_sorted)))
                ax.set_yticklabels([
                    _fmt_val(y_param, v) if (i % label_every == 0 or i in {0, len(ys_sorted)-1}) else ""
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
                fig.suptitle(fig_title, y=1.02, fontsize=bfs * 1.15)
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
                out = save_dir / f"{save_name}_{met_key}_{type}{logsavename}{inlogsavename}.{fmt}"
                fig.savefig(out, bbox_inches="tight")
                print(f"[SAVED] {out}")
            if show:
                plt.show()
            else:
                plt.close(fig)
