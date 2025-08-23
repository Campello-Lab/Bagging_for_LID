import re
from collections import defaultdict
from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Iterable, Tuple, Union
##############################################################################################################################BAR PLOT###############################################################################################################################

__all__ = [
    "parse_experiment_name",
    "plot_bias_variance_bars",
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

    # Baseline *k* needs scaling so that its bars align with non‑baseline runs
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


# ---------------------------------------------------------------------------
# Main plotting routine
# ---------------------------------------------------------------------------

def _auto_grid(n_plots: int) -> Tuple[int, int]:
    """Return (n_rows, n_cols) so that rows ≥ cols and grid ≈ square."""
    if n_plots <= 0:
        raise ValueError("Number of sub‑plots must be positive")

    n_cols = max(1, int(math.floor(math.sqrt(n_plots))))
    n_rows = int(math.ceil(n_plots / n_cols))

    # Guarantee the spec: *more rows than columns* when not square
    while n_rows < n_cols:
        n_cols -= 1
        n_rows = int(math.ceil(n_plots / n_cols))
    return n_rows, n_cols


def plot_bias_variance_bars(
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
    label_every: int = 1,
    show: bool = False,
    save_name: str = "sr_bar_plot",
    save_dir: Union[str, Path] = "./plots",
    formats: Tuple[str, ...] = ("pdf",),
):
    """Stacked bars of Bias² + Variance vs. *sampling‑rate* at a fixed *k*.

    Parameters
    ----------
    label_every : int, default=1
        Show only every *nth* x‑tick label to reduce clutter.  E.g. set to 5 to
        display 0.02, 0.12, 0.22, … if your rates are spaced by 0.02.
    """
    data_by_dataset: dict[str, list[dict]] = defaultdict(list)
    for name, result in zip(name_list, results_list):
        try:
            parsed = parse_experiment_name(name, adjust_k=adjust_k)
        except ValueError:
            continue
        if parsed["k"] != fixed_k:
            continue
        if allowed_methods is not None:
            mtype = parsed["method_type"]
            if mtype not in allowed_methods:
                continue
            rates_allowed = allowed_methods[mtype]
            if isinstance(rates_allowed, list):
                if parsed["sampling_rate"] not in rates_allowed:
                    continue
            elif rates_allowed is False:
                continue
        rate_val = float(parsed["sampling_rate"] or 1.0)
        rate_lbl = parsed["sampling_rate"] or "1.0"
        for dataset, values in result.items():
            mse, b2, var = _extract_metrics(values, partial_key=partial_key)
            if log:
                b2, var, mse = np.log10([b2, var, mse])
            data_by_dataset[dataset].append(
                {
                    "rate_val": rate_val,
                    "rate_lbl": rate_lbl,
                    "bias2": b2,
                    "var": var,
                }
            )

    datasets = sorted(data_by_dataset)
    if not datasets:
        raise ValueError("No data matched – nothing to plot.")

    n_rows, n_cols = _auto_grid(len(datasets)) if grid and len(datasets) > 1 else (len(datasets), 1)
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)
    if base_fontsize is None:
        base_fontsize = max(6, 0.9 * min(figsize) + 2)

    rc = {
        "axes.titlesize": base_fontsize * 1.4,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize * 0.7,
        "ytick.labelsize": base_fontsize * 1.2,
        "legend.fontsize": base_fontsize * 1.2,
    }

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(rc):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=False)
        axes = np.array(axes).reshape(-1)
        for ax in axes[len(datasets):]:
            ax.axis("off")

        for ax, dataset in zip(axes, datasets):
            entries = sorted(data_by_dataset[dataset], key=lambda d: d["rate_val"])
            rate_labels = [f"{float(e['rate_lbl']):.2f}" if e['rate_lbl'] is not None else f"{float(1):.2f}" for e in entries]
            bias_vals = [e["bias2"] for e in entries]
            var_vals = [e["var"] for e in entries]
            x = np.arange(len(rate_labels))
            ax.bar(x, bias_vals, width=0.6, color=colors[0], label="Bias²")
            ax.bar(x, var_vals, width=0.6, bottom=bias_vals, color=colors[1], label="Variance")
            ax.set_xticks(x)
            # show only every *label_every* label
            display_labels = [lbl if idx % label_every == 0 else "" for idx, lbl in enumerate(rate_labels)]
            ax.set_xticklabels(display_labels, rotation=45, ha="right")
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