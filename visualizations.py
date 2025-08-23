# param_viz.py
# Visualize random samples of maps f: [0,1]^n_params -> R^d (now supports d ≥ 1, incl. 4D via color)

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional, Sequence

try:
    # noqa: F401 - needed to enable 3D projection in matplotlib
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore
except Exception:
    pass

import matplotlib.colors as mcolors


def _call_map_fn(map_fn: Callable, P: np.ndarray) -> np.ndarray:
    """Call map_fn either as map_fn(P) or map_fn(*cols)."""
    try:
        Y = map_fn(P)
    except TypeError:
        Y = map_fn(*[P[:, i] for i in range(P.shape[1])])
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return Y


def visualize_unit_cube_map(
    map_fn: Callable,
    n_params: int,
    samples: int = 50_000,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    marker_size: float = 2.0,
    alpha: float = 0.25,
    seed: int = 0,
    equal_axes: bool = True,
    # --- New options for 4D/HD projections ---
    axes: Sequence[int] = (0, 1, 2),   # which components to use as x,y,(z)
    color_dim: Optional[int] = None,   # which component to color by (e.g., 3 for 4D)
    cmap: str = "viridis",
    colorbar: bool = True,
) -> Tuple[plt.Figure, Optional[str]]:
    """
    Sample a map f: [0,1]^n_params -> R^d and plot it.

    Supports:
      d = 1  -> scatter along index
      d = 2  -> 2D scatter
      d = 3  -> 3D scatter
      d >= 4 -> pick 3 dims via `axes` and use `color_dim` for color
    Also works for d > 4 if you specify `axes` (length 2 or 3) and optionally `color_dim`.

    map_fn can be written in either style:
      1) map_fn(P): P is (N, n_params) -> returns (N, d)
      2) map_fn(x0, x1, ..., x_{n-1}): each is length N -> returns (N, d) or 1-D

    Returns (fig, save_path).
    """
    rng = np.random.default_rng(seed)
    P = rng.random((samples, n_params))
    Y = _call_map_fn(map_fn, P)
    N, d = Y.shape

    # Choose sensible defaults for projection if not provided
    axes = tuple(axes)
    if len(axes) not in (2, 3):
        raise ValueError("`axes` must have length 2 or 3 (for 2D/3D plotting).")

    # Auto-select color_dim if none and we have an extra dimension
    if color_dim is None and d >= len(axes) + 1:
        # Use the first component not in axes, otherwise fall back to last
        remaining = [i for i in range(d) if i not in axes]
        color_dim = remaining[0] if remaining else d - 1

    # Build plot title
    if title is None:
        base = f"Parametric visualization (n_params={n_params}, d={d})"
        if d >= 4:
            title = f"{base} — axes={axes}, color_dim={color_dim}"
        else:
            title = base

    # Extract coordinates and (optional) color
    coords = Y[:, list(axes)]
    C = None
    if color_dim is not None and 0 <= color_dim < d:
        C = Y[:, color_dim]

    # ----- Plotting -----
    if len(axes) == 2:
        fig = plt.figure(figsize=(6, 6))
        scatter_kwargs = dict(s=marker_size, alpha=alpha)
        if C is not None:
            scatter_kwargs.update(dict(c=C, cmap=cmap))
        plt.scatter(coords[:, 0], coords[:, 1], **scatter_kwargs)
        if equal_axes:
            plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel(f"dim {axes[0]}")
        plt.ylabel(f"dim {axes[1]}")
        plt.title(title)
        if colorbar and C is not None:
            mappable = plt.cm.ScalarMappable(
                norm=mcolors.Normalize(vmin=np.min(C), vmax=np.max(C)),
                cmap=cmap,
            )
            mappable.set_array([])
            plt.colorbar(mappable, label=f"dim {color_dim}")
        plt.tight_layout()

    else:  # 3D plot
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        scatter_kwargs = dict(s=marker_size, alpha=alpha)
        if C is not None:
            scatter_kwargs.update(dict(c=C, cmap=cmap))
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], **scatter_kwargs)
        ax.set_xlabel(f"dim {axes[0]}")
        ax.set_ylabel(f"dim {axes[1]}")
        ax.set_zlabel(f"dim {axes[2]}")
        ax.set_title(title)

        # Equal-ish aspect from data ranges
        if equal_axes and hasattr(ax, "set_box_aspect"):
            ranges = np.ptp(coords, axis=0)
            ranges[ranges == 0] = 1.0
            ax.set_box_aspect(ranges)

        # Colorbar
        if colorbar and C is not None:
            # Create a dummy mappable just for the colorbar
            mappable = plt.cm.ScalarMappable(
                norm=mcolors.Normalize(vmin=np.min(C), vmax=np.max(C)),
                cmap=cmap,
            )
            mappable.set_array([])
            cb = plt.colorbar(mappable, pad=0.1)
            cb.set_label(f"dim {color_dim}")

        plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=160)
    plt.show()
    return fig, save_path


# ------------------ Plotly (interactive) ------------------

import plotly.graph_objects as go

def visualize_unit_cube_map_plotly(
    map_fn: Callable,
    n_params: int,
    samples: int = 50_000,
    title: Optional[str] = None,
    seed: int = 0,
    save_html: Optional[str] = None,
    axes: Sequence[int] = (0, 1, 2),
    color_dim: Optional[int] = None,
    marker_size: int = 2,
    opacity: float = 0.35,
    colorscale: str = "Viridis",
):
    """
    Plotly version (interactive). Supports projecting higher-D outputs with color for a 4th dim.
    """
    rng = np.random.default_rng(seed)
    P = rng.random((samples, n_params))
    Y = _call_map_fn(map_fn, P)
    N, d = Y.shape

    axes = tuple(axes)
    if len(axes) not in (2, 3):
        raise ValueError("`axes` must have length 2 or 3 (for 2D/3D plotting).")

    if color_dim is None and d >= len(axes) + 1:
        remaining = [i for i in range(d) if i not in axes]
        color_dim = remaining[0] if remaining else d - 1

    if title is None:
        base = f"Parametric visualization (n_params={n_params}, d={d})"
        if d >= 4:
            title = f"{base} — axes={axes}, color_dim={color_dim}"
        else:
            title = base

    coords = Y[:, list(axes)]
    C = None if color_dim is None or not (0 <= color_dim < d) else Y[:, color_dim]

    if len(axes) == 2:
        fig = go.Figure(go.Scattergl(
            x=coords[:, 0], y=coords[:, 1],
            mode="markers",
            marker=dict(size=marker_size, opacity=opacity,
                        color=C, colorscale=colorscale, showscale=C is not None),
        ))
        fig.update_layout(
            title=title,
            xaxis_title=f"dim {axes[0]}",
            yaxis_title=f"dim {axes[1]}",
            yaxis_scaleanchor="x",
            yaxis_scaleratio=1,
        )
    else:
        fig = go.Figure(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(size=marker_size, opacity=opacity,
                        color=C, colorscale=colorscale, showscale=C is not None),
        ))
        fig.update_layout(
            title=title,
            scene=dict(aspectmode="data",
                       xaxis_title=f"dim {axes[0]}",
                       yaxis_title=f"dim {axes[1]}",
                       zaxis_title=f"dim {axes[2]}"),
        )

    if save_html:
        fig.write_html(save_html, include_plotlyjs="cdn", full_html=True)
    fig.show()
    return fig

# -----------------------
# Example maps you asked for
# -----------------------

# (cos(2πx0), sin(2πx0))  — unit circle
def map_circle(x0):
    return np.c_[np.cos(2 * np.pi * x0), np.sin(2 * np.pi * x0)]

# (x1^2 cos(2πx0), x2^2 sin(2πx0))  — filled unit disk
def map_disk(x0, x1, x2):
    return np.c_[
        (x1 ** 2) * np.cos(2 * np.pi * x0),
        (x2 ** 2) * np.sin(2 * np.pi * x0),
    ]

# (x1^2 cos(2πx0), x2^2 sin(2πx0), x1 + x2 + (x1 - x3)^2)  — 3D dome
def map_3d(x0, x1, x2, x3):
    X = (x1 ** 2) * np.cos(2 * np.pi * x0)
    Y = (x2 ** 2) * np.sin(2 * np.pi * x0)
    Z = x1 + x2 + (x1 - x3) ** 2
    return np.c_[X, Y, Z]

def map_3d2(x0, x1, x2, x3):
    X = (x1 ** 2) * np.cos(2 * np.pi * x0)
    Y = (x2 ** 2) * np.sin(2 * np.pi * x0)
    Z = x1 - 2*x2 + (x0 - x3) ** 2
    return np.c_[X, Y, Z]

def map_3d3(x0, x1, x2, x3):
    X = (x1 ** 2) * np.cos(2 * np.pi * x0)
    Y = (x2 ** 2) * np.sin(2 * np.pi * x0)
    Z = -x1 - 2*x2 + (x2-x3)**2
    return np.c_[X, Y, Z]

def map_3d4(x0, x1, x2, x3):
    X = (x1 ** 2) * np.cos(2 * np.pi * x0)
    Y = (x2 ** 2) * np.sin(2 * np.pi * x0)
    Z = x0**2 -x1**2 +x2**2-x3**2
    return np.c_[X, Y, Z]

def map_3d5(x0, x1, x2, x3):
    X = x1 - 2*x2 + (x0 - x3) ** 2
    Y = -x1 - 2*x2 + (x2-x3)**2
    Z = x0**2 -x1**2 +x2**2-x3**2
    return np.c_[X, Y, Z]

def map_3d6(x0, x1, x2, x3):
    X = x1 + x2 + (x1 - x3) ** 2
    Y = x1 - 2*x2 + (x0 - x3) ** 2
    Z = -x1 - 2*x2 + (x2-x3)**2
    return np.c_[X, Y, Z]

def map_weird1(x0, x1):
    X = x1*np.cos(2*np.pi*x0)
    Y = x1*np.sin(2*np.pi*x0)
    Z = x0*np.cos(2*np.pi*x1)
    return np.c_[X, Y, Z]

def map_weird2(x0, x1):
    X = x1*np.cos(2*np.pi*x0)
    Y = x1*np.sin(2*np.pi*x0)
    Z = x0*np.sin(2*np.pi*x1)
    return np.c_[X, Y, Z]

def map_4d(x0, x1, x2, x3):
    X = (x1 ** 2) * np.cos(2 * np.pi * x0)
    Y = (x2 ** 2) * np.sin(2 * np.pi * x0)
    Z = x1 + x2 + (x1 - x3) ** 2
    W = x0 ** 2 - x1 ** 2 + x2 ** 2 - x3 ** 2
    return np.c_[X, Y, Z, W]

def map_weird_full(x0, x1):
    X = x1*np.cos(2*np.pi*x0)
    Y = x1*np.sin(2*np.pi*x0)
    Z = x0*np.cos(2*np.pi*x1)
    W = x0*np.sin(2*np.pi*x1)
    return np.c_[X, Y, Z, W]

def map_weird_full2(x0, x1):
    X = x1*np.cos(2*np.pi*x0)
    Y = x1*np.sin(2*np.pi*x0)
    Z = x1*np.cos(2*np.pi*x0)
    W = x1*np.sin(2*np.pi*x0)
    return np.c_[X, Y, Z, W]

def map_weird_full3(x0, x1):
    X = x1*np.cos(2*np.pi*x0)
    Y = x1*np.sin(2*np.pi*x0)
    Z = x1*np.cos(2*np.pi*x0)
    W = x0*np.sin(2*np.pi*x1)
    return np.c_[X, Y, Z, W]

def helix(x0, x1):
    x0 = x0*10*np.pi
    x1 = x1*10*np.pi
    X = x0*np.cos(x1)
    Y = x0*np.sin(x1)
    Z = 0.5*x1
    W = 0*x1
    return np.c_[X, Y, Z, W]

def roll(x0, x1):
    x0 = x0*3*np.pi + 1.5*np.pi
    x1 = x1*21
    X = x0*np.cos(x0)
    Y = x1
    Z = x0*np.sin(x0)
    W = 0*x1
    return np.c_[X, Y, Z, W]

def affine3to5(x0, x1, x2):
    x0, x1, x2 = x0*4, x1*4, x2*4
    X = 1.2*x0-0.5*x1+3
    Y = 0.5*x0 + 0.9*x1 -1
    Z = -0.5*x0 -0.2*x1 +x2
    W = 0.4*x0 -0.9*x1 -  0.1*x2
    return np.c_[X, Y, Z, W]

def m9affine(x0, x1, x2, x3):
    x0, x1, x2, x3 = (x0-0.5)*5, (x1-0.5)*5, (x2-0.5)*5, (x3-0.5)*5
    X = x0
    Y = x1
    Z = x2
    W = x3
    return np.c_[X, Y, Z, W]

def moebius(x0, x1):
    x0 = x0*np.pi*2
    x1 = (x1-0.5)*2
    X = (1+0.5*x1*np.cos(5*x0))*np.cos(x0)
    Y = (1+0.5*x1*np.cos(5*x0))*np.sin(x0)
    Z = 0.5*x1*np.sin(5*x0)
    return np.c_[X, Y, Z]

def scurve(x0, x1):
    x0 = (x0-0.5)*1.5*np.pi*2
    x1 = x1*2
    X = np.sin(x0)
    Y = x1
    Z = np.sign(x0)*(np.cos(x0)-1)
    W = x0*0
    return np.c_[X, Y, Z, W]

def mn_nonlinear1(x0, x1, x2, x3):
    X = np.tan(x0*np.cos(x3))
    Y = np.arctan(x3*np.sin(x0))
    Z = np.tan(x0*np.cos(x3))
    W = np.arctan(x3*np.sin(x0))
    return np.c_[X, Y, Z, W]

def mn_nonlinear2(x0, x1, x2, x3):
    X = np.tan(x0*np.cos(x3))
    Y = np.arctan(x3*np.sin(x0))
    Z = np.tan(x1*np.cos(x2))
    W = np.arctan(x1*np.sin(x2))
    return np.c_[X, Y, Z, W]

def mn_nonlinear3(x0, x1, x2, x3):
    X = np.tan(x0*np.cos(x3))
    Y = np.arctan(x3*np.sin(x0))
    W = np.tan(x1*np.cos(x2))
    Z = np.arctan(x1*np.sin(x2))
    return np.c_[X, Y, Z, W]

def mn_nonlinear_full_1d(x0):
    X = np.tan(x0*np.cos(x0))
    Y = np.arctan(x0*np.sin(x0))
    Z = np.tan(x0*np.cos(x0))
    W = np.arctan(x0*np.sin(x0))
    return np.c_[X, Y, Z, W]

def mn_nonlinear_2d(x0, x1):
    X = np.tan(x0*np.cos(x1))
    Y = np.arctan(x0*np.sin(x1))
    Z = np.tan(x1*np.cos(x0))
    W = np.arctan(x1*np.sin(x0))
    return np.c_[X, Y, Z, W]

def mn_nonlinear_3d(x0, x1, x2):
    X = np.tan(x0*np.cos(x2))
    Y = np.tan(x1*np.cos(x1))
    Z = np.tan(x2*np.cos(x0))
    W = np.arctan(x2*np.sin(x0))
    return np.c_[X, Y, Z, W]

def mn_nonlinear_4d(x0, x1, x2, x3):
    X = np.tan(x0*np.cos(x3))
    Y = np.tan(x1*np.cos(x2))
    Z = np.tan(x2*np.cos(x1))
    W = np.tan(x3*np.cos(x0))
    return np.c_[X, Y, Z, W]

def mn_nonlinear_4d_2(x0, x1, x2, x3):
    X = np.tan(x0*np.cos(x3))
    Y = np.tan(x1*np.cos(x2))
    Z = np.arctan(x2*np.cos(x1))
    W = np.arctan(x3*np.cos(x0))
    return np.c_[X, Y, Z, W]



if __name__ == "__main__":

    visualize_unit_cube_map_plotly(mn_nonlinear_4d_2, n_params=4, samples=80000, axes=(0,1,2), color_dim=3,
                               title="mn_nonlinear_4d_2", seed=3, save_html="mn_nonlinear_4d_2.html")