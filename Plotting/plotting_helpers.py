from __future__ import annotations
import importlib.metadata as _imeta
from typing import Any
import numpy as np

#This figures out how to arrange the per dataset subplots so that it's not too wide
def _auto_grid(n: int) -> tuple[int, int]:
    cols = int(np.floor(np.sqrt(n))) or 1
    rows = int(np.ceil(n / cols))
    while rows < cols:
        cols -= 1
        rows = int(np.ceil(n / cols))
    return rows, cols

#Tries to figure out fontsize
def _auto_fontsize(figsize: tuple[float, float], base: int | float | None) -> float:
    return float(base) if base is not None else max(6.0, 0.9 * min(figsize) + 2)

#This is just for different labeling of different varying params (where to cut the decimals)
def _fmt_val(p: str, v: Any) -> str:
    if v is None:
        return "None"
    if p in {"sr", "t"}:
        return f"{float(v):.3f}"
    if p in {"n", "k", "Nbag", "lid", "dim"}:
        return str(int(v))
    return str(v)

# is num a float or not
def isfloat(num):
    if num is not None:
        try:
            float(num)
            return True
        except ValueError:
            return False
    else:
        return False

#hThese handle kaleido for saving figures
def _write_figure(fig, path, fmt, **size_kw):
    img_bytes = fig.to_image(format=fmt, engine="kaleido", **size_kw)
    if not img_bytes:
        raise RuntimeError("Kaleido returned zero bytes – export aborted.")
    with open(path, "wb") as fh:
        fh.write(img_bytes)

def _check_versions():
    pv = tuple(map(int, _imeta.version("plotly").split(".")[:2]))
    kv = tuple(map(int, _imeta.version("kaleido").split(".")[:2]))
    if pv >= (6, 0) and kv < (0, 2):          # Plotly ≥ 6 wants Kaleido ≥ 0.2
        raise RuntimeError(
            f"Plotly {pv} requires Kaleido ≥ 0.2.*, "
            f"but Kaleido {kv} is installed. "
            "Run  pip install -U 'kaleido>=0.2.1,<1'."
        )