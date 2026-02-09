from __future__ import annotations
from typing import Any
import numpy as np

#This figures out how to arrange the per dataset subplots so that it's not too wide
def auto_grid(n: int) -> tuple[int, int]:
    cols = int(np.floor(np.sqrt(n))) or 1
    rows = int(np.ceil(n / cols))
    while rows < cols:
        cols -= 1
        rows = int(np.ceil(n / cols))
    return rows, cols

#Tries to figure out fontsize
def auto_fontsize(figsize: tuple[float, float], base: int | float | None) -> float:
    return float(base) if base is not None else max(6.0, 0.9 * min(figsize) + 2)

#This is just for different labeling of different varying params (where to cut the decimals)
def fmt_val(p: str, v: Any) -> str:
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