from __future__ import annotations
from plotly.colors import get_colorscale

def _parse_rgba(col: str):
    col = str(col).strip()
    if col.startswith("#"):
        h = col[1:]
        if len(h) == 3:
            r, g, b = [int(ch*2, 16) for ch in h]
        else:
            r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return (r, g, b, 1.0)
    if col.lower().startswith("rgba"):
        a = col[col.find("(")+1:col.find(")")].split(",")
        return (int(float(a[0])), int(float(a[1])), int(float(a[2])), float(a[3]))
    if col.lower().startswith("rgb"):
        a = col[col.find("(")+1:col.find(")")].split(",")
        return (int(float(a[0])), int(float(a[1])), int(float(a[2])), 1.0)
    return _parse_rgba(get_colorscale([(0.0, col), (1.0, col)])[-1][1])

def _lerp(a, b, t): return a + (b - a) * t

def _color_at(stops, p):
    p = max(0.0, min(1.0, float(p)))
    for i in range(len(stops)-1):
        p0, c0 = stops[i]
        p1, c1 = stops[i+1]
        if p0 <= p <= p1:
            t = 0.0 if p1 == p0 else (p - p0) / (p1 - p0)
            r0,g0,b0,a0 = _parse_rgba(c0); r1,g1,b1,a1 = _parse_rgba(c1)
            r = int(round(_lerp(r0, r1, t))); g = int(round(_lerp(g0, g1, t)))
            b = int(round(_lerp(b0, b1, t))); a = _lerp(a0, a1, t)
            return f"rgba({r},{g},{b},{a:.3f})"
    r,g,b,a = _parse_rgba(stops[-1][1])
    return f"rgba({r},{g},{b},{a:.3f})"

def truncate_and_stretch(cs_like="Reds", cut_top=0.20):
    base = get_colorscale(cs_like) if isinstance(cs_like, str) else list(cs_like)
    base = sorted([(float(p), str(c)) for p,c in base], key=lambda x: x[0])
    alpha = max(1e-6, 1.0 - float(cut_top))
    kept = [(p/alpha, c) for (p,c) in base if p <= alpha]
    end_col = _color_at(base, alpha)
    if not kept or kept[-1][0] < 1.0 - 1e-9:
        kept.append((1.0, end_col))
    else:
        kept[-1] = (1.0, end_col)
    start_col = _color_at(base, 0.0)
    kept[0] = (0.0, "rgba(255,255,255,1.0)")
    return kept