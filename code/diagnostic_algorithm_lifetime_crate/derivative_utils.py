"""Numerical utilities for smoothing, differentiating, and resampling traces."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def unique_sorted_xy(x, y, eps: float = 1e-12):
    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.array([]), np.array([])
    xs = x[m]
    ys = y[m]
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    keep = np.r_[True, np.diff(xs) > eps]
    return xs[keep], ys[keep]


def _odd_window(n: int, requested: int, polyorder: int) -> int:
    if n <= polyorder + 2:
        return 0
    w = min(int(requested), int(n if n % 2 == 1 else n - 1))
    if w % 2 == 0:
        w -= 1
    if w <= polyorder:
        w = polyorder + 3 if (polyorder + 3) % 2 == 1 else polyorder + 4
    return w if w < n else (n - 1 if (n - 1) % 2 == 1 else n - 2)


def smooth_then_grad(x, y, window_length: int = 801, polyorder: int = 3):
    xs, ys = unique_sorted_xy(x, y)
    if xs.size < 5:
        return xs, ys, np.full_like(xs, np.nan)
    w = _odd_window(xs.size, window_length, polyorder)
    if w > 0:
        ys_s = savgol_filter(ys, window_length=w, polyorder=polyorder, mode="interp")
    else:
        ys_s = ys.copy()
    dydx = np.gradient(ys_s, xs)
    return xs, ys_s, dydx


def resample_clamped(x, y, xq):
    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    xq = np.asarray(xq, float).reshape(-1)
    xs, ys = unique_sorted_xy(x, y)
    if xs.size < 2:
        return np.full_like(xq, np.nan)
    return np.interp(xq, xs, ys)
