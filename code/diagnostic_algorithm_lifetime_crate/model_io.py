"""Model-evaluation helpers for reconstructed full-cell voltage curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

OCP3Fn = Callable[[pd.Series, float], Tuple[float, float, float]]


@dataclass
class VoltageModelCurve:
    rpt: int
    Q_grid: np.ndarray
    Vfit: np.ndarray
    dVdQ: Optional[np.ndarray] = None
    meta: Optional[dict] = None


def evaluate_ocp_vectorized(ocp_3: OCP3Fn, res: pd.Series, Q_eval: np.ndarray):
    Q_eval = np.asarray(Q_eval, float).reshape(-1)
    try:
        out = ocp_3(res, Q_eval)
        if isinstance(out, (tuple, list)) and len(out) == 3:
            v, va, vc = out
            v = np.asarray(v, float).reshape(-1)
            va = np.asarray(va, float).reshape(-1)
            vc = np.asarray(vc, float).reshape(-1)
            if v.size == Q_eval.size:
                return v, va, vc
    except Exception:
        pass

    Vfit = np.empty_like(Q_eval)
    Van = np.empty_like(Q_eval)
    Vca = np.empty_like(Q_eval)
    for i, q in enumerate(Q_eval):
        v, va, vc = ocp_3(res, float(q))
        Vfit[i] = float(v)
        Van[i] = float(va)
        Vca[i] = float(vc)
    return Vfit, Van, Vca


def params_to_series(
    Cn_Si: float,
    Cn_Gr: float,
    x100: float,
    Cp: float,
    y100: float,
    si_scale_a: float = 1.0,
    si_shift_b: float = 0.0,
) -> pd.Series:
    """Return the seven-parameter vector in the internal table schema.

    Internal aliases are kept for compatibility with saved outputs:
    x100 = x_n,100, y100 = x_p,100, si_scale_a = s_V, and
    si_shift_b = U_off.
    """
    return pd.Series([Cn_Si, Cn_Gr, x100, Cp, y100, si_scale_a, si_shift_b], dtype=float)
