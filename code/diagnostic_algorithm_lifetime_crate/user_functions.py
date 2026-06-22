"""Load OCP references and define the Si/Gr/NMC voltage model functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import interpolate
from scipy.signal import savgol_filter
from .ocp_alignment import align_ocp_to_common_p_axis
from scipy.interpolate import PchipInterpolator

def process_ocp_curve(df, x_col="sto", y_col="p",
                      window_length=31, polyorder=3,
                      enforce_monotone=True,
                      monotone_direction="decreasing"):
    x = np.asarray(df[x_col], float)
    y = np.asarray(df[y_col], float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_u, idx = np.unique(x, return_index=True)
    y_u = y[idx]

    if enforce_monotone:
        if monotone_direction == "decreasing":
            y_mono = np.minimum.accumulate(y_u)
        elif monotone_direction == "increasing":
            y_mono = np.maximum.accumulate(y_u)
        else:
            raise ValueError("monotone_direction must be 'decreasing' or 'increasing'")
    else:
        y_mono = y_u.copy()

    w = min(window_length, len(x_u) if len(x_u) % 2 == 1 else len(x_u) - 1)
    if w < polyorder + 3:
        w = polyorder + 3
    if w % 2 == 0:
        w += 1
    if w > len(x_u):
        w = len(x_u) if len(x_u) % 2 == 1 else len(x_u) - 1

    y_smooth = savgol_filter(y_mono, window_length=w, polyorder=polyorder, mode="interp")

    if enforce_monotone:
        if monotone_direction == "decreasing":
            y_smooth = np.minimum.accumulate(y_smooth)
        else:
            y_smooth = np.maximum.accumulate(y_smooth)

    dUdx = np.gradient(y_smooth, x_u)

    return pd.DataFrame({
        "sto": x_u,
        "p_raw": y_u,
        "p_monotone": y_mono,
        "p": y_smooth,
        "dUdx": dUdx,
    })

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "cell_ocp"

GR_CSV = DATA_DIR / "Graphite_OCP_Lithiation.csv"

SI_CSV = DATA_DIR / "Silicon_OCP_Lithiation_MSMR.csv"

NMC_CSV = DATA_DIR / "NMC622_OCP_Delithiation.csv"

for p in (GR_CSV, SI_CSV, NMC_CSV):
    if not p.exists():
        raise FileNotFoundError(f"Missing required OCP file: {p}")

Gr_OCP_raw = pd.read_csv(GR_CSV)
Si_OCP_raw = pd.read_csv(SI_CSV)
NMC_OCP_raw = pd.read_csv(NMC_CSV)

Gr_OCP = process_ocp_curve(
    Gr_OCP_raw,
    x_col="sto",
    y_col="p",
    window_length=1,
    polyorder=3,
    enforce_monotone=True,
    monotone_direction="decreasing",
)

NMC_OCP = process_ocp_curve(
    NMC_OCP_raw,
    x_col="sto",
    y_col="p",
    window_length=11, #11
    polyorder=3,
    enforce_monotone=True,
    monotone_direction="decreasing", 
)

p_common = np.linspace(0.0373, 1.0, 2000)

Gr_OCP, GR_ALIGN_META = align_ocp_to_common_p_axis(
    Gr_OCP,
    p_common=p_common,
    source_p_col="p",
    source_sto_col="sto",
    allow_extrapolation=False,
)

Si_OCP, SI_ALIGN_META = align_ocp_to_common_p_axis(
    Si_OCP_raw,
    p_common=p_common,
    source_p_col="p",
    source_sto_col="sto",
    allow_extrapolation=False,
)
# Gr_OCP = Gr_OCP_raw.copy()
# NMC_OCP = pd.read_csv(NMC_CSV)

for df in (Gr_OCP, Si_OCP, NMC_OCP, Gr_OCP_raw, Si_OCP_raw):
    df["sto"] = pd.to_numeric(df["sto"], errors="coerce")
    df["p"] = pd.to_numeric(df["p"], errors="coerce")

Gr_OCP = Gr_OCP.dropna(subset=["sto", "p"]).reset_index(drop=True)
Si_OCP = Si_OCP.dropna(subset=["sto", "p"]).reset_index(drop=True)
Gr_OCP_raw = Gr_OCP_raw.dropna(subset=["sto", "p"]).reset_index(drop=True)
Si_OCP_raw = Si_OCP_raw.dropna(subset=["sto", "p"]).reset_index(drop=True)
NMC_OCP = NMC_OCP.dropna(subset=["sto", "p"]).reset_index(drop=True)

_Up = interpolate.interp1d(
    NMC_OCP["sto"].to_numpy(),
    NMC_OCP["p"].to_numpy(),
    kind="linear",
    bounds_error=False,
    fill_value="extrapolate",
)


def Up(sto):
    sto = np.asarray(sto, dtype=float)
    lo = float(NMC_OCP["sto"].min())
    hi = float(NMC_OCP["sto"].max())
    return _Up(np.clip(sto, lo, hi))


def apply_si_model_a_to_aligned_p(p_aligned, si_scale_a: float = 1.0, si_shift_b: float = 0.0):
    """Apply the manuscript's Si-OCP deformation: U_si,new = s_V U_si,base + U_off."""
    p_aligned = np.asarray(p_aligned, dtype=float)
    return float(si_scale_a) * p_aligned + float(si_shift_b)


def build_effective_si_ocp(
    si_scale_a: float = 1.0,
    si_shift_b: float = 0.0,
    reconstruction_mode: str = "direct",
    si_voltage_high: float = 1.0,
    si_voltage_low: float = 0.0373,
    si_left_x: float = -0.08,
    si_right_x: float = 1.12,
    si_left_k: float = 8.0,
    si_right_k: float = 25.0, #25
    n_points_full: int = 2000,
    n_points_window: int = 1000,
):

    # Base aligned Si OCP
    x_si_raw = np.asarray(Si_OCP["sto"], float)
    p_si_aligned = np.asarray(Si_OCP["p"], float)

    order = np.argsort(x_si_raw)
    x_si_raw = x_si_raw[order]
    p_si_aligned = p_si_aligned[order]

    x_si_raw, idxu = np.unique(x_si_raw, return_index=True)
    p_si_aligned = p_si_aligned[idxu]

    p_si_target = apply_si_model_a_to_aligned_p(
        p_si_aligned,
        si_scale_a=si_scale_a,
        si_shift_b=si_shift_b,
    )

    if reconstruction_mode == "direct":
        return {
            "sto": x_si_raw.copy(),
            "p": p_si_target.copy(),
        }

    if reconstruction_mode != "reconstructed":
        raise ValueError(f"Unknown reconstruction_mode: {reconstruction_mode}")

    # reconstructed mode
    # exactly on [0,1], and only add tails outside.
    pchip_target = PchipInterpolator(x_si_raw, p_si_target, extrapolate=False)

    u0 = float(pchip_target(0.0))
    m0 = float(pchip_target.derivative(1)(0.0))

    u1 = float(pchip_target(1.0))
    m1 = float(pchip_target.derivative(1)(1.0))

    sto_ext = np.linspace(si_left_x, si_right_x, n_points_full)
    p_ext = np.empty_like(sto_ext)

    # left tail: x < 0
    mask_left = sto_ext < 0.0
    x_left = sto_ext[mask_left]
    p_ext[mask_left] = u0 + m0 * x_left - si_left_k * x_left**2

    # middle: 0 <= x <= 1
    mask_mid = (sto_ext >= 0.0) & (sto_ext <= 1.0)
    p_ext[mask_mid] = np.interp(sto_ext[mask_mid], x_si_raw, p_si_target)

    # right tail: x > 1
    mask_right = sto_ext > 1.0
    dx = sto_ext[mask_right] - 1.0
    p_ext[mask_right] = u1 + m1 * dx - si_right_k * dx**3

    p_ext = np.maximum(p_ext, 0.0)

    # enforce monotone decreasing
    if not np.all(np.diff(p_ext) <= 1e-10):
        p_proj = p_ext.copy()
        for i in range(1, len(p_proj)):
            if p_proj[i] > p_proj[i - 1]:
                p_proj[i] = p_proj[i - 1]
        p_ext = p_proj

    orderU = np.argsort(p_ext)
    p_sorted = p_ext[orderU]
    sto_sorted = sto_ext[orderU]

    p_sorted_unique, idxp = np.unique(p_sorted, return_index=True)
    sto_sorted_unique = sto_sorted[idxp]

    high_eff = min(si_voltage_high, float(np.max(p_sorted_unique)))
    low_eff = max(si_voltage_low, float(np.min(p_sorted_unique)))

    if low_eff >= high_eff:
        raise ValueError(
            f"Requested silicon voltage window [{si_voltage_high}, {si_voltage_low}] "
            f"does not overlap reconstructed Si OCP range "
            f"[{float(np.max(p_sorted_unique))}, {float(np.min(p_sorted_unique))}]."
        )

    sto_at_high = float(np.interp(high_eff, p_sorted_unique, sto_sorted_unique))
    sto_at_low = float(np.interp(low_eff, p_sorted_unique, sto_sorted_unique))

    x_lo = min(sto_at_high, sto_at_low)
    x_hi = max(sto_at_high, sto_at_low)

    sto_win = np.linspace(x_lo, x_hi, n_points_window)
    p_win = np.interp(sto_win, sto_ext, p_ext)
    sto_norm = (sto_win - x_lo) / max(x_hi - x_lo, 1e-12)

    sto_norm[0] = 0.0
    sto_norm[-1] = 1.0
    p_win[0] = high_eff
    p_win[-1] = low_eff

    return {
        "sto": sto_norm,
        "p": p_win,
        "sto_raw_window": sto_win,
        "p_full": p_ext,
        "sto_full": sto_ext,
        "x_lo": x_lo,
        "x_hi": x_hi,
        "p_high_eff": high_eff,
        "p_low_eff": low_eff,
    }


def Un(
    sto,
    alpha,
    si_scale_a: float = 1.0,
    si_shift_b: float = 0.0,
    use_reconstructed_si_ocp: bool = False,
):
    sto = np.asarray(sto, dtype=float)

    p_common = np.asarray(Gr_OCP["p"], float)
    x_gr = np.asarray(Gr_OCP["sto"], float)

    if use_reconstructed_si_ocp:
        si_ocp_eff = build_effective_si_ocp(
            si_scale_a=si_scale_a,
            si_shift_b=si_shift_b,
            reconstruction_mode="reconstructed",
        )
        x_si = np.asarray(si_ocp_eff["sto"], float)
        p_si_new = np.asarray(si_ocp_eff["p"], float)
    else:
        si_ocp_eff = build_effective_si_ocp(
            si_scale_a=si_scale_a,
            si_shift_b=si_shift_b,
            reconstruction_mode="direct",
        )
        x_si = np.asarray(si_ocp_eff["sto"], float)
        p_si_new = np.asarray(si_ocp_eff["p"], float)

    order_si = np.argsort(p_si_new)
    p_si_sorted = p_si_new[order_si]
    x_si_sorted = x_si[order_si]

    p_si_sorted, idx_si = np.unique(p_si_sorted, return_index=True)
    x_si_sorted = x_si_sorted[idx_si]

    x_si_on_common = np.interp(
        p_common,
        p_si_sorted,
        x_si_sorted,
        left=float(x_si_sorted[0]),
        right=float(x_si_sorted[-1]),
    )

    x_n = float(alpha) * x_si_on_common + (1.0 - float(alpha)) * x_gr

    order_n = np.argsort(x_n)
    x_n_sorted = x_n[order_n]
    p_n_sorted = p_common[order_n]

    x_n_sorted, idx_n = np.unique(x_n_sorted, return_index=True)
    p_n_sorted = p_n_sorted[idx_n]

    clipped = np.clip(sto, float(x_n_sorted[0]), float(x_n_sorted[-1]))
    return np.interp(clipped, x_n_sorted, p_n_sorted)


def ocp_3(res: pd.Series, Q, use_reconstructed_si_ocp: bool = False):
    r = np.asarray(res, dtype=float).reshape(-1)
    S = r[0] + r[1]
    x100 = r[2]
    Cp = r[3]
    y100 = r[4]
    si_scale_a = float(r[5]) if r.size >= 6 else 1.0
    si_shift_b = float(r[6]) if r.size >= 7 else 0.0
    alpha = r[0] / max(S, 1e-12)
    Q = np.asarray(Q, dtype=float)

    sto_p = np.clip(y100 + Q / Cp, 0.0, 1.0)
    sto_n = np.clip(x100 - Q / S, 0.0, 1.0)

    ocp_cathode = Up(sto_p)
    ocp_anode = Un(
        sto_n,
        alpha,
        si_scale_a=si_scale_a,
        si_shift_b=si_shift_b,
        use_reconstructed_si_ocp=use_reconstructed_si_ocp,
    )
    ocp = ocp_cathode - ocp_anode
    return ocp, ocp_anode, ocp_cathode
