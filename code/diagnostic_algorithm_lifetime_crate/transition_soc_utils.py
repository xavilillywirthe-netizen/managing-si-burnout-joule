"""Compute silicon current share and silicon-dominant transition SoC."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
from pathlib import Path

def save_transition_soc_plot(
    result_dict,
    save_path,
    title=None,
    threshold=0.5,
):
    """
    Save Si current share vs SoC plot with transition SoC marked.

    Parameters
    ----------
    result_dict : dict
        Output from compute_si_current_share_and_transition_7p(...)
    save_path : str or Path
        Output figure path
    title : str or None
        Plot title
    threshold : float
        Threshold for transition detection, default 0.5
    """
    details_df = result_dict["details_df"].copy()
    transition_soc = result_dict.get("transition_soc", np.nan)
    transition_reason = result_dict.get("transition_reason", "")

    plot_df = details_df[["SoC", "si_share"]].dropna().sort_values("SoC")

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(plot_df["SoC"], plot_df["si_share"], label="Si current share")
    plt.axhline(threshold, linestyle="--", linewidth=1, label=f"threshold = {threshold:.2f}")

    if np.isfinite(transition_soc):
        plt.axvline(
            transition_soc,
            linestyle="--",
            linewidth=1.5,
            label=f"transition SoC = {transition_soc:.4f}",
        )

    plt.xlabel("SoC")
    plt.ylabel("Si current share")
    if title is None:
        title = f"Si current share vs SoC\nreason: {transition_reason}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def smooth_ocp_curve(
    df: pd.DataFrame,
    x_col: str = "sto",
    y_col: str = "p",
    window_length: int = 31,
    polyorder: int = 3,
    enforce_monotone: bool = True,
) -> pd.DataFrame:
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
        y_mono = np.minimum.accumulate(y_u)
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
        y_smooth = np.minimum.accumulate(y_smooth)

    dUdx_smooth = np.gradient(y_smooth, x_u)

    return pd.DataFrame(
        {
            "sto": x_u,
            "p": y_u,
            "p_monotone": y_mono,
            "p_smooth": y_smooth,
            "dUdx_smooth": dUdx_smooth,
        }
    )


def detect_transition_soc_from_share(
    soc,
    si_share,
    threshold=0.5,
    persistence_window=0.10,
    tol=0.01,
    min_points_after=3,
):
    import numpy as np

    def _portion_above_threshold_before_transition(soc_arr, share_arr, transition_soc, threshold):
        """
        Compute the fraction of the SoC interval [soc_min, transition_soc]
        where si_share > threshold.

        Uses linear interpolation and inserts exact threshold-crossing points
        so the length fraction is computed cleanly.
        """
        soc_arr = np.asarray(soc_arr, dtype=float)
        share_arr = np.asarray(share_arr, dtype=float)

        m = np.isfinite(soc_arr) & np.isfinite(share_arr)
        soc_arr = soc_arr[m]
        share_arr = share_arr[m]

        if len(soc_arr) < 2 or not np.isfinite(transition_soc):
            return np.nan

        order = np.argsort(soc_arr)
        soc_arr = soc_arr[order]
        share_arr = share_arr[order]

        soc_min = soc_arr.min()
        soc_max = transition_soc

        if soc_max <= soc_min:
            return 0.0

        keep = (soc_arr >= soc_min) & (soc_arr <= soc_max)
        xs = soc_arr[keep]
        ys = share_arr[keep]

        if len(xs) == 0 or xs[-1] < soc_max:
            y_end = np.interp(soc_max, soc_arr, share_arr)
            xs = np.append(xs, soc_max)
            ys = np.append(ys, y_end)

        if len(xs) == 0 or xs[0] > soc_min:
            y_start = np.interp(soc_min, soc_arr, share_arr)
            xs = np.insert(xs, 0, soc_min)
            ys = np.insert(ys, 0, y_start)

        new_x = [xs[0]]
        new_y = [ys[0]]

        for i in range(len(xs) - 1):
            x1, x2 = xs[i], xs[i + 1]
            y1, y2 = ys[i], ys[i + 1]

            crosses = (y1 - threshold) * (y2 - threshold) < 0
            if crosses and y2 != y1:
                x_cross = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
                new_x.append(x_cross)
                new_y.append(threshold)

            new_x.append(x2)
            new_y.append(y2)

        xs = np.asarray(new_x, dtype=float)
        ys = np.asarray(new_y, dtype=float)

        total_len = soc_max - soc_min
        if total_len <= 0:
            return 0.0

        above_len = 0.0
        for i in range(len(xs) - 1):
            x1, x2 = xs[i], xs[i + 1]
            y_mid = 0.5 * (ys[i] + ys[i + 1])
            if y_mid > threshold:
                above_len += (x2 - x1)

        return above_len / total_len

    soc = np.asarray(soc, dtype=float)
    si_share = np.asarray(si_share, dtype=float)

    m = np.isfinite(soc) & np.isfinite(si_share)
    soc = soc[m]
    si_share = si_share[m]

    if len(soc) < 3:
        return {
            "transition_soc": np.nan,
            "transition_idx_left": None,
            "candidate_crossings": [],
            "valid_crossings": [],
            "reason": "not_enough_points",
        }

    order = np.argsort(soc)
    soc = soc[order]
    si_share = si_share[order]

    candidate_crossings = []
    valid_crossings = []

    for i in range(len(soc) - 1):
        s1, s2 = soc[i], soc[i + 1]
        y1, y2 = si_share[i], si_share[i + 1]

        # downward crossing: > threshold to <= threshold
        if (y1 > threshold) and (y2 <= threshold):
            if y2 == y1:
                soc_cross = s1
            else:
                soc_cross = s1 + (threshold - y1) * (s2 - s1) / (y2 - y1)

            cand = {
                "idx_left": int(i),
                "soc_cross": float(soc_cross),
                "share_left": float(y1),
                "share_right": float(y2),
            }

            # Rule 1: persistence after crossing
            soc_end = soc_cross + persistence_window
            future_mask = (soc >= soc_cross) & (soc <= soc_end)

            if np.sum(future_mask) < min_points_after:
                future_mask = soc >= soc_cross

            future_soc = soc[future_mask]
            future_share = si_share[future_mask]

            if len(future_share) < min_points_after:
                cand["valid"] = False
                cand["reject_reason"] = "insufficient_future_points"
                cand["future_soc_end_checked"] = float(np.nanmax(future_soc)) if len(future_soc) else np.nan
                cand["future_share_max"] = float(np.nanmax(future_share)) if len(future_share) else np.nan
                cand["pre_transition_above_threshold_portion"] = np.nan
                candidate_crossings.append(cand.copy())
                continue

            if not np.all(future_share <= threshold + tol):
                cand["valid"] = False
                cand["reject_reason"] = "share_rebounds_above_threshold"
                cand["future_soc_end_checked"] = float(np.nanmax(future_soc))
                cand["future_share_max"] = float(np.nanmax(future_share))
                cand["pre_transition_above_threshold_portion"] = np.nan
                candidate_crossings.append(cand.copy())
                continue

            # Rule 2: before crossing, >threshold must occupy more than half the interval
            portion_above_before = _portion_above_threshold_before_transition(
                soc, si_share, soc_cross, threshold
            )

            cand["pre_transition_above_threshold_portion"] = float(portion_above_before)

            if not np.isfinite(portion_above_before) or portion_above_before <= 0.5:
                cand["valid"] = False
                cand["reject_reason"] = "pre_transition_above_threshold_portion_not_more_than_half"
                cand["future_soc_end_checked"] = float(np.nanmax(future_soc))
                cand["future_share_max"] = float(np.nanmax(future_share))
                candidate_crossings.append(cand.copy())
                continue

            # if both rules pass
            cand["valid"] = True
            cand["reject_reason"] = ""
            cand["future_soc_end_checked"] = float(np.nanmax(future_soc))
            cand["future_share_max"] = float(np.nanmax(future_share))

            candidate_crossings.append(cand.copy())
            valid_crossings.append(cand.copy())

    # normal case: choose the first crossing that satisfies BOTH rules
    if len(valid_crossings) > 0:
        selected = valid_crossings[0]
        return {
            "transition_soc": float(selected["soc_cross"]),
            "transition_idx_left": int(selected["idx_left"]),
            "candidate_crossings": candidate_crossings,
            "valid_crossings": valid_crossings,
            "reason": "success",
            "pre_transition_above_threshold_portion": float(selected["pre_transition_above_threshold_portion"]),
        }

    # fallback case 1: all below threshold -> transition at 0
    if np.all(si_share <= threshold + tol):
        return {
            "transition_soc": 0.0,
            "transition_idx_left": None,
            "candidate_crossings": candidate_crossings,
            "valid_crossings": valid_crossings,
            "reason": "no_crossing_all_below_threshold_set_to_zero",
            "pre_transition_above_threshold_portion": 0.0,
        }

    # fallback case 2:
    # if there is some above-threshold region but no valid transition,
    # decide between 0 and 1 based on how much of the SoC interval is above threshold
    total_soc_span = soc.max() - soc.min()
    if np.any(si_share > threshold + tol) and total_soc_span > 0:
        above_len_total = 0.0

        for i in range(len(soc) - 1):
            x1, x2 = soc[i], soc[i + 1]
            y1, y2 = si_share[i], si_share[i + 1]

            if (y1 > threshold) and (y2 > threshold):
                above_len_total += (x2 - x1)

            elif (y1 - threshold) * (y2 - threshold) < 0 and y2 != y1:
                x_cross = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
                if y1 > threshold:
                    above_len_total += (x_cross - x1)
                else:
                    above_len_total += (x2 - x_cross)

        portion_above_total = above_len_total / total_soc_span

        if portion_above_total > 0.5:
            return {
                "transition_soc": 1.0,
                "transition_idx_left": None,
                "candidate_crossings": candidate_crossings,
                "valid_crossings": valid_crossings,
                "reason": "no_valid_transition_but_above_threshold_portion_more_than_half_set_to_one",
                "pre_transition_above_threshold_portion": float(portion_above_total),
            }
        else:
            return {
                "transition_soc": 0.0,
                "transition_idx_left": None,
                "candidate_crossings": candidate_crossings,
                "valid_crossings": valid_crossings,
                "reason": "no_valid_transition_and_above_threshold_portion_not_more_than_half_set_to_zero",
                "pre_transition_above_threshold_portion": float(portion_above_total),
            }

    return {
        "transition_soc": np.nan,
        "transition_idx_left": None,
        "candidate_crossings": candidate_crossings,
        "valid_crossings": valid_crossings,
        "reason": "unclassified_case",
        "pre_transition_above_threshold_portion": np.nan,
    }

def compute_si_current_share_and_transition_7p(
    params: Iterable[float],
    *,
    Qdata_expand: Iterable[float],
    ocp_3,
    smooth_then_grad,
    build_effective_si_ocp,
    Gr_OCP_smooth: pd.DataFrame,
    SI_RECONSTRUCTION_MODE: str = "reconstructed",
    CATHODE_P_MIN: float = 3.53,
    CATHODE_P_MAX: float = 4.2900,
    ANODE_P_MIN: float = 0.0375,
    ANODE_P_MAX: float = 0.999,
    win: int = 31,
    poly: int = 3,
    threshold: float = 0.5,
    persistence_window: float = 0.10,
    tol: float = 0.01,
) -> Dict[str, Any]:
    """Compute silicon current share and transition SoC from the seven-parameter model.

    The input vector uses the repository's saved-output schema:
    [Cn_Si, Cn_Gr, x100, Cp, y100, si_scale_a, si_shift_b], where
    x100 = x_n,100, y100 = x_p,100, si_scale_a = s_V, and
    si_shift_b = U_off in the manuscript notation.
    """
    Cn_Si, Cn_Gr, x100, Cp, y100, si_scale_a, si_shift_b = map(float, params)
    Qdata_expand = np.asarray(Qdata_expand, dtype=float)

    def _unique_sorted_xy(x, y):
        x = np.asarray(x, float).reshape(-1)
        y = np.asarray(y, float).reshape(-1)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size == 0:
            return x, y
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        x_u, idx = np.unique(x, return_index=True)
        y_u = y[idx]
        return x_u, y_u

    def inv_or_nan(arr, eps=1e-12):
        arr = np.asarray(arr, float)
        out = np.full_like(arr, np.nan, dtype=float)
        m = np.isfinite(arr) & (np.abs(arr) > eps)
        out[m] = 1.0 / arr[m]
        return out

    def apply_nan_mask(arr, mask):
        arr = np.asarray(arr, float).copy()
        arr[~mask] = np.nan
        return arr

    def build_si_inverse_map(si_scale_a, si_shift_b, reconstruction_mode="direct"):
        si_ocp_eff = build_effective_si_ocp(
            si_scale_a=si_scale_a,
            si_shift_b=si_shift_b,
            reconstruction_mode=reconstruction_mode,
        )
        x_si = np.asarray(si_ocp_eff["sto"], float)
        p_si = np.asarray(si_ocp_eff["p"], float)
        order = np.argsort(p_si)
        p_si = p_si[order]
        x_si = x_si[order]
        p_si, idx = np.unique(p_si, return_index=True)
        x_si = x_si[idx]
        f_si_sto = interp1d(p_si, x_si, bounds_error=False, fill_value=np.nan)
        return si_ocp_eff, f_si_sto

    def build_gr_inverse_map():
        x_gr = np.asarray(Gr_OCP_smooth["sto"], float)
        p_gr = np.asarray(Gr_OCP_smooth["p_smooth"], float)
        order = np.argsort(p_gr)
        p_gr = p_gr[order]
        x_gr = x_gr[order]
        p_gr, idx = np.unique(p_gr, return_index=True)
        x_gr = x_gr[idx]
        f_gr_sto = interp1d(p_gr, x_gr, bounds_error=False, fill_value=np.nan)
        return f_gr_sto

    f_gr_sto = build_gr_inverse_map()

    Vfit_exp = np.array(
        [ocp_3(params, Q, use_reconstructed_si_ocp=(SI_RECONSTRUCTION_MODE == "reconstructed"))[0] for Q in Qdata_expand],
        dtype=float,
    )
    Vfit_Anode_exp = np.array(
        [ocp_3(params, Q, use_reconstructed_si_ocp=(SI_RECONSTRUCTION_MODE == "reconstructed"))[1] for Q in Qdata_expand],
        dtype=float,
    )
    Vfit_Cathode_exp = np.array(
        [ocp_3(params, Q, use_reconstructed_si_ocp=(SI_RECONSTRUCTION_MODE == "reconstructed"))[2] for Q in Qdata_expand],
        dtype=float,
    )

    cathode_valid = np.isfinite(Vfit_Cathode_exp) & (Vfit_Cathode_exp >= CATHODE_P_MIN) & (Vfit_Cathode_exp <= CATHODE_P_MAX)
    anode_valid = np.isfinite(Vfit_Anode_exp) & (Vfit_Anode_exp >= ANODE_P_MIN) & (Vfit_Anode_exp <= ANODE_P_MAX)
    full_valid = cathode_valid & anode_valid & np.isfinite(Vfit_exp)

    Vfit_Cathode_exp = apply_nan_mask(Vfit_Cathode_exp, cathode_valid)
    Vfit_Anode_exp = apply_nan_mask(Vfit_Anode_exp, anode_valid)
    Vfit_exp = apply_nan_mask(Vfit_exp, full_valid)

    q_full, _, dVdQ_fit_full_native = smooth_then_grad(Qdata_expand, Vfit_exp, window_length=win, polyorder=poly)
    q_an, Vfit_Anode_smooth, dVdQ_fit_Anode_native = smooth_then_grad(Qdata_expand, Vfit_Anode_exp, window_length=win, polyorder=poly)
    q_ca, _, dVdQ_fit_Cathode_native = smooth_then_grad(Qdata_expand, Vfit_Cathode_exp, window_length=win, polyorder=poly)

    dVdQ_fit_exp = np.interp(Qdata_expand, q_full, dVdQ_fit_full_native, left=np.nan, right=np.nan)
    Vfit_Anode_aligned = np.interp(Qdata_expand, q_an, Vfit_Anode_smooth, left=np.nan, right=np.nan)
    dVdQ_fit_Anode_exp = np.interp(Qdata_expand, q_an, dVdQ_fit_Anode_native, left=np.nan, right=np.nan)
    dVdQ_fit_Cathode_exp = np.interp(Qdata_expand, q_ca, dVdQ_fit_Cathode_native, left=np.nan, right=np.nan)

    dVdQ_fit_exp = apply_nan_mask(dVdQ_fit_exp, full_valid)
    Vfit_Anode_aligned = apply_nan_mask(Vfit_Anode_aligned, anode_valid)
    dVdQ_fit_Anode_exp = apply_nan_mask(dVdQ_fit_Anode_exp, anode_valid)
    dVdQ_fit_Cathode_exp = apply_nan_mask(dVdQ_fit_Cathode_exp, cathode_valid)

    si_ocp_eff, f_si_sto = build_si_inverse_map(
        si_scale_a=si_scale_a,
        si_shift_b=si_shift_b,
        reconstruction_mode=SI_RECONSTRUCTION_MODE,
    )

    x_si_on_common = np.asarray(f_si_sto(Vfit_Anode_aligned), float)
    x_gr_on_common = np.asarray(f_gr_sto(Vfit_Anode_aligned), float)
    x_si_on_common = apply_nan_mask(x_si_on_common, anode_valid)
    x_gr_on_common = apply_nan_mask(x_gr_on_common, anode_valid)

    Q_si = x_si_on_common * Cn_Si
    Q_gr = x_gr_on_common * Cn_Gr

    q_si_native, _, dVdQ_fit_si_native = smooth_then_grad(Q_si, Vfit_exp, window_length=win, polyorder=poly)
    q_gr_native, _, dVdQ_fit_gr_native = smooth_then_grad(Q_gr, Vfit_exp, window_length=win, polyorder=poly)

    dVdQ_fit_si_exp = np.full_like(Qdata_expand, np.nan, dtype=float)
    dVdQ_fit_gr_exp = np.full_like(Qdata_expand, np.nan, dtype=float)

    mask_si = np.isfinite(Q_si) & full_valid
    if len(q_si_native) >= 5 and np.isfinite(dVdQ_fit_si_native).any():
        qsi_u, dsi_u = _unique_sorted_xy(q_si_native, dVdQ_fit_si_native)
        if len(qsi_u) >= 5:
            dVdQ_fit_si_exp[mask_si] = np.interp(Q_si[mask_si], qsi_u, dsi_u, left=np.nan, right=np.nan)

    mask_gr = np.isfinite(Q_gr) & full_valid
    if len(q_gr_native) >= 5 and np.isfinite(dVdQ_fit_gr_native).any():
        qgr_u, dgr_u = _unique_sorted_xy(q_gr_native, dVdQ_fit_gr_native)
        if len(qgr_u) >= 5:
            dVdQ_fit_gr_exp[mask_gr] = np.interp(Q_gr[mask_gr], qgr_u, dgr_u, left=np.nan, right=np.nan)

    dVdQ_fit_si_exp = apply_nan_mask(dVdQ_fit_si_exp, full_valid)
    dVdQ_fit_gr_exp = apply_nan_mask(dVdQ_fit_gr_exp, full_valid)

    dqdv_fit = inv_or_nan(dVdQ_fit_exp)
    dqdv_fit_si = inv_or_nan(dVdQ_fit_si_exp)
    dqdv_fit_gr = inv_or_nan(dVdQ_fit_gr_exp)
    dqdv_fit = apply_nan_mask(dqdv_fit, full_valid)
    dqdv_fit_si = apply_nan_mask(dqdv_fit_si, full_valid)
    dqdv_fit_gr = apply_nan_mask(dqdv_fit_gr, full_valid)

    si_share = np.full_like(Qdata_expand, np.nan, dtype=float)
    m_share = np.isfinite(dVdQ_fit_exp) & np.isfinite(dVdQ_fit_si_exp) & (np.abs(dVdQ_fit_si_exp) > 1e-12)
    si_share[m_share] = -dVdQ_fit_exp[m_share] / dVdQ_fit_si_exp[m_share]

    gr_share = np.full_like(Qdata_expand, np.nan, dtype=float)
    m_gr_share = np.isfinite(dVdQ_fit_exp) & np.isfinite(dVdQ_fit_gr_exp) & (np.abs(dVdQ_fit_gr_exp) > 1e-12)
    gr_share[m_gr_share] = -dVdQ_fit_exp[m_gr_share] / dVdQ_fit_gr_exp[m_gr_share]

    si_share = apply_nan_mask(si_share, full_valid)
    gr_share = apply_nan_mask(gr_share, full_valid)

    mask_clean = np.isfinite(Vfit_exp)
    Vfit_clean = Vfit_exp[mask_clean]
    Qin_clean = np.asarray(Qdata_expand)[mask_clean]
    if len(Vfit_clean) == 0:
        return {"transition_soc": np.nan, "transition_reason": "no_finite_voltage_points"}

    idx_4_2 = np.argmin(np.abs(Vfit_clean - 4.2))
    idx_2_9 = np.argmin(np.abs(Vfit_clean - 2.9))
    Q_ref_4_2 = float(Qin_clean[idx_4_2])
    Q_ref_2_9 = float(Qin_clean[idx_2_9])

    Q_shifted = -(np.asarray(Qdata_expand) - Q_ref_2_9)
    SoCs = (np.asarray(Qdata_expand) - Q_ref_2_9) / (-Q_ref_2_9 + Q_ref_4_2)

    transition_result = detect_transition_soc_from_share(
        soc=SoCs,
        si_share=si_share,
        threshold=threshold,
        persistence_window=persistence_window,
        tol=tol,
    )

    details_df = pd.DataFrame(
        {
            "Q_exp": Q_shifted,
            "SoC": SoCs,
            "V_fit": Vfit_exp,
            "V_fit_anode": Vfit_Anode_aligned,
            "V_fit_cathode": Vfit_Cathode_exp,
            "dvdq_fit": dVdQ_fit_exp,
            "dvdq_fit_si": dVdQ_fit_si_exp,
            "dvdq_fit_gr": dVdQ_fit_gr_exp,
            "dqdv_fit": dqdv_fit,
            "dqdv_fit_si": dqdv_fit_si,
            "dqdv_fit_gr": dqdv_fit_gr,
            "x_si": x_si_on_common,
            "x_gr": x_gr_on_common,
            "Q_si": Q_si,
            "Q_gr": Q_gr,
            "si_share": si_share,
            "gr_share": gr_share,
        }
    )

    return {
        "params": {
            "Cn_Si": Cn_Si,
            "Cn_Gr": Cn_Gr,
            "x100": x100,
            "Cp": Cp,
            "y100": y100,
            "si_scale_a": si_scale_a,
            "si_shift_b": si_shift_b,
        },
        "transition_soc": transition_result["transition_soc"],
        "transition_reason": transition_result["reason"],
        "transition_idx_left": transition_result["transition_idx_left"],
        "candidate_crossings": transition_result["candidate_crossings"],
        "valid_crossings": transition_result["valid_crossings"],
        "Q_ref_2_9": Q_ref_2_9,
        "Q_ref_4_2": Q_ref_4_2,
        "details_df": details_df,
        "si_ocp_eff": si_ocp_eff,
    }


def append_transition_soc_to_summary_df(
    df: pd.DataFrame,
    *,
    Qdata_expand,
    ocp_3,
    smooth_then_grad,
    build_effective_si_ocp,
    Gr_OCP_smooth: pd.DataFrame,
    SI_RECONSTRUCTION_MODE: str = "reconstructed",
    win: int = 31,
    poly: int = 3,
    threshold: float = 0.5,
    persistence_window: float = 0.10,
    tol: float = 0.01,
) -> pd.DataFrame:
    out_df = df.copy()

    required_cols = ["Cn_Si", "Cn_Gr", "x100", "Cp", "y100", "si_scale_a", "si_shift_b"]
    missing = [c for c in required_cols if c not in out_df.columns]
    if missing:
        raise KeyError(f"Missing required parameter columns: {missing}")

    transition_soc_list = []
    transition_reason_list = []
    transition_n_candidates_list = []
    transition_n_valid_list = []

    for _, row in out_df.iterrows():
        params = [
            row["Cn_Si"],
            row["Cn_Gr"],
            row["x100"],
            row["Cp"],
            row["y100"],
            row["si_scale_a"],
            row["si_shift_b"],
        ]

        try:
            res = compute_si_current_share_and_transition_7p(
                params=params,
                Qdata_expand=Qdata_expand,
                ocp_3=ocp_3,
                smooth_then_grad=smooth_then_grad,
                build_effective_si_ocp=build_effective_si_ocp,
                Gr_OCP_smooth=Gr_OCP_smooth,
                SI_RECONSTRUCTION_MODE=SI_RECONSTRUCTION_MODE,
                win=win,
                poly=poly,
                threshold=threshold,
                persistence_window=persistence_window,
                tol=tol,
            )
            transition_soc_list.append(res["transition_soc"])
            transition_reason_list.append(res["transition_reason"])
            transition_n_candidates_list.append(len(res.get("candidate_crossings", [])))
            transition_n_valid_list.append(len(res.get("valid_crossings", [])))
        except Exception as e:
            transition_soc_list.append(np.nan)
            transition_reason_list.append(f"ERROR: {repr(e)}")
            transition_n_candidates_list.append(np.nan)
            transition_n_valid_list.append(np.nan)

    out_df["transition_soc"] = transition_soc_list
    out_df["transition_reason"] = transition_reason_list
    out_df["transition_n_candidates"] = transition_n_candidates_list
    out_df["transition_n_valid"] = transition_n_valid_list

    return out_df
