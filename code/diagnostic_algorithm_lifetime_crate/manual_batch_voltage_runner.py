
"""Run lifetime-cell voltage diagnostics and export per-cell fit summaries."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "managing_si_burnout_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import diagnostic_algorithm_lifetime_crate.config as cfgmod
import diagnostic_algorithm_lifetime_crate.estimation as estmod
import diagnostic_algorithm_lifetime_crate.measured_io as miomod
import diagnostic_algorithm_lifetime_crate.user_functions as ufmod
from diagnostic_algorithm_lifetime_crate.user_functions import build_effective_si_ocp

try:
    import diagnostic_algorithm_lifetime_crate.plot as plotmod
except Exception:
    plotmod = None

importlib.reload(cfgmod)
importlib.reload(estmod)
importlib.reload(miomod)
importlib.reload(ufmod)
if plotmod is not None:
    importlib.reload(plotmod)

from diagnostic_algorithm_lifetime_crate.config import VoltageFitConfig
from diagnostic_algorithm_lifetime_crate.measured_io import load_processed_voltage
from diagnostic_algorithm_lifetime_crate.estimation import fit_voltage_rpt
from diagnostic_algorithm_lifetime_crate.derivative_utils import resample_clamped, smooth_then_grad, unique_sorted_xy
from scipy.signal import find_peaks


def _jsonable(obj: Any):
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def make_default_fit_cfg() -> VoltageFitConfig:
    fit_cfg = VoltageFitConfig(
        v_q_fit_frac_min=0.05,
        v_q_fit_frac_max=1.0,
        v_weight_inside=1.0,
        v_weight_outside=0.1,
        use_dvdq=True,
        w_dvdq=0.1,
        dvdq_q_fit_frac_min=0.05,
        dvdq_q_fit_frac_max=1.0,
        dvdq_weight_inside=1.0,
        dvdq_weight_outside=0.1,
        use_peak_alignment=False,
        w_peak=0.0,
        peak_regions=((0.1, 0.35), (0.6, 0.9)),
        peak_regions_are_relative=True,
        optimizer="de",
        de_maxiter=60,
        de_popsize=10,
        de_seed=42,
        de_workers=1,
        do_lsq_polish_after_de=True,
        truncate_at_cv_onset=False,
        cv_voltage_cutoff_v=4.195,
        dvdq_sg_window=31,
        bounds={
        "Cn_Si": (0.05, 1.4),
        "Cn_Gr": (0.5, 1.6),
        "x100": (0.7, 1.0),
        "Cp": (2.6, 2.8),
        "y100": (0.0, 0.12),
        "si_scale_a": (0.3, 1.1),
        "si_shift_b": (0, 0.08),
        }
    )
    if hasattr(fit_cfg, "enable_si_deformation"):
        fit_cfg.enable_si_deformation = True
    if hasattr(fit_cfg, "si_deformation_model"):
        fit_cfg.si_deformation_model = "model_a"
    if hasattr(fit_cfg, "enable_si_drift"):
        fit_cfg.enable_si_drift = True
    return fit_cfg


def save_plot_data_csv(plot_data: dict, out_prefix: Path):
    out_prefix = Path(out_prefix)


    # voltage panel: measured axis
    vp = plot_data.get("voltage_panel", {})
    pd.DataFrame({
        "Q_meas": np.asarray(vp.get("Q_meas", []), float),
        "V_meas": np.asarray(vp.get("V_meas", []), float),
        "Q_model": np.asarray(vp.get("Q_model", []), float),
        "V_model": np.asarray(vp.get("V_model", []), float),
    }).to_csv(out_prefix.with_name(out_prefix.name + "_voltage_panel.csv"), index=False)


    # voltage panel: native model grid

    q_model_native = np.asarray(vp.get("Q_model_native", []), float)
    v_model_native = np.asarray(vp.get("V_model_native", []), float)
    if q_model_native.size > 0 and v_model_native.size > 0:
        pd.DataFrame({
            "Q_model_native": q_model_native,
            "V_model_native": v_model_native,
        }).to_csv(out_prefix.with_name(out_prefix.name + "_voltage_panel_native.csv"), index=False)


    # derivative panel: native derivative
    dp = plot_data.get("derivative_panel", {})
    q_meas_native = np.asarray(dp.get("Q_meas_native", []), float)
    dvdq_meas_native = np.asarray(dp.get("dVdQ_meas_native", []), float)
    q_model_native = np.asarray(dp.get("Q_model_native", []), float)
    dvdq_model_native = np.asarray(dp.get("dVdQ_model_native", []), float)

    if q_meas_native.size > 0 and dvdq_meas_native.size > 0:
        pd.DataFrame({
            "Q_meas_native": q_meas_native,
            "dVdQ_meas_native": dvdq_meas_native,
        }).to_csv(out_prefix.with_name(out_prefix.name + "_derivative_meas_native.csv"), index=False)

    if q_model_native.size > 0 and dvdq_model_native.size > 0:
        pd.DataFrame({
            "Q_model_native": q_model_native,
            "dVdQ_model_native": dvdq_model_native,
        }).to_csv(out_prefix.with_name(out_prefix.name + "_derivative_model_native.csv"), index=False)


    # derivative panel: plotted on same Q axis
    q_plot = np.asarray(dp.get("Q_plot", []), float)
    y_meas_plot = np.asarray(dp.get("y_meas_plot", []), float)
    y_model_plot = np.asarray(dp.get("y_model_plot", []), float)
    if q_plot.size > 0 and y_meas_plot.size > 0 and y_model_plot.size > 0:
        pd.DataFrame({
            "Q_plot": q_plot,
            "y_meas_plot": y_meas_plot,
            "y_model_plot": y_model_plot,
        }).to_csv(out_prefix.with_name(out_prefix.name + "_derivative_panel.csv"), index=False)


    # feature points
    fq_meas = np.asarray(dp.get("feature_q_meas", []), float)
    fy_meas = np.asarray(dp.get("feature_y_meas", []), float)
    fq_model = np.asarray(dp.get("feature_q_model", []), float)
    fy_model = np.asarray(dp.get("feature_y_model", []), float)

    n_feat = max(fq_meas.size, fy_meas.size, fq_model.size, fy_model.size)
    if n_feat > 0:
        def _pad(x, n):
            x = np.asarray(x, float)
            if x.size < n:
                x = np.r_[x, np.full(n - x.size, np.nan)]
            return x

        pd.DataFrame({
            "feature_q_meas": _pad(fq_meas, n_feat),
            "feature_y_meas": _pad(fy_meas, n_feat),
            "feature_q_model": _pad(fq_model, n_feat),
            "feature_y_model": _pad(fy_model, n_feat),
        }).to_csv(out_prefix.with_name(out_prefix.name + "_features.csv"), index=False)


    # regions
    regions = np.asarray(dp.get("resolved_regions", np.empty((0, 2))), float)
    if regions.size > 0:
        pd.DataFrame(regions, columns=["q_lo", "q_hi"]).to_csv(
            out_prefix.with_name(out_prefix.name + "_regions.csv"),
            index=False
        )


    # si ocp panel
    sp = plot_data.get("si_ocp_panel", {})
    if sp:
        sto_raw = np.asarray(sp.get("sto_raw", []), float)
        U_raw = np.asarray(sp.get("U_raw", []), float)
        U_target = np.asarray(sp.get("U_target", []), float)

        if sto_raw.size > 0 and U_raw.size > 0 and U_target.size > 0:
            pd.DataFrame({
                "sto_raw": sto_raw,
                "U_raw": U_raw,
                "U_target": U_target,
            }).to_csv(out_prefix.with_name(out_prefix.name + "_si_ocp_raw_target.csv"), index=False)

        if "sto_full" in sp and "U_full" in sp:
            sto_full = np.asarray(sp.get("sto_full", []), float)
            U_full = np.asarray(sp.get("U_full", []), float)
            if sto_full.size > 0 and U_full.size > 0:
                pd.DataFrame({
                    "sto_full": sto_full,
                    "U_full": U_full,
                }).to_csv(out_prefix.with_name(out_prefix.name + "_si_ocp_full.csv"), index=False)

        if "sto_norm" in sp and "U_norm" in sp:
            sto_norm = np.asarray(sp.get("sto_norm", []), float)
            U_norm = np.asarray(sp.get("U_norm", []), float)
            if sto_norm.size > 0 and U_norm.size > 0:
                pd.DataFrame({
                    "sto_norm": sto_norm,
                    "U_norm": U_norm,
                }).to_csv(out_prefix.with_name(out_prefix.name + "_si_ocp_norm.csv"), index=False)

def _extract_cc_segment_for_plot(meas, cc_current_threshold_a=0.05, min_cc_points=30):
    Q = np.asarray(meas.Q_Ah, float).reshape(-1)
    V = np.asarray(meas.V_V, float).reshape(-1)
    I = None if getattr(meas, "I_A", None) is None else np.asarray(meas.I_A, float).reshape(-1)

    if np.isfinite(Q).any():
        Q = Q - float(Q[np.where(np.isfinite(Q))[0][0]])

    m = np.isfinite(Q) & np.isfinite(V)
    if I is not None:
        m_cc = m & np.isfinite(I) & (I > float(cc_current_threshold_a))
    else:
        m_cc = m
    if np.sum(m_cc) < min_cc_points:
        m_cc = m

    Qcc = Q[m_cc]
    Vcc = V[m_cc]
    Qcc, Vcc = unique_sorted_xy(Qcc, Vcc)
    return Qcc, Vcc


def _resolve_feature_regions_for_plot(Q, feature_regions, feature_regions_are_relative=True):
    Q = np.asarray(Q, float).reshape(-1)
    if Q.size == 0 or feature_regions is None:
        return tuple()
    if not feature_regions_are_relative:
        return tuple((float(lo), float(hi)) for lo, hi in feature_regions)
    qmin = float(np.nanmin(Q))
    qmax = float(np.nanmax(Q))
    span = max(qmax - qmin, 1e-12)
    out = []
    for lo, hi in feature_regions:
        lo_abs = qmin + float(lo) * span
        hi_abs = qmin + float(hi) * span
        if lo_abs > hi_abs:
            lo_abs, hi_abs = hi_abs, lo_abs
        out.append((lo_abs, hi_abs))
    return tuple(out)


def _compute_features_for_plot(Q, y, regions, *, mode="peak", peak_min_prominence=0.02):
    Q = np.asarray(Q, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    q_out, y_out = [], []
    if Q.size < 5 or y.size != Q.size:
        return np.asarray([]), np.asarray([])
    for (lo, hi) in regions:
        m = np.isfinite(Q) & np.isfinite(y) & (Q >= lo) & (Q <= hi)
        if np.sum(m) < 5:
            continue
        q_r = Q[m]
        y_r = y[m]
        if mode in ("peak", "first_peak", "last_peak"):
            idx, props = find_peaks(y_r, prominence=float(peak_min_prominence))
            if len(idx) == 0:
                j = int(np.nanargmax(y_r))
            else:
                if mode == "first_peak":
                    j = int(idx[0])
                elif mode == "last_peak":
                    j = int(idx[-1])
                else:
                    prom = props.get("prominences", np.zeros(len(idx)))
                    j = int(idx[np.argmax(prom)])
        else:
            idx, props = find_peaks(-y_r, prominence=float(peak_min_prominence))
            if len(idx) == 0:
                j = int(np.nanargmin(y_r))
            else:
                if mode == "first_valley":
                    j = int(idx[0])
                elif mode == "last_valley":
                    j = int(idx[-1])
                elif mode == "deepest_valley":
                    j = int(idx[np.argmin(y_r[idx])])
                else:
                    prom = props.get("prominences", np.zeros(len(idx)))
                    j = int(idx[np.argmax(prom)])
        q_out.append(float(q_r[j]))
        y_out.append(float(y_r[j]))
    return np.asarray(q_out), np.asarray(y_out)


def _get_si_ocp_for_plot():
    si_plot = ufmod.Si_OCP_raw.copy()
    si_plot["sto"] = pd.to_numeric(si_plot["sto"], errors="coerce")
    si_plot["p"] = pd.to_numeric(si_plot["p"], errors="coerce")
    si_plot = si_plot.dropna(subset=["sto", "p"]).sort_values("sto").reset_index(drop=True)
    return si_plot["sto"].to_numpy(), si_plot["p"].to_numpy()


def plot_voltage_fit_3panel(meas, curve, *, fit_cfg=None, title="Voltage fit", derivative_mode="dvdq"):
    if plotmod is not None and hasattr(plotmod, "plot_voltage_fit"):
        try:
            return plotmod.plot_voltage_fit(
                meas,
                curve,
                fit_cfg=fit_cfg,
                title=title,
                show_si_ocp_panel=True,
                derivative_mode=derivative_mode,
            )
        except TypeError:
            pass

    Qcc, Vcc = _extract_cc_segment_for_plot(meas)
    Qcc_max_meas = float(curve.meta.get("Qcc_max_meas", Qcc[-1]))
    Q_model_mapped = Qcc_max_meas - np.asarray(curve.Q_grid, float)
    V_model_on_meas = resample_clamped(Q_model_mapped, np.asarray(curve.Vfit, float), Qcc)

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    ax[0].plot(Qcc, Vcc, label="Measured", lw=1.8)
    ax[0].plot(Qcc, V_model_on_meas, label="Model", lw=1.8)
    ax[0].set_xlabel("Q [Ah]")
    ax[0].set_ylabel("Voltage [V]")
    ax[0].set_title(title)
    ax[0].legend()

    sg_window = int(getattr(fit_cfg, "dvdq_sg_window", 31)) if fit_cfg is not None else 31
    sg_poly = int(getattr(fit_cfg, "dvdq_sg_poly", 3)) if fit_cfg is not None else 3
    feature_regions = getattr(fit_cfg, "peak_regions", ()) if fit_cfg is not None else ()
    feature_regions_are_relative = getattr(fit_cfg, "peak_regions_are_relative", True) if fit_cfg is not None else True
    peak_min_prominence = getattr(fit_cfg, "peak_min_prominence", 0.02) if fit_cfg is not None else 0.02

    Qs_m, _, dVdQ_m = smooth_then_grad(Qcc, Vcc, sg_window, sg_poly)
    Qs_f, _, dVdQ_f = smooth_then_grad(Qcc, V_model_on_meas, sg_window, sg_poly)
    if Qs_m.size and Qs_f.size:
        dVdQ_f_on_m = resample_clamped(Qs_f, dVdQ_f, Qs_m)
        if derivative_mode == "minus_dvdq":
            y_m = -dVdQ_m
            y_f = -dVdQ_f_on_m
            ylabel = r"$-dV/dQ$ [V/Ah]"
            region_modes = ["first_valley", "last_valley"]
        else:
            y_m = dVdQ_m
            y_f = dVdQ_f_on_m
            ylabel = r"$dV/dQ$ [V/Ah]"
            region_modes = ["first_valley", "last_valley"]

        ax[1].plot(Qs_m, y_m, label="Measured", lw=1.8)
        ax[1].plot(Qs_m, y_f, label="Model", lw=1.8)
        ax[1].set_xlabel("Q [Ah]")
        ax[1].set_ylabel(ylabel)
        ax[1].set_title("Derivative and detected features")
        

        resolved_regions = _resolve_feature_regions_for_plot(
            Qs_m, feature_regions, feature_regions_are_relative=feature_regions_are_relative
        )
        for i, (lo, hi) in enumerate(resolved_regions, start=1):
            ax[1].axvspan(lo, hi, alpha=0.12)
            ax[1].text(0.5 * (lo + hi), 0.98, f"R{i}", transform=ax[1].get_xaxis_transform(),
                       ha="center", va="top", fontsize=9)

        qpk_m_list, ypk_m_list = [], []
        qpk_f_list, ypk_f_list = [], []
        for i, reg in enumerate(resolved_regions):
            mode_i = region_modes[i] if i < len(region_modes) else "valley"
            q1, y1 = _compute_features_for_plot(Qs_m, y_m, [reg], mode=mode_i, peak_min_prominence=peak_min_prominence)
            q2, y2 = _compute_features_for_plot(Qs_m, y_f, [reg], mode=mode_i, peak_min_prominence=peak_min_prominence)
            if q1.size:
                qpk_m_list.append(q1[0]); ypk_m_list.append(y1[0])
            if q2.size:
                qpk_f_list.append(q2[0]); ypk_f_list.append(y2[0])
        if qpk_m_list:
            ax[1].scatter(np.asarray(qpk_m_list), np.asarray(ypk_m_list), s=45, marker="o", label="Measured features", zorder=5)
        if qpk_f_list:
            ax[1].scatter(np.asarray(qpk_f_list), np.asarray(ypk_f_list), s=55, marker="x", label="Model features", zorder=5)
        ax[1].legend()
        
        
        # ---- panel 3: reconstructed Si OCP ----
        sto_si, U_si_raw = _get_si_ocp_for_plot()
        a = float(curve.meta.get("si_scale_a", 1.0))
        b = float(curve.meta.get("si_shift_b", 0.0))
        use_reconstructed_si_ocp = bool(curve.meta.get("use_reconstructed_si_ocp", False))

        # target = only shift + compress
        U_si_target = a * U_si_raw + b

        ax[2].plot(sto_si, U_si_raw, label="Raw Si OCP", lw=2.0, color="black")

        if use_reconstructed_si_ocp:
            si_eff = build_effective_si_ocp(
                si_scale_a=a,
                si_shift_b=b,
                reconstruction_mode="reconstructed",
            )

            sto_full = np.asarray(si_eff["sto_full"], float)
            p_full = np.asarray(si_eff["p_full"], float)
            sto_norm = np.asarray(si_eff["sto"], float)
            p_norm = np.asarray(si_eff["p"], float)

            mask_left = sto_full < 0.0
            mask_mid = (sto_full >= 0.0) & (sto_full <= 1.0)
            mask_right = sto_full > 1.0

            # faint target reference
            ax[2].plot(
                sto_si,
                U_si_target,
                label="Target (aU+b)",
                lw=1.8,
                color="tab:orange",
                alpha=0.45,
            )

            if np.any(mask_left):
                ax[2].plot(
                    sto_full[mask_left],
                    p_full[mask_left],
                    lw=2.4,
                    color="tab:blue",
                    label="Left tail",
                )
            if np.any(mask_mid):
                ax[2].plot(
                    sto_full[mask_mid],
                    p_full[mask_mid],
                    lw=2.4,
                    color="tab:orange",
                    label="Core target kept",
                )
            if np.any(mask_right):
                ax[2].plot(
                    sto_full[mask_right],
                    p_full[mask_right],
                    lw=2.4,
                    color="tab:red",
                    label="Right tail",
                )

            # show the actual normalized curve used in blending
            ax[2].plot(
                sto_norm,
                p_norm,
                lw=2.2,
                color="tab:purple",
                linestyle="--",
                label="Normalized OCP used",
            )

            p_high = float(si_eff.get("p_high_eff", np.nan))
            p_low = float(si_eff.get("p_low_eff", np.nan))
            if np.isfinite(p_high):
                ax[2].axhline(p_high, color="gray", linestyle=":", lw=1.0)
            if np.isfinite(p_low):
                ax[2].axhline(p_low, color="gray", linestyle=":", lw=1.0)

            ax[2].axvline(0.0, color="gray", linestyle="--", lw=1.0, alpha=0.7)
            ax[2].axvline(1.0, color="gray", linestyle="--", lw=1.0, alpha=0.7)
            ax[2].set_title(f"Reconstructed Si OCP (s_V={a:.3f}, U_off={b*1000:.1f} mV)")
        else:
            U_si_curr = U_si_target
            ax[2].plot(sto_si, U_si_curr, label="Current Si OCP", lw=2.0, color="tab:red")
            ax[2].set_title(f"Si OCP (s_V={a:.3f}, U_off={b*1000:.1f} mV)")

        ax[2].set_xlabel("Silicon stoichiometry [-]")
        ax[2].set_ylabel("Potential [V vs. Li/Li+]")
        ax[2].grid(True, alpha=0.3)
        ax[2].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_lifetime_si_ocp_shift(results_df: pd.DataFrame, out_path: Path):
    from matplotlib import cm, colors
    sto, U_base = _get_si_ocp_for_plot()
    dfp = results_df.sort_values("rpt_seq").reset_index(drop=True)
    plt.close("all")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sto, U_base, color="black", linewidth=2.5, label="Raw Si OCP")
    norm = colors.Normalize(vmin=float(dfp["rpt_seq"].min()), vmax=float(dfp["rpt_seq"].max()))
    cmap = cm.get_cmap("viridis")
    for _, row in dfp.iterrows():
        a = float(row.get("si_scale_a", 1.0))
        b = float(row.get("si_shift_b", 0.0))
        U_new = a * U_base + b
        ax.plot(sto, U_new, color=cmap(norm(float(row["rpt_seq"]))), linewidth=2)
    ax.set_xlabel("Silicon stoichiometry [-]")
    ax.set_ylabel("Potential [V vs. Li/Li+]")
    ax.set_title("Model A silicon OCP evolution across lifetime")
    ax.grid(True, alpha=0.3)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("RPT sequence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_trends_1x3(results_df: pd.DataFrame, out_path: Path):
    dfp = results_df.sort_values("Ah_throughput").reset_index(drop=True).copy()
    if "Cn" not in dfp.columns and {"Cn_Si", "Cn_Gr"}.issubset(dfp.columns):
        dfp["Cn"] = dfp["Cn_Si"] + dfp["Cn_Gr"]

    x = dfp["Ah_throughput"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=False)

    # Row 1, Col 1
    for col in [c for c in ["Qcc_max_meas", "Cn", "Cp"] if c in dfp.columns]:
        y = dfp[col].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        axes[0, 0].plot(x[m], y[m], marker="o", linewidth=2, label=col)
    axes[0, 0].set_title("Cell / electrode capacity", fontsize=13)
    axes[0, 0].set_xlabel("Ah throughput [Ah]", fontsize=12)
    axes[0, 0].set_ylabel("Capacity", fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=10)

    # Row 1, Col 2
    for col in [c for c in ["Cn_Si", "Cn_Gr"] if c in dfp.columns]:
        y = dfp[col].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        axes[0, 1].plot(x[m], y[m], marker="o", linewidth=2, label=col)
    axes[0, 1].set_title("Anode material capacities", fontsize=13)
    axes[0, 1].set_xlabel("Ah throughput [Ah]", fontsize=12)
    axes[0, 1].set_ylabel("Capacity", fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=10)

    # Row 1, Col 3
    if "LLI" in dfp.columns:
        y = dfp["LLI"].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        axes[0, 2].plot(x[m], y[m], marker="o", linewidth=2, label="LLI")
    axes[0, 2].set_title("Lithium inventory loss", fontsize=13)
    axes[0, 2].set_xlabel("Ah throughput [Ah]", fontsize=12)
    axes[0, 2].set_ylabel("LLI", fontsize=12)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend(fontsize=10)

    # Row 2, Col 1
    if "si_scale_a" in dfp.columns:
        y = dfp["si_scale_a"].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        axes[1, 0].plot(x[m], y[m], marker="o", linewidth=2, label="s_V")
    axes[1, 0].set_title("Si-OCP voltage scaling", fontsize=13)
    axes[1, 0].set_xlabel("Ah throughput [Ah]", fontsize=12)
    axes[1, 0].set_ylabel("s_V", fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=10)

    # Row 2, Col 2
    if "si_shift_b" in dfp.columns:
        y = dfp["si_shift_b"].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        axes[1, 1].plot(x[m], y[m], marker="o", linewidth=2, label="U_off")
    axes[1, 1].set_title("Si-OCP voltage offset", fontsize=13)
    axes[1, 1].set_xlabel("Ah throughput [Ah]", fontsize=12)
    axes[1, 1].set_ylabel("U_off [V]", fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=10)

    # Row 2, Col 3
    for col in [c for c in ["rmse_v_global", "rmse_dvdq_global"] if c in dfp.columns]:
        y = dfp[col].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        axes[1, 2].plot(x[m], y[m], marker="o", linewidth=2, label=col)
    axes[1, 2].set_title("Fit error", fontsize=13)
    axes[1, 2].set_xlabel("Ah throughput [Ah]", fontsize=12)
    axes[1, 2].set_ylabel("RMSE", fontsize=12)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend(fontsize=10)

    for ax in axes.ravel():
        ax.tick_params(labelsize=10)

    fig.suptitle("Capacity, eSOH, deformation parameters, and fit error vs Ah throughput", fontsize=15, y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_manual_batch(
    *,
    cell: int,
    project_dir: Path,
    voltaiq_root: Path,
    schema: str = "auto",
    direction: str = "charge",
    protocol_keyword: str | None = "C/20",
    fit_cfg: VoltageFitConfig | None = None,
    rpt_filter: list[int] | None = None,
    refresh_cache: bool = False,
    out_dir: Path | None = None,
    save_plots: bool = True,
):
    project_dir = Path(project_dir)
    cache_dir = project_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if fit_cfg is None:
        fit_cfg = make_default_fit_cfg()

    measured = load_processed_voltage(
        cell,
        schema=schema,
        voltaiq_root=voltaiq_root,
        direction=direction,
        protocol_keyword=protocol_keyword,
        cache_dir=cache_dir,
        use_cache=True,
        refresh_cache=refresh_cache,
    )

    rpt_keys = sorted(measured.keys())
    if rpt_filter is not None:
        rpt_keys = [k for i, k in enumerate(rpt_keys) if i in set(rpt_filter)]

    if out_dir is None:
        vmin = getattr(fit_cfg, "v_q_fit_frac_min", None)
        vmax = getattr(fit_cfg, "v_q_fit_frac_max", None)
        vout = getattr(fit_cfg, "v_weight_outside", None)
        w_d = getattr(fit_cfg, "w_dvdq", None)
        dmin = getattr(fit_cfg, "dvdq_q_fit_frac_min", None)
        dmax = getattr(fit_cfg, "dvdq_q_fit_frac_max", None)
        dout = getattr(fit_cfg, "dvdq_weight_outside", None)
        drift = getattr(fit_cfg, "enable_si_drift", None)
        reconstr = getattr(fit_cfg, "use_reconstructed_si_ocp", None)
        out_dir = project_dir / "batch_results" / "A01_cell_lifetime_diagnostics" / f"cell{cell:03d}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    curves = {}
    p0_current = {
        "Cn_Si": 1.36,
        "Cn_Gr": 1.34,
        "x100": 0.91,
        "Cp": 2.61,
        "y100": 0.02,
        "si_scale_a": 1.0,
        "si_shift_b": 0.0,
    }

    plot_dir = out_dir / "rpt_plots"
    if save_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    for rpt_seq, rpt_key in enumerate(rpt_keys):
        print(f"=== Fitting RPT seq {rpt_seq} (raw key {rpt_key}) ===")
        meas = measured[rpt_key]
        est, curve = fit_voltage_rpt(meas, p0=p0_current, ocp_3=ufmod.ocp_3, fit_cfg=fit_cfg)

        row = {
            "cell": cell,
            "rpt_seq": rpt_seq,
            "rpt_key": rpt_key,
            "Ah_throughput": getattr(meas, "Ah_throughput", np.nan),
            **est,
        }
        rows.append(row)
        curves[rpt_seq] = curve

        p0_current = {
            "Cn_Si": est["Cn_Si"],
            "Cn_Gr": est["Cn_Gr"],
            "x100": est["x100"],
            "Cp": est["Cp"],
            "y100": est["y100"],
            "si_scale_a": est.get("si_scale_a", 1.0),
            "si_shift_b": est.get("si_shift_b", 0.0),
        }

        if save_plots:
            base_name = plot_dir / f"cell{cell:03d}_rptseq{rpt_seq:03d}"

            if plotmod is not None and hasattr(plotmod, "plot_voltage_fit"):
                fig, plot_data = plotmod.plot_voltage_fit(
                    meas,
                    curve,
                    fit_cfg=fit_cfg,
                    title=f"Cell {cell:03d} | RPT seq {rpt_seq}",
                    show_si_ocp_panel=True,
                    derivative_mode="dvdq",
                    return_plot_data=True,
                )
            else:
                fig = plot_voltage_fit_3panel(
                    meas,
                    curve,
                    fit_cfg=fit_cfg,
                    title=f"Cell {cell:03d} | RPT seq {rpt_seq}",
                )
                plot_data = None

            fig.savefig(
                base_name.with_name(base_name.name + "_fit.png"),
                dpi=220,
                bbox_inches="tight"
            )
            plt.close(fig)
            # save_plot_csv: bool = False
            if plot_data is not None:
                save_plot_data_csv(plot_data, base_name)

    results_df = pd.DataFrame(rows).sort_values(["rpt_seq"]).reset_index(drop=True)

    summary_cols = [c for c in [
        "rpt_seq", "rpt_key", "Ah_throughput",
        "rmse_v_global", "rmse_dvdq_global",
        "Cn_Si", "Cn_Gr", "Cn", "Cp", "LLI",
        "x100", "y100", "si_scale_a", "si_shift_b"
    ] if c in results_df.columns]

    results_df.to_csv(out_dir / f"cell{cell:03d}_voltage_fit_summary.csv", index=False)
    results_df[summary_cols].to_json(out_dir / f"cell{cell:03d}_voltage_fit_summary.json",
                                     orient="records", indent=2)

    fit_cfg_dict = _jsonable(fit_cfg)
    with open(out_dir / "fit_cfg.json", "w") as f:
        json.dump(fit_cfg_dict, f, indent=2)

    if not results_df.empty:
        final_params = _jsonable(results_df.iloc[-1].to_dict())
        with open(out_dir / f"cell{cell:03d}_final_parameters.json", "w") as f:
            json.dump(final_params, f, indent=2)

    if save_plots and not results_df.empty:
        plot_lifetime_si_ocp_shift(results_df, out_dir / f"cell{cell:03d}_si_ocp_lifetime_gradient.png")
        plot_parameter_trends_1x3(results_df, out_dir / f"cell{cell:03d}_parameter_trends_1x3.png")

    return results_df, curves, out_dir


def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", type=int, required=True)
    p.add_argument("--project-dir", type=Path, default=Path.cwd())
    p.add_argument("--voltaiq-root", type=Path, required=True)
    p.add_argument("--schema", type=str, default="auto")
    p.add_argument("--direction", type=str, default="charge")
    p.add_argument("--protocol-keyword", type=str, default="C/20")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p


def main():
    args = _build_parser().parse_args()
    fit_cfg = make_default_fit_cfg()
    results_df, curves, out_dir = run_manual_batch(
        cell=args.cell,
        project_dir=args.project_dir,
        voltaiq_root=args.voltaiq_root,
        schema=args.schema,
        direction=args.direction,
        protocol_keyword=args.protocol_keyword,
        fit_cfg=fit_cfg,
        refresh_cache=args.refresh_cache,
        save_plots=not args.no_plots,
    )
    print("Saved outputs to:", out_dir)
    print(results_df.head())


if __name__ == "__main__":
    main()
