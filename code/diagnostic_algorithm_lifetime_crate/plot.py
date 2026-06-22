"""Plot diagnostic voltage fits, dV/dQ comparisons, and silicon-OCP panels."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from .derivative_utils import resample_clamped, smooth_then_grad, unique_sorted_xy
from .user_functions import Si_OCP_raw
from .user_functions import build_effective_si_ocp


def _extract_cc_segment_for_plot_same_as_fit(meas, fit_cfg):
    Q = np.asarray(meas.Q_Ah, float).reshape(-1)
    V = np.asarray(meas.V_V, float).reshape(-1)
    I = None if meas.I_A is None else np.asarray(meas.I_A, float).reshape(-1)

    if np.isfinite(Q).any():
        Q = Q - float(Q[np.where(np.isfinite(Q))[0][0]])

    cc_current_threshold_a = float(getattr(fit_cfg, "cc_current_threshold_a", 0.01))
    min_cc_points = int(getattr(fit_cfg, "min_cc_points", 10))
    voltage_min_v = float(getattr(fit_cfg, "voltage_min_v", -np.inf))
    voltage_max_v = float(getattr(fit_cfg, "voltage_max_v", np.inf))
    truncate_at_cv_onset = bool(getattr(fit_cfg, "truncate_at_cv_onset", False))
    cv_voltage_cutoff_v = getattr(fit_cfg, "cv_voltage_cutoff_v", None)

    m = np.isfinite(Q) & np.isfinite(V)
    m &= (V >= voltage_min_v) & (V <= voltage_max_v)

    if I is not None:
        m_cc = m & np.isfinite(I) & (I > cc_current_threshold_a)
    else:
        m_cc = m.copy()

    if truncate_at_cv_onset and (cv_voltage_cutoff_v is not None):
        idx = np.where(m_cc)[0]
        if idx.size > 0:
            hit = np.where(V[idx] >= float(cv_voltage_cutoff_v))[0]
            if hit.size > 0:
                first_hit_global = idx[int(hit[0])]
                m_cc[first_hit_global:] = False

    if np.sum(m_cc) < min_cc_points:
        m_cc = m.copy()

    Qcc = Q[m_cc]
    Vcc = V[m_cc]
    Icc = I[m_cc] if I is not None else None

    Qcc, Vcc = unique_sorted_xy(Qcc, Vcc)

    if Icc is not None and Qcc.size > 0:
        Icc = resample_clamped(Q[m_cc], I[m_cc], Qcc)

    # identical to estimation.py
    if Qcc.size > 500:
        q_new = np.linspace(float(Qcc[0]), float(Qcc[-1]), 800)
        Vcc = np.interp(q_new, Qcc, Vcc)
        if Icc is not None:
            Icc = np.interp(q_new, Qcc, Icc)
        Qcc = q_new

    return Qcc, Vcc

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


def _compute_features_for_plot(
    Q,
    y,
    regions,
    *,
    mode="peak",
    peak_min_prominence=0.02,
):
    Q = np.asarray(Q, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)

    q_out = []
    y_out = []

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

        elif mode in ("valley", "first_valley", "last_valley", "deepest_valley"):
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
        else:
            raise ValueError(f"Unknown mode: {mode}")

        q_out.append(float(q_r[j]))
        y_out.append(float(y_r[j]))

    return np.asarray(q_out), np.asarray(y_out)


def _get_si_ocp_for_plot():
    si_plot = Si_OCP_raw.copy()
    si_plot["sto"] = np.asarray(si_plot["sto"], float)
    si_plot["p"] = np.asarray(si_plot["p"], float)
    si_plot = si_plot.dropna(subset=["sto", "p"]).sort_values("sto").reset_index(drop=True)
    return si_plot["sto"].to_numpy(), si_plot["p"].to_numpy()


def plot_voltage_fit(
    meas,
    curve,
    *,
    fit_cfg=None,
    title: str = "Voltage fit",
    show_features: bool = True,
    show_regions: bool = True,
    derivative_mode: str = "dvdq",
    annotate_features: bool = True,
    ylim_derivative=None,
    show_si_ocp_panel: bool = True,
    return_plot_data: bool = True,
):
    if fit_cfg is not None:
        Qcc, Vcc = _extract_cc_segment_for_plot_same_as_fit(meas, fit_cfg)
    else:
        Qcc, Vcc = _extract_cc_segment_for_plot(meas)

    Qcc_max_meas = float(curve.meta.get("Qcc_max_meas", Qcc[-1]))

    Q_model_mapped = Qcc_max_meas - np.asarray(curve.Q_grid, float)
    V_model_on_meas = resample_clamped(Q_model_mapped, np.asarray(curve.Vfit, float), Qcc)

    if show_si_ocp_panel:
        fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.5))

    plot_data = {
        "voltage_panel": {
            "Q_meas": np.asarray(Qcc, float),
            "V_meas": np.asarray(Vcc, float),
            "Q_model": np.asarray(Qcc, float),
            "V_model": np.asarray(V_model_on_meas, float),
            "Q_model_native": np.asarray(Q_model_mapped, float),
            "V_model_native": np.asarray(curve.Vfit, float),
            "Qcc_max_meas": float(Qcc_max_meas),
        }
    }

    # panel 1: voltage
    ax[0].plot(Qcc, Vcc, label="Measured", lw=1.8)
    ax[0].plot(Qcc, V_model_on_meas, label="Model", lw=1.8)
    ax[0].set_xlabel("Q [Ah]")
    ax[0].set_ylabel("Voltage [V]")
    ax[0].set_title(title)
    ax[0].legend()

    sg_window = 301
    sg_poly = 3
    feature_regions = ()
    feature_regions_are_relative = True
    peak_min_prominence = 0.02

    if fit_cfg is not None:
        sg_window = int(getattr(fit_cfg, "dvdq_sg_window", sg_window))
        sg_poly = int(getattr(fit_cfg, "dvdq_sg_poly", sg_poly))
        feature_regions = getattr(fit_cfg, "peak_regions", ())
        feature_regions_are_relative = getattr(fit_cfg, "peak_regions_are_relative", True)
        peak_min_prominence = getattr(fit_cfg, "peak_min_prominence", peak_min_prominence)

    Qs_m = np.asarray([])
    dVdQ_m = np.asarray([])
    Qs_f = np.asarray([])
    dVdQ_f = np.asarray([])
    dVdQ_f_on_m = np.asarray([])
    y_m = np.asarray([])
    y_f = np.asarray([])
    qpk_m = np.asarray([])
    ypk_m = np.asarray([])
    qpk_f = np.asarray([])
    ypk_f = np.asarray([])
    resolved_regions = ()

    # panel 2: derivative
    Qs_m, _, dVdQ_m = smooth_then_grad(Qcc, Vcc, sg_window, sg_poly)
    Qs_f, _, dVdQ_f = smooth_then_grad(Qcc, V_model_on_meas, sg_window, sg_poly)

    if Qs_m.size and Qs_f.size:
        dVdQ_f_on_m = resample_clamped(Qs_f, dVdQ_f, Qs_m)

        if derivative_mode == "minus_dvdq":
            y_m = -dVdQ_m
            y_f = -dVdQ_f_on_m
            ylabel = "-dV/dQ [V/Ah]"
            dtitle = "Derivative and detected features (-dV/dQ)"
            region_modes = ["first_valley", "last_valley"]
        else:
            y_m = dVdQ_m
            y_f = dVdQ_f_on_m
            ylabel = "dV/dQ [V/Ah]"
            dtitle = "Derivative and detected features (dV/dQ)"
            region_modes = ["first_valley", "last_valley"]

        ax[1].plot(Qs_m, y_m, label="Measured", lw=1.8)
        ax[1].plot(Qs_m, y_f, label="Model", lw=1.8)
        ax[1].set_xlabel("Q [Ah]")
        ax[1].set_ylabel(ylabel)
        ax[1].set_title(dtitle)

        resolved_regions = _resolve_feature_regions_for_plot(
            Qs_m, feature_regions, feature_regions_are_relative=feature_regions_are_relative
        )

        if show_regions and len(resolved_regions) > 0:
            for i, (lo, hi) in enumerate(resolved_regions, start=1):
                ax[1].axvspan(lo, hi, alpha=0.12)
                ax[1].text(
                    0.5 * (lo + hi),
                    0.98,
                    f"R{i}",
                    transform=ax[1].get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=9,
                )

        if show_features and len(resolved_regions) > 0:
            qpk_m_list, ypk_m_list = [], []
            qpk_f_list, ypk_f_list = [], []

            for i, reg in enumerate(resolved_regions):
                mode_i = region_modes[i] if i < len(region_modes) else "valley"

                q1, y1 = _compute_features_for_plot(
                    Qs_m, y_m, [reg],
                    mode=mode_i,
                    peak_min_prominence=peak_min_prominence,
                )
                q2, y2 = _compute_features_for_plot(
                    Qs_m, y_f, [reg],
                    mode=mode_i,
                    peak_min_prominence=peak_min_prominence,
                )

                if q1.size:
                    qpk_m_list.append(q1[0])
                    ypk_m_list.append(y1[0])
                if q2.size:
                    qpk_f_list.append(q2[0])
                    ypk_f_list.append(y2[0])

            qpk_m = np.asarray(qpk_m_list, float)
            ypk_m = np.asarray(ypk_m_list, float)
            qpk_f = np.asarray(qpk_f_list, float)
            ypk_f = np.asarray(ypk_f_list, float)

            if qpk_m.size:
                ax[1].scatter(qpk_m, ypk_m, s=45, marker="o", label="Measured features", zorder=5)
                for q, y in zip(qpk_m, ypk_m):
                    ax[1].axvline(q, ls="--", lw=0.8, alpha=0.5)
                    if annotate_features:
                        ax[1].annotate(
                            f"{q:.2f}",
                            (q, y),
                            textcoords="offset points",
                            xytext=(0, 8),
                            ha="center",
                            fontsize=8,
                        )

            if qpk_f.size:
                ax[1].scatter(qpk_f, ypk_f, s=55, marker="x", label="Model features", zorder=5)
                for q, y in zip(qpk_f, ypk_f):
                    if annotate_features:
                        ax[1].annotate(
                            f"{q:.2f}",
                            (q, y),
                            textcoords="offset points",
                            xytext=(0, -12),
                            ha="center",
                            fontsize=8,
                        )

        if ylim_derivative is not None:
            ax[1].set_ylim(ylim_derivative)
        else:
            ax[1].set_ylim([0, 0.8])

        ax[1].legend()

    plot_data["derivative_panel"] = {
        "Q_meas_native": np.asarray(Qs_m, float),
        "dVdQ_meas_native": np.asarray(dVdQ_m, float),
        "Q_model_native": np.asarray(Qs_f, float),
        "dVdQ_model_native": np.asarray(dVdQ_f, float),
        "Q_plot": np.asarray(Qs_m, float),
        "y_meas_plot": np.asarray(y_m, float),
        "y_model_plot": np.asarray(y_f, float),
        "feature_q_meas": np.asarray(qpk_m, float),
        "feature_y_meas": np.asarray(ypk_m, float),
        "feature_q_model": np.asarray(qpk_f, float),
        "feature_y_model": np.asarray(ypk_f, float),
        "resolved_regions": np.asarray(resolved_regions, float) if len(resolved_regions) > 0 else np.empty((0, 2)),
        "derivative_mode": derivative_mode,
        "sg_window": int(sg_window),
        "sg_poly": int(sg_poly),
    }

    # panel 3: Si OCP
    if show_si_ocp_panel:
        sto_si, U_si_raw = _get_si_ocp_for_plot()
        a = float(curve.meta.get("si_scale_a", 1.0))
        b = float(curve.meta.get("si_shift_b", 0.0))
        use_reconstructed_si_ocp = bool(curve.meta.get("use_reconstructed_si_ocp", False))

        U_si_target = a * U_si_raw + b

        ax[2].plot(sto_si, U_si_raw, label="Raw Si OCP", lw=2.0, color="black")

        si_ocp_data = {
            "sto_raw": np.asarray(sto_si, float),
            "U_raw": np.asarray(U_si_raw, float),
            "U_target": np.asarray(U_si_target, float),
            "a": float(a),
            "b": float(b),
            "use_reconstructed_si_ocp": bool(use_reconstructed_si_ocp),
        }

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
            ax[2].set_xlim([-0.08, 1.12])
            ax[2].set_ylim([0.02, 1])
            ax[2].set_title(f"Reconstructed Si OCP (s_V={a:.3f}, U_off={b*1000:.1f} mV)")

            si_ocp_data.update({
                "sto_full": np.asarray(sto_full, float),
                "U_full": np.asarray(p_full, float),
                "sto_norm": np.asarray(sto_norm, float),
                "U_norm": np.asarray(p_norm, float),
                "mask_left": np.asarray(mask_left, bool),
                "mask_mid": np.asarray(mask_mid, bool),
                "mask_right": np.asarray(mask_right, bool),
                "x_lo": float(si_eff.get("x_lo", np.nan)),
                "x_hi": float(si_eff.get("x_hi", np.nan)),
                "p_high_eff": float(si_eff.get("p_high_eff", np.nan)),
                "p_low_eff": float(si_eff.get("p_low_eff", np.nan)),
            })
        else:
            ax[2].plot(sto_si, U_si_target, label="Current Si OCP", lw=2.0, color="tab:red")
            ax[2].set_title(f"Si OCP (s_V={a:.3f}, U_off={b*1000:.1f} mV)")

        ax[2].set_xlabel("Silicon stoichiometry [-]")
        ax[2].set_ylabel("Potential [V vs. Li/Li+]")
        ax[2].grid(True, alpha=0.3)
        ax[2].legend(fontsize=8)

        plot_data["si_ocp_panel"] = si_ocp_data

    fig.tight_layout()

    if return_plot_data:
        return fig, plot_data
    return fig
