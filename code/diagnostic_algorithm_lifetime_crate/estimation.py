"""Fit the material-specific eSOH model to measured voltage and dV/dQ traces."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, differential_evolution
from scipy.signal import find_peaks
from scipy import interpolate

from .config import VoltageFitConfig
from .derivative_utils import resample_clamped, smooth_then_grad, unique_sorted_xy
from .measured_io import VoltageTrace
from .model_io import VoltageModelCurve, evaluate_ocp_vectorized, params_to_series
from .user_functions import Un, Up, Si_OCP, Gr_OCP, build_effective_si_ocp


def _safe_std(x, eps: float = 1e-6) -> float:
    s = float(np.nanstd(np.asarray(x, float)))
    return s if np.isfinite(s) and s >= eps else eps


def _rmse(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) == 0:
        return np.nan
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def _weighted_rmse(a, b, w):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    w = np.asarray(w, float)
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(w)
    if np.sum(m) == 0 or np.sum(w[m]) <= 0:
        return np.nan
    return float(np.sqrt(np.sum(w[m] * (a[m] - b[m]) ** 2) / np.sum(w[m])))


def _extract_cc_segment(
    meas: VoltageTrace,
    *,
    cc_current_threshold_a: float,
    min_cc_points: int,
    voltage_min_v: float,
    voltage_max_v: float,
    truncate_at_cv_onset: bool,
    cv_voltage_cutoff_v: Optional[float],
):
    Q = np.asarray(meas.Q_Ah, float).reshape(-1)
    V = np.asarray(meas.V_V, float).reshape(-1)
    I = None if meas.I_A is None else np.asarray(meas.I_A, float).reshape(-1)

    if np.isfinite(Q).any():
        Q = Q - float(Q[np.where(np.isfinite(Q))[0][0]])

    m = np.isfinite(Q) & np.isfinite(V)
    m &= (V >= float(voltage_min_v)) & (V <= float(voltage_max_v))
    if I is not None:
        m_cc = m & np.isfinite(I) & (I > float(cc_current_threshold_a))
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
    if Qcc.size > 500:
        q_new = np.linspace(float(Qcc[0]), float(Qcc[-1]), 800)
        Vcc = np.interp(q_new, Qcc, Vcc)
        if Icc is not None:
            Icc = np.interp(q_new, Qcc, Icc)
        Qcc = q_new
    return Qcc, Vcc, Icc


def _build_early_weights(n: int, frac: float, early_weight: float, mode: str) -> np.ndarray:
    w = np.ones(int(n), dtype=float)
    if n <= 0:
        return w
    frac = max(0.0, min(float(frac), 0.95))
    k0 = int(np.floor(frac * n))
    if k0 <= 0 or early_weight >= 1.0:
        return w
    mode = str(mode).lower().strip()
    if mode == "step":
        w[:k0] = early_weight
    elif mode == "linear":
        t = np.linspace(0.0, 1.0, k0)
        w[:k0] = early_weight + (1.0 - early_weight) * t
    else:
        t = np.linspace(0.0, 1.0, k0)
        w[:k0] = early_weight + (1.0 - early_weight) * (0.5 - 0.5 * np.cos(np.pi * t))
    return w


def _build_window_weights(Q, q_fit_min, q_fit_max, q_fit_frac_min, q_fit_frac_max, win, wout):
    Q = np.asarray(Q, float).reshape(-1)
    if Q.size == 0:
        return np.asarray([], dtype=float)

    w = np.full_like(Q, float(wout), dtype=float)
    qmin = float(np.nanmin(Q))
    qmax = float(np.nanmax(Q))
    span = max(qmax - qmin, 1e-12)

    use_frac = (q_fit_frac_min is not None) or (q_fit_frac_max is not None)
    if use_frac:
        Q_rel = (Q - qmin) / span
        lo = 0.0 if q_fit_frac_min is None else float(q_fit_frac_min)
        hi = 1.0 if q_fit_frac_max is None else float(q_fit_frac_max)
        lo = max(0.0, min(lo, 1.0))
        hi = max(0.0, min(hi, 1.0))
        if lo > hi:
            lo, hi = hi, lo
        mask = (Q_rel >= lo) & (Q_rel <= hi)
    else:
        lo = qmin if q_fit_min is None else float(q_fit_min)
        hi = qmax if q_fit_max is None else float(q_fit_max)
        if lo > hi:
            lo, hi = hi, lo
        mask = (Q >= lo) & (Q <= hi)

    w[mask] = float(win)
    return w


def _resolve_feature_regions(Q: np.ndarray, regions, are_relative: bool):
    Q = np.asarray(Q, float).reshape(-1)
    if Q.size == 0:
        return tuple()
    if not are_relative:
        return tuple((float(lo), float(hi)) for lo, hi in regions)

    qmin = float(np.nanmin(Q))
    qmax = float(np.nanmax(Q))
    span = max(qmax - qmin, 1e-12)
    out = []
    for lo, hi in regions:
        lo_abs = qmin + float(lo) * span
        hi_abs = qmin + float(hi) * span
        if lo_abs > hi_abs:
            lo_abs, hi_abs = hi_abs, lo_abs
        out.append((lo_abs, hi_abs))
    return tuple(out)


def _compute_features_for_estimation(
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
            q_out.append(np.nan)
            y_out.append(np.nan)
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


def _flatten_feature_dict(prefix: str, q_arr: np.ndarray, y_arr: np.ndarray):
    out = {}
    n = max(len(q_arr), len(y_arr))
    for i in range(n):
        out[f"{prefix}{i+1}_q"] = float(q_arr[i]) if i < len(q_arr) and np.isfinite(q_arr[i]) else np.nan
        out[f"{prefix}{i+1}_y"] = float(y_arr[i]) if i < len(y_arr) and np.isfinite(y_arr[i]) else np.nan
    return out


def fit_voltage_rpt(meas: VoltageTrace, *, p0: Dict[str, float], ocp_3, fit_cfg: VoltageFitConfig):
    Qcc, Vcc, _ = _extract_cc_segment(
        meas,
        cc_current_threshold_a=fit_cfg.cc_current_threshold_a,
        min_cc_points=fit_cfg.min_cc_points,
        voltage_min_v=fit_cfg.voltage_min_v,
        voltage_max_v=fit_cfg.voltage_max_v,
        truncate_at_cv_onset=fit_cfg.truncate_at_cv_onset,
        cv_voltage_cutoff_v=fit_cfg.cv_voltage_cutoff_v,
    )
    if Qcc.size < 10:
        raise ValueError("Too few valid CC points for voltage fitting.")

    Qcc_max_meas = float(Qcc[-1])
    early_w = _build_early_weights(
        Qcc.size, fit_cfg.voltage_early_frac, fit_cfg.voltage_early_weight, fit_cfg.voltage_weight_mode
    )
    win_w = _build_window_weights(
        Qcc,
        fit_cfg.v_q_fit_min,
        fit_cfg.v_q_fit_max,
        fit_cfg.v_q_fit_frac_min,
        fit_cfg.v_q_fit_frac_max,
        fit_cfg.v_weight_inside,
        fit_cfg.v_weight_outside,
    )
    point_w = early_w * win_w

    if fit_cfg.opt_grid_mode == "uniform":
        Q_opt = np.arange(0.0, Qcc_max_meas + 0.5 * fit_cfg.opt_dQ, fit_cfg.opt_dQ)
    else:
        Q_opt = Qcc.copy()

    use_reconstructed_si_ocp = bool(getattr(fit_cfg, "use_reconstructed_si_ocp", False))

    # Internal aliases match saved outputs: x100 = x_n,100, y100 = x_p,100,
    # si_scale_a = s_V, and si_shift_b = U_off in the manuscript.
    names = ["Cn_Si", "Cn_Gr", "x100", "Cp", "y100"]
    if getattr(fit_cfg, "enable_si_drift", False) or getattr(fit_cfg, "enable_si_deformation", False):
        names += ["si_scale_a", "si_shift_b"]
    x0_init = np.array([p0[k] for k in names], dtype=float)
    lb = np.array([fit_cfg.bounds[k][0] for k in names], dtype=float)
    ub = np.array([fit_cfg.bounds[k][1] for k in names], dtype=float)
    bounds_list = [(float(lb[i]), float(ub[i])) for i in range(len(names))]

    def ocp_3_local(res: pd.Series, Q):
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

    def build_model(x: np.ndarray, Q_eval: np.ndarray):
        x = np.asarray(x, float).reshape(-1)
        if len(x) == 5:
            res = params_to_series(x[0], x[1], x[2], x[3], x[4], 1.0, 0.0)
        elif len(x) == 7:
            res = params_to_series(x[0], x[1], x[2], x[3], x[4], x[5], x[6])
        else:
            raise ValueError(f"Unexpected parameter length: {len(x)}")
        return evaluate_ocp_vectorized(ocp_3_local, res, Q_eval)

    def residual_parts(x: np.ndarray):
        Vfit_opt, _, _ = build_model(x, Q_opt)
        Q_mapped = Qcc_max_meas - Q_opt
        V_model_on_meas = resample_clamped(Q_mapped, Vfit_opt, Qcc)

        sigV = _safe_std(Vcc)
        rV = np.sqrt(point_w) * (Vcc - V_model_on_meas) / sigV * float(fit_cfg.w_voltage)

        rD = None
        Qs_m = dVdQ_m = dVdQ_f_on_m = w_d = None
        if fit_cfg.use_dvdq:
            Qs_m, _, dVdQ_m = smooth_then_grad(Qcc, Vcc, fit_cfg.dvdq_sg_window, fit_cfg.dvdq_sg_poly)
            Qs_f, _, dVdQ_f = smooth_then_grad(Qcc, V_model_on_meas, fit_cfg.dvdq_sg_window, fit_cfg.dvdq_sg_poly)
            if Qs_m.size >= 5 and Qs_f.size >= 5:
                dVdQ_f_on_m = resample_clamped(Qs_f, dVdQ_f, Qs_m)
                w_d = _build_window_weights(
                    Qs_m,
                    fit_cfg.dvdq_q_fit_min,
                    fit_cfg.dvdq_q_fit_max,
                    fit_cfg.dvdq_q_fit_frac_min,
                    fit_cfg.dvdq_q_fit_frac_max,
                    fit_cfg.dvdq_weight_inside,
                    fit_cfg.dvdq_weight_outside,
                )
                sigd = _safe_std(dVdQ_m)
                rD = np.sqrt(w_d) * (dVdQ_m - dVdQ_f_on_m) / sigd * float(fit_cfg.w_dvdq)

        rP = None
        qpk_m = ypk_m = qpk_f = ypk_f = np.asarray([])
        if fit_cfg.use_peak_alignment and fit_cfg.w_peak > 0:
            if Qs_m is None or dVdQ_m is None or dVdQ_f_on_m is None:
                Qs_m, _, dVdQ_m = smooth_then_grad(Qcc, Vcc, fit_cfg.dvdq_sg_window, fit_cfg.dvdq_sg_poly)
                Qs_f, _, dVdQ_f = smooth_then_grad(Qcc, V_model_on_meas, fit_cfg.dvdq_sg_window, fit_cfg.dvdq_sg_poly)
                if Qs_m.size >= 5 and Qs_f.size >= 5:
                    dVdQ_f_on_m = resample_clamped(Qs_f, dVdQ_f, Qs_m)

            if Qs_m is not None and dVdQ_m is not None and dVdQ_f_on_m is not None and Qs_m.size >= 5:
                feature_regions = _resolve_feature_regions(Qs_m, fit_cfg.peak_regions, fit_cfg.peak_regions_are_relative)

                qpk_m_list, ypk_m_list = [], []
                qpk_f_list, ypk_f_list = [], []

                for i, reg in enumerate(feature_regions):
                    if i == 0:
                        mode_i = "first_valley"
                    elif i == 1:
                        mode_i = "last_valley"
                    else:
                        mode_i = "valley"

                    q1, y1 = _compute_features_for_estimation(
                        Qs_m, dVdQ_m, [reg],
                        mode=mode_i,
                        peak_min_prominence=fit_cfg.peak_min_prominence,
                    )
                    q2, y2 = _compute_features_for_estimation(
                        Qs_m, dVdQ_f_on_m, [reg],
                        mode=mode_i,
                        peak_min_prominence=fit_cfg.peak_min_prominence,
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

                m = np.isfinite(qpk_m) & np.isfinite(qpk_f)
                if np.any(m):
                    rP = (qpk_m[m] - qpk_f[m]) * float(fit_cfg.w_peak)

        rPrior = None
        if fit_cfg.use_prior:
            prior = []
            for i, name in enumerate(names):
                sigma = float(fit_cfg.prior_sigma.get(name, 0.0))
                if sigma > 0:
                    prior.append((x[i] - float(p0[name])) / sigma)
            if prior:
                rPrior = np.asarray(prior, float)

        return {
            "V_model_on_meas": V_model_on_meas,
            "rV": rV,
            "rD": rD,
            "rP": rP,
            "rPrior": rPrior,
            "Qs_m": Qs_m,
            "dVdQ_m": dVdQ_m,
            "dVdQ_f_on_m": dVdQ_f_on_m,
            "w_d": w_d,
            "qpk_m": qpk_m,
            "ypk_m": ypk_m,
            "qpk_f": qpk_f,
            "ypk_f": ypk_f,
        }

    def residual_vector(x: np.ndarray):
        parts = residual_parts(x)
        arrs = [parts["rV"], parts["rD"], parts["rP"], parts["rPrior"]]
        r = np.r_[tuple(a for a in arrs if a is not None)]
        r = r[np.isfinite(r)]
        return r if r.size else np.array([1e6], dtype=float)

    def scalar_objective_for_de(x: np.ndarray):
        r = residual_vector(np.asarray(x, float))
        return float(np.sum(r ** 2))

    if str(fit_cfg.optimizer).lower() == "de":
        de_res = differential_evolution(
            scalar_objective_for_de,
            bounds_list,
            maxiter=int(fit_cfg.de_maxiter),
            popsize=int(fit_cfg.de_popsize),
            seed=int(fit_cfg.de_seed),
            polish=bool(fit_cfg.de_polish),
            tol=float(fit_cfg.de_tol),
            workers=int(fit_cfg.de_workers),
            updating="deferred" if int(fit_cfg.de_workers) != 1 else "immediate",
        )
        x_start = np.asarray(de_res.x, float)
        opt_success = bool(de_res.success)
        opt_message = str(de_res.message)
        opt_nfev = int(getattr(de_res, "nfev", -1))
        opt_cost = float(de_res.fun)

        if fit_cfg.do_lsq_polish_after_de:
            lsq_res = least_squares(
                residual_vector,
                x_start,
                bounds=(lb, ub),
                method=fit_cfg.lsq_method,
                loss=fit_cfg.lsq_loss,
                f_scale=float(fit_cfg.lsq_f_scale),
            )
            x_opt = np.asarray(lsq_res.x, float)
            opt_success = bool(lsq_res.success)
            opt_message = f"DE -> LSQ polish | {lsq_res.message}"
            opt_nfev = int(lsq_res.nfev)
            opt_cost = float(lsq_res.cost)
        else:
            x_opt = x_start.copy()
    else:
        lsq_res = least_squares(
            residual_vector,
            x0_init,
            bounds=(lb, ub),
            method=fit_cfg.lsq_method,
            loss=fit_cfg.lsq_loss,
            f_scale=float(fit_cfg.lsq_f_scale),
        )
        x_opt = np.asarray(lsq_res.x, float)
        opt_success = bool(lsq_res.success)
        opt_message = str(lsq_res.message)
        opt_nfev = int(lsq_res.nfev)
        opt_cost = float(lsq_res.cost)

    parts_final = residual_parts(x_opt)
    V_model_on_meas = parts_final["V_model_on_meas"]

    Q_grid = fit_cfg.make_Q_grid()
    Vfit, Van, Vca = build_model(x_opt, Q_grid)
    dVdQ_grid = None
    Qs, _, dVdQ_tmp = smooth_then_grad(Q_grid, Vfit, fit_cfg.dvdq_sg_window, fit_cfg.dvdq_sg_poly)
    if Qs.size > 0:
        dVdQ_grid = resample_clamped(Qs, dVdQ_tmp, Q_grid)

    rmse_v_global = _rmse(Vcc, V_model_on_meas)
    rmse_v_weighted = _weighted_rmse(Vcc, V_model_on_meas, point_w)

    rmse_dvdq_global = np.nan
    rmse_dvdq_weighted = np.nan

    Qs_m_eval, _, dVdQ_m_eval = smooth_then_grad(Qcc, Vcc, fit_cfg.dvdq_sg_window, fit_cfg.dvdq_sg_poly)
    Qs_f_eval, _, dVdQ_f_eval = smooth_then_grad(Qcc, V_model_on_meas, fit_cfg.dvdq_sg_window, fit_cfg.dvdq_sg_poly)

    if Qs_m_eval.size >= 5 and Qs_f_eval.size >= 5:
        dVdQ_f_on_m_eval = resample_clamped(Qs_f_eval, dVdQ_f_eval, Qs_m_eval)

        w_d_eval = _build_window_weights(
            Qs_m_eval,
            fit_cfg.dvdq_q_fit_min,
            fit_cfg.dvdq_q_fit_max,
            fit_cfg.dvdq_q_fit_frac_min,
            fit_cfg.dvdq_q_fit_frac_max,
            fit_cfg.dvdq_weight_inside,
            fit_cfg.dvdq_weight_outside,
        )

        rmse_dvdq_global = _rmse(dVdQ_m_eval, dVdQ_f_on_m_eval)
        rmse_dvdq_weighted = _weighted_rmse(dVdQ_m_eval, dVdQ_f_on_m_eval, w_d_eval)

    Cn_Si = float(x_opt[0])
    Cn_Gr = float(x_opt[1])
    x100 = float(x_opt[2])
    Cp = float(x_opt[3])
    y100 = float(x_opt[4])
    si_scale_a = float(x_opt[5]) if len(x_opt) >= 6 else 1.0
    si_shift_b = float(x_opt[6]) if len(x_opt) >= 7 else 0.0
    Cn = Cn_Si + Cn_Gr
    LLI = float(x100 * Cn + y100 * Cp)

    Cap = Qcc_max_meas
    alpha = Cn_Si / max(Cn, 1e-12)

    x0 = x100 - Cap / max(Cn, 1e-12)
    y0 = y100 + Cap / max(Cp, 1e-12)

    R = np.nan

    Un0 = float(Un(
        x0,
        alpha,
        si_scale_a=si_scale_a,
        si_shift_b=si_shift_b,
        use_reconstructed_si_ocp=use_reconstructed_si_ocp,
    ))
    Un100 = float(Un(
        x100,
        alpha,
        si_scale_a=si_scale_a,
        si_shift_b=si_shift_b,
        use_reconstructed_si_ocp=use_reconstructed_si_ocp,
    ))

    if use_reconstructed_si_ocp:
        si_ocp_eff = build_effective_si_ocp(
            si_scale_a=si_scale_a,
            si_shift_b=si_shift_b,
            reconstruction_mode="reconstructed",
        )
    else:
        si_ocp_eff = build_effective_si_ocp(
            si_scale_a=si_scale_a,
            si_shift_b=si_shift_b,
            reconstruction_mode="direct",
        )

    si_p = np.asarray(si_ocp_eff["p"], float)
    si_sto = np.asarray(si_ocp_eff["sto"], float)

    order_si_eff = np.argsort(si_p)
    si_p = si_p[order_si_eff]
    si_sto = si_sto[order_si_eff]
    si_p, idx_eff = np.unique(si_p, return_index=True)
    si_sto = si_sto[idx_eff]

    f_si_p_sto = interpolate.interp1d(
        si_p,
        si_sto,
        bounds_error=False,
        fill_value=(float(si_sto[0]), float(si_sto[-1])),
    )
    f_gr_p_sto = interpolate.interp1d(
        Gr_OCP["p"],
        Gr_OCP["sto"],
        bounds_error=False,
        fill_value=(float(Gr_OCP["sto"].iloc[0]), float(Gr_OCP["sto"].iloc[-1])),
    )

    x0_si = float(f_si_p_sto(Un0))
    x0_gr = float(f_gr_p_sto(Un0))
    x100_si = float(f_si_p_sto(Un100))
    x100_gr = float(f_gr_p_sto(Un100))

    lli_theta = float(x100 * Cn + y100 * Cp)

    feature_meas_dict = _flatten_feature_dict("feature_meas_", parts_final["qpk_m"], parts_final["ypk_m"])
    feature_model_dict = _flatten_feature_dict("feature_model_", parts_final["qpk_f"], parts_final["ypk_f"])

    peak_meas_dict = _flatten_feature_dict("peak_meas_", parts_final["qpk_m"], parts_final["ypk_m"])
    peak_model_dict = _flatten_feature_dict("peak_model_", parts_final["qpk_f"], parts_final["ypk_f"])

    flag_bad_qmax = bool(Qcc_max_meas > float(fit_cfg.qcc_max_suspect_ah))
    flag_peak_missing = False
    if fit_cfg.use_peak_alignment:
        qpkm = parts_final["qpk_m"]
        qpkf = parts_final["qpk_f"]
        flag_peak_missing = bool((qpkm.size == 0) or (qpkf.size == 0) or np.any(~np.isfinite(qpkm)) or np.any(~np.isfinite(qpkf)))

    est = {
        "Cn_Si": Cn_Si,
        "Cn_Gr": Cn_Gr,
        "x100": x100,
        "Cp": Cp,
        "y100": y100,
        "Cn": Cn,
        "LLI": LLI,
        "x0": x0,
        "y0": y0,
        "alpha": alpha,
        "R": R,
        "Un0": Un0,
        "Un100": Un100,
        "x0_si": x0_si,
        "x0_gr": x0_gr,
        "x100_si": x100_si,
        "x100_gr": x100_gr,
        "lli_theta": lli_theta,
        "si_scale_a": si_scale_a,
        "si_shift_b": si_shift_b,
        "success": bool(opt_success),
        "cost": float(opt_cost),
        "nfev": int(opt_nfev),
        "message": str(opt_message),
        "Qcc_max_meas": Qcc_max_meas,
        "rmse_v_global": rmse_v_global,
        "rmse_v_weighted": rmse_v_weighted,
        "rmse_dvdq_global": rmse_dvdq_global,
        "rmse_dvdq_weighted": rmse_dvdq_weighted,
        "flag_bad_qmax": flag_bad_qmax,
        "flag_peak_missing": flag_peak_missing,
        "flag_fit_suspect": bool(flag_bad_qmax or (np.isfinite(rmse_v_global) and rmse_v_global > 0.03)),
        "feature_definition": "dV/dQ valleys | R1=first_valley, R2=last_valley",
        "dvdq_sg_window_used": int(fit_cfg.dvdq_sg_window),
        "dvdq_sg_poly_used": int(fit_cfg.dvdq_sg_poly),
    }
    est.update(feature_meas_dict)
    est.update(feature_model_dict)
    est.update(peak_meas_dict)
    est.update(peak_model_dict)

    curve = VoltageModelCurve(
        rpt=int(getattr(meas, "rpt", -1)),
        Q_grid=Q_grid,
        Vfit=Vfit,
        dVdQ=dVdQ_grid,
        meta={
            "V_anode": Van,
            "V_cathode": Vca,
            "Qcc_max_meas": Qcc_max_meas,
            "Ah_throughput": getattr(meas, "Ah_throughput", np.nan),
            "fit_config": asdict(fit_cfg),
            "opt_success": bool(opt_success),
            "opt_cost": float(opt_cost),
            "opt_message": str(opt_message),
            "opt_nfev": int(opt_nfev),
            "V_model_on_meas": V_model_on_meas,
            "Qcc_meas": Qcc,
            "Vcc_meas": Vcc,
            "feature_definition": "dV/dQ valleys | R1=first_valley, R2=last_valley",
            "dvdq_sg_window_used": int(fit_cfg.dvdq_sg_window),
            "dvdq_sg_poly_used": int(fit_cfg.dvdq_sg_poly),
            "feature_regions_resolved": list(_resolve_feature_regions(Qcc, fit_cfg.peak_regions, fit_cfg.peak_regions_are_relative)),
            "si_scale_a": si_scale_a,
            "si_shift_b": si_shift_b,
            "use_reconstructed_si_ocp": use_reconstructed_si_ocp,
        },
    )
    return est, curve
