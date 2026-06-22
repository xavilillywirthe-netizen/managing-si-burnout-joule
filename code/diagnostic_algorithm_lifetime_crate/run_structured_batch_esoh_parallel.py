"""Run structured C-rate diagnostic batches with optional parallel execution."""

from __future__ import annotations

import copy
import json
import os
import pickle
import re
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "managing_si_burnout_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnostic_algorithm_lifetime_crate.config import VoltageFitConfig
from diagnostic_algorithm_lifetime_crate.estimation import fit_voltage_rpt
from diagnostic_algorithm_lifetime_crate.plot import plot_voltage_fit
import diagnostic_algorithm_lifetime_crate.user_functions as uf

from .crate_io import structure_segments_from_files
from .ir_correction import apply_ir_correction_to_trace, get_R_from_label

def save_plot_data_csv(plot_data: dict, out_prefix: Path):
    out_prefix = Path(out_prefix)

    vp = plot_data.get("voltage_panel", {})
    pd.DataFrame({
        "Q_meas": np.asarray(vp.get("Q_meas", []), float),
        "V_meas": np.asarray(vp.get("V_meas", []), float),
        "Q_model": np.asarray(vp.get("Q_model", []), float),
        "V_model": np.asarray(vp.get("V_model", []), float),
    }).to_csv(out_prefix.with_name(out_prefix.name + "_voltage_panel.csv"), index=False)

    q_model_native = np.asarray(vp.get("Q_model_native", []), float)
    v_model_native = np.asarray(vp.get("V_model_native", []), float)
    if q_model_native.size > 0 and v_model_native.size > 0:
        pd.DataFrame({
            "Q_model_native": q_model_native,
            "V_model_native": v_model_native,
        }).to_csv(out_prefix.with_name(out_prefix.name + "_voltage_panel_native.csv"), index=False)

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

    q_plot = np.asarray(dp.get("Q_plot", []), float)
    y_meas_plot = np.asarray(dp.get("y_meas_plot", []), float)
    y_model_plot = np.asarray(dp.get("y_model_plot", []), float)
    if q_plot.size > 0 and y_meas_plot.size > 0 and y_model_plot.size > 0:
        pd.DataFrame({
            "Q_plot": q_plot,
            "y_meas_plot": y_meas_plot,
            "y_model_plot": y_model_plot,
        }).to_csv(out_prefix.with_name(out_prefix.name + "_derivative_panel.csv"), index=False)

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

    regions = np.asarray(dp.get("resolved_regions", np.empty((0, 2))), float)
    if regions.size > 0:
        pd.DataFrame(regions, columns=["q_lo", "q_hi"]).to_csv(
            out_prefix.with_name(out_prefix.name + "_regions.csv"),
            index=False
        )

    # si ocp
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
                
def _save_curve_artifacts(curve, curve_dir: Path, stem: str):
    q = np.asarray(curve.Q_grid, float)
    v = np.asarray(curve.Vfit, float)
    dvdq = None if curve.dVdQ is None else np.asarray(curve.dVdQ, float)

    df = pd.DataFrame({"Q_grid_Ah": q, "Vfit_V": v})
    if dvdq is not None and len(dvdq) == len(df):
        df["dVdQ_V_per_Ah"] = dvdq

    df.to_csv(curve_dir / f"{stem}_curve.csv", index=False)
    with open(curve_dir / f"{stem}_curve.pkl", "wb") as f:
        pickle.dump(curve, f)


def _fmt_tag(x):
    if x is None:
        return "none"
    if isinstance(x, bool):
        return "T" if x else "F"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.3g}".replace(".", "p")
    return str(x).replace(".", "p").replace("/", "_")


def _infer_cell_tag(file_paths: Sequence[str | Path]) -> str:
    if not file_paths:
        return "structured"
    stems = [Path(p).stem for p in file_paths]
    matches = []
    for s in stems:
        m = re.search(r"CELL(\d+)", s, flags=re.IGNORECASE)
        if m:
            matches.append(int(m.group(1)))
    if not matches:
        return "structured"
    unique = sorted(set(matches))
    if len(unique) == 1:
        return f"cell{unique[0]:03d}"
    return "multi_" + "-".join(f"{u:03d}" for u in unique[:5])


def _auto_run_name(file_paths: Sequence[str | Path], fit_cfg: VoltageFitConfig) -> str:
    vmin = getattr(fit_cfg, "v_q_fit_frac_min", None)
    vmax = getattr(fit_cfg, "v_q_fit_frac_max", None)
    vout = getattr(fit_cfg, "v_weight_outside", None)
    w_d = getattr(fit_cfg, "w_dvdq", None)
    dmin = getattr(fit_cfg, "dvdq_q_fit_frac_min", None)
    dmax = getattr(fit_cfg, "dvdq_q_fit_frac_max", None)
    dout = getattr(fit_cfg, "dvdq_weight_outside", None)
    drift = getattr(fit_cfg, "enable_si_drift", None)
    deform = getattr(fit_cfg, "enable_si_deformation", None)
    recon = getattr(fit_cfg, "use_reconstructed_si_ocp", None)

    prefix = _infer_cell_tag(file_paths)
    return (
        f"{prefix}"
        f"_drift{_fmt_tag(drift)}"
        f"_deform{_fmt_tag(deform)}"
        f"_recon{_fmt_tag(recon)}"
        f"_v{_fmt_tag(vmin)}to{_fmt_tag(vmax)}"
        f"_vout{_fmt_tag(vout)}"
        f"_d{_fmt_tag(dmin)}to{_fmt_tag(dmax)}"
        f"_dout{_fmt_tag(dout)}"
        f"_wd{_fmt_tag(w_d)}"
    )


def _crate_label_to_c(label: str):
    if label is None:
        return np.nan
    s = str(label).strip().upper()

    m = re.match(r"C\s*/\s*([0-9.]+)", s)
    if m:
        denom = float(m.group(1))
        return 1.0 / denom if denom != 0 else np.nan

    m = re.match(r"([0-9.]+)\s*C", s)
    if m:
        return float(m.group(1))

    return np.nan


def _aggregate_by_crate(results_df: pd.DataFrame) -> pd.DataFrame:
    df = results_df.copy()
    df["crate_c"] = df["crate_label"].apply(_crate_label_to_c)
    df = df[np.isfinite(df["crate_c"])].copy()
    if df.empty:
        return pd.DataFrame()

    agg_spec = {
        "segment_global_id": "count",
        "Cn_Si": "mean",
        "Cn_Gr": "mean",
        "Cn": "mean",
        "Cp": "mean",
        "LLI": "mean",
        "rmse_v_global": "mean",
        "rmse_dvdq_global": "mean",
    }
    if "si_scale_a" in df.columns:
        agg_spec["si_scale_a"] = "mean"
    if "si_shift_b" in df.columns:
        agg_spec["si_shift_b"] = "mean"

    agg = (
        df.groupby(["crate_c", "crate_label"], as_index=False)
        .agg(agg_spec)
        .rename(columns={"segment_global_id": "n_segments"})
        .sort_values("crate_c", ascending=False)
        .reset_index(drop=True)
    )
    return agg


def save_crate_parameter_trend_plot(results_df: pd.DataFrame, out_path: Path):
    agg = _aggregate_by_crate(results_df)
    if agg.empty:
        print("No valid C-rate labels found; skip crate trend plot.")
        return

    x = agg["crate_c"].to_numpy()

    plt.close("all")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    plot_items = [
        ("Cn_Si", "Cn_Si"),
        ("Cn_Gr", "Cn_Gr"),
        ("Cn", "Cn"),
        ("Cp", "Cp"),
        ("LLI", "LLI"),
    ]

    for ax, (col, title) in zip(axes.flat[:5], plot_items):
        ax.plot(x, agg[col].to_numpy(), marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("C-rate")
        ax.grid(True, alpha=0.3)

    ax = axes.flat[5]
    if "si_scale_a" in agg.columns:
        ax.plot(x, agg["si_scale_a"].to_numpy(), marker="o", linewidth=2, label="s_V")
    if "si_shift_b" in agg.columns:
        ax.plot(x, agg["si_shift_b"].to_numpy(), marker="s", linewidth=2, label="U_off")
    ax.set_title("Si-OCP deformation")
    ax.set_xlabel("C-rate")
    ax.grid(True, alpha=0.3)
    if ("si_scale_a" in agg.columns) or ("si_shift_b" in agg.columns):
        ax.legend()

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.3g}C" for v in x], rotation=30)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_crate_reference_tables(results_df: pd.DataFrame, out_dir: Path):
    agg = _aggregate_by_crate(results_df)
    if agg.empty:
        print("No valid C-rate labels found; skip reference tables.")
        return None, None, None

    agg.to_csv(out_dir / "crate_parameter_summary_by_crate.csv", index=False)

    ref_candidates = agg[np.isclose(agg["crate_c"], 0.01, atol=1e-9)]
    if ref_candidates.empty:
        ref_idx = int(np.argmin(np.abs(agg["crate_c"].to_numpy() - 0.01)))
        ref_row = agg.iloc[ref_idx]
    else:
        ref_row = ref_candidates.iloc[0]

    params5 = ["Cn_Si", "Cn_Gr", "Cn", "Cp", "LLI"]

    rows = []
    for _, row in agg.iterrows():
        out = {
            "crate_c": float(row["crate_c"]),
            "crate_label": row["crate_label"],
            "reference_crate_c": float(ref_row["crate_c"]),
            "reference_crate_label": ref_row["crate_label"],
        }
        abs_pct_list = []
        for p in params5:
            ref_val = float(ref_row[p])
            cur_val = float(row[p])
            if np.isfinite(ref_val) and abs(ref_val) > 1e-12:
                pct = 100.0 * (cur_val - ref_val) / ref_val
                abs_pct = abs(pct)
            else:
                pct = np.nan
                abs_pct = np.nan
            out[f"{p}_pct_vs_C100"] = pct
            out[f"{p}_abs_pct_vs_C100"] = abs_pct
            if np.isfinite(abs_pct):
                abs_pct_list.append(abs_pct)

        if "si_scale_a" in agg.columns:
            ref_val = float(ref_row["si_scale_a"])
            cur_val = float(row["si_scale_a"])
            out["si_scale_a_pct_vs_C100"] = 100.0 * (cur_val - ref_val) / ref_val if np.isfinite(ref_val) and abs(ref_val) > 1e-12 else np.nan
        if "si_shift_b" in agg.columns:
            ref_val = float(ref_row["si_shift_b"])
            cur_val = float(row["si_shift_b"])
            out["si_shift_b_pct_vs_C100"] = 100.0 * (cur_val - ref_val) / ref_val if np.isfinite(ref_val) and abs(ref_val) > 1e-12 else np.nan

        out["mean_absolute_error_percentage_5params_vs_C100"] = float(np.mean(abs_pct_list)) if abs_pct_list else np.nan
        rows.append(out)

    pct_df = pd.DataFrame(rows).sort_values("crate_c", ascending=False).reset_index(drop=True)
    pct_csv = out_dir / "crate_vs_C100_percent_change.csv"
    pct_df.to_csv(pct_csv, index=False)

    mae_df = pct_df[[
        "crate_c",
        "crate_label",
        "reference_crate_c",
        "reference_crate_label",
        "mean_absolute_error_percentage_5params_vs_C100",
    ]].copy()
    mae_csv = out_dir / "crate_vs_C100_mae_percentage.csv"
    mae_df.to_csv(mae_csv, index=False)

    return agg, pct_df, mae_df


def _run_one_segment_worker(
    *,
    seg_global_id: int,
    seg,
    fit_cfg: VoltageFitConfig,
    p0_default: Dict[str, float],
    R_by_label: Dict[str, float],
    default_R_ohm: Optional[float],
    plot_dir: Path,
    curve_dir: Path,
    save_plots: bool,
):
    R_ohm = get_R_from_label(seg.crate_label, R_by_label, default_R_ohm=default_R_ohm)
    seg_corr = apply_ir_correction_to_trace(seg, R_ohm=R_ohm)

    est, curve = fit_voltage_rpt(seg_corr, p0=p0_default.copy(), ocp_3=uf.ocp_3, fit_cfg=fit_cfg)

    stem = f"seg{seg_global_id:03d}_{Path(seg.file).stem}_{seg.crate_label.replace('/', '_')}"
    if save_plots:
        fig, plot_data = plot_voltage_fit(
            seg_corr,
            curve,
            fit_cfg=fit_cfg,
            title=f"{seg.file} | seg {seg.segment_id} | {seg.crate_label} | R={R_ohm:.6f}Ω",
            show_features=True,
            show_regions=True,
            derivative_mode="dvdq",
            annotate_features=True,
            ylim_derivative=(0, 1.2),
            return_plot_data=True,
        )
        fig.savefig(plot_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        save_plot_data_csv(plot_data, plot_dir / stem)

    _save_curve_artifacts(curve, curve_dir, stem)

    row = {
        "segment_global_id": seg_global_id,
        "file": seg.file,
        "segment_id": seg.segment_id,
        "crate_label": seg.crate_label,
        "I_mean_A": seg.I_mean_A,
        "R_used_ohm": R_ohm,
        "Ah_throughput": seg.Ah_throughput,
        **est,
    }
    return row


def run_structured_batch_esoh(
    *,
    file_paths: Sequence[str | Path],
    project_dir: Path,
    fit_cfg: VoltageFitConfig,
    R_by_label: Dict[str, float],
    nominal_capacity_ah: float = 2.5,
    run_name: Optional[str] = None,
    output_group: Optional[str] = None,
    charge_threshold_a: float = 0.01,
    min_segment_points: int = 300,
    cc_only: bool = True,
    cc_rel_tol: float = 0.05,
    save_plots: bool = True,
    default_R_ohm: Optional[float] = None,
    p0_default: Optional[Dict[str, float]] = None,
    parallel: bool = False,
    n_jobs: int = 4,
):
    if p0_default is None:
        p0_default = {
            "Cn_Si": 1.36,
            "Cn_Gr": 1.34,
            "x100": 0.91,
            "Cp": 2.61,
            "y100": 0.02,
            "si_scale_a": 1.0,
            "si_shift_b": 0.0,
        }

    if run_name is None:
        run_name = _auto_run_name(file_paths, fit_cfg)

    batch_root = Path(project_dir) / "batch_results"
    if output_group:
        batch_root = batch_root / output_group
    out_dir = batch_root / run_name
    plot_dir = out_dir / "plots"
    curve_dir = out_dir / "curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    curve_dir.mkdir(parents=True, exist_ok=True)

    segments = structure_segments_from_files(
        list(file_paths),
        nominal_capacity_ah=nominal_capacity_ah,
        charge_threshold_a=charge_threshold_a,
        min_segment_points=min_segment_points,
        cc_only=cc_only,
        cc_rel_tol=cc_rel_tol,
        resample_on_q=True,
        n_resample=800,
        cache_dir=project_dir / "_crate_cache",
        refresh_cache=False,
    )

    print(f"Loaded {len(segments)} charge segments")

    rows = []
    failed = []

    if parallel:
        fit_cfg_local = copy.deepcopy(fit_cfg)
        if hasattr(fit_cfg_local, "de_workers"):
            fit_cfg_local.de_workers = 1

        futures = []
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
            for idx, seg in enumerate(segments):
                print(f"Submitting segment {idx}: {seg.file} | {seg.crate_label}")
                futures.append(
                    ex.submit(
                        _run_one_segment_worker,
                        seg_global_id=idx,
                        seg=seg,
                        fit_cfg=fit_cfg_local,
                        p0_default=p0_default.copy(),
                        R_by_label=R_by_label,
                        default_R_ohm=default_R_ohm,
                        plot_dir=plot_dir,
                        curve_dir=curve_dir,
                        save_plots=save_plots,
                    )
                )

            for fut in as_completed(futures):
                try:
                    row = fut.result()
                    rows.append(row)
                    print(f"Finished segment {row['segment_global_id']}: {row['file']} | {row['crate_label']}")
                except Exception as e:
                    failed.append({
                        "segment_global_id": np.nan,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    })
                    print(f"Parallel worker failed: {e}")

    else:
        for idx, seg in enumerate(segments):
            try:
                row = _run_one_segment_worker(
                    seg_global_id=idx,
                    seg=seg,
                    fit_cfg=fit_cfg,
                    p0_default=p0_default.copy(),
                    R_by_label=R_by_label,
                    default_R_ohm=default_R_ohm,
                    plot_dir=plot_dir,
                    curve_dir=curve_dir,
                    save_plots=save_plots,
                )
                rows.append(row)
                print(f"Finished segment {row['segment_global_id']}: {row['file']} | {row['crate_label']}")
            except Exception as e:
                failed.append({
                    "segment_global_id": idx,
                    "file": getattr(seg, "file", None),
                    "segment_id": getattr(seg, "segment_id", None),
                    "crate_label": getattr(seg, "crate_label", None),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                print(f"Segment {idx} failed: {e}")

    if len(rows) == 0:
        if failed:
            pd.DataFrame(failed).to_csv(out_dir / "structured_batch_esoh_failed.csv", index=False)
        raise RuntimeError(
            "All parallel workers failed. Check structured_batch_esoh_failed.csv in the output folder "
            "for the first real traceback."
        )

    results_df = pd.DataFrame(rows).sort_values(["segment_global_id"]).reset_index(drop=True)
    results_df.to_csv(out_dir / "structured_batch_esoh_summary.csv", index=False)

    with open(out_dir / "fit_cfg.json", "w") as f:
        json.dump(fit_cfg.__dict__, f, indent=2, default=str)
    with open(out_dir / "R_by_label.json", "w") as f:
        json.dump({str(k): float(v) for k, v in R_by_label.items()}, f, indent=2)

    save_crate_parameter_trend_plot(results_df, out_dir / "crate_parameter_trends.png")
    save_crate_reference_tables(results_df, out_dir)

    if failed:
        pd.DataFrame(failed).to_csv(out_dir / "structured_batch_esoh_failed.csv", index=False)

    print(f"Saved summary to {out_dir / 'structured_batch_esoh_summary.csv'}")
    return results_df, out_dir
