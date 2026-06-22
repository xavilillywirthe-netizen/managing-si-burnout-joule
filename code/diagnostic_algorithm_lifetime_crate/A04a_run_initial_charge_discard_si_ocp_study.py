"""Run the discarded-initial-charge and silicon-OCP-treatment C-rate study.

This script supports the Supplementary Information analysis comparing:
1) the portion of initial charge data excluded from the fit, swept from
   0% to 10%, and
2) two silicon OCP treatments: deformation enabled vs. deformation disabled.

The same script can be configured for BOL or EOL data and for runs with or
without kinetic compensation. Kinetic compensation is applied through the
C-rate-specific resistance map R_by_label before the steady-state voltage fit.
"""

from __future__ import annotations

import copy
import importlib
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def find_repo_root(start: Path) -> Path:
    """Return the repository root that contains the shared code and data folders."""
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "code" / "diagnostic_algorithm_lifetime_crate").is_dir() and (
            candidate / "data" / "bol_eol_crate_data"
        ).is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the repository root from this script.")


REPO_ROOT = find_repo_root(Path(__file__))
CODE_DIR = REPO_ROOT / "code"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
CRATE_DATA_DIR = REPO_ROOT / "data" / "bol_eol_crate_data"

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "managing_si_burnout_matplotlib"))

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import diagnostic_algorithm_lifetime_crate.config as cfgmod
import diagnostic_algorithm_lifetime_crate.estimation as estmod
import diagnostic_algorithm_lifetime_crate.plot as plotmod
import diagnostic_algorithm_lifetime_crate.user_functions as ufmod

importlib.reload(cfgmod)
importlib.reload(estmod)
importlib.reload(plotmod)
importlib.reload(ufmod)

from diagnostic_algorithm_lifetime_crate.config import VoltageFitConfig
from diagnostic_algorithm_lifetime_crate.derivative_utils import smooth_then_grad
from diagnostic_algorithm_lifetime_crate.run_structured_batch_esoh_parallel import run_structured_batch_esoh
from diagnostic_algorithm_lifetime_crate.transition_soc_utils import (
    append_transition_soc_to_summary_df,
    smooth_ocp_curve,
)
from diagnostic_algorithm_lifetime_crate.user_functions import Gr_OCP, build_effective_si_ocp, ocp_3


# -----------------------------------------------------------------------------
# Study selection
# -----------------------------------------------------------------------------
# Use "BOL" for the beginning-of-life C-rate dataset or "EOL" for the aged
# cell-020 C-rate dataset. Set KINETIC_COMPENSATION to False to reproduce the
# uncompensated steady-state analysis.
DATASET = "BOL"
KINETIC_COMPENSATION = True

# The paper compares deformation-enabled and deformation-disabled fits. The
# fixed-(s_V, U_off) case is retained as an optional diagnostic but is disabled
# by default because it is not part of the two-case figure comparison.
INCLUDE_FIXED_AB_CASE = False


DATASETS = {
    "BOL": {
        "standard_bol": {
            "Cn_Si": 1.234015,
            "Cn_Gr": 1.280356,
            "Cn": 2.514370,
            "Cp": 2.728159,
            "LLI": 2.397999,
            "transition_soc": 0.55775,
        },
        "standard_current": {
            "Cn_Si": 1.234015,
            "Cn_Gr": 1.280356,
            "Cn": 2.514370,
            "Cp": 2.728159,
            "LLI": 2.397999,
            "transition_soc": 0.55775,
        },
        "fixed_a": 0.961458,
        "fixed_b": 0.031372,
        "file_paths": [
            CRATE_DATA_DIR / "GMFEB23S_CELL076_C10C5C3Expansion_1_P25C_15P0PSI_20260206_R0_36_1_2_2818579691.xlsx",
            CRATE_DATA_DIR / "GMFEB23S_CELL076_C20Expansion_1_P25C_15P0PSI_20260124_R0_36_1_2_2818579686.xlsx",
            CRATE_DATA_DIR / "GMFEB23S_CELL076_C100Expansion_1_P25C_15P0PSI_20260124_R0_36_1_2_2818579684.xlsx",
        ],
        "R_with_kinetic": {
            "C/200": 0.0,
            "C/100": 0.0,
            "C/20": 0.0,
            "C/10": 0.02,
            "C/5": 0.02,
            "C/3": 0.02,
        },
    },
    "EOL": {
        "standard_bol": {
            "Cn_Si": 1.234015,
            "Cn_Gr": 1.280356,
            "Cn": 2.514370,
            "Cp": 2.728159,
            "LLI": 2.397999,
            "transition_soc": 0.55775,
        },
        "standard_current": {
            "Cn_Si": 0.444368655,
            "Cn_Gr": 1.317862677,
            "Cn": 1.762231332,
            "Cp": 2.71626892,
            "LLI": 1.79191851,
            "transition_soc": 0.3836124234719959,
        },
        "fixed_a": 0.618179094,
        "fixed_b": 0.08,
        "file_paths": [
            CRATE_DATA_DIR / "GMFEB23S_CELL020_Cby100Cby20Cby10Cby5Cby3_20251222_R0_36_2_1_2818579680.xlsx",
        ],
        "R_with_kinetic": {
            "C/200": 0.0,
            "C/100": 0.0,
            "C/20": 0.0,
            "C/10": 0.18,
            "C/5": 0.096,
            "C/3": 0.0876,
        },
    },
}


if DATASET not in DATASETS:
    raise ValueError(f"DATASET must be one of {sorted(DATASETS)}")

DATASET_CFG = DATASETS[DATASET]
STANDARD_SET_BOL = DATASET_CFG["standard_bol"]
STANDARD_SET_Current = DATASET_CFG["standard_current"]
FIXED_A = DATASET_CFG["fixed_a"]
FIXED_B = DATASET_CFG["fixed_b"]
FILE_PATHS = DATASET_CFG["file_paths"]
R_by_label = (
    DATASET_CFG["R_with_kinetic"]
    if KINETIC_COMPENSATION
    else {label: 0.0 for label in DATASET_CFG["R_with_kinetic"]}
)

RUN_TAG = f"{DATASET.lower()}_{'with_kinetic_compensation' if KINETIC_COMPENSATION else 'without_kinetic_compensation'}"

Qdata_expand = np.linspace(-3, 3, 600)
QFIT_MINS = np.round(np.arange(0.0, 0.1001, 0.01), 2)
PARAM_COLS = ["Cn_Si", "Cn_Gr", "Cn", "Cp", "LLI"]
FIX_EPS = 1e-8

SI_RECONSTRUCTION_MODE = "reconstructed"
TRANSITION_THRESHOLD = 0.5
TRANSITION_PERSISTENCE_WINDOW = 0.15
TRANSITION_TOL = 0.01

CASES = [
    {
        "case_name": "deformation_enabled",
        "enable_deformation": True,
        "use_reconstructed_si_ocp": True,
        "fix_ab": False,
    },
    {
        "case_name": "deformation_disabled",
        "enable_deformation": False,
        "use_reconstructed_si_ocp": False,
        "fix_ab": False,
    },
]

if INCLUDE_FIXED_AB_CASE:
    CASES.append(
        {
            "case_name": "deformation_enabled_fixed_ab",
            "enable_deformation": True,
            "use_reconstructed_si_ocp": True,
            "fix_ab": True,
        }
    )


def build_base_fit_cfg():
    """Build the common voltage/dVdQ fit configuration used in each sweep case."""
    fit_cfg = VoltageFitConfig()

    fit_cfg.v_q_fit_frac_min = 0.0
    fit_cfg.v_q_fit_frac_max = 1.0
    fit_cfg.v_weight_inside = 1.0
    fit_cfg.v_weight_outside = 0.0

    fit_cfg.use_dvdq = True
    fit_cfg.w_dvdq = 0.1
    fit_cfg.dvdq_q_fit_frac_min = 0.0
    fit_cfg.dvdq_q_fit_frac_max = 1.0
    fit_cfg.dvdq_weight_inside = 1.0
    fit_cfg.dvdq_weight_outside = 0.0

    if hasattr(fit_cfg, "enable_si_deformation"):
        fit_cfg.enable_si_deformation = True
    if hasattr(fit_cfg, "si_deformation_model"):
        fit_cfg.si_deformation_model = "model_a"
    if hasattr(fit_cfg, "enable_si_drift"):
        fit_cfg.enable_si_drift = True

    fit_cfg.use_reconstructed_si_ocp = True
    fit_cfg.bounds = {
        "Cn_Si": (0.05, 1.6),
        "Cn_Gr": (0.5, 1.6),
        "x100": (0.5, 1.0),
        "Cp": (2.4, 2.8),
        "y100": (0.0, 0.12),
        "si_scale_a": (0.3, 1.1),
        "si_shift_b": (0.0, 0.08),
    }

    fit_cfg.optimizer = "de"
    fit_cfg.de_maxiter = 60
    fit_cfg.de_popsize = 10
    fit_cfg.de_seed = 42
    fit_cfg.de_workers = 1
    fit_cfg.do_lsq_polish_after_de = True
    fit_cfg.truncate_at_cv_onset = False
    fit_cfg.cv_voltage_cutoff_v = 4.195
    fit_cfg.dvdq_sg_window = 31
    return fit_cfg


def validate_study_inputs():
    """Check standards and input files before launching the expensive sweep."""
    for key in PARAM_COLS + ["transition_soc"]:
        if STANDARD_SET_BOL.get(key, None) is None:
            raise ValueError(f"STANDARD_SET_BOL[{key!r}] is missing.")
        if STANDARD_SET_Current.get(key, None) is None:
            raise ValueError(f"STANDARD_SET_Current[{key!r}] is missing.")

    missing = [path for path in FILE_PATHS if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing C-rate input files:\n" + "\n".join(str(p) for p in missing))


def clone_cfg(cfg):
    return copy.deepcopy(cfg)


def set_deformation_flags(cfg, enable_deformation, use_reconstructed_si_ocp):
    if hasattr(cfg, "enable_si_deformation"):
        cfg.enable_si_deformation = enable_deformation
    if hasattr(cfg, "enable_si_drift"):
        cfg.enable_si_drift = enable_deformation
    if hasattr(cfg, "si_deformation_model") and enable_deformation:
        cfg.si_deformation_model = "model_a"
    if hasattr(cfg, "use_reconstructed_si_ocp"):
        cfg.use_reconstructed_si_ocp = use_reconstructed_si_ocp
    return cfg


def set_qfit_window(cfg, qmin):
    cfg.v_q_fit_frac_min = float(qmin)
    cfg.dvdq_q_fit_frac_min = float(qmin)
    return cfg


def set_ab_behavior_by_bounds(cfg, fix_ab=False, fixed_a=FIXED_A, fixed_b=FIXED_B, eps=FIX_EPS):
    cfg.bounds = copy.deepcopy(cfg.bounds)
    if fix_ab:
        cfg.bounds["si_scale_a"] = (fixed_a - eps, fixed_a + eps)
        cfg.bounds["si_shift_b"] = (fixed_b - eps, fixed_b + eps)
    else:
        cfg.bounds["si_scale_a"] = (0.3, 1.1)
        cfg.bounds["si_shift_b"] = (0.0, 0.08)
    return cfg


def _crate_label_to_numeric(label):
    """Convert labels such as C/100 and C/3 to numeric C-rate values."""
    if pd.isna(label):
        return np.nan
    s = str(label).strip().upper()
    match = re.match(r"C\s*/\s*([0-9.]+)", s)
    if match:
        denom = float(match.group(1))
        return 1.0 / denom if denom != 0 else np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def ensure_required_columns(summary_df):
    summary_df = summary_df.copy()

    if "Cn" not in summary_df.columns and all(c in summary_df.columns for c in ["Cn_Si", "Cn_Gr"]):
        summary_df["Cn"] = summary_df["Cn_Si"] + summary_df["Cn_Gr"]

    rename_map = {}
    if "LLI%" in summary_df.columns and "LLI" not in summary_df.columns:
        rename_map["LLI%"] = "LLI"
    if "rmse_voltage_global" in summary_df.columns and "rmse_v_global" not in summary_df.columns:
        rename_map["rmse_voltage_global"] = "rmse_v_global"
    if "rmse_dVdQ_global" in summary_df.columns and "rmse_dvdq_global" not in summary_df.columns:
        rename_map["rmse_dVdQ_global"] = "rmse_dvdq_global"
    if rename_map:
        summary_df = summary_df.rename(columns=rename_map)

    if "crate_c" not in summary_df.columns and "crate_label" in summary_df.columns:
        summary_df["crate_c"] = summary_df["crate_label"].apply(_crate_label_to_numeric)

    return summary_df


def load_summary_from_run_dir(run_dir, fallback_results_df=None):
    """Load the structured batch summary, falling back to grouped results if needed."""
    candidate_paths = [
        Path(run_dir) / "structured_batch_esoh_summary.csv",
        Path(run_dir) / "crate_parameter_summary_by_crate.csv",
    ]

    for summary_csv in candidate_paths:
        if summary_csv.exists():
            return ensure_required_columns(pd.read_csv(summary_csv))

    if fallback_results_df is None:
        raise FileNotFoundError(f"Could not find a summary CSV in {run_dir}.")

    df = fallback_results_df.copy()
    for col in ["crate_c", "crate_label"]:
        if col not in df.columns:
            raise KeyError(f"results_df is missing required column: {col}")

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != "crate_c"]
    grouped = df.groupby(["crate_c", "crate_label"], as_index=False)[numeric_cols].mean(numeric_only=True)
    counts = df.groupby(["crate_c", "crate_label"], as_index=False).size().rename(columns={"size": "n_segments"})
    return ensure_required_columns(grouped.merge(counts, on=["crate_c", "crate_label"], how="left"))


def compute_standard_comparison(summary_df):
    """Compare C-rate estimates with the C/100-like standard for six metrics."""
    summary_df = ensure_required_columns(summary_df).copy()

    missing = [col for col in PARAM_COLS + ["transition_soc"] if col not in summary_df.columns]
    if missing:
        raise KeyError(f"summary_df missing required columns: {missing}")

    pct_rows = []
    mae_rows = []

    for _, row in summary_df.iterrows():
        out_row = {
            "crate_c": row.get("crate_c", np.nan),
            "crate_label": row["crate_label"],
            "reference_crate_label": "C/100_standard",
        }

        abs_errors_for_mae = []
        for param in PARAM_COLS:
            pct_change = (row[param] - STANDARD_SET_Current[param]) / STANDARD_SET_BOL[param] * 100.0
            abs_pct = abs(pct_change)
            out_row[f"{param}_pct_vs_C100_standard"] = pct_change
            out_row[f"{param}_abs_pct_vs_C100_standard"] = abs_pct
            abs_errors_for_mae.append(abs_pct)

        transition_error_pct_soc = abs(row["transition_soc"] - STANDARD_SET_Current["transition_soc"]) * 100.0
        out_row["transition_soc_abs_error_pct_soc_vs_standard"] = transition_error_pct_soc
        abs_errors_for_mae.append(transition_error_pct_soc)
        pct_rows.append(out_row)

        mae_rows.append(
            {
                "crate_c": row.get("crate_c", np.nan),
                "crate_label": row["crate_label"],
                "reference_crate_label": "C/100_standard",
                "mean_absolute_error_6metrics_vs_C100_standard": float(np.mean(abs_errors_for_mae)),
            }
        )

    return pd.DataFrame(pct_rows), pd.DataFrame(mae_rows)


def attach_case_metadata(df, case_name, q_fit_min):
    out = df.copy()
    out["case_name"] = case_name
    out["q_fit_min"] = q_fit_min
    out["dataset"] = DATASET
    out["kinetic_compensation"] = bool(KINETIC_COMPENSATION)
    return out


def safe_select_columns(df, desired_cols):
    return df[[col for col in desired_cols if col in df.columns]]


def main():
    validate_study_inputs()

    base_fit_cfg = build_base_fit_cfg()
    study_root = PROJECT_DIR / "batch_results" / f"A04_initial_charge_discard_{RUN_TAG}"
    study_root.mkdir(parents=True, exist_ok=True)

    Gr_OCP_smooth = smooth_ocp_curve(
        Gr_OCP,
        x_col="sto",
        y_col="p",
        window_length=51,
        polyorder=3,
        enforce_monotone=True,
    )

    all_summary_list = []
    all_pct_list = []
    all_mae_list = []
    failures = []

    print("Dataset:", DATASET)
    print("Kinetic compensation:", KINETIC_COMPENSATION)
    print("Study output folder:", study_root)

    for case in CASES:
        case_name = case["case_name"]

        for qmin in QFIT_MINS:
            print("=" * 110)
            print(f"Running {case_name}, discarded initial charge fraction = {qmin:.2f}")

            cfg_i = clone_cfg(base_fit_cfg)
            cfg_i = set_deformation_flags(
                cfg_i,
                enable_deformation=case["enable_deformation"],
                use_reconstructed_si_ocp=case["use_reconstructed_si_ocp"],
            )
            cfg_i = set_qfit_window(cfg_i, qmin=qmin)
            cfg_i = set_ab_behavior_by_bounds(
                cfg_i,
                fix_ab=case["fix_ab"],
                fixed_a=FIXED_A,
                fixed_b=FIXED_B,
                eps=FIX_EPS,
            )

            run_name = f"{case_name}_discard_initial_{qmin:.2f}".replace(".", "p")

            try:
                results_df, out_dir = run_structured_batch_esoh(
                    file_paths=FILE_PATHS,
                    project_dir=study_root,
                    fit_cfg=cfg_i,
                    R_by_label=R_by_label,
                    nominal_capacity_ah=2.5,
                    run_name=run_name,
                    charge_threshold_a=0.01,
                    min_segment_points=300,
                    cc_only=True,
                    cc_rel_tol=0.05,
                    save_plots=True,
                    default_R_ohm=0.0,
                    parallel=True,
                    n_jobs=4,
                )

                out_dir = Path(out_dir)
                summary_df = load_summary_from_run_dir(out_dir, fallback_results_df=results_df)
                summary_df = append_transition_soc_to_summary_df(
                    summary_df,
                    Qdata_expand=Qdata_expand,
                    ocp_3=ocp_3,
                    smooth_then_grad=smooth_then_grad,
                    build_effective_si_ocp=build_effective_si_ocp,
                    Gr_OCP_smooth=Gr_OCP_smooth,
                    SI_RECONSTRUCTION_MODE=SI_RECONSTRUCTION_MODE,
                    win=11,
                    poly=3,
                    threshold=TRANSITION_THRESHOLD,
                    persistence_window=TRANSITION_PERSISTENCE_WINDOW,
                    tol=TRANSITION_TOL,
                )
                summary_df = attach_case_metadata(summary_df, case_name, qmin)

                pct_df, mae_df = compute_standard_comparison(summary_df)
                pct_df = attach_case_metadata(pct_df, case_name, qmin)
                mae_df = attach_case_metadata(mae_df, case_name, qmin)

                summary_with_mae = summary_df.merge(
                    mae_df[
                        [
                            "crate_c",
                            "crate_label",
                            "case_name",
                            "q_fit_min",
                            "mean_absolute_error_6metrics_vs_C100_standard",
                        ]
                    ],
                    on=["crate_c", "crate_label", "case_name", "q_fit_min"],
                    how="left",
                )

                summary_with_mae.to_csv(
                    out_dir / "crate_parameter_summary_by_crate_with_standard_mae_6metrics.csv",
                    index=False,
                )
                pct_df.to_csv(out_dir / "crate_vs_C100_standard_percent_change_6metrics.csv", index=False)
                mae_df.to_csv(out_dir / "crate_vs_C100_standard_mae_6metrics.csv", index=False)

                compact_cols = [
                    "crate_c",
                    "crate_label",
                    "case_name",
                    "q_fit_min",
                    "dataset",
                    "kinetic_compensation",
                    "Cn_Si",
                    "Cn_Gr",
                    "Cn",
                    "Cp",
                    "LLI",
                    "transition_soc",
                    "rmse_v_global",
                    "rmse_dvdq_global",
                    "si_scale_a",
                    "si_shift_b",
                    "mean_absolute_error_6metrics_vs_C100_standard",
                    "transition_reason",
                    "transition_n_candidates",
                    "transition_n_valid",
                ]
                safe_select_columns(summary_with_mae, compact_cols).to_csv(
                    out_dir / "compact_summary_for_ranking_6metrics.csv",
                    index=False,
                )

                all_summary_list.append(summary_with_mae)
                all_pct_list.append(pct_df)
                all_mae_list.append(mae_df)
                print("Finished run:", out_dir)

            except Exception as exc:
                failures.append({"case_name": case_name, "q_fit_min": qmin, "error": repr(exc)})
                print(f"[FAILED] {case_name}, q_fit_min={qmin:.2f}: {exc}")

    if not all_summary_list:
        raise RuntimeError("No successful runs completed.")

    all_summary = pd.concat(all_summary_list, ignore_index=True)
    all_pct = pd.concat(all_pct_list, ignore_index=True)
    all_mae = pd.concat(all_mae_list, ignore_index=True)

    all_summary.to_csv(study_root / "ALL_summary_with_standard_mae_6metrics.csv", index=False)
    all_pct.to_csv(study_root / "ALL_pct_vs_C100_standard_6metrics.csv", index=False)
    all_mae.to_csv(study_root / "ALL_mae_vs_C100_standard_6metrics.csv", index=False)

    if failures:
        pd.DataFrame(failures).to_csv(study_root / "FAILED_runs.csv", index=False)

    rank_mae = all_summary.sort_values(
        ["mean_absolute_error_6metrics_vs_C100_standard", "rmse_v_global", "rmse_dvdq_global"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    rank_mae.to_csv(study_root / "RANK_overall_by_standard_mae_6metrics.csv", index=False)

    best_by_crate_mae = (
        all_summary.sort_values(
            ["crate_c", "mean_absolute_error_6metrics_vs_C100_standard", "rmse_v_global", "rmse_dvdq_global"],
            ascending=[True, True, True, True],
        )
        .groupby(["crate_c", "crate_label"], as_index=False)
        .first()
    )
    best_by_crate_mae.to_csv(study_root / "BEST_by_crate_standard_mae_6metrics.csv", index=False)

    print("\n================ OVERALL BEST BY STANDARD MAE (6 metrics) ================\n")
    print(
        safe_select_columns(
            rank_mae.head(30),
            [
                "crate_label",
                "case_name",
                "q_fit_min",
                "Cn_Si",
                "Cn_Gr",
                "Cn",
                "Cp",
                "LLI",
                "transition_soc",
                "rmse_v_global",
                "rmse_dvdq_global",
                "si_scale_a",
                "si_shift_b",
                "mean_absolute_error_6metrics_vs_C100_standard",
            ],
        ).to_string(index=False)
    )

    print("\nStudy output folder:", study_root)
    print("Saved files:")
    for path in sorted(study_root.glob("*")):
        print(" -", path.name)


if __name__ == "__main__":
    main()
