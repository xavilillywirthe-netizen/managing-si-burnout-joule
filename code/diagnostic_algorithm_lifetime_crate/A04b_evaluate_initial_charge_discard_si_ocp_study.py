"""Evaluate the discarded-initial-charge and silicon-OCP-treatment study.

This script reads the aggregate CSV produced by
A04a_run_initial_charge_discard_si_ocp_study.py and generates tables/figures
that compare deformation-enabled and deformation-disabled silicon OCP treatment
as a function of the discarded initial charge fraction.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_repo_root(start: Path) -> Path:
    """Return the repository root that contains the shared code and data folders."""
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "code" / "diagnostic_algorithm_lifetime_crate").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the repository root from this script.")


REPO_ROOT = find_repo_root(Path(__file__))
SCRIPT_DIR = REPO_ROOT / "code" / "diagnostic_algorithm_lifetime_crate"

# Match these settings to the A04a run you want to evaluate.
DATASET = "BOL"
KINETIC_COMPENSATION = True

RUN_TAG = f"{DATASET.lower()}_{'with_kinetic_compensation' if KINETIC_COMPENSATION else 'without_kinetic_compensation'}"
STUDY_DIR = SCRIPT_DIR / "batch_results" / f"A04_initial_charge_discard_{RUN_TAG}"
INPUT_CSV = STUDY_DIR / "ALL_summary_with_standard_mae_6metrics.csv"
OUTPUT_DIR = STUDY_DIR / "analysis_outputs_2case_compare"

PRIMARY_METRIC = "mean_absolute_error_6metrics_vs_C100_standard"
SECONDARY_METRICS = ["rmse_v_global", "rmse_dvdq_global"]

GROUP_COL = "crate_label"
CASE_COL = "case_name"
SWEEP_COL = "q_fit_min"

PARAM_COLS = [
    "Cn_Si",
    "Cn_Gr",
    "Cn",
    "Cp",
    "LLI",
    "transition_soc",
    # Internal output aliases for the manuscript's s_V and U_off parameters.
    "si_scale_a",
    "si_shift_b",
]

CASE_ORDER = ["deformation_enabled", "deformation_disabled"]
CASE_DISPLAY_NAME = {
    "deformation_enabled": "Enable Si OCP deformation",
    "deformation_disabled": "Disable Si OCP deformation",
}
GROUP_ORDER = ["C/100", "C/20", "C/10", "C/5", "C/3"]

TOP_K = 30
TITLE_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 14
LINEWIDTH = 2.5
MARKERSIZE = 7


def safe_select(df, cols):
    return df[[col for col in cols if col in df.columns]]


def require_columns(df, cols):
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def save_table(df, name):
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def save_fig(name):
    path = OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def rename_case_for_display(series):
    return series.map(lambda value: CASE_DISPLAY_NAME.get(value, value))


def step0_filter_cases(df):
    """Keep the two silicon-OCP treatments used in the paper comparison."""
    print("\n" + "=" * 80)
    print("STEP 0: FILTER TO TWO SILICON-OCP TREATMENTS")
    print("=" * 80)

    require_columns(df, [CASE_COL])
    df2 = df[df[CASE_COL].isin(CASE_ORDER)].copy()
    if df2.empty:
        raise ValueError(
            f"No rows found for CASE_ORDER={CASE_ORDER}. "
            f"Available cases: {sorted(df[CASE_COL].dropna().unique().tolist())}"
        )

    df2[CASE_COL] = pd.Categorical(df2[CASE_COL], categories=CASE_ORDER, ordered=True)
    df2 = df2.sort_values([GROUP_COL, CASE_COL, SWEEP_COL]).reset_index(drop=True)
    save_table(df2, "step0_filtered_two_cases.csv")
    return df2


def step1_validate(df):
    """Validate metrics and save basic descriptive statistics."""
    print("\n" + "=" * 80)
    print("STEP 1: DEFINE PRIMARY AND SECONDARY METRICS")
    print("=" * 80)

    required = [GROUP_COL, CASE_COL, SWEEP_COL, PRIMARY_METRIC] + SECONDARY_METRICS
    require_columns(df, required)

    metric_cols = [PRIMARY_METRIC] + SECONDARY_METRICS
    desc = df[metric_cols].describe().T
    save_table(desc.reset_index().rename(columns={"index": "metric"}), "step1_metric_descriptives_2case.csv")
    return desc


def step2_best_tables(df):
    """Save ranked tables for the best configurations overall and by C-rate."""
    print("\n" + "=" * 80)
    print("STEP 2: GENERATE BEST-RESULT TABLES")
    print("=" * 80)

    sort_cols = [PRIMARY_METRIC] + SECONDARY_METRICS
    overall_best = df.sort_values(sort_cols, ascending=[True] * len(sort_cols)).reset_index(drop=True)
    save_table(overall_best, "step2_overall_rank_2case.csv")

    best_by_group_case = (
        df.sort_values([GROUP_COL, CASE_COL] + sort_cols, ascending=[True, True] + [True] * len(sort_cols))
        .groupby([GROUP_COL, CASE_COL], as_index=False)
        .first()
    )
    save_table(best_by_group_case, "step2_best_by_group_and_case_2case.csv")

    best_by_group = (
        df.sort_values([GROUP_COL] + sort_cols, ascending=[True] + [True] * len(sort_cols))
        .groupby(GROUP_COL, as_index=False)
        .first()
    )
    save_table(best_by_group, "step2_best_by_group_2case.csv")

    print("\nTop overall rows:")
    print(
        safe_select(
            overall_best.head(TOP_K),
            [GROUP_COL, CASE_COL, SWEEP_COL, PRIMARY_METRIC] + SECONDARY_METRICS + PARAM_COLS,
        ).to_string(index=False)
    )
    return overall_best, best_by_group_case, best_by_group


def step3_case_comparison(df):
    """Compare deformation-enabled and deformation-disabled cases by C-rate."""
    print("\n" + "=" * 80)
    print("STEP 3: DIRECT TWO-CASE COMPARISON")
    print("=" * 80)

    summary = (
        df.groupby([GROUP_COL, CASE_COL], as_index=False)
        .agg(
            n_rows=(PRIMARY_METRIC, "count"),
            primary_mean=(PRIMARY_METRIC, "mean"),
            primary_std=(PRIMARY_METRIC, "std"),
            rmse_v_mean=("rmse_v_global", "mean"),
            rmse_dvdq_mean=("rmse_dvdq_global", "mean"),
        )
    )

    summary[CASE_COL] = summary[CASE_COL].astype(str)
    summary["case_display"] = rename_case_for_display(summary[CASE_COL])
    save_table(summary, "step3_summary_by_group_case_2case.csv")

    pivot_primary = summary.pivot(index=GROUP_COL, columns=CASE_COL, values="primary_mean")
    compare = pd.DataFrame(index=pivot_primary.index)
    compare.index.name = GROUP_COL

    enabled, disabled = CASE_ORDER
    if enabled in pivot_primary.columns and disabled in pivot_primary.columns:
        compare[f"{enabled}_primary_mean"] = pivot_primary[enabled]
        compare[f"{disabled}_primary_mean"] = pivot_primary[disabled]
        compare["delta_primary_mean_(disabled-enabled)"] = pivot_primary[disabled] - pivot_primary[enabled]

    compare = compare.reset_index()
    save_table(compare, "step3_direct_case_comparison_by_group_2case.csv")
    print(compare.to_string(index=False))
    return summary, compare


def step4_visualizations(df):
    """Plot MAE versus discarded initial charge data for each C-rate."""
    print("\n" + "=" * 80)
    print("STEP 4: VISUALIZE TWO-CASE COMPARISON")
    print("=" * 80)

    groups_present = df[GROUP_COL].dropna().unique().tolist()
    groups = [group for group in GROUP_ORDER if group in groups_present] or sorted(groups_present)

    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(9.2, max(7.5, 2.8 * n)), sharex=True, sharey=False)
    if n == 1:
        axes = [axes]

    for ax, group in zip(axes, groups):
        sub = df[df[GROUP_COL] == group].copy()
        for case_name in CASE_ORDER:
            ss = sub[sub[CASE_COL].astype(str) == case_name].sort_values(SWEEP_COL)
            if ss.empty:
                continue
            ax.plot(
                100 * ss[SWEEP_COL],
                ss[PRIMARY_METRIC],
                marker="o",
                markersize=MARKERSIZE,
                linewidth=LINEWIDTH,
                label=CASE_DISPLAY_NAME.get(case_name, case_name),
            )
        ax.set_title(group, fontsize=TITLE_FONTSIZE, pad=10)
        ax.set_ylabel("MAE (%)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=LEGEND_FONTSIZE, loc="best", frameon=True)

    axes[-1].set_xlabel("Discarded initial charge data (%)", fontsize=AXIS_LABEL_FONTSIZE)
    fig.suptitle(
        f"{DATASET}: {'with' if KINETIC_COMPENSATION else 'without'} kinetic compensation",
        fontsize=TITLE_FONTSIZE + 2,
        y=0.995,
    )
    save_fig("step4_primary_metric_lineplots_2case.png")

    best_rows = (
        df.sort_values([GROUP_COL, CASE_COL, PRIMARY_METRIC], ascending=[True, True, True])
        .groupby([GROUP_COL, CASE_COL], as_index=False)
        .first()
    )
    best_rows[CASE_COL] = best_rows[CASE_COL].astype(str)
    best_rows["case_display"] = rename_case_for_display(best_rows[CASE_COL])

    pivot = best_rows.pivot(index=GROUP_COL, columns="case_display", values=PRIMARY_METRIC).reindex(groups)
    plt.figure(figsize=(9.2, 5.6))
    pivot.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Best MAE (%)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.xlabel("C-rate", fontsize=AXIS_LABEL_FONTSIZE)
    plt.title("Best MAE by C-rate and silicon-OCP treatment", fontsize=TITLE_FONTSIZE)
    plt.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=0, fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    save_fig("step4_best_primary_bar_by_group_2case.png")


def step5_writing_support(df, best_by_group_case, compare_table):
    """Save concise notes that support manuscript figure interpretation."""
    print("\n" + "=" * 80)
    print("STEP 5: GENERATE WRITING SUPPORT TABLES")
    print("=" * 80)

    case_summary = (
        df.groupby(CASE_COL, as_index=False)
        .agg(
            n_rows=(PRIMARY_METRIC, "count"),
            primary_mean=(PRIMARY_METRIC, "mean"),
            primary_std=(PRIMARY_METRIC, "std"),
            primary_min=(PRIMARY_METRIC, "min"),
            primary_max=(PRIMARY_METRIC, "max"),
            rmse_v_mean=("rmse_v_global", "mean"),
            rmse_dvdq_mean=("rmse_dvdq_global", "mean"),
        )
        .sort_values("primary_mean", ascending=True)
    )
    case_summary["case_display"] = rename_case_for_display(case_summary[CASE_COL].astype(str))
    save_table(case_summary, "step5_case_summary_overall_2case.csv")

    notes = [
        "Discarded initial charge and silicon-OCP treatment comparison",
        "=============================================================",
        f"Dataset: {DATASET}",
        f"Kinetic compensation: {KINETIC_COMPENSATION}",
        f"Primary metric: {PRIMARY_METRIC}",
        "",
        "Overall case summary:",
    ]
    for _, row in case_summary.iterrows():
        notes.append(
            f"- {row['case_display']}: mean MAE={row['primary_mean']:.4g}, "
            f"std={row['primary_std']:.4g}, mean rmse_v={row['rmse_v_mean']:.4g}, "
            f"mean rmse_dvdq={row['rmse_dvdq_mean']:.4g}"
        )

    notes.append("")
    notes.append("Best configuration by C-rate and case:")
    for _, row in best_by_group_case.iterrows():
        notes.append(
            f"- {row.get(GROUP_COL, 'NA')} / "
            f"{CASE_DISPLAY_NAME.get(str(row.get(CASE_COL, 'NA')), row.get(CASE_COL, 'NA'))}: "
            f"discarded={100 * row.get(SWEEP_COL, np.nan):.0f}%, "
            f"MAE={row.get(PRIMARY_METRIC, np.nan):.4g}"
        )

    notes.append("")
    notes.append("Direct group-level comparison:")
    for _, row in compare_table.iterrows():
        delta_primary = row.get("delta_primary_mean_(disabled-enabled)", np.nan)
        notes.append(
            f"- {row[GROUP_COL]}: disabled-enabled mean MAE delta = {delta_primary:.4g}"
            if pd.notna(delta_primary)
            else f"- {row[GROUP_COL]}: comparison unavailable"
        )

    notes_path = OUTPUT_DIR / "step5_analysis_notes_2case.txt"
    notes_path.write_text("\n".join(notes), encoding="utf-8")
    print(f"Saved: {notes_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Could not find input CSV: {INPUT_CSV}\n"
            "Run A04a first with matching DATASET and KINETIC_COMPENSATION settings."
        )

    df = pd.read_csv(INPUT_CSV)
    df = step0_filter_cases(df)
    step1_validate(df)
    overall_best, best_by_group_case, best_by_group = step2_best_tables(df)
    summary_by_group_case, compare_table = step3_case_comparison(df)
    step4_visualizations(df)
    step5_writing_support(df, best_by_group_case, compare_table)

    print("\nDone.")
    print(f"All outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
