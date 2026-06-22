"""Align material OCP references onto common potential grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class OCPAlignmentMeta:
    method: str
    source_p_col: str
    source_sto_col: str
    ref_p_col: str
    ref_sto_col: str
    allow_extrapolation: bool
    n_source: int
    n_ref: int
    n_aligned: int
    p_min_source: float
    p_max_source: float
    p_min_ref: float
    p_max_ref: float
    n_nan_outside_overlap: int


def _clean_ocp_df(df: pd.DataFrame, *, p_col: str = "p", sto_col: str = "sto") -> pd.DataFrame:
    out = df[[p_col, sto_col]].copy()
    out[p_col] = pd.to_numeric(out[p_col], errors="coerce")
    out[sto_col] = pd.to_numeric(out[sto_col], errors="coerce")
    out = out.dropna(subset=[p_col, sto_col]).sort_values(p_col).reset_index(drop=True)

    if len(out) >= 2:
        keep = np.r_[True, np.diff(out[p_col].to_numpy(float)) > 1e-12]
        out = out.loc[keep].reset_index(drop=True)
    return out


def _interp_on_target_p(
    source_p: np.ndarray,
    source_sto: np.ndarray,
    target_p: np.ndarray,
    *,
    allow_extrapolation: bool = False,
) -> np.ndarray:
    source_p = np.asarray(source_p, float).reshape(-1)
    source_sto = np.asarray(source_sto, float).reshape(-1)
    target_p = np.asarray(target_p, float).reshape(-1)

    if source_p.size < 2:
        return np.full_like(target_p, np.nan, dtype=float)

    if allow_extrapolation:
        return np.interp(target_p, source_p, source_sto)

    out = np.interp(target_p, source_p, source_sto)

    lo = float(np.min(source_p))
    hi = float(np.max(source_p))

    left_mask = target_p < lo
    right_mask = target_p > hi

    if np.any(left_mask):
        out[left_mask] = float(source_sto[0])
    if np.any(right_mask):
        out[right_mask] = float(source_sto[-1])

    return out


def align_ocp_to_reference_p_axis(
    source_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    *,
    source_p_col: str = "p",
    source_sto_col: str = "sto",
    ref_p_col: str = "p",
    ref_sto_col: str = "sto",
    allow_extrapolation: bool = False,
) -> Tuple[pd.DataFrame, OCPAlignmentMeta]:
    src = _clean_ocp_df(source_df, p_col=source_p_col, sto_col=source_sto_col)
    ref = _clean_ocp_df(ref_df, p_col=ref_p_col, sto_col=ref_sto_col)

    source_p = src[source_p_col].to_numpy(float)
    source_sto = src[source_sto_col].to_numpy(float)
    ref_p = ref[ref_p_col].to_numpy(float)

    aligned_sto = _interp_on_target_p(
        source_p,
        source_sto,
        ref_p,
        allow_extrapolation=allow_extrapolation,
    )

    aligned_df = pd.DataFrame({
        "p": ref_p,
        "sto": aligned_sto,
    })

    meta = OCPAlignmentMeta(
        method="align_ocp_to_reference_p_axis",
        source_p_col=source_p_col,
        source_sto_col=source_sto_col,
        ref_p_col=ref_p_col,
        ref_sto_col=ref_sto_col,
        allow_extrapolation=bool(allow_extrapolation),
        n_source=int(len(src)),
        n_ref=int(len(ref)),
        n_aligned=int(len(aligned_df)),
        p_min_source=float(np.min(source_p)),
        p_max_source=float(np.max(source_p)),
        p_min_ref=float(np.min(ref_p)),
        p_max_ref=float(np.max(ref_p)),
        n_nan_outside_overlap=int(np.sum(~np.isfinite(aligned_sto))),
    )
    return aligned_df, meta

def align_ocp_to_common_p_axis(
    source_df: pd.DataFrame,
    *,
    p_common: np.ndarray,
    source_p_col: str = "p",
    source_sto_col: str = "sto",
    allow_extrapolation: bool = False,
) -> Tuple[pd.DataFrame, OCPAlignmentMeta]:
    src = _clean_ocp_df(source_df, p_col=source_p_col, sto_col=source_sto_col)

    source_p = src[source_p_col].to_numpy(float)
    source_sto = src[source_sto_col].to_numpy(float)
    p_common = np.asarray(p_common, float)

    aligned_sto = _interp_on_target_p(
        source_p,
        source_sto,
        p_common,
        allow_extrapolation=allow_extrapolation,
    )

    aligned_df = pd.DataFrame({
        "p": p_common,
        "sto": aligned_sto,
    })

    meta = OCPAlignmentMeta(
        method="align_ocp_to_common_p_axis",
        source_p_col=source_p_col,
        source_sto_col=source_sto_col,
        ref_p_col="p_common",
        ref_sto_col="",
        allow_extrapolation=bool(allow_extrapolation),
        n_source=int(len(src)),
        n_ref=int(len(p_common)),
        n_aligned=int(len(aligned_df)),
        p_min_source=float(np.min(source_p)),
        p_max_source=float(np.max(source_p)),
        p_min_ref=float(np.min(p_common)),
        p_max_ref=float(np.max(p_common)),
        n_nan_outside_overlap=int(np.sum(~np.isfinite(aligned_sto))),
    )
    return aligned_df, meta


def build_composite_anode_ocp_from_aligned_components(
    si_aligned_df: pd.DataFrame,
    gr_df: pd.DataFrame,
    *,
    alpha: float,
    p_col: str = "p",
    sto_col: str = "sto",
) -> pd.DataFrame:
    si = _clean_ocp_df(si_aligned_df, p_col=p_col, sto_col=sto_col)
    gr = _clean_ocp_df(gr_df, p_col=p_col, sto_col=sto_col)

    p_si = si[p_col].to_numpy(float)
    p_gr = gr[p_col].to_numpy(float)

    if len(p_si) != len(p_gr) or not np.allclose(p_si, p_gr, rtol=0, atol=1e-12):
        raise ValueError("si_aligned_df and gr_df must already share the same p axis.")

    x_si = si[sto_col].to_numpy(float)
    x_gr = gr[sto_col].to_numpy(float)

    x_mix = float(alpha) * x_si + (1.0 - float(alpha)) * x_gr
    return pd.DataFrame({"p": p_gr, "sto": x_mix})


def summarize_alignment(meta: OCPAlignmentMeta) -> str:
    return (
        f"method={meta.method}, "
        f"source_p=[{meta.p_min_source:.4f}, {meta.p_max_source:.4f}], "
        f"ref_p=[{meta.p_min_ref:.4f}, {meta.p_max_ref:.4f}], "
        f"n_nan_outside_overlap={meta.n_nan_outside_overlap}"
    )
