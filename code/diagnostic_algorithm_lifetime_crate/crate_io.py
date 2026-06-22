"""Load C-rate Excel files and extract constant-current charge segments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import hashlib
import os
import pickle

import numpy as np
import pandas as pd


@dataclass
class SegmentTrace:
    rpt: int
    Q_Ah: np.ndarray
    V_V: np.ndarray
    I_A: np.ndarray
    t_hr: np.ndarray
    Ah_throughput: float
    test_name: Optional[str] = None
    crate_label: Optional[str] = None
    file: Optional[str] = None
    segment_id: Optional[int] = None
    I_mean_A: Optional[float] = None


def _make_cache_key(
    file_path: str | Path,
    *,
    nominal_capacity_ah: float,
    charge_threshold_a: float,
    min_segment_points: int,
    cc_only: bool,
    cc_rel_tol: float,
    resample_on_q: bool,
    n_resample: Optional[int],
) -> str:
    fp = Path(file_path).expanduser().resolve()
    stat = fp.stat()
    raw = "|".join(
        [
            str(fp),
            str(stat.st_mtime_ns),
            str(stat.st_size),
            f"nomcap={nominal_capacity_ah}",
            f"th={charge_threshold_a}",
            f"minpts={min_segment_points}",
            f"cconly={cc_only}",
            f"cctol={cc_rel_tol}",
            f"resample={resample_on_q}",
            f"nres={n_resample}",
        ]
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _cache_path_for_file(
    file_path: str | Path,
    *,
    cache_dir: str | Path,
    nominal_capacity_ah: float,
    charge_threshold_a: float,
    min_segment_points: int,
    cc_only: bool,
    cc_rel_tol: float,
    resample_on_q: bool,
    n_resample: Optional[int],
) -> Path:
    fp = Path(file_path).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _make_cache_key(
        fp,
        nominal_capacity_ah=nominal_capacity_ah,
        charge_threshold_a=charge_threshold_a,
        min_segment_points=min_segment_points,
        cc_only=cc_only,
        cc_rel_tol=cc_rel_tol,
        resample_on_q=resample_on_q,
        n_resample=n_resample,
    )
    return cache_dir / f"{fp.stem}_{key}_segments.pkl"


def load_and_clean_record_xlsx(file_path: str | Path, sheet_name: str = "record") -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("（", "(", regex=False)
        .str.replace("）", ")", regex=False)
        .str.replace("\uFF08", "(", regex=False)
        .str.replace("\uFF09", ")", regex=False)
    )

    rename_map = {}
    if "Current(A)" not in df.columns:
        for c in df.columns:
            if c.lower().replace(" ", "") in {"current(a)", "current"}:
                rename_map[c] = "Current(A)"
    if "Voltage(V)" not in df.columns:
        for c in df.columns:
            if c.lower().replace(" ", "") in {"voltage(v)", "voltage"}:
                rename_map[c] = "Voltage(V)"
    if "Total Time" not in df.columns:
        for c in df.columns:
            if c.lower().replace(" ", "") in {"totaltime"}:
                rename_map[c] = "Total Time"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = ["Current(A)", "Voltage(V)", "Total Time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    df["Current(A)"] = pd.to_numeric(df["Current(A)"], errors="coerce")
    df["Voltage(V)"] = pd.to_numeric(df["Voltage(V)"], errors="coerce")
    df["Total Time"] = pd.to_timedelta(df["Total Time"])
    df["Time_s"] = df["Total Time"].dt.total_seconds()

    return df.dropna(subset=["Current(A)", "Voltage(V)", "Time_s"]).reset_index(drop=True)


def extract_charge_segments(df: pd.DataFrame, threshold_a: float = 0.01, min_points: int = 300) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    in_charge = False
    start_idx = None

    I = df["Current(A)"].to_numpy(float)
    for i in range(len(df)):
        if (I[i] > threshold_a) and (not in_charge):
            start_idx = i
            in_charge = True
        elif in_charge and (I[i] < threshold_a):
            end_idx = i - 1
            if start_idx is not None and (end_idx - start_idx + 1) >= min_points:
                segments.append((start_idx, end_idx))
            in_charge = False
            start_idx = None

    if in_charge and start_idx is not None:
        end_idx = len(df) - 1
        if (end_idx - start_idx + 1) >= min_points:
            segments.append((start_idx, end_idx))
    return segments


def classify_crate(I_mean_A: float, nominal_capacity_ah: float = 2.5, tol_frac: float = 0.18) -> str:
    if nominal_capacity_ah <= 0:
        return f"{abs(I_mean_A):.3f}A"
    crate = abs(I_mean_A) / float(nominal_capacity_ah)

    known = [
        (1/200, "C/200"),
        (1/100, "C/100"),
        (1/50, "C/50"),
        (1/20, "C/20"),
        (1/10, "C/10"),
        (1/5, "C/5"),
        (1/3, "C/3"),
        (1/2, "C/2"),
        (1.0, "1C"),
    ]
    for val, label in known:
        if abs(crate - val) <= tol_frac * max(val, 1e-12):
            return label
    return f"{crate:.3f}C"


def trim_cc_portion(seg_df: pd.DataFrame, rel_tol: float = 0.05, min_points: int = 20) -> pd.DataFrame:
    seg = seg_df.copy().reset_index(drop=True)
    if len(seg) < min_points:
        return seg

    I0 = float(seg["Current(A)"].iloc[0])
    if abs(I0) < 1e-12:
        return seg

    cc_mask = np.abs(seg["Current(A)"].to_numpy(float) - I0) < (rel_tol * abs(I0))
    cc_idx = np.where(cc_mask)[0]
    if len(cc_idx) < min_points:
        return seg
    return seg.iloc[: cc_idx[-1] + 1].reset_index(drop=True)


def _resample_segment_on_q(
    t_s: np.ndarray,
    I_A: np.ndarray,
    V_V: np.ndarray,
    *,
    n_resample: int,
):
    t_s = np.asarray(t_s, float)
    I_A = np.asarray(I_A, float)
    V_V = np.asarray(V_V, float)

    if len(t_s) < 3 or n_resample <= 0 or len(t_s) <= n_resample:
        dt = np.gradient(t_s)
        Q_Ah = np.cumsum(I_A * dt) / 3600.0
        if len(Q_Ah):
            Q_Ah = Q_Ah - float(Q_Ah[0])
        return t_s, I_A, V_V, Q_Ah

    dt = np.gradient(t_s)
    Q_Ah = np.cumsum(I_A * dt) / 3600.0
    Q_Ah = Q_Ah - float(Q_Ah[0])

    keep = np.r_[True, np.diff(Q_Ah) > 1e-12]
    q_u = Q_Ah[keep]
    v_u = V_V[keep]
    i_u = I_A[keep]
    t_u = t_s[keep]

    if len(q_u) < 3:
        return t_s, I_A, V_V, Q_Ah

    q_new = np.linspace(float(q_u[0]), float(q_u[-1]), int(n_resample))
    v_new = np.interp(q_new, q_u, v_u)
    i_new = np.interp(q_new, q_u, i_u)
    t_new = np.interp(q_new, q_u, t_u)
    return t_new, i_new, v_new, q_new


def _build_segment_trace(
    seg: pd.DataFrame,
    *,
    file: str,
    segment_id: int,
    nominal_capacity_ah: float,
    resample_on_q: bool,
    n_resample: Optional[int],
) -> SegmentTrace:
    out = seg.copy().reset_index(drop=True)
    t_s = out["Time_s"].to_numpy(float)
    I_A = out["Current(A)"].to_numpy(float)
    V_V = out["Voltage(V)"].to_numpy(float)

    if resample_on_q:
        t_s, I_A, V_V, Q_Ah = _resample_segment_on_q(
            t_s, I_A, V_V, n_resample=int(n_resample or 800)
        )
    else:
        dt = np.gradient(t_s)
        Q_Ah = np.cumsum(I_A * dt) / 3600.0
        if len(Q_Ah):
            Q_Ah = Q_Ah - float(Q_Ah[0])

    I_mean = float(np.nanmean(I_A[: min(100, len(I_A))]))
    crate_label = classify_crate(I_mean, nominal_capacity_ah=nominal_capacity_ah)

    return SegmentTrace(
        rpt=int(segment_id),
        Q_Ah=np.asarray(Q_Ah, float),
        V_V=np.asarray(V_V, float),
        I_A=np.asarray(I_A, float),
        t_hr=np.asarray((t_s - t_s[0]) / 3600.0, float),
        Ah_throughput=float(Q_Ah[-1]) if len(Q_Ah) else np.nan,
        test_name=f"{Path(file).name}::segment_{segment_id}",
        crate_label=crate_label,
        file=os.path.basename(file),
        segment_id=int(segment_id),
        I_mean_A=I_mean,
    )


def structure_segments_from_file(
    file_path: str | Path,
    *,
    nominal_capacity_ah: float = 2.5,
    charge_threshold_a: float = 0.01,
    min_segment_points: int = 300,
    cc_only: bool = True,
    cc_rel_tol: float = 0.05,
    resample_on_q: bool = True,
    n_resample: Optional[int] = 800,
    cache_dir: str | Path | None = None,
    refresh_cache: bool = False,
) -> List[SegmentTrace]:
    cache_path = None
    if cache_dir is not None:
        cache_path = _cache_path_for_file(
            file_path,
            cache_dir=cache_dir,
            nominal_capacity_ah=nominal_capacity_ah,
            charge_threshold_a=charge_threshold_a,
            min_segment_points=min_segment_points,
            cc_only=cc_only,
            cc_rel_tol=cc_rel_tol,
            resample_on_q=resample_on_q,
            n_resample=n_resample,
        )
        if cache_path.exists() and not refresh_cache:
            with open(cache_path, "rb") as f:
                out = pickle.load(f)
            print(f"[cache hit] {Path(file_path).name} -> {cache_path.name}")
            return out

    df = load_and_clean_record_xlsx(file_path)
    segments = extract_charge_segments(df, threshold_a=charge_threshold_a, min_points=min_segment_points)

    out: List[SegmentTrace] = []
    for seg_id, (start, end) in enumerate(segments):
        seg = df.iloc[start : end + 1].copy()
        if cc_only:
            seg = trim_cc_portion(seg, rel_tol=cc_rel_tol)
        out.append(
            _build_segment_trace(
                seg,
                file=str(file_path),
                segment_id=seg_id,
                nominal_capacity_ah=nominal_capacity_ah,
                resample_on_q=resample_on_q,
                n_resample=n_resample,
            )
        )

    if cache_path is not None:
        with open(cache_path, "wb") as f:
            pickle.dump(out, f)
        print(f"[cache save] {Path(file_path).name} -> {cache_path.name}")

    return out


def structure_segments_from_files(
    file_paths: List[str | Path],
    *,
    nominal_capacity_ah: float = 2.5,
    charge_threshold_a: float = 0.01,
    min_segment_points: int = 300,
    cc_only: bool = True,
    cc_rel_tol: float = 0.05,
    resample_on_q: bool = True,
    n_resample: Optional[int] = 800,
    cache_dir: str | Path | None = None,
    refresh_cache: bool = False,
) -> List[SegmentTrace]:
    all_segments: List[SegmentTrace] = []
    for fp in file_paths:
        all_segments.extend(
            structure_segments_from_file(
                fp,
                nominal_capacity_ah=nominal_capacity_ah,
                charge_threshold_a=charge_threshold_a,
                min_segment_points=min_segment_points,
                cc_only=cc_only,
                cc_rel_tol=cc_rel_tol,
                resample_on_q=resample_on_q,
                n_resample=n_resample,
                cache_dir=cache_dir,
                refresh_cache=refresh_cache,
            )
        )
    return all_segments
