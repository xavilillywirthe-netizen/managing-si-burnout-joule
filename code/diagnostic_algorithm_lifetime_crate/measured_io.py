"""Read processed lifetime-cell data and assemble diagnostic voltage traces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import gzip
import hashlib
import pickle

import numpy as np
import pandas as pd

from .config import PATHS
from .schemas import SCHEMA_TYPE1, SCHEMA_TYPE2, schema_id_for_cell


@dataclass
class VoltageTrace:
    rpt: int
    Q_Ah: np.ndarray
    V_V: np.ndarray
    t_hr: np.ndarray
    I_A: Optional[np.ndarray] = None
    Ah_throughput: Optional[float] = None
    test_name: Optional[str] = None
    direction: Optional[str] = None


def _read_pkl_gz(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _unwrap_df(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        for k in ("data", "df", "CD", "cd", "table"):
            v = obj.get(k, None)
            if isinstance(v, pd.DataFrame):
                return v
        for v in obj.values():
            if isinstance(v, pd.DataFrame):
                return v
    raise TypeError(f"Expected DataFrame or dict containing DataFrame, got {type(obj)}")


def _get_schema(schema_id: str) -> dict:
    if schema_id == "type1":
        return SCHEMA_TYPE1
    if schema_id == "type2":
        return SCHEMA_TYPE2
    raise ValueError(f"Unknown schema_id={schema_id}")


def _choose_schema_id(cell: int, schema: str) -> str:
    if schema in ("type1", "type2"):
        return schema
    sid = schema_id_for_cell(cell)
    return sid if sid in ("type1", "type2") else "type2"


def _choose_root(schema_id: str, voltaiq_root: Optional[Path]) -> Path:
    if voltaiq_root is not None:
        return Path(voltaiq_root)
    return PATHS.voltaiq_processed_root_type1 if schema_id == "type1" else PATHS.voltaiq_processed_root_type2


def _cell_dir_candidates(root: Path, cell_no: str) -> List[Path]:
    return [
        root / f"{root.name}_CELL{cell_no}",
        root / f"GMFEB23S_CELL{cell_no}",
        root / f"GMFEB23s_CELL{cell_no}",
        root / f"CELL{cell_no}",
        root / cell_no,
    ]


def _resolve_cell_dir(root: Path, cell: int) -> Path:
    cell_no = f"{cell:03d}"
    for d in _cell_dir_candidates(root, cell_no):
        if d.exists():
            return d
    hits = [p for p in Path(root).glob(f"*{cell_no}*") if p.is_dir()]
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(f"Cannot locate CELL{cell_no} under root={root}")


def _pick_col(df: pd.DataFrame, candidates: List[str], *, required: bool = True) -> Optional[str]:
    for c in candidates:
        if c and c in df.columns:
            return c
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c and c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise KeyError(f"Missing any of {candidates}. Available cols: {list(df.columns)[:40]}")
    return None


def _to_datetime_series(s: pd.Series) -> pd.Series:
    s = pd.Series(s)
    if pd.api.types.is_datetime64_any_dtype(s.dtype) or pd.api.types.is_datetime64tz_dtype(s.dtype):
        out = s.copy()
        if not pd.api.types.is_datetime64tz_dtype(out.dtype):
            out = out.dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
        return out
    if pd.api.types.is_numeric_dtype(s.dtype):
        x = pd.to_numeric(s, errors="coerce")
        if np.nanmedian(x) > 1e11:
            return pd.to_datetime(x, unit="ms", utc=True, errors="coerce")
        return pd.to_datetime(x, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce")


def _to_epoch_ms(dt_series: pd.Series) -> np.ndarray:
    dt = _to_datetime_series(dt_series)
    ns = dt.view("int64")
    ms = (ns // 1_000_000).to_numpy(dtype="int64")
    ms[~np.isfinite(ns.to_numpy(dtype=float))] = -1
    return ms


def _ensure_sorted_with_timekey(df: pd.DataFrame, *, time_mode: str, time_col: str) -> pd.DataFrame:
    if time_col not in df.columns:
        time_col = _pick_col(df, [time_col], required=True)
    out = df.copy()
    if time_mode == "ms_col":
        out["_t_ms_"] = pd.to_numeric(out[time_col], errors="coerce").to_numpy(dtype=float)
    elif time_mode == "timestamp":
        out["_t_ms_"] = _to_epoch_ms(out[time_col]).astype(float)
    else:
        raise ValueError("time_mode must be 'ms_col' or 'timestamp'")
    out = out[np.isfinite(out["_t_ms_"]) & (out["_t_ms_"] >= 0)].sort_values("_t_ms_").reset_index(drop=True)
    return out


def _slice_by_time_sorted(df_sorted: pd.DataFrame, start_ms: float, end_ms: float) -> pd.DataFrame:
    t = df_sorted["_t_ms_"].to_numpy(dtype=float)
    i0 = int(np.searchsorted(t, start_ms, side="left"))
    i1 = int(np.searchsorted(t, end_ms, side="right"))
    return df_sorted.iloc[i0:i1].reset_index(drop=True)


def _event_type_aliases(value: str) -> set[str]:
    value = str(value)
    aliases = {value}
    if value.startswith("_"):
        aliases.add(value[1:])
    else:
        aliases.add(f"_{value}")
    return aliases


def _prepare_cd_table(cd: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, str, Optional[str], Optional[str]]:
    time_col = _pick_col(cd, [schema["cd_time_col"]], required=True)
    v_col = _pick_col(cd, [schema["cd_v_col"]], required=True)
    i_col = _pick_col(cd, [schema["cd_i_col"]], required=False)
    q_col = _pick_col(cd, [schema["cd_ah_col"]], required=False)

    keep_cols = [time_col, v_col]
    if i_col is not None:
        keep_cols.append(i_col)
    if q_col is not None:
        keep_cols.append(q_col)
    keep_cols = list(dict.fromkeys(keep_cols))

    cd_small = cd.loc[:, keep_cols].copy()
    cd_small = _ensure_sorted_with_timekey(cd_small, time_mode=schema["cd_time_mode"], time_col=time_col)
    return cd_small, v_col, i_col, q_col


def _build_event_table(cell_dir: Path, schema: dict) -> pd.DataFrame:
    ccm = _unwrap_df(_read_pkl_gz(cell_dir / schema["ccm_filename"]))
    ccm = _ensure_sorted_with_timekey(
        ccm,
        time_mode=schema["ccm_time_mode"],
        time_col=schema["ccm_time_col"],
    )

    type_col = _pick_col(ccm, [schema["ccm_test_type_col"]], required=True)
    prot_col = _pick_col(ccm, [schema["ccm_protocol_col"]], required=False)
    name_col = _pick_col(ccm, [schema["ccm_test_name_col"]], required=False)
    ah_col = _pick_col(ccm, [schema["ccm_ahthr_col"]], required=False)

    cyc_val = str(schema["ccm_test_type_value"])
    rpt_val = str(schema.get("ccm_rpt_type_value", "RPT"))
    boundary_types = _event_type_aliases(cyc_val) | _event_type_aliases(rpt_val) | {"_F", "F"}
    rpt_types = _event_type_aliases(rpt_val)

    ccm_evt = ccm[ccm[type_col].astype(str).isin(boundary_types)].copy().reset_index(drop=True)
    if len(ccm_evt) < 2:
        return pd.DataFrame(columns=["rpt", "start_ms", "end_ms", "protocol", "direction", "test_name", "AhThr"])

    if name_col is not None and name_col in ccm_evt.columns:
        ccm_evt = ccm_evt.drop_duplicates(subset=[name_col, "_t_ms_"], keep="first").reset_index(drop=True)
    else:
        ccm_evt = ccm_evt.drop_duplicates(subset=["_t_ms_"], keep="first").reset_index(drop=True)

    rows = []
    rpt_seq = 0
    for i in range(len(ccm_evt) - 1):
        start = float(ccm_evt.loc[i, "_t_ms_"])
        end = float(ccm_evt.loc[i + 1, "_t_ms_"])
        if end <= start:
            continue

        row_type = str(ccm_evt.loc[i, type_col])
        if row_type not in rpt_types:
            continue

        protocol = str(ccm_evt.loc[i, prot_col]) if (prot_col is not None and prot_col in ccm_evt.columns) else ""
        p = protocol.lower()
        direction = None
        if "charge" in p and "discharge" not in p:
            direction = "charge"
        elif "discharge" in p:
            direction = "discharge"

        test_name = str(ccm_evt.loc[i, name_col]) if (name_col is not None and name_col in ccm_evt.columns) else None
        ahthr = float(ccm_evt.loc[i, ah_col]) if (ah_col is not None and ah_col in ccm_evt.columns and pd.notna(ccm_evt.loc[i, ah_col])) else np.nan

        rows.append({
            "rpt": rpt_seq,
            "start_ms": start,
            "end_ms": end,
            "protocol": protocol,
            "direction": direction,
            "test_name": test_name,
            "AhThr": ahthr,
        })
        rpt_seq += 1

    return pd.DataFrame(rows)

# cache
def _cache_key(cell: int, schema_id: str, root: Path, direction: str, protocol_keyword: Optional[str]) -> str:
    payload = f"cell={cell:03d}|schema={schema_id}|root={str(root.resolve())}|dir={direction}|protocol={protocol_keyword}"
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]
    safe_protocol = "all" if not protocol_keyword else str(protocol_keyword).replace(" ", "_").replace("/", "_")
    return f"cell{cell:03d}_{schema_id}_{direction}_{safe_protocol}_{digest}.pkl"


def load_processed_voltage(
    cell: int,
    *,
    schema: str = "auto",
    voltaiq_root: Optional[Path] = None,
    direction: str = "charge",
    protocol_keyword: Optional[str] = "C/20",
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> Dict[int, VoltageTrace]:
    schema_id = _choose_schema_id(cell, schema)
    sch = _get_schema(schema_id)
    root = _choose_root(schema_id, voltaiq_root)

    if cache_dir is None:
        cache_dir = PATHS.cache_dir
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _cache_key(cell, schema_id, root, direction, protocol_keyword)

    if use_cache and cache_path.exists() and not refresh_cache:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    cell_dir = _resolve_cell_dir(root, cell)
    events = _build_event_table(cell_dir, sch)
    if protocol_keyword:
        events = events[events["protocol"].astype(str).str.contains(protocol_keyword, case=False, na=False)]
    if direction:
        events = events[events["direction"] == direction]
    if "test_name" in events.columns:
        events = events.drop_duplicates(subset=["test_name", "protocol", "direction"], keep="first")
    events = events.reset_index(drop=True)

    cd = _unwrap_df(_read_pkl_gz(cell_dir / sch["cd_filename"]))
    cd, v_col, i_col, q_col = _prepare_cd_table(cd, sch)

    out: Dict[int, VoltageTrace] = {}
    for _, row in events.iterrows():
        seg = _slice_by_time_sorted(cd, float(row.start_ms), float(row.end_ms))
        if len(seg) < 10:
            continue

        V = pd.to_numeric(seg[v_col], errors="coerce").to_numpy(dtype=float)
        I = pd.to_numeric(seg[i_col], errors="coerce").to_numpy(dtype=float) if i_col else None
        t_hr = (seg["_t_ms_"].to_numpy(dtype=float) - float(seg["_t_ms_"].iloc[0])) / 1000.0 / 3600.0

        if q_col is not None:
            Q = pd.to_numeric(seg[q_col], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(Q).any():
                Q = Q - float(Q[np.where(np.isfinite(Q))[0][0]])
            else:
                Q = np.full_like(V, np.nan)
        elif I is not None:
            dt_hr = np.diff(t_hr)
            Q = np.concatenate([[0.0], np.cumsum(0.5 * (np.abs(I[1:]) + np.abs(I[:-1])) * dt_hr)])
        else:
            continue

        out[int(row.rpt)] = VoltageTrace(
            rpt=int(row.rpt),
            Q_Ah=Q,
            V_V=V,
            t_hr=t_hr,
            I_A=I,
            Ah_throughput=float(row.AhThr) if np.isfinite(row.AhThr) else None,
            test_name=row.test_name,
            direction=row.direction,
        )

    with open(cache_path, "wb") as f:
        pickle.dump(out, f)
    return out
