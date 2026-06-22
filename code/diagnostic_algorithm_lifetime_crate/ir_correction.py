"""Apply ohmic kinetic compensation to measured charge-voltage traces."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Optional

import numpy as np


def apply_ir_correction(V_V: np.ndarray, I_A: np.ndarray, R_ohm: float) -> np.ndarray:
    V = np.asarray(V_V, float)
    I = np.asarray(I_A, float)
    return V - I * float(R_ohm)


def get_R_from_label(crate_label: str, R_by_label: Mapping[str, float], default_R_ohm: Optional[float] = None) -> float:
    if crate_label in R_by_label:
        return float(R_by_label[crate_label])
    if default_R_ohm is not None:
        return float(default_R_ohm)
    raise KeyError(f"No R provided for crate label '{crate_label}'")


def apply_ir_correction_to_trace(trace, *, R_ohm: float):
    corrected_V = apply_ir_correction(trace.V_V, trace.I_A, R_ohm)
    cls = trace.__class__
    new_trace = cls(
        rpt=getattr(trace, "rpt", -1),
        Q_Ah=np.asarray(trace.Q_Ah, float).copy(),
        V_V=np.asarray(corrected_V, float),
        I_A=np.asarray(trace.I_A, float).copy() if getattr(trace, "I_A", None) is not None else None,
        t_hr=np.asarray(trace.t_hr, float).copy() if getattr(trace, "t_hr", None) is not None else None,
        Ah_throughput=float(getattr(trace, "Ah_throughput", np.nan)),
        test_name=getattr(trace, "test_name", None),
        crate_label=getattr(trace, "crate_label", None),
        file=getattr(trace, "file", None),
        segment_id=getattr(trace, "segment_id", None),
        I_mean_A=float(getattr(trace, "I_mean_A", np.nan)),
    )
    setattr(new_trace, "V_V_raw", np.asarray(trace.V_V, float).copy())
    setattr(new_trace, "V_V_corr", np.asarray(corrected_V, float).copy())
    setattr(new_trace, "R_used_ohm", float(R_ohm))
    return new_trace
