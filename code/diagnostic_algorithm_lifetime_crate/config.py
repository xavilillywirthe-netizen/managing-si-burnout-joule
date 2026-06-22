"""Configuration objects and default paths for voltage-based eSOH fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np


def _env_path(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default)).expanduser()


@dataclass(frozen=True)
class Paths:
    voltaiq_processed_root_type1: Path = _env_path(
        "VOLTAIQ_PROCESSED_ROOT_TYPE1",
        "./data/cell_lifetime_data/type1",
    )
    voltaiq_processed_root_type2: Path = _env_path(
        "VOLTAIQ_PROCESSED_ROOT_TYPE2",
        "./data/cell_lifetime_data/type2",
    )
    voltaiq_processed_root: Path = _env_path(
        "VOLTAIQ_PROCESSED_ROOT",
        "./data/cell_lifetime_data",
    )
    esoh_folder: Path = _env_path("ESOH_FOLDER", ".")
    cache_dir: Path = _env_path("VOLTAGE_ONLY_CACHE", "./cache_voltage_only")
    output_dir: Path = _env_path("VOLTAGE_ONLY_OUTPUT", "./out_voltage_only")


@dataclass
class VoltageFitConfig:
    q_fit_min: Optional[float] = None
    q_fit_max: Optional[float] = None

    q_fit_frac_min: Optional[float] = None
    q_fit_frac_max: Optional[float] = None

    weight_inside: float = 1.0
    weight_outside: float = 0.2

    cc_current_threshold_a: float = 0.099
    min_cc_points: int = 50
    voltage_min_v: float = 2.7
    voltage_max_v: float = 4.25

    truncate_at_cv_onset: bool = False
    cv_voltage_cutoff_v: Optional[float] = 4.195

    opt_grid_mode: str = "meas_cc"
    opt_dQ: float = 0.01
    Q_grid_min: float = -3.0
    Q_grid_max: float = 6.0
    Q_grid_n: int = 900

    voltage_early_frac: float = 0
    voltage_early_weight: float = 1
    voltage_weight_mode: str = "cosine"

    use_dvdq: bool = True
    w_voltage: float = 1.0
    w_dvdq: float = 0.1
    dvdq_sg_window: int = 31
    dvdq_sg_poly: int = 3

    v_q_fit_min: Optional[float] = None
    v_q_fit_max: Optional[float] = None
    v_q_fit_frac_min: float = 0
    v_q_fit_frac_max: float = 1
    v_weight_inside: float = 1.0
    v_weight_outside: float = 1.0

    dvdq_q_fit_min: Optional[float] = None
    dvdq_q_fit_max: Optional[float] = None
    dvdq_q_fit_frac_min: float = 0
    dvdq_q_fit_frac_max: float = 1
    dvdq_weight_inside: float = 1.0
    dvdq_weight_outside: float = 0

    use_peak_alignment: bool = False
    w_peak: float = 0.0
    peak_min_prominence: float = 0.03
    peak_regions: Tuple[Tuple[float, float], ...] = ((0, 0.3), (0.6, 1))
    peak_regions_are_relative: bool = True

    # Optional silicon-OCP drift term for sensitivity studies.
    enable_si_drift: bool = False
    si_drift_center: float = 0.5
    use_reconstructed_si_ocp = True

    use_prior: bool = False
    # Historical output-field aliases used by the code and saved tables:
    # x100 -> x_n,100; y100 -> x_p,100; si_scale_a -> s_V;
    # si_shift_b -> U_off.
    prior_sigma: Dict[str, float] = field(default_factory=lambda: {
        "Cn_Si": 0.05,
        "Cn_Gr": 0.2,
        "x100": 0.05,
        "Cp": 0.05,
        "y100": 0.02,
        "si_shift_v": 0.01,
        "si_tilt_v": 0.02,
    })

    bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Cn_Si": (0.05, 1.4),
        "Cn_Gr": (0.5, 1.6),
        "x100": (0.7, 1.0),
        "Cp": (2.6, 2.8),
        "y100": (0.0, 0.12),
        "si_scale_a": (0.3, 1.1),
        "si_shift_b": (0, 0.08),
    })

    optimizer: str = "de"

    de_maxiter: int = 60
    de_popsize: int = 10
    de_seed: int = 42
    de_polish: bool = False
    de_tol: float = 0.01
    de_workers: int = 1

    lsq_method: str = "trf"
    lsq_loss: str = "soft_l1"
    lsq_f_scale: float = 1.0

    do_lsq_polish_after_de: bool = True

    qcc_max_suspect_ah: float = 10.0

    def make_Q_grid(self) -> np.ndarray:
        return np.linspace(self.Q_grid_min, self.Q_grid_max, self.Q_grid_n)


PATHS = Paths()
