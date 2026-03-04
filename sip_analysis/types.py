from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlotConfig:
    geometric_factor: float = 1.0
    fmin_hz: float | None = None
    fmax_hz: float | None = None
    show_mean_std: bool = False
    smoothing_points: int = 300


@dataclass(frozen=True)
class FitConfig:
    model_name: str
    geometric_factor: float = 1.0
    fmin_hz: float | None = None
    fmax_hz: float | None = None
    uncertainty_mode: str = "None"
    n_samples: int = 80
    ci_level: float = 95.0
    random_seed: int = 42


@dataclass(frozen=True)
class PeakResult:
    name: str
    f_peak_hz: float
    sigma_q_peak_uS_cm: float
    tau_peak_s: float


@dataclass
class FitResult:
    name: str
    model: str
    params_summary: str
    rmse: float
    f_peak_hz: float
    sigma_q_peak_uS_cm: float
    tau_peak_s: float
    extra: dict[str, Any] = field(default_factory=dict)
