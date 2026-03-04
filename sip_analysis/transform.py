from __future__ import annotations

import numpy as np
import pandas as pd


def add_conductivity_columns(df: pd.DataFrame, geometric_factor: float) -> pd.DataFrame:
    if geometric_factor <= 0:
        raise ValueError("Geometric factor must be positive.")

    out = df.copy()
    base = geometric_factor / out["impedance_ohm"].astype(float)
    out["sigma_in_phase_uS_cm"] = base * np.cos(out["phase_rad"]) * 1e6
    out["sigma_quadrature_uS_cm"] = -base * np.sin(out["phase_rad"]) * 1e6
    out["minus_phase_mrad"] = -out["phase_rad"] * 1000.0
    out["geometric_factor"] = geometric_factor
    return out


def filter_frequency_range(df: pd.DataFrame, fmin: float | None, fmax: float | None) -> pd.DataFrame:
    out = df.copy()
    if fmin is not None:
        out = out[out["frequency_hz"] >= fmin]
    if fmax is not None:
        out = out[out["frequency_hz"] <= fmax]
    return out
