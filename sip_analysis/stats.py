from __future__ import annotations

import numpy as np
import pandas as pd


def compute_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("No data available for mean/std calculation.")
    work = df.copy()
    work = work[np.isfinite(work["frequency_hz"]) & (work["frequency_hz"] > 0)].copy()
    if work.empty:
        raise ValueError("No positive frequency points available for mean/std calculation.")

    metric_cols = ["sigma_in_phase_uS_cm", "sigma_quadrature_uS_cm", "minus_phase_mrad"]
    freq_grid = np.array(sorted(work["frequency_hz"].astype(float).map(lambda v: float(f"{v:.8g}")).unique()))
    log_grid = np.log10(freq_grid)

    by_file: dict[str, pd.DataFrame] = {}
    for file_name, g in work.groupby("file", sort=True):
        gg = g.groupby("frequency_hz", as_index=False)[metric_cols].mean().sort_values("frequency_hz")
        x = gg["frequency_hz"].to_numpy(dtype=float)
        if len(x) < 2:
            continue
        xlog = np.log10(x)
        if np.any(np.diff(xlog) <= 0):
            continue
        by_file[file_name] = gg

    if not by_file:
        raise ValueError("Need at least one file with 2+ valid frequency points for mean/std.")

    interpolated: dict[str, np.ndarray] = {}
    for col in metric_cols:
        mat = np.full((len(by_file), len(freq_grid)), np.nan, dtype=float)
        for i, gg in enumerate(by_file.values()):
            xlog = np.log10(gg["frequency_hz"].to_numpy(dtype=float))
            y = gg[col].to_numpy(dtype=float)
            mat[i, :] = np.interp(log_grid, xlog, y, left=np.nan, right=np.nan)
        interpolated[col] = mat

    n = np.sum(~np.isnan(interpolated["sigma_in_phase_uS_cm"]), axis=0)
    min_required = 2 if len(by_file) > 1 else 1
    keep = n >= min_required
    if not np.any(keep):
        raise ValueError("No overlapping frequency points across selected files in range.")
    out = pd.DataFrame({"frequency_hz": freq_grid[keep], "n": n[keep].astype(int)})

    for col, mean_name, std_name in [
        ("sigma_in_phase_uS_cm", "sigma_in_phase_mean_uS_cm", "sigma_in_phase_std_uS_cm"),
        ("sigma_quadrature_uS_cm", "sigma_quadrature_mean_uS_cm", "sigma_quadrature_std_uS_cm"),
        ("minus_phase_mrad", "minus_phase_mean_mrad", "minus_phase_std_mrad"),
    ]:
        mat = interpolated[col][:, keep]
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0, ddof=1)
        std[out["n"].to_numpy() <= 1] = 0.0
        std = np.nan_to_num(std, nan=0.0)
        out[mean_name] = mean
        out[std_name] = std

    return out.sort_values("frequency_hz").reset_index(drop=True)


def interpolate_log_curve(x: np.ndarray, y: np.ndarray, points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return x, y
    xlog = np.log10(x)
    if np.any(np.diff(xlog) <= 0):
        return x, y
    dense_x = np.logspace(xlog[0], xlog[-1], points)
    dense_y = np.interp(np.log10(dense_x), xlog, y)
    return dense_x, dense_y


def peak_from_curve(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or not np.isfinite(y).any():
        raise ValueError("Cannot find peak on empty/non-finite curve.")
    idx = int(np.nanargmax(y))
    f_peak = float(x[idx])
    sigma_peak = float(y[idx])
    tau_peak = 1.0 / (2.0 * np.pi * f_peak)
    return f_peak, sigma_peak, tau_peak
