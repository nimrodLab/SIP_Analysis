from __future__ import annotations

import numpy as np
import pandas as pd

from sip_analysis.stats import compute_mean_std, interpolate_log_curve, peak_from_curve


def test_compute_mean_std_interpolates_and_aggregates():
    df = pd.DataFrame(
        {
            "file": ["a", "a", "b", "b"],
            "frequency_hz": [1.0, 10.0, 1.0, 10.0],
            "sigma_in_phase_uS_cm": [10.0, 30.0, 14.0, 34.0],
            "sigma_quadrature_uS_cm": [2.0, 6.0, 4.0, 8.0],
            "minus_phase_mrad": [1.0, 3.0, 2.0, 4.0],
        }
    )
    out = compute_mean_std(df)
    assert len(out) == 2
    assert np.all(out["n"].to_numpy() == 2)
    assert np.isclose(out.loc[0, "sigma_in_phase_mean_uS_cm"], 12.0)


def test_interpolate_log_curve_returns_dense_monotonic_x():
    x = np.array([1.0, 10.0, 100.0])
    y = np.array([1.0, 3.0, 2.0])
    xx, yy = interpolate_log_curve(x, y, points=50)
    assert len(xx) == 50
    assert np.all(np.diff(xx) > 0)
    assert np.isfinite(yy).all()


def test_peak_from_curve_deterministic():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 5.0, 2.0])
    f_peak, s_peak, tau_peak = peak_from_curve(x, y)
    assert np.isclose(f_peak, 2.0)
    assert np.isclose(s_peak, 5.0)
    assert np.isclose(tau_peak, 1.0 / (2.0 * np.pi * 2.0))
