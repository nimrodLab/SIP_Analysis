from __future__ import annotations

import numpy as np
import pandas as pd

from sip_analysis.transform import add_conductivity_columns


def test_add_conductivity_columns_basic():
    df = pd.DataFrame(
        {
            "impedance_ohm": [100.0, 200.0],
            "phase_rad": [0.0, np.pi / 2],
        }
    )
    out = add_conductivity_columns(df, geometric_factor=1.0)
    assert "sigma_in_phase_uS_cm" in out.columns
    assert "sigma_quadrature_uS_cm" in out.columns
    assert np.isclose(out.loc[0, "sigma_in_phase_uS_cm"], 10000.0)
    assert np.isclose(out.loc[1, "sigma_in_phase_uS_cm"], 0.0, atol=1e-9)


def test_add_conductivity_columns_rejects_nonpositive_gf():
    df = pd.DataFrame({"impedance_ohm": [100.0], "phase_rad": [0.0]})
    try:
        add_conductivity_columns(df, geometric_factor=0.0)
        assert False
    except ValueError:
        assert True
