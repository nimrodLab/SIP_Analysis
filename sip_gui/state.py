from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class AppState:
    datasets: dict[str, pd.DataFrame] = field(default_factory=dict)
    combined_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    mean_std_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    fit_results: list[dict[str, object]] = field(default_factory=list)
    fit_paths: list[str] = field(default_factory=list)
