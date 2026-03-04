#!/usr/bin/env python3
"""Desktop GUI for SIP CSV analysis."""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sip_analysis import io as analysis_io
from sip_analysis import models as analysis_models
from sip_analysis import plotting as analysis_plotting
from sip_analysis import stats as analysis_stats
from sip_analysis import transform as analysis_transform
from sip_gui.state import AppState


COLORBLIND_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]


def _to_float(value: str) -> float:
    if value is None:
        raise ValueError("None is not a number")
    text = str(value).strip().replace("\ufeff", "")
    if not text:
        raise ValueError("empty string")

    if text.count(",") == 1 and text.count(".") == 0:
        text = text.replace(",", ".")
    elif text.count(",") > 1 and text.count(".") == 0:
        text = text.replace(",", "")
    else:
        text = text.replace(",", "")

    return float(text)


def _guess_delimiter(lines: list[str]) -> str:
    candidates = [",", ";", "\t"]
    scores = {d: 0 for d in candidates}
    for line in lines[:50]:
        for d in candidates:
            scores[d] += line.count(d)
    return max(scores, key=scores.get)


def _normalize_header(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _detect_timestamp(lines: list[str]) -> str | None:
    joined = "\n".join(lines[:120])

    patterns = [
        r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)",
        r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}[ T]\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)",
    ]
    for pat in patterns:
        match = re.search(pat, joined)
        if match:
            ts = _parse_timestamp_string(match.group(1))
            if ts:
                return ts

    date_match = re.search(r"(?:date|measurement\s*date)\s*[:=]\s*([^\n\r]+)", joined, re.I)
    time_match = re.search(r"(?:time|measurement\s*time)\s*[:=]\s*([^\n\r]+)", joined, re.I)
    if date_match and time_match:
        ts = _parse_timestamp_string(f"{date_match.group(1).strip()} {time_match.group(1).strip()}")
        if ts:
            return ts

    date_value = None
    time_value = None
    for line in lines[:80]:
        parts = [p.strip() for p in re.split(r"[,\t;]", line) if p.strip()]
        if len(parts) < 2:
            continue
        key = parts[0].lower()
        value = parts[1].strip()
        if date_value is None and ("date" == key or "measurement date" in key):
            date_value = value
        if time_value is None and ("time" == key or "measurement time" in key):
            time_value = value

    if date_value and time_value:
        ts = _parse_timestamp_string(f"{date_value} {time_value}")
        if ts:
            return ts

    return None


def _parse_timestamp_string(text: str) -> str | None:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return None

    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%y %H:%M:%S",
        "%m/%d/%y %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(text, fmt).isoformat(sep=" ")
        except ValueError:
            pass

    dt = pd.to_datetime(text, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        dt = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.notna(dt):
        return pd.Timestamp(dt).isoformat(sep=" ")
    return None


def _find_data_layout(rows: list[list[str]]) -> tuple[int, int, int, int, str]:
    header_idx = -1
    header_cells: list[str] = []

    for i, row in enumerate(rows[:120]):
        if len(row) < 3:
            continue
        norm = [_normalize_header(cell) for cell in row]

        has_freq = any("freq" in c for c in norm)
        has_phase = any("phase" in c or c == "phi" for c in norm)
        has_imp = any(("imp" in c) or ("z" in c and "hz" not in c) for c in norm)

        if has_freq and has_phase and has_imp:
            header_idx = i
            header_cells = row
            break

    if header_idx == -1:
        for i, row in enumerate(rows):
            if len(row) < 3:
                continue
            try:
                _to_float(row[0])
                _to_float(row[1])
                _to_float(row[2])
                return i, 0, 1, 2, "unknown"
            except ValueError:
                continue
        raise ValueError("Could not locate SIP data columns (frequency/impedance/phase).")

    norm = [_normalize_header(c) for c in header_cells]

    def pick_idx(predicate):
        for idx, cell in enumerate(norm):
            if predicate(cell):
                return idx
        return -1

    freq_idx = pick_idx(lambda c: "freq" in c)
    imp_idx = pick_idx(
        lambda c: (
            ("imp" in c)
            or c in {"z", "absz", "modz", "zmag", "impedanceohm"}
            or "zohm" in c
            or "zmod" in c
            or "absolutedance" in c
        )
        and "real" not in c
        and "imag" not in c
    )
    phase_idx = pick_idx(lambda c: "phase" in c or c == "phi")

    if min(freq_idx, imp_idx, phase_idx) < 0:
        raise ValueError("Found data header, but missing frequency/impedance/phase columns.")

    phase_header = norm[phase_idx]
    if "mrad" in phase_header:
        phase_unit = "mrad"
    elif "deg" in phase_header:
        phase_unit = "deg"
    elif "rad" in phase_header:
        phase_unit = "rad"
    else:
        phase_unit = "unknown"

    return header_idx + 1, freq_idx, imp_idx, phase_idx, phase_unit


def parse_sip_csv(path: str) -> pd.DataFrame:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        raise ValueError("File is empty.")

    timestamp = _detect_timestamp(lines)
    delimiter = _guess_delimiter(lines)
    rows = list(csv.reader(lines, delimiter=delimiter))

    start_row, freq_idx, imp_idx, phase_idx, phase_unit = _find_data_layout(rows)

    data = []
    for row in rows[start_row:]:
        max_idx = max(freq_idx, imp_idx, phase_idx)
        if len(row) <= max_idx:
            continue
        try:
            freq = _to_float(row[freq_idx])
            imp = _to_float(row[imp_idx])
            phase_raw = _to_float(row[phase_idx])
        except ValueError:
            continue

        if phase_unit == "deg":
            phase_rad = np.deg2rad(phase_raw)
        elif phase_unit == "mrad":
            phase_rad = phase_raw / 1000.0
        elif phase_unit == "rad":
            phase_rad = phase_raw
        else:
            phase_rad = np.deg2rad(phase_raw) if abs(phase_raw) > 3.2 else phase_raw

        data.append(
            {
                "file": os.path.basename(path),
                "filepath": str(path),
                "timestamp": timestamp,
                "frequency_hz": freq,
                "impedance_ohm": imp,
                "phase_rad": phase_rad,
            }
        )

    if not data:
        raise ValueError("No numeric SIP rows found in file.")

    return pd.DataFrame(data).sort_values("frequency_hz").reset_index(drop=True)


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


def compute_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("No data available for mean/std calculation.")
    work = df.copy()
    work = work[np.isfinite(work["frequency_hz"]) & (work["frequency_hz"] > 0)].copy()
    if work.empty:
        raise ValueError("No positive frequency points available for mean/std calculation.")

    metric_cols = ["sigma_in_phase_uS_cm", "sigma_quadrature_uS_cm", "minus_phase_mrad"]

    # Build one shared frequency grid (from selected files/range only), then
    # interpolate each file onto that grid in log-frequency space.
    freq_grid = np.array(sorted(work["frequency_hz"].astype(float).map(lambda v: float(f"{v:.8g}")).unique()))
    log_grid = np.log10(freq_grid)
    by_file: dict[str, pd.DataFrame] = {}
    for file_name, g in work.groupby("file", sort=True):
        gg = (
            g.groupby("frequency_hz", as_index=False)[metric_cols]
            .mean()
            .sort_values("frequency_hz")
        )
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
        raise ValueError("No overlapping frequency points across the selected files in the chosen range.")
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


class SipGuiApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SIP Data Analyzer")
        self.root.geometry("1380x860")

        self.state = AppState()
        self.datasets: dict[str, pd.DataFrame] = self.state.datasets
        self.legend_name_vars: dict[str, tk.StringVar] = {}
        self.combined_df = self.state.combined_df
        self.mean_std_df = self.state.mean_std_df
        self.last_fig = None
        self.compare_last_fig = None
        self.fit_last_fig = None
        self.fit_paths: list[str] = self.state.fit_paths
        self.fit_results: list[dict[str, object]] = self.state.fit_results
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._busy = False
        self._settings_path = Path(__file__).resolve().parent / "settings.json"
        self._settings = self._load_settings()
        self._last_dir = self._settings.get("last_dir", str(Path.home()))
        self._canvases: dict[str, FigureCanvasTkAgg] = {}
        self._toolbars: dict[str, NavigationToolbar2Tk] = {}
        self._figure_hosts: dict[str, ttk.Frame] = {}

        self._build_ui()
        self._build_menu()
        self.root.bind_all("<Control-o>", lambda _e: self.choose_files())
        self.root.bind_all("<Control-s>", lambda _e: self.save_plot_png())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_menu(self) -> None:
        menu = tk.Menu(self.root)
        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="Export Debug Bundle", command=self.export_debug_bundle)
        menu.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menu)

    def export_debug_bundle(self) -> None:
        out_dir = filedialog.askdirectory(title="Choose debug bundle destination", initialdir=self._last_dir)
        if not out_dir:
            return
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        bundle = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "settings": self._settings,
            "dataset_count": len(self.datasets),
            "fit_result_count": len(self.fit_results),
        }
        (out / "sip_debug_bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        try:
            if self._settings_path.exists():
                (out / "settings.json").write_text(self._settings_path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass
        messagebox.showinfo("Debug bundle", f"Exported debug bundle to:\n{out}")

    def _on_close(self) -> None:
        self._save_settings()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()

    def _load_settings(self) -> dict[str, object]:
        try:
            if self._settings_path.exists():
                return json.loads(self._settings_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {"geom_factor": "1.0", "fit_geom_factor": "1.0", "last_dir": str(Path.home())}

    def _save_settings(self) -> None:
        self._settings["geom_factor"] = self.gf_var.get().strip()
        self._settings["fit_geom_factor"] = self.fit_gf_var.get().strip()
        self._settings["last_dir"] = self._last_dir
        try:
            self._settings_path.write_text(json.dumps(self._settings, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _set_busy(self, busy: bool, text: str | None = None) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        for btn in [self.main_plot_btn, self.main_load_btn, self.fit_load_btn, self.fit_run_btn]:
            btn.config(state=state)
        self.progress.configure(mode="indeterminate")
        if busy:
            self.progress.start(8)
        else:
            self.progress.stop()
        if text:
            self.global_status_var.set(text)

    def _run_background(self, fn, on_done, busy_text: str) -> None:
        if self._busy:
            return
        self._set_busy(True, busy_text)
        fut = self._executor.submit(fn)

        def _check():
            if not fut.done():
                self.root.after(120, _check)
                return
            self._set_busy(False, "Ready.")
            try:
                result = fut.result()
            except Exception as exc:
                messagebox.showerror("Task failed", str(exc))
                return
            on_done(result)

        self.root.after(120, _check)

    def _build_ui(self) -> None:
        root_frame = ttk.Frame(self.root)
        root_frame.pack(fill=tk.BOTH, expand=True)
        notebook = ttk.Notebook(root_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.main_tab = ttk.Frame(notebook, padding=12)
        self.compare_tab = ttk.Frame(notebook, padding=12)
        self.fit_tab = ttk.Frame(notebook, padding=12)
        notebook.add(self.main_tab, text="Analyze SIP")
        notebook.add(self.compare_tab, text="Compare Mean DFs")
        notebook.add(self.fit_tab, text="Fit Models")

        self._build_main_tab()
        self._build_compare_tab()
        self._build_fit_tab()
        status_bar = ttk.Frame(root_frame)
        status_bar.pack(fill=tk.X, pady=(2, 0))
        self.global_status_var = tk.StringVar(value="Ready.")
        ttk.Label(status_bar, textvariable=self.global_status_var).pack(side=tk.LEFT, padx=(6, 8))
        self.progress = ttk.Progressbar(status_bar, mode="indeterminate", length=220)
        self.progress.pack(side=tk.RIGHT, padx=(8, 8), pady=(0, 2))

    def _build_main_tab(self) -> None:
        layout = ttk.Panedwindow(self.main_tab, orient=tk.HORIZONTAL)
        layout.pack(fill=tk.BOTH, expand=True)
        sidebar = ttk.Frame(layout, padding=(6, 6))
        content = ttk.Frame(layout, padding=(6, 6))
        layout.add(sidebar, weight=1)
        layout.add(content, weight=3)

        controls = ttk.LabelFrame(sidebar, text="Controls")
        controls.pack(fill=tk.X)
        self.main_load_btn = ttk.Button(controls, text="Load SIP File(s)", command=self.choose_files)
        self.main_load_btn.pack(fill=tk.X, pady=(6, 4))
        ttk.Label(controls, text="Geometric factor (1/cm):").pack(anchor=tk.W)
        self.gf_var = tk.StringVar(value=str(self._settings.get("geom_factor", "1.0")))
        ttk.Entry(controls, textvariable=self.gf_var, width=14).pack(fill=tk.X, pady=(0, 4))
        ttk.Label(controls, text="Min freq (Hz):").pack(anchor=tk.W)
        self.fmin_var = tk.StringVar(value="")
        ttk.Entry(controls, textvariable=self.fmin_var, width=14).pack(fill=tk.X, pady=(0, 4))
        ttk.Label(controls, text="Max freq (Hz):").pack(anchor=tk.W)
        self.fmax_var = tk.StringVar(value="")
        ttk.Entry(controls, textvariable=self.fmax_var, width=14).pack(fill=tk.X, pady=(0, 6))
        self.mean_std_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls,
            text="Mean ± std only",
            variable=self.mean_std_var,
            command=self._toggle_mean_save,
        ).pack(anchor=tk.W, pady=(0, 6))
        self.main_plot_btn = ttk.Button(controls, text="Plot", command=self.plot_data)
        self.main_plot_btn.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(controls, text="Save Plot PNG", command=self.save_plot_png).pack(fill=tk.X, pady=(0, 4))
        self.save_mean_btn = ttk.Button(
            controls, text="Save Mean±Std DF", command=self.save_mean_std_csv, state=tk.DISABLED
        )
        self.save_mean_btn.pack(fill=tk.X, pady=(0, 6))

        self.status_var = tk.StringVar(value="No files loaded.")
        ttk.Label(sidebar, textvariable=self.status_var, wraplength=300).pack(fill=tk.X, pady=(8, 4))

        legend_frame = ttk.LabelFrame(sidebar, text="Legend Labels")
        legend_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 4))
        self.legend_editor = ttk.Frame(legend_frame)
        self.legend_editor.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._figure_hosts["main"] = ttk.LabelFrame(content, text="Plot Preview")
        self._figure_hosts["main"].pack(fill=tk.BOTH, expand=True)

        table_frame = ttk.Frame(content)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        columns = ("file", "timestamp", "rows")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        for col, width in zip(columns, (260, 240, 80)):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width, anchor=tk.W)
        self.tree.pack(fill=tk.X, expand=False)

        peak_frame = ttk.LabelFrame(table_frame, text="Peak Summary (Selected Range)")
        peak_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        peak_cols = ("name", "f_peak_hz", "sigma_q_peak_uS_cm", "tau_peak_s")
        self.peak_tree = ttk.Treeview(peak_frame, columns=peak_cols, show="headings", height=6)
        self.peak_tree.heading("name", text="name")
        self.peak_tree.heading("f_peak_hz", text="f_peak (Hz)")
        self.peak_tree.heading("sigma_q_peak_uS_cm", text="sigma''_peak (uS/cm)")
        self.peak_tree.heading("tau_peak_s", text="tau_peak (s)")
        self.peak_tree.column("name", width=220, anchor=tk.W)
        self.peak_tree.column("f_peak_hz", width=140, anchor=tk.W)
        self.peak_tree.column("sigma_q_peak_uS_cm", width=200, anchor=tk.W)
        self.peak_tree.column("tau_peak_s", width=140, anchor=tk.W)
        self.peak_tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._refresh_legend_editor()

    def _build_compare_tab(self) -> None:
        top = ttk.Frame(self.compare_tab)
        top.pack(fill=tk.X)

        ttk.Button(top, text="Choose Mean DF CSV(s)", command=self.choose_mean_df_files).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(top, text="Plot Mean DF(s)", command=self.plot_mean_comparison).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(top, text="Save Mean Plot PNG", command=self.save_compare_png).pack(side=tk.LEFT)

        self.compare_status = tk.StringVar(value="No mean DF files selected.")
        ttk.Label(self.compare_tab, textvariable=self.compare_status).pack(fill=tk.X, pady=(10, 8))

        self.mean_df_paths: list[str] = []
        self.compare_list = tk.Listbox(self.compare_tab, selectmode=tk.EXTENDED, height=8)
        self.compare_list.pack(fill=tk.X, expand=False)
        self._figure_hosts["compare"] = ttk.LabelFrame(self.compare_tab, text="Plot Preview")
        self._figure_hosts["compare"].pack(fill=tk.BOTH, expand=True, pady=(8, 0))

    def _build_fit_tab(self) -> None:
        split = ttk.Panedwindow(self.fit_tab, orient=tk.HORIZONTAL)
        split.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(split)
        right = ttk.Frame(split)
        split.add(left, weight=2)
        split.add(right, weight=3)

        left_split = ttk.Panedwindow(left, orient=tk.VERTICAL)
        left_split.pack(fill=tk.BOTH, expand=True)
        top_left = ttk.Frame(left_split)
        bottom_left = ttk.LabelFrame(left_split, text="Fit Inputs")
        left_split.add(top_left, weight=7)
        left_split.add(bottom_left, weight=3)

        top = ttk.Frame(top_left)
        top.pack(fill=tk.X)
        self.fit_load_btn = ttk.Button(top, text="Choose File(s)", command=self.choose_fit_files)
        self.fit_load_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.fit_run_btn = ttk.Button(top, text="Fit Selected/All", command=self.fit_selected_datasets)
        self.fit_run_btn.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(top, text="Save Fit Plot PNG", command=self.save_fit_png).pack(side=tk.LEFT)
        ttk.Button(top, text="Save Fit Results CSV", command=self.save_fit_results_csv).pack(side=tk.LEFT, padx=(10, 0))

        model_row = ttk.Frame(top_left)
        model_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(model_row, text="Model:").pack(side=tk.LEFT)
        self.fit_model_var = tk.StringVar(value="Cole-Cole")
        model_combo = ttk.Combobox(
            model_row,
            textvariable=self.fit_model_var,
            state="readonly",
            width=30,
            values=[
                "Cole-Cole",
                "Double Cole-Cole",
                "Havriliak-Negami",
                "Double Havriliak-Negami",
                "Neural Network (experimental)",
                "CVNN (paper repo, local train)",
            ],
        )
        model_combo.pack(side=tk.LEFT, padx=(6, 0))

        filters = ttk.Frame(top_left)
        filters.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(filters, text="Geometric factor (1/cm):").pack(side=tk.LEFT)
        self.fit_gf_var = tk.StringVar(value=str(self._settings.get("fit_geom_factor", "1.0")))
        ttk.Entry(filters, textvariable=self.fit_gf_var, width=10).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(filters, text="Min freq (Hz):").pack(side=tk.LEFT)
        self.fit_fmin_var = tk.StringVar(value="")
        ttk.Entry(filters, textvariable=self.fit_fmin_var, width=10).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(filters, text="Max freq (Hz):").pack(side=tk.LEFT)
        self.fit_fmax_var = tk.StringVar(value="")
        ttk.Entry(filters, textvariable=self.fit_fmax_var, width=10).pack(side=tk.LEFT, padx=(6, 0))

        cvnn_cfg = ttk.Frame(top_left)
        cvnn_cfg.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(cvnn_cfg, text="CVNN train N:").pack(side=tk.LEFT)
        self.fit_cvnn_ntrain_var = tk.StringVar(value="3000")
        ttk.Entry(cvnn_cfg, textvariable=self.fit_cvnn_ntrain_var, width=8).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(cvnn_cfg, text="CVNN epochs:").pack(side=tk.LEFT)
        self.fit_cvnn_epochs_var = tk.StringVar(value="800")
        ttk.Entry(cvnn_cfg, textvariable=self.fit_cvnn_epochs_var, width=8).pack(side=tk.LEFT, padx=(6, 0))

        unc = ttk.Frame(top_left)
        unc.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(unc, text="Uncertainty:").pack(side=tk.LEFT)
        self.fit_uncertainty_var = tk.StringVar(value="None")
        ttk.Combobox(
            unc,
            textvariable=self.fit_uncertainty_var,
            state="readonly",
            width=24,
            values=["None", "Bootstrap CI", "MCMC (experimental)"],
        ).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(unc, text="N samples/steps:").pack(side=tk.LEFT)
        self.fit_unc_samples_var = tk.StringVar(value="80")
        ttk.Entry(unc, textvariable=self.fit_unc_samples_var, width=8).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(unc, text="CI (%):").pack(side=tk.LEFT)
        self.fit_ci_level_var = tk.StringVar(value="95")
        ttk.Entry(unc, textvariable=self.fit_ci_level_var, width=6).pack(side=tk.LEFT, padx=(6, 0))

        note = (
            "Neural-network option is experimental (paper-inspired settings). "
            "Please cite: https://doi.org/10.1093/gji/ggaf348 ; code/data DOI: https://doi.org/10.5281/zenodo.16950274 "
            "(paper CVNN mode requires PyTorch)."
        )
        ttk.Label(top_left, text=note, foreground="#444").pack(fill=tk.X, pady=(8, 8))

        self.fit_status_var = tk.StringVar(value="No files selected for fitting.")
        ttk.Label(top_left, textvariable=self.fit_status_var).pack(fill=tk.X, pady=(0, 8))

        bottom_left.rowconfigure(0, weight=1)
        bottom_left.columnconfigure(0, weight=1)
        self.fit_list = tk.Listbox(bottom_left, selectmode=tk.EXTENDED, height=8)
        fit_list_y = ttk.Scrollbar(bottom_left, orient=tk.VERTICAL, command=self.fit_list.yview)
        fit_list_x = ttk.Scrollbar(bottom_left, orient=tk.HORIZONTAL, command=self.fit_list.xview)
        self.fit_list.configure(yscrollcommand=fit_list_y.set, xscrollcommand=fit_list_x.set)
        self.fit_list.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=(6, 0))
        fit_list_y.grid(row=0, column=1, sticky="ns", padx=(2, 6), pady=(6, 0))
        fit_list_x.grid(row=1, column=0, sticky="ew", padx=(6, 0), pady=(2, 6))

        right_split = ttk.Panedwindow(right, orient=tk.VERTICAL)
        right_split.pack(fill=tk.BOTH, expand=True)

        plot_frame = ttk.LabelFrame(right_split, text="Plot Preview")
        right_split.add(plot_frame, weight=3)
        self._figure_hosts["fit"] = plot_frame
        self._set_figure_placeholder("fit", "Run fitting to display model and data curves.")

        results_frame = ttk.LabelFrame(right_split, text="Fit Summary")
        right_split.add(results_frame, weight=2)
        cols = (
            "name",
            "model",
            "sigma_dc_uS_cm",
            "sigma_inf_uS_cm",
            "rmse",
            "rmse_ci",
            "p_m1",
            "p_tau1_s",
            "p_c1",
            "p_beta1",
            "p_m2",
            "p_tau2_s",
            "p_c2",
            "p_beta2",
            "f_peak_hz",
            "f_peak_hz_ci",
            "sigma_q_peak_uS_cm",
            "sigma_q_peak_uS_cm_ci",
            "tau_peak_s",
            "tau_peak_s_ci",
        )
        self.fit_results_tree = ttk.Treeview(results_frame, columns=cols, show="headings", height=10)
        self.fit_results_tree.heading("name", text="Dataset")
        self.fit_results_tree.heading("model", text="Model")
        self.fit_results_tree.heading("sigma_dc_uS_cm", text="sigma_dc\n(uS/cm)")
        self.fit_results_tree.heading("sigma_inf_uS_cm", text="sigma_inf\n(uS/cm)")
        self.fit_results_tree.heading("rmse", text="RMSE\n(uS/cm)")
        self.fit_results_tree.heading("rmse_ci", text="RMSE CI\n(uS/cm)")
        self.fit_results_tree.heading("p_m1", text="m1\n(-)")
        self.fit_results_tree.heading("p_tau1_s", text="tau1\n(s)")
        self.fit_results_tree.heading("p_c1", text="c/a1\n(-)")
        self.fit_results_tree.heading("p_beta1", text="beta1\n(-)")
        self.fit_results_tree.heading("p_m2", text="m2\n(-)")
        self.fit_results_tree.heading("p_tau2_s", text="tau2\n(s)")
        self.fit_results_tree.heading("p_c2", text="c/a2\n(-)")
        self.fit_results_tree.heading("p_beta2", text="beta2\n(-)")
        self.fit_results_tree.heading("f_peak_hz", text="f_peak\n(Hz)")
        self.fit_results_tree.heading("f_peak_hz_ci", text="f_peak CI\n(Hz)")
        self.fit_results_tree.heading("sigma_q_peak_uS_cm", text="sigma''_peak\n(uS/cm)")
        self.fit_results_tree.heading("sigma_q_peak_uS_cm_ci", text="sigma'' CI\n(uS/cm)")
        self.fit_results_tree.heading("tau_peak_s", text="tau_peak\n(s)")
        self.fit_results_tree.heading("tau_peak_s_ci", text="tau_peak CI\n(s)")
        self.fit_results_tree.column("name", width=160, anchor=tk.W)
        self.fit_results_tree.column("model", width=160, anchor=tk.W)
        self.fit_results_tree.column("sigma_dc_uS_cm", width=96, anchor=tk.CENTER)
        self.fit_results_tree.column("sigma_inf_uS_cm", width=96, anchor=tk.CENTER)
        self.fit_results_tree.column("rmse", width=84, anchor=tk.CENTER)
        self.fit_results_tree.column("rmse_ci", width=120, anchor=tk.CENTER)
        self.fit_results_tree.column("p_m1", width=68, anchor=tk.CENTER)
        self.fit_results_tree.column("p_tau1_s", width=88, anchor=tk.CENTER)
        self.fit_results_tree.column("p_c1", width=68, anchor=tk.CENTER)
        self.fit_results_tree.column("p_beta1", width=68, anchor=tk.CENTER)
        self.fit_results_tree.column("p_m2", width=68, anchor=tk.CENTER)
        self.fit_results_tree.column("p_tau2_s", width=88, anchor=tk.CENTER)
        self.fit_results_tree.column("p_c2", width=68, anchor=tk.CENTER)
        self.fit_results_tree.column("p_beta2", width=68, anchor=tk.CENTER)
        self.fit_results_tree.column("f_peak_hz", width=90, anchor=tk.CENTER)
        self.fit_results_tree.column("f_peak_hz_ci", width=120, anchor=tk.CENTER)
        self.fit_results_tree.column("sigma_q_peak_uS_cm", width=105, anchor=tk.CENTER)
        self.fit_results_tree.column("sigma_q_peak_uS_cm_ci", width=120, anchor=tk.CENTER)
        self.fit_results_tree.column("tau_peak_s", width=90, anchor=tk.CENTER)
        self.fit_results_tree.column("tau_peak_s_ci", width=120, anchor=tk.CENTER)
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)
        fit_results_y = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.fit_results_tree.yview)
        fit_results_x = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL, command=self.fit_results_tree.xview)
        self.fit_results_tree.configure(yscrollcommand=fit_results_y.set, xscrollcommand=fit_results_x.set)
        self.fit_results_tree.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=(6, 0))
        fit_results_y.grid(row=0, column=1, sticky="ns", padx=(2, 6), pady=(6, 0))
        fit_results_x.grid(row=1, column=0, sticky="ew", padx=(6, 0), pady=(2, 6))

    def choose_fit_files(self) -> None:
        file_paths = filedialog.askopenfilenames(
            title="Select raw SIP CSV(s) or mean DF CSV(s)",
            initialdir=self._last_dir,
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not file_paths:
            return
        self._last_dir = str(Path(file_paths[0]).parent)
        self._save_settings()
        self.fit_paths = list(file_paths)
        self.state.fit_paths = self.fit_paths
        self.fit_list.delete(0, tk.END)
        for p in self.fit_paths:
            self.fit_list.insert(tk.END, p)
        self.fit_status_var.set(f"Loaded {len(self.fit_paths)} file(s) for fitting.")

    def _selected_fit_paths(self) -> list[str]:
        idxs = self.fit_list.curselection()
        if not idxs:
            return self.fit_paths
        return [self.fit_list.get(i) for i in idxs]

    def _parse_fit_frequency_range(self) -> tuple[float | None, float | None]:
        fmin_text = self.fit_fmin_var.get().strip()
        fmax_text = self.fit_fmax_var.get().strip()
        fmin = float(fmin_text) if fmin_text else None
        fmax = float(fmax_text) if fmax_text else None
        if fmin is not None and fmin <= 0:
            raise ValueError("Fit min frequency must be > 0.")
        if fmax is not None and fmax <= 0:
            raise ValueError("Fit max frequency must be > 0.")
        if fmin is not None and fmax is not None and fmin > fmax:
            raise ValueError("Fit min frequency must be <= fit max frequency.")
        return fmin, fmax

    def _parse_fit_uncertainty(self) -> tuple[str, int, float]:
        mode = self.fit_uncertainty_var.get().strip()
        try:
            n_samples = int(self.fit_unc_samples_var.get().strip())
        except Exception as exc:
            raise ValueError("N samples/steps must be an integer.") from exc
        if n_samples <= 0:
            raise ValueError("N samples/steps must be > 0.")
        try:
            ci_level = float(self.fit_ci_level_var.get().strip())
        except Exception as exc:
            raise ValueError("CI (%) must be numeric.") from exc
        if not (50.0 <= ci_level < 100.0):
            raise ValueError("CI (%) must be in [50, 100).")
        return mode, n_samples, ci_level

    def _parse_cvnn_train_config(self) -> tuple[int, int]:
        try:
            n_train = int(self.fit_cvnn_ntrain_var.get().strip())
        except Exception as exc:
            raise ValueError("CVNN train N must be an integer.") from exc
        try:
            epochs = int(self.fit_cvnn_epochs_var.get().strip())
        except Exception as exc:
            raise ValueError("CVNN epochs must be an integer.") from exc
        if n_train < 200:
            raise ValueError("CVNN train N must be >= 200.")
        if epochs < 50:
            raise ValueError("CVNN epochs must be >= 50.")
        return n_train, epochs

    def _load_fit_dataset(self, path: str) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        name = os.path.basename(path)
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.DataFrame()

        mean_cols = {
            "frequency_hz",
            "sigma_in_phase_mean_uS_cm",
            "sigma_quadrature_mean_uS_cm",
        }
        cond_cols = {"frequency_hz", "sigma_in_phase_uS_cm", "sigma_quadrature_uS_cm"}

        if mean_cols.issubset(df.columns):
            work = df.sort_values("frequency_hz").copy()
            freq = work["frequency_hz"].to_numpy(dtype=float)
            sigma_in = work["sigma_in_phase_mean_uS_cm"].to_numpy(dtype=float)
            sigma_q = work["sigma_quadrature_mean_uS_cm"].to_numpy(dtype=float)
            return name, freq, sigma_in, sigma_q

        if cond_cols.issubset(df.columns):
            work = df.sort_values("frequency_hz").copy()
            freq = work["frequency_hz"].to_numpy(dtype=float)
            sigma_in = work["sigma_in_phase_uS_cm"].to_numpy(dtype=float)
            sigma_q = work["sigma_quadrature_uS_cm"].to_numpy(dtype=float)
            return name, freq, sigma_in, sigma_q

        gf = float(self.fit_gf_var.get().strip())
        raw = analysis_io.parse_sip_csv(path)
        raw = analysis_transform.add_conductivity_columns(raw, gf).sort_values("frequency_hz")
        freq = raw["frequency_hz"].to_numpy(dtype=float)
        sigma_in = raw["sigma_in_phase_uS_cm"].to_numpy(dtype=float)
        sigma_q = raw["sigma_quadrature_uS_cm"].to_numpy(dtype=float)
        return name, freq, sigma_in, sigma_q

    def _extract_fit_param_columns(
        self,
        best_params: np.ndarray | None,
        n_terms: int,
        beta_free: bool,
    ) -> dict[str, float | None]:
        out: dict[str, float | None] = {
            "p_m1": None,
            "p_tau1_s": None,
            "p_c1": None,
            "p_beta1": None,
            "p_m2": None,
            "p_tau2_s": None,
            "p_c2": None,
            "p_beta2": None,
        }
        if best_params is None:
            return out

        idx = 1
        for term in range(1, min(2, n_terms) + 1):
            m = float(best_params[idx])
            tau = float(np.exp(best_params[idx + 1]))
            c_or_a = float(best_params[idx + 2])
            idx += 3
            beta = float(best_params[idx]) if beta_free else 1.0
            if beta_free:
                idx += 1
            out[f"p_m{term}"] = m
            out[f"p_tau{term}_s"] = tau
            out[f"p_c{term}"] = c_or_a
            out[f"p_beta{term}"] = beta
        return out

    def _refresh_fit_results_table(self) -> None:
        def fmt_ci(value: object) -> str:
            if not value:
                return "-"
            lo, hi = value
            return f"[{float(lo):.4g}, {float(hi):.4g}]"

        def fmt_num(value: object) -> str:
            if value is None:
                return "-"
            return f"{float(value):.6g}"

        for item in self.fit_results_tree.get_children():
            self.fit_results_tree.delete(item)
        for r in self.fit_results:
            self.fit_results_tree.insert(
                "",
                tk.END,
                values=(
                    r["name"],
                    r["model"],
                    fmt_num(r.get("sigma_dc_uS_cm")),
                    fmt_num(r.get("sigma_inf_uS_cm")),
                    fmt_num(r.get("rmse")),
                    fmt_ci(r.get("rmse_ci")),
                    fmt_num(r.get("p_m1")),
                    fmt_num(r.get("p_tau1_s")),
                    fmt_num(r.get("p_c1")),
                    fmt_num(r.get("p_beta1")),
                    fmt_num(r.get("p_m2")),
                    fmt_num(r.get("p_tau2_s")),
                    fmt_num(r.get("p_c2")),
                    fmt_num(r.get("p_beta2")),
                    fmt_num(r.get("f_peak_hz")),
                    fmt_ci(r.get("f_peak_hz_ci")),
                    fmt_num(r.get("sigma_q_peak_uS_cm")),
                    fmt_ci(r.get("sigma_q_peak_uS_cm_ci")),
                    fmt_num(r.get("tau_peak_s")),
                    fmt_ci(r.get("tau_peak_s_ci")),
                ),
            )

    def fit_selected_datasets(self) -> None:
        paths = self._selected_fit_paths()
        if not paths:
            messagebox.showerror("Cannot fit", "No input files selected.")
            return
        self.fit_status_var.set("Running fits...")
        self._set_figure_placeholder("fit", "Running fits...")

        model_name = self.fit_model_var.get().strip()
        try:
            fmin, fmax = self._parse_fit_frequency_range()
            unc_mode, unc_n, ci_level = self._parse_fit_uncertainty()
            cvnn_ntrain, cvnn_epochs = (0, 0)
            if model_name == "CVNN (paper repo, local train)":
                cvnn_ntrain, cvnn_epochs = self._parse_cvnn_train_config()
        except Exception as exc:
            messagebox.showerror("Cannot fit", str(exc))
            return
        def worker():
            fit_results: list[dict[str, object]] = []
            skipped: list[str] = []
            plot_rows: list[dict[str, object]] = []

            for path in paths:
                try:
                    name, freq, sigma_in, sigma_q = self._load_fit_dataset(path)
                    mask = np.isfinite(freq) & np.isfinite(sigma_in) & np.isfinite(sigma_q) & (freq > 0)
                    if fmin is not None:
                        mask &= freq >= fmin
                    if fmax is not None:
                        mask &= freq <= fmax
                    freq = freq[mask]
                    sigma_in = sigma_in[mask]
                    sigma_q = sigma_q[mask]
                    if len(freq) < 5:
                        raise ValueError("Need at least 5 valid points in selected frequency range.")

                    order = np.argsort(freq)
                    freq = freq[order]
                    sigma_in = sigma_in[order]
                    sigma_q = sigma_q[order]
                    uniq = pd.DataFrame({"frequency_hz": freq, "sigma_in": sigma_in, "sigma_q": sigma_q}).groupby(
                        "frequency_hz", as_index=False
                    ).mean()
                    freq = uniq["frequency_hz"].to_numpy(dtype=float)
                    sigma_in = uniq["sigma_in"].to_numpy(dtype=float)
                    sigma_q = uniq["sigma_q"].to_numpy(dtype=float)

                    best_params = None
                    n_terms = 0
                    beta_free = False
                    sigma_dc_uS_cm = None
                    sigma_inf_uS_cm = None
                    if model_name == "Neural Network (experimental)":
                        pred_in, pred_q, rmse = analysis_models.fit_neural_network(freq, sigma_in, sigma_q)
                    elif model_name == "CVNN (paper repo, local train)":
                        pred_in, pred_q, rmse, best_params = analysis_models.fit_cvnn_paper_local(
                            freq, sigma_in, sigma_q, n_train=cvnn_ntrain, n_epoch=cvnn_epochs
                        )
                        n_terms, beta_free = 2, False
                        sigma_dc_uS_cm, sigma_inf_uS_cm = analysis_models.conductivity_endpoints_from_params(
                            best_params, n_terms, beta_free
                        )
                    else:
                        pred_in, pred_q, rmse, best_params, n_terms, beta_free = analysis_models.fit_relaxation_model_with_params(
                            model_name, freq, sigma_in, sigma_q
                        )
                        sigma_dc_uS_cm, sigma_inf_uS_cm = analysis_models.conductivity_endpoints_from_params(
                            best_params, n_terms, beta_free
                        )

                    pred_phase = -1000.0 * np.arctan2(pred_q, np.maximum(np.abs(pred_in), 1e-12))
                    raw_phase = -1000.0 * np.arctan2(sigma_q, np.maximum(np.abs(sigma_in), 1e-12))
                    x_dense, pred_q_dense = analysis_stats.interpolate_log_curve(freq, pred_q, points=600)
                    peak_idx = int(np.nanargmax(pred_q_dense))
                    f_peak = float(x_dense[peak_idx])
                    sigma_peak = float(pred_q_dense[peak_idx])
                    tau_peak = 1.0 / (2.0 * np.pi * f_peak)
                    ci_result: dict[str, tuple[float, float]] = {}
                    if unc_mode == "Bootstrap CI":
                        if model_name == "CVNN (paper repo, local train)":
                            raise ValueError("Bootstrap CI is not supported for paper CVNN mode yet.")
                        ci_result = analysis_models.bootstrap_uncertainty_for_fit(
                            model_name=model_name,
                            frequency_hz=freq,
                            sigma_in=sigma_in,
                            sigma_q=sigma_q,
                            n_boot=unc_n,
                            ci_level=ci_level,
                        )
                    elif unc_mode == "MCMC (experimental)":
                        if model_name in {"Neural Network (experimental)", "CVNN (paper repo, local train)"}:
                            raise ValueError("MCMC uncertainty is not supported for neural-network fits.")
                        if best_params is None:
                            raise RuntimeError("MCMC setup failed (missing fitted parameters).")
                        ci_result = analysis_models.mcmc_uncertainty_for_fit(
                            model_name=model_name,
                            frequency_hz=freq,
                            sigma_in=sigma_in,
                            sigma_q=sigma_q,
                            best_params=best_params,
                            n_terms=n_terms,
                            beta_free=beta_free,
                            n_steps=unc_n,
                            ci_level=ci_level,
                        )

                    resid = np.sqrt((pred_in - sigma_in) ** 2 + (pred_q - sigma_q) ** 2)
                    plot_rows.append(
                        {
                            "name": name,
                            "freq": freq,
                            "sigma_in": sigma_in,
                            "sigma_q": sigma_q,
                            "pred_in": pred_in,
                            "pred_q": pred_q,
                            "raw_phase": raw_phase,
                            "pred_phase": pred_phase,
                            "resid": resid,
                            "f_peak": f_peak,
                        }
                    )
                    param_cols = self._extract_fit_param_columns(best_params, n_terms, beta_free)
                    fit_results.append(
                        {
                            "name": name,
                            "model": model_name,
                            "sigma_dc_uS_cm": sigma_dc_uS_cm,
                            "sigma_inf_uS_cm": sigma_inf_uS_cm,
                            "rmse": rmse,
                            "f_peak_hz": f_peak,
                            "sigma_q_peak_uS_cm": sigma_peak,
                            "tau_peak_s": tau_peak,
                            **param_cols,
                            **ci_result,
                        }
                    )
                except Exception as exc:
                    skipped.append(f"{os.path.basename(path)}: {exc}")

            if not fit_results:
                return {"fit_results": [], "plot_rows": [], "skipped": skipped}

            return {"fit_results": fit_results, "plot_rows": plot_rows, "skipped": skipped}

        def on_done(payload):
            self.fit_results = payload["fit_results"]
            self.state.fit_results = self.fit_results
            self._refresh_fit_results_table()
            if not self.fit_results:
                detail = "\n".join(payload.get("skipped", []))
                msg = "No files could be fitted."
                if detail:
                    msg += f"\n\nDetails:\n{detail}"
                self.fit_status_var.set("No files could be fitted. See error details.")
                self._set_figure_placeholder("fit", msg)
                messagebox.showerror("Cannot fit", msg)
                return

            fig, axs = analysis_plotting.init_2x2_layout(f"Model Fit: {model_name}")
            handles: list[object] = []
            labels: list[str] = []
            for row in payload["plot_rows"]:
                name = str(row["name"])
                color, marker = self._style_for_name(name)
                freq = np.asarray(row["freq"], dtype=float)
                sigma_in = np.asarray(row["sigma_in"], dtype=float)
                sigma_q = np.asarray(row["sigma_q"], dtype=float)
                pred_in = np.asarray(row["pred_in"], dtype=float)
                pred_q = np.asarray(row["pred_q"], dtype=float)
                raw_phase = np.asarray(row["raw_phase"], dtype=float)
                pred_phase = np.asarray(row["pred_phase"], dtype=float)
                resid = np.asarray(row["resid"], dtype=float)
                f_peak = float(row["f_peak"])

                axs[0, 0].plot(freq, sigma_in, linestyle="None", marker=marker, markersize=4, color=color)
                h = axs[0, 0].plot(freq, pred_in, linestyle="-", linewidth=1.8, color=color, label=name)[0]
                axs[0, 1].plot(freq, sigma_q, linestyle="None", marker=marker, markersize=4, color=color)
                axs[0, 1].plot(freq, pred_q, linestyle="-", linewidth=1.8, color=color)
                axs[0, 1].axvline(f_peak, color=color, linestyle="--", alpha=0.35, linewidth=1.0)
                axs[1, 0].plot(freq, raw_phase, linestyle="None", marker=marker, markersize=4, color=color)
                axs[1, 0].plot(freq, pred_phase, linestyle="-", linewidth=1.8, color=color)
                axs[1, 1].plot(freq, resid, color=color, linewidth=1.1)
                handles.append(h)
                labels.append(name)

            axs[1, 1].axis("on")
            axs[1, 1].set_xscale("log")
            axs[1, 1].set_title("Residual Magnitude")
            axs[1, 1].set_xlabel("Frequency (Hz)")
            axs[1, 1].set_ylabel("|residual| (uS/cm)")
            axs[1, 1].grid(True, which="both", linestyle="--", alpha=0.35)
            axs[0, 0].legend(handles, labels, loc="best", frameon=False)
            fig.tight_layout()
            self.fit_last_fig = fig
            self.fit_status_var.set(f"Fitted {len(self.fit_results)} dataset(s).")
            self._display_figure("fit", fig)
            skipped = payload.get("skipped", [])
            if skipped:
                messagebox.showwarning("Some fits failed", "\n".join(skipped))

        self._run_background(worker, on_done, "Running fits...")

    def save_fit_png(self) -> None:
        if self.fit_last_fig is None:
            messagebox.showerror("Cannot save", "Create a fit plot first.")
            return
        out_path = filedialog.asksaveasfilename(
            title="Save fit PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile="sip_fit_plot.png",
        )
        if not out_path:
            return
        self.fit_last_fig.savefig(out_path, dpi=300, bbox_inches="tight")
        self.fit_status_var.set(f"Saved fit plot: {out_path}")

    def save_fit_results_csv(self) -> None:
        if not self.fit_results:
            messagebox.showerror("Cannot save", "Run fitting first.")
            return
        out_path = filedialog.asksaveasfilename(
            title="Save fit results CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="sip_fit_results.csv",
        )
        if not out_path:
            return

        rows: list[dict[str, object]] = []
        for r in self.fit_results:
            row = dict(r)
            for key in ["rmse_ci", "f_peak_hz_ci", "sigma_q_peak_uS_cm_ci", "tau_peak_s_ci"]:
                ci = row.get(key)
                if ci:
                    lo, hi = ci
                    row[f"{key}_lo"] = lo
                    row[f"{key}_hi"] = hi
                else:
                    row[f"{key}_lo"] = np.nan
                    row[f"{key}_hi"] = np.nan
            rows.append(row)

        pd.DataFrame(rows).to_csv(out_path, index=False)
        self.fit_status_var.set(f"Saved fit results: {out_path}")

    def _toggle_mean_save(self) -> None:
        state = tk.NORMAL if self.mean_std_var.get() else tk.DISABLED
        self.save_mean_btn.config(state=state)

    def _parse_geometric_factor(self) -> float:
        try:
            gf = float(self.gf_var.get().strip())
        except ValueError as exc:
            raise ValueError("Geometric factor must be numeric.") from exc
        if gf <= 0:
            raise ValueError("Geometric factor must be > 0.")
        return gf

    def _parse_frequency_range(self) -> tuple[float | None, float | None]:
        fmin_text = self.fmin_var.get().strip()
        fmax_text = self.fmax_var.get().strip()

        fmin = None
        fmax = None

        if fmin_text:
            fmin = float(fmin_text)
            if fmin <= 0:
                raise ValueError("Min frequency must be > 0 for log-scale plotting.")
        if fmax_text:
            fmax = float(fmax_text)
            if fmax <= 0:
                raise ValueError("Max frequency must be > 0 for log-scale plotting.")

        if fmin is not None and fmax is not None and fmin > fmax:
            raise ValueError("Min frequency must be <= max frequency.")

        return fmin, fmax

    def choose_files(self) -> None:
        file_paths = filedialog.askopenfilenames(
            title="Select SIP CSV file(s)",
            initialdir=self._last_dir,
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not file_paths:
            return
        self._last_dir = str(Path(file_paths[0]).parent)
        self._save_settings()

        def worker():
            loaded = {}
            errors = []
            for path in file_paths:
                try:
                    loaded[path] = analysis_io.parse_sip_csv(path)
                except Exception as exc:
                    errors.append(f"{os.path.basename(path)}: {exc}")
            return loaded, errors

        def on_done(result):
            loaded_map, errors = result
            self.datasets.update(loaded_map)
            self._rebuild_combined_df()
            self._refresh_tree()
            parts = [f"Loaded {len(loaded_map)} file(s)."]
            if errors:
                parts.append(f"Failed {len(errors)} file(s).")
            self.status_var.set(" ".join(parts))
            if errors:
                messagebox.showwarning("Some files failed", "\n".join(errors))

        self._run_background(worker, on_done, "Loading files...")

    def _rebuild_combined_df(self) -> None:
        if not self.datasets:
            self.combined_df = pd.DataFrame()
            self.mean_std_df = pd.DataFrame()
            self.state.combined_df = self.combined_df
            self.state.mean_std_df = self.mean_std_df
            return

        gf = self._parse_geometric_factor()
        merged = pd.concat(self.datasets.values(), ignore_index=True)
        self.combined_df = analysis_transform.add_conductivity_columns(merged, gf)
        self.state.combined_df = self.combined_df

    def _refresh_tree(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        for path, df in self.datasets.items():
            ts_values = df["timestamp"].dropna().unique()
            ts = ts_values[0] if len(ts_values) else "N/A"
            self.tree.insert("", tk.END, values=(os.path.basename(path), ts, len(df)))
        self._refresh_legend_editor()

    def _refresh_legend_editor(self) -> None:
        existing = {os.path.basename(p) for p in self.datasets}
        for name in list(self.legend_name_vars):
            if name not in existing:
                del self.legend_name_vars[name]
        for name in sorted(existing):
            if name not in self.legend_name_vars:
                self.legend_name_vars[name] = tk.StringVar(value=name)

        for child in self.legend_editor.winfo_children():
            child.destroy()

        if not existing:
            ttk.Label(self.legend_editor, text="Load files to edit legend names.").pack(anchor=tk.W)
            return

        for row, name in enumerate(sorted(existing)):
            ttk.Label(self.legend_editor, text=name).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            ttk.Entry(self.legend_editor, textvariable=self.legend_name_vars[name], width=28).grid(
                row=row, column=1, sticky="w", pady=2
            )

    def _legend_name_for_file(self, file_name: str) -> str:
        var = self.legend_name_vars.get(file_name)
        if var is None:
            return file_name
        text = var.get().strip()
        return text if text else file_name

    def _set_peak_rows(self, rows: list[tuple[str, float, float, float]]) -> None:
        for item in self.peak_tree.get_children():
            self.peak_tree.delete(item)
        for name, f_peak, sigma_peak, tau_peak in rows:
            self.peak_tree.insert(
                "",
                tk.END,
                values=(name, f"{f_peak:.6g}", f"{sigma_peak:.6g}", f"{tau_peak:.6g}"),
            )

    def _selected_or_all(self) -> pd.DataFrame:
        if self.combined_df.empty:
            raise ValueError("No data loaded.")

        selected = self.tree.selection()
        if not selected:
            out = self.combined_df.copy()
        else:
            selected_files = {self.tree.item(item, "values")[0] for item in selected}
            out = self.combined_df[self.combined_df["file"].isin(selected_files)].copy()
            if out.empty:
                raise ValueError("Selected files have no rows.")

        fmin, fmax = self._parse_frequency_range()
        if fmin is not None:
            out = out[out["frequency_hz"] >= fmin]
        if fmax is not None:
            out = out[out["frequency_hz"] <= fmax]

        if out.empty:
            raise ValueError("No rows remain after applying frequency range.")

        return out

    def _style_for_name(self, name: str) -> tuple[str, str]:
        # Stable deterministic mapping per name across runs.
        idx = sum(ord(c) for c in name)
        color = COLORBLIND_COLORS[idx % len(COLORBLIND_COLORS)]
        marker = MARKERS[(idx // len(COLORBLIND_COLORS)) % len(MARKERS)]
        return color, marker

    def _interp_log_curve(self, x: np.ndarray, y: np.ndarray, points: int = 300) -> tuple[np.ndarray, np.ndarray]:
        if len(x) < 2:
            return x, y
        xlog = np.log10(x)
        if np.any(np.diff(xlog) <= 0):
            return x, y
        dense_x = np.logspace(xlog[0], xlog[-1], points)
        dense_y = np.interp(np.log10(dense_x), xlog, y)
        return dense_x, dense_y

    def _init_2x2_layout(self, title: str):
        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle(title)

        for ax in [axs[0, 0], axs[0, 1], axs[1, 0]]:
            ax.set_xscale("log")
            ax.grid(True, which="both", linestyle="--", alpha=0.35)

        axs[0, 0].set_ylabel("In-phase conductivity (uS/cm)")
        axs[0, 1].set_ylabel("Quadrature conductivity (uS/cm)")
        axs[1, 0].set_ylabel("-Phase (mrad)")
        axs[1, 0].set_xlabel("Frequency (Hz)")

        axs[1, 1].axis("off")
        return fig, axs

    def _display_figure(self, key: str, fig) -> None:
        host = self._figure_hosts.get(key)
        if host is None:
            return
        for child in host.winfo_children():
            try:
                child.destroy()
            except Exception:
                pass
        if key in self._canvases:
            try:
                self._canvases[key].get_tk_widget().destroy()
            except Exception:
                pass
        if key in self._toolbars:
            try:
                self._toolbars[key].destroy()
            except Exception:
                pass
        canvas = FigureCanvasTkAgg(fig, master=host)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, host, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._canvases[key] = canvas
        self._toolbars[key] = toolbar

    def _set_figure_placeholder(self, key: str, text: str) -> None:
        host = self._figure_hosts.get(key)
        if host is None:
            return
        if key in self._canvases:
            self._canvases.pop(key, None)
        if key in self._toolbars:
            self._toolbars.pop(key, None)
        for child in host.winfo_children():
            try:
                child.destroy()
            except Exception:
                pass
        ttk.Label(host, text=text, anchor="center", justify=tk.LEFT).pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

    def plot_data(self) -> None:
        try:
            self._rebuild_combined_df()
            df = self._selected_or_all()
        except Exception as exc:
            self._set_peak_rows([])
            messagebox.showerror("Cannot plot", str(exc))
            return

        fig, axs = analysis_plotting.init_2x2_layout("SIP Results")

        handles = []
        labels = []
        mean_only = self.mean_std_var.get()

        if not mean_only:
            peak_rows: list[tuple[str, float, float, float]] = []
            for file_name, g in df.groupby("file", sort=True):
                legend_name = self._legend_name_for_file(file_name)
                g = (
                    g.groupby("frequency_hz", as_index=False)[
                        ["sigma_in_phase_uS_cm", "sigma_quadrature_uS_cm", "minus_phase_mrad"]
                    ]
                    .mean()
                    .sort_values("frequency_hz")
                )
                color, marker = self._style_for_name(file_name)
                x_raw = g["frequency_hz"].to_numpy(dtype=float)
                y_in_raw = g["sigma_in_phase_uS_cm"].to_numpy(dtype=float)
                y_q_raw = g["sigma_quadrature_uS_cm"].to_numpy(dtype=float)
                y_p_raw = g["minus_phase_mrad"].to_numpy(dtype=float)

                x_i, y_in_i = analysis_stats.interpolate_log_curve(x_raw, y_in_raw)
                _, y_q_i = analysis_stats.interpolate_log_curve(x_raw, y_q_raw)
                _, y_p_i = analysis_stats.interpolate_log_curve(x_raw, y_p_raw)

                axs[0, 0].plot(
                    x_raw,
                    y_in_raw,
                    linestyle="None",
                    marker=marker,
                    markersize=5,
                    color=color,
                )
                h = axs[0, 0].plot(x_i, y_in_i, linestyle="-", linewidth=1.6, color=color, label=legend_name)[0]
                axs[0, 1].plot(
                    x_raw,
                    y_q_raw,
                    linestyle="None",
                    marker=marker,
                    markersize=5,
                    color=color,
                )
                axs[0, 1].plot(x_i, y_q_i, linestyle="-", linewidth=1.6, color=color)
                axs[1, 0].plot(
                    x_raw,
                    y_p_raw,
                    linestyle="None",
                    marker=marker,
                    markersize=5,
                    color=color,
                )
                axs[1, 0].plot(x_i, y_p_i, linestyle="-", linewidth=1.6, color=color)

                if len(x_i) > 0 and np.isfinite(y_q_i).any():
                    idx_peak = int(np.nanargmax(y_q_i))
                    f_peak = float(x_i[idx_peak])
                    sigma_q_peak = float(y_q_i[idx_peak])
                    tau_peak = 1.0 / (2.0 * np.pi * f_peak)
                    axs[0, 1].axvline(f_peak, color=color, linestyle="--", alpha=0.35, linewidth=1.0)
                    peak_rows.append((legend_name, f_peak, sigma_q_peak, tau_peak))

                handles.append(h)
                labels.append(legend_name)
            self._set_peak_rows(peak_rows)

        if mean_only:
            self.mean_std_df = analysis_stats.compute_mean_std(df)
            self.state.mean_std_df = self.mean_std_df
            m = self.mean_std_df
            x = m["frequency_hz"].to_numpy(dtype=float)
            for ax, mean_col, std_col in [
                (axs[0, 0], "sigma_in_phase_mean_uS_cm", "sigma_in_phase_std_uS_cm"),
                (axs[0, 1], "sigma_quadrature_mean_uS_cm", "sigma_quadrature_std_uS_cm"),
                (axs[1, 0], "minus_phase_mean_mrad", "minus_phase_std_mrad"),
            ]:
                y = m[mean_col].to_numpy(dtype=float)
                s = m[std_col].to_numpy(dtype=float)
                ax.fill_between(x, y - s, y + s, color="black", alpha=0.15, linewidth=0)
                line = ax.plot(x, y, linestyle="-", linewidth=1.8, color="black")[0]
            handles.append(line)
            labels.append("Mean ± STD")
            self._set_peak_rows([])
        else:
            self.mean_std_df = pd.DataFrame()
            self.state.mean_std_df = self.mean_std_df

        axs[1, 1].legend(handles, labels, loc="center", frameon=False)

        fig.tight_layout()
        self.last_fig = fig
        self._display_figure("main", fig)

    def save_plot_png(self) -> None:
        if self.last_fig is None:
            messagebox.showerror("Cannot save", "Create a plot first.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Save plot PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile="sip_plot.png",
        )
        if not out_path:
            return

        self.last_fig.savefig(out_path, dpi=300, bbox_inches="tight")
        self.status_var.set(f"Saved plot: {out_path}")

    def save_combined_csv(self) -> None:
        try:
            self._rebuild_combined_df()
            if self.combined_df.empty:
                raise ValueError("No data loaded.")
        except Exception as exc:
            messagebox.showerror("Cannot save", str(exc))
            return

        out_path = filedialog.asksaveasfilename(
            title="Save combined dataframe",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="sip_combined_dataframe.csv",
        )
        if not out_path:
            return

        self.combined_df.to_csv(out_path, index=False)
        self.status_var.set(f"Saved combined dataframe: {out_path}")

    def save_mean_std_csv(self) -> None:
        if not self.mean_std_var.get():
            messagebox.showerror("Cannot save", "Enable 'Plot mean ± std' first.")
            return

        if self.mean_std_df.empty:
            messagebox.showerror("Cannot save", "Create a mean ± std plot first.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Save mean ± std dataframe",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="sip_mean_std.csv",
        )
        if not out_path:
            return

        self.mean_std_df.to_csv(out_path, index=False)
        self.status_var.set(f"Saved mean±std dataframe: {out_path}")

    def choose_mean_df_files(self) -> None:
        file_paths = filedialog.askopenfilenames(
            title="Select saved mean DF file(s)",
            initialdir=self._last_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not file_paths:
            return
        self._last_dir = str(Path(file_paths[0]).parent)
        self._save_settings()

        self.mean_df_paths = list(file_paths)
        self.compare_list.delete(0, tk.END)
        for p in self.mean_df_paths:
            self.compare_list.insert(tk.END, p)

        self.compare_status.set(f"Loaded {len(self.mean_df_paths)} mean DF file(s).")

    def _selected_mean_paths(self) -> list[str]:
        idxs = self.compare_list.curselection()
        if not idxs:
            return self.mean_df_paths
        return [self.compare_list.get(i) for i in idxs]

    def plot_mean_comparison(self) -> None:
        paths = self._selected_mean_paths()
        if not paths:
            messagebox.showerror("Cannot plot", "No mean DF files selected.")
            return

        fig, axs = analysis_plotting.init_2x2_layout("Saved Mean ± Std DF Plot")
        handles = []
        labels = []

        for p in paths:
            try:
                df = pd.read_csv(p)
            except Exception as exc:
                messagebox.showwarning("Skipping file", f"{os.path.basename(p)}: {exc}")
                continue

            required = {
                "frequency_hz",
                "sigma_in_phase_mean_uS_cm",
                "sigma_quadrature_mean_uS_cm",
                "minus_phase_mean_mrad",
            }
            if not required.issubset(df.columns):
                messagebox.showwarning("Skipping file", f"{os.path.basename(p)} missing required mean columns.")
                continue

            label = os.path.basename(p)
            color, marker = self._style_for_name(label)
            df = df.sort_values("frequency_hz")

            h = axs[0, 0].plot(
                df["frequency_hz"],
                df["sigma_in_phase_mean_uS_cm"],
                linestyle="None",
                marker=marker,
                markersize=6,
                color=color,
                label=label,
            )[0]
            axs[0, 1].plot(
                df["frequency_hz"],
                df["sigma_quadrature_mean_uS_cm"],
                linestyle="None",
                marker=marker,
                markersize=6,
                color=color,
            )
            axs[1, 0].plot(
                df["frequency_hz"],
                df["minus_phase_mean_mrad"],
                linestyle="None",
                marker=marker,
                markersize=6,
                color=color,
            )

            # If std columns exist, overlay as error bars.
            if "sigma_in_phase_std_uS_cm" in df.columns:
                axs[0, 0].errorbar(
                    df["frequency_hz"],
                    df["sigma_in_phase_mean_uS_cm"],
                    yerr=df["sigma_in_phase_std_uS_cm"],
                    linestyle="None",
                    marker="None",
                    ecolor=color,
                    alpha=0.7,
                    capsize=2,
                )
            if "sigma_quadrature_std_uS_cm" in df.columns:
                axs[0, 1].errorbar(
                    df["frequency_hz"],
                    df["sigma_quadrature_mean_uS_cm"],
                    yerr=df["sigma_quadrature_std_uS_cm"],
                    linestyle="None",
                    marker="None",
                    ecolor=color,
                    alpha=0.7,
                    capsize=2,
                )
            if "minus_phase_std_mrad" in df.columns:
                axs[1, 0].errorbar(
                    df["frequency_hz"],
                    df["minus_phase_mean_mrad"],
                    yerr=df["minus_phase_std_mrad"],
                    linestyle="None",
                    marker="None",
                    ecolor=color,
                    alpha=0.7,
                    capsize=2,
                )

            handles.append(h)
            labels.append(label)

        if not handles:
            messagebox.showerror("Cannot plot", "No valid mean DF files to plot.")
            plt.close(fig)
            return

        axs[1, 1].legend(handles, labels, loc="center", frameon=False)
        fig.tight_layout()
        self.compare_last_fig = fig
        self._display_figure("compare", fig)

    def save_compare_png(self) -> None:
        if self.compare_last_fig is None:
            messagebox.showerror("Cannot save", "Create a mean DF plot first.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Save mean plot PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile="sip_mean_plot.png",
        )
        if not out_path:
            return

        self.compare_last_fig.savefig(out_path, dpi=300, bbox_inches="tight")
        self.compare_status.set(f"Saved mean plot: {out_path}")


def main() -> None:
    root = tk.Tk()
    app = SipGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
