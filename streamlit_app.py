#!/usr/bin/env python3
"""Streamlit web app for SIP plotting, comparison, and model fitting."""

from __future__ import annotations

from io import BytesIO
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sip_analysis import io as analysis_io
from sip_analysis import models as analysis_models
from sip_analysis import plotting as analysis_plotting
from sip_analysis import stats as analysis_stats
from sip_analysis import transform as analysis_transform


MODEL_OPTIONS = [
    "Cole-Cole",
    "Double Cole-Cole",
    "Havriliak-Negami",
    "Double Havriliak-Negami",
    "Neural Network (experimental)",
    "CVNN (paper repo, local train)",
]

UNCERTAINTY_OPTIONS = ["None", "Bootstrap CI", "MCMC (experimental)"]

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


def _style_for_name(name: str) -> tuple[str, str]:
    idx = sum(ord(c) for c in name)
    color = COLORBLIND_COLORS[idx % len(COLORBLIND_COLORS)]
    marker = MARKERS[(idx // len(COLORBLIND_COLORS)) % len(MARKERS)]
    return color, marker


def _extract_param_columns(best_params: np.ndarray | None, n_terms: int, beta_free: bool) -> dict[str, float | None]:
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


def _normalize_cond_df(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    out = df.copy()
    if "file" not in out.columns:
        out["file"] = file_name

    required = {"frequency_hz", "sigma_in_phase_uS_cm", "sigma_quadrature_uS_cm"}
    if not required.issubset(out.columns):
        raise ValueError("Missing required conductivity columns.")

    if "minus_phase_mrad" not in out.columns:
        sigma_in = out["sigma_in_phase_uS_cm"].to_numpy(dtype=float)
        sigma_q = out["sigma_quadrature_uS_cm"].to_numpy(dtype=float)
        out["minus_phase_mrad"] = -1000.0 * np.arctan2(sigma_q, np.maximum(np.abs(sigma_in), 1e-12))

    return out[["file", "frequency_hz", "sigma_in_phase_uS_cm", "sigma_quadrature_uS_cm", "minus_phase_mrad"]].copy()


def _load_uploaded_plot_df(uploaded_file, gf: float) -> pd.DataFrame:
    name = uploaded_file.name
    raw_bytes = uploaded_file.getvalue()

    try:
        parsed_csv = pd.read_csv(BytesIO(raw_bytes))
    except Exception:
        parsed_csv = pd.DataFrame()

    mean_cols = {"frequency_hz", "sigma_in_phase_mean_uS_cm", "sigma_quadrature_mean_uS_cm"}
    cond_cols = {"frequency_hz", "sigma_in_phase_uS_cm", "sigma_quadrature_uS_cm"}

    if mean_cols.issubset(parsed_csv.columns):
        work = parsed_csv.copy()
        work["sigma_in_phase_uS_cm"] = work["sigma_in_phase_mean_uS_cm"]
        work["sigma_quadrature_uS_cm"] = work["sigma_quadrature_mean_uS_cm"]
        if "minus_phase_mean_mrad" in work.columns:
            work["minus_phase_mrad"] = work["minus_phase_mean_mrad"]
        return _normalize_cond_df(work, name)

    if cond_cols.issubset(parsed_csv.columns):
        return _normalize_cond_df(parsed_csv, name)

    with NamedTemporaryFile(suffix=".csv") as tmp:
        tmp.write(raw_bytes)
        tmp.flush()
        parsed = analysis_io.parse_sip_csv(tmp.name)

    parsed = analysis_transform.add_conductivity_columns(parsed, gf)
    return _normalize_cond_df(parsed, name)


def _combine_plot_uploads(uploaded_files, gf: float, fmin: float | None, fmax: float | None):
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []

    for uploaded_file in uploaded_files:
        try:
            df = _load_uploaded_plot_df(uploaded_file, gf)
            mask = (
                np.isfinite(df["frequency_hz"])
                & np.isfinite(df["sigma_in_phase_uS_cm"])
                & np.isfinite(df["sigma_quadrature_uS_cm"])
                & np.isfinite(df["minus_phase_mrad"])
                & (df["frequency_hz"] > 0)
            )
            if fmin is not None:
                mask &= df["frequency_hz"] >= fmin
            if fmax is not None:
                mask &= df["frequency_hz"] <= fmax
            df = df.loc[mask].copy()
            if df.empty:
                raise ValueError("No valid rows remain after filtering.")
            frames.append(df)
        except Exception as exc:
            skipped.append(f"{uploaded_file.name}: {exc}")

    if not frames:
        return pd.DataFrame(), skipped

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["file", "frequency_hz"]).reset_index(drop=True)
    return out, skipped


def _build_plot_only_figure(df: pd.DataFrame, mean_only: bool):
    fig, axs = analysis_plotting.init_2x2_layout("SIP Results")
    handles = []
    labels = []
    peak_rows: list[dict[str, float | str]] = []

    mean_std_df = pd.DataFrame()
    if mean_only:
        mean_std_df = analysis_stats.compute_mean_std(df)
        x = mean_std_df["frequency_hz"].to_numpy(dtype=float)

        line = None
        for ax, mean_col, std_col in [
            (axs[0, 0], "sigma_in_phase_mean_uS_cm", "sigma_in_phase_std_uS_cm"),
            (axs[0, 1], "sigma_quadrature_mean_uS_cm", "sigma_quadrature_std_uS_cm"),
            (axs[1, 0], "minus_phase_mean_mrad", "minus_phase_std_mrad"),
        ]:
            y = mean_std_df[mean_col].to_numpy(dtype=float)
            s = mean_std_df[std_col].to_numpy(dtype=float)
            ax.fill_between(x, y - s, y + s, color="black", alpha=0.15, linewidth=0)
            line = ax.plot(x, y, linestyle="-", linewidth=1.8, color="black")[0]

        if line is not None:
            handles.append(line)
            labels.append("Mean ± STD")
    else:
        for file_name, group in df.groupby("file", sort=True):
            agg = (
                group.groupby("frequency_hz", as_index=False)[
                    ["sigma_in_phase_uS_cm", "sigma_quadrature_uS_cm", "minus_phase_mrad"]
                ]
                .mean()
                .sort_values("frequency_hz")
            )
            color, marker = _style_for_name(str(file_name))
            x_raw = agg["frequency_hz"].to_numpy(dtype=float)
            y_in_raw = agg["sigma_in_phase_uS_cm"].to_numpy(dtype=float)
            y_q_raw = agg["sigma_quadrature_uS_cm"].to_numpy(dtype=float)
            y_p_raw = agg["minus_phase_mrad"].to_numpy(dtype=float)

            x_i, y_in_i = analysis_stats.interpolate_log_curve(x_raw, y_in_raw)
            _, y_q_i = analysis_stats.interpolate_log_curve(x_raw, y_q_raw)
            _, y_p_i = analysis_stats.interpolate_log_curve(x_raw, y_p_raw)

            axs[0, 0].plot(x_raw, y_in_raw, linestyle="None", marker=marker, markersize=5, color=color)
            h = axs[0, 0].plot(x_i, y_in_i, linestyle="-", linewidth=1.6, color=color, label=str(file_name))[0]
            axs[0, 1].plot(x_raw, y_q_raw, linestyle="None", marker=marker, markersize=5, color=color)
            axs[0, 1].plot(x_i, y_q_i, linestyle="-", linewidth=1.6, color=color)
            axs[1, 0].plot(x_raw, y_p_raw, linestyle="None", marker=marker, markersize=5, color=color)
            axs[1, 0].plot(x_i, y_p_i, linestyle="-", linewidth=1.6, color=color)

            if len(x_i) > 0 and np.isfinite(y_q_i).any():
                idx_peak = int(np.nanargmax(y_q_i))
                f_peak = float(x_i[idx_peak])
                sigma_q_peak = float(y_q_i[idx_peak])
                tau_peak = 1.0 / (2.0 * np.pi * f_peak)
                axs[0, 1].axvline(f_peak, color=color, linestyle="--", alpha=0.35, linewidth=1.0)
                peak_rows.append(
                    {
                        "File": str(file_name),
                        "f_peak (Hz)": f_peak,
                        "sigma''_peak (uS/cm)": sigma_q_peak,
                        "tau_peak (s)": tau_peak,
                    }
                )

            handles.append(h)
            labels.append(str(file_name))

    if handles:
        axs[1, 1].legend(handles, labels, loc="center", frameon=False)

    fig.tight_layout()
    return fig, mean_std_df, pd.DataFrame(peak_rows)


def _load_uploaded_mean_df(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(BytesIO(uploaded_file.getvalue()))
    required = {
        "frequency_hz",
        "sigma_in_phase_mean_uS_cm",
        "sigma_quadrature_mean_uS_cm",
        "minus_phase_mean_mrad",
    }
    if not required.issubset(df.columns):
        raise ValueError("Missing required mean columns.")
    return df.sort_values("frequency_hz")


def _build_mean_comparison_plot(uploaded_files):
    fig, axs = analysis_plotting.init_2x2_layout("Saved Mean ± Std DF Plot")
    handles = []
    labels = []
    skipped: list[str] = []

    for uploaded_file in uploaded_files:
        name = uploaded_file.name
        try:
            df = _load_uploaded_mean_df(uploaded_file)
        except Exception as exc:
            skipped.append(f"{name}: {exc}")
            continue

        color, marker = _style_for_name(name)
        h = axs[0, 0].plot(
            df["frequency_hz"],
            df["sigma_in_phase_mean_uS_cm"],
            linestyle="None",
            marker=marker,
            markersize=6,
            color=color,
            label=name,
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
        labels.append(name)

    if not handles:
        plt.close(fig)
        raise ValueError("No valid mean DF files to plot.")

    axs[1, 1].legend(handles, labels, loc="center", frameon=False)
    fig.tight_layout()
    return fig, skipped


def _load_uploaded_dataset(uploaded_file, gf: float) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    name = uploaded_file.name
    raw_bytes = uploaded_file.getvalue()
    df: pd.DataFrame
    try:
        df = pd.read_csv(BytesIO(raw_bytes))
    except Exception:
        df = pd.DataFrame()

    mean_cols = {"frequency_hz", "sigma_in_phase_mean_uS_cm", "sigma_quadrature_mean_uS_cm"}
    cond_cols = {"frequency_hz", "sigma_in_phase_uS_cm", "sigma_quadrature_uS_cm"}

    if mean_cols.issubset(df.columns):
        work = df.sort_values("frequency_hz").copy()
        return (
            name,
            work["frequency_hz"].to_numpy(dtype=float),
            work["sigma_in_phase_mean_uS_cm"].to_numpy(dtype=float),
            work["sigma_quadrature_mean_uS_cm"].to_numpy(dtype=float),
        )

    if cond_cols.issubset(df.columns):
        work = df.sort_values("frequency_hz").copy()
        return (
            name,
            work["frequency_hz"].to_numpy(dtype=float),
            work["sigma_in_phase_uS_cm"].to_numpy(dtype=float),
            work["sigma_quadrature_uS_cm"].to_numpy(dtype=float),
        )

    with NamedTemporaryFile(suffix=".csv") as tmp:
        tmp.write(raw_bytes)
        tmp.flush()
        parsed = analysis_io.parse_sip_csv(tmp.name)
    parsed = analysis_transform.add_conductivity_columns(parsed, gf).sort_values("frequency_hz")
    return (
        name,
        parsed["frequency_hz"].to_numpy(dtype=float),
        parsed["sigma_in_phase_uS_cm"].to_numpy(dtype=float),
        parsed["sigma_quadrature_uS_cm"].to_numpy(dtype=float),
    )


def _fmt_ci(value: object) -> str:
    if not value:
        return "-"
    lo, hi = value
    return f"[{float(lo):.4g}, {float(hi):.4g}]"


def _fit_uploaded_files(
    uploaded_files,
    model_name: str,
    gf: float,
    fmin: float | None,
    fmax: float | None,
    unc_mode: str,
    unc_n: int,
    ci_level: float,
    cvnn_ntrain: int,
    cvnn_epochs: int,
):
    fit_results: list[dict[str, object]] = []
    skipped: list[str] = []
    plot_rows: list[dict[str, object]] = []

    for uploaded_file in uploaded_files:
        try:
            name, freq, sigma_in, sigma_q = _load_uploaded_dataset(uploaded_file, gf)
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
                    **_extract_param_columns(best_params, n_terms, beta_free),
                    **ci_result,
                }
            )
        except Exception as exc:
            skipped.append(f"{uploaded_file.name}: {exc}")

    return fit_results, plot_rows, skipped


def _results_dataframe(fit_results: list[dict[str, object]]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for r in fit_results:
        records.append(
            {
                "Dataset": r["name"],
                "Model": r["model"],
                "sigma_dc (uS/cm)": r.get("sigma_dc_uS_cm"),
                "sigma_inf (uS/cm)": r.get("sigma_inf_uS_cm"),
                "RMSE (uS/cm)": r.get("rmse"),
                "RMSE CI (uS/cm)": _fmt_ci(r.get("rmse_ci")),
                "m1 (-)": r.get("p_m1"),
                "tau1 (s)": r.get("p_tau1_s"),
                "c/a1 (-)": r.get("p_c1"),
                "beta1 (-)": r.get("p_beta1"),
                "m2 (-)": r.get("p_m2"),
                "tau2 (s)": r.get("p_tau2_s"),
                "c/a2 (-)": r.get("p_c2"),
                "beta2 (-)": r.get("p_beta2"),
                "f_peak (Hz)": r.get("f_peak_hz"),
                "f_peak CI (Hz)": _fmt_ci(r.get("f_peak_hz_ci")),
                "sigma''_peak (uS/cm)": r.get("sigma_q_peak_uS_cm"),
                "sigma''_peak CI (uS/cm)": _fmt_ci(r.get("sigma_q_peak_uS_cm_ci")),
                "tau_peak (s)": r.get("tau_peak_s"),
                "tau_peak CI (s)": _fmt_ci(r.get("tau_peak_s_ci")),
            }
        )
    return pd.DataFrame.from_records(records)


def _build_fit_plot(model_name: str, plot_rows: list[dict[str, object]]):
    fig, axs = analysis_plotting.init_2x2_layout(f"Model Fit: {model_name}")
    for idx, row in enumerate(plot_rows):
        color = f"C{idx % 10}"
        freq = np.asarray(row["freq"], dtype=float)
        sigma_in = np.asarray(row["sigma_in"], dtype=float)
        sigma_q = np.asarray(row["sigma_q"], dtype=float)
        pred_in = np.asarray(row["pred_in"], dtype=float)
        pred_q = np.asarray(row["pred_q"], dtype=float)
        raw_phase = np.asarray(row["raw_phase"], dtype=float)
        pred_phase = np.asarray(row["pred_phase"], dtype=float)
        resid = np.asarray(row["resid"], dtype=float)
        f_peak = float(row["f_peak"])

        axs[0, 0].plot(freq, sigma_in, linestyle="None", marker="o", markersize=4, color=color)
        axs[0, 0].plot(freq, pred_in, linestyle="-", linewidth=1.8, color=color, label=str(row["name"]))
        axs[0, 1].plot(freq, sigma_q, linestyle="None", marker="o", markersize=4, color=color)
        axs[0, 1].plot(freq, pred_q, linestyle="-", linewidth=1.8, color=color)
        axs[0, 1].axvline(f_peak, color=color, linestyle="--", alpha=0.35, linewidth=1.0)
        axs[1, 0].plot(freq, raw_phase, linestyle="None", marker="o", markersize=4, color=color)
        axs[1, 0].plot(freq, pred_phase, linestyle="-", linewidth=1.8, color=color)
        axs[1, 1].plot(freq, resid, color=color, linewidth=1.1)

    axs[1, 1].axis("on")
    axs[1, 1].set_xscale("log")
    axs[1, 1].set_title("Residual Magnitude")
    axs[1, 1].set_xlabel("Frequency (Hz)")
    axs[1, 1].set_ylabel("|residual| (uS/cm)")
    axs[1, 1].grid(True, which="both", linestyle="--", alpha=0.35)
    axs[0, 0].legend(loc="best", frameon=False)
    fig.tight_layout()
    return fig


def _render_plot_compare_tab() -> None:
    st.subheader("Plot and Compare")
    st.caption("Plot raw/processed SIP files, compute Mean ± STD across files, and compare saved mean-DF CSVs.")

    c1, c2, c3 = st.columns(3)
    with c1:
        gf = st.number_input("Geometric factor (1/cm)", min_value=1e-12, value=1.0, format="%.6g", key="plot_gf")
    with c2:
        fmin = st.number_input("Min freq (Hz)", min_value=0.0, value=0.0, format="%.6g", key="plot_fmin")
    with c3:
        fmax = st.number_input("Max freq (Hz)", min_value=0.0, value=0.0, format="%.6g", key="plot_fmax")

    mean_only = st.checkbox("Mean ± STD only", value=False, key="plot_mean_only")
    uploaded_plot_files = st.file_uploader(
        "Upload SIP CSV(s) for plotting",
        type=["csv", "txt"],
        accept_multiple_files=True,
        key="plot_uploader",
    )

    if uploaded_plot_files:
        st.write(f"Loaded files: {len(uploaded_plot_files)}")
        st.dataframe(
            pd.DataFrame({"filename": [f.name for f in uploaded_plot_files]}),
            use_container_width=True,
            height=180,
        )

    if st.button("Plot Uploaded Files", type="primary", key="plot_button"):
        fmin_val = fmin if fmin > 0 else None
        fmax_val = fmax if fmax > 0 else None
        if fmin_val is not None and fmax_val is not None and fmin_val > fmax_val:
            st.error("Min frequency must be <= max frequency.")
            return

        if not uploaded_plot_files:
            st.error("Upload at least one file for plotting.")
            return

        combined_df, skipped = _combine_plot_uploads(uploaded_plot_files, gf=float(gf), fmin=fmin_val, fmax=fmax_val)
        if skipped:
            st.warning("Some files failed:\n\n" + "\n".join(skipped))
        if combined_df.empty:
            st.error("No valid rows available for plotting.")
            return

        try:
            fig, mean_std_df, peak_df = _build_plot_only_figure(combined_df, mean_only=mean_only)
        except Exception as exc:
            st.error(f"Plot failed: {exc}")
            return

        st.pyplot(fig, use_container_width=True)

        combined_csv = combined_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download combined dataframe CSV",
            data=combined_csv,
            file_name="sip_combined_dataframe.csv",
            mime="text/csv",
            key="dl_combined_csv",
        )

        if not peak_df.empty:
            st.subheader("Peak Summary")
            st.dataframe(peak_df, use_container_width=True)

        if not mean_std_df.empty:
            st.subheader("Mean ± STD")
            st.dataframe(mean_std_df, use_container_width=True)
            mean_csv = mean_std_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download mean±std CSV",
                data=mean_csv,
                file_name="sip_mean_std.csv",
                mime="text/csv",
                key="dl_mean_std_csv",
            )

        plot_png = BytesIO()
        fig.savefig(plot_png, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        st.download_button(
            "Download plot PNG",
            data=plot_png.getvalue(),
            file_name="sip_plot.png",
            mime="image/png",
            key="dl_plot_png",
        )

    st.divider()
    st.subheader("Compare Saved Mean DF CSV(s)")
    compare_files = st.file_uploader(
        "Upload saved mean-DF CSV(s)",
        type=["csv"],
        accept_multiple_files=True,
        key="compare_uploader",
    )

    if st.button("Plot Mean DF(s)", key="compare_plot_button"):
        if not compare_files:
            st.error("Upload at least one mean-DF CSV.")
            return

        try:
            fig, skipped = _build_mean_comparison_plot(compare_files)
        except Exception as exc:
            st.error(f"Comparison plot failed: {exc}")
            return

        if skipped:
            st.warning("Skipped files:\n\n" + "\n".join(skipped))

        st.pyplot(fig, use_container_width=True)
        comp_png = BytesIO()
        fig.savefig(comp_png, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        st.download_button(
            "Download mean comparison PNG",
            data=comp_png.getvalue(),
            file_name="sip_mean_plot.png",
            mime="image/png",
            key="dl_mean_plot_png",
        )


def _render_fit_tab() -> None:
    st.subheader("Model Fitting")
    st.caption("Upload SIP CSV files, run relaxation-model fitting, review 2x2 fit plots, and export results.")

    c1, c2, c3 = st.columns(3)
    with c1:
        gf = st.number_input("Geometric factor (1/cm)", min_value=1e-12, value=1.0, format="%.6g", key="fit_gf")
    with c2:
        fmin = st.number_input("Min freq (Hz)", min_value=0.0, value=0.0, format="%.6g", key="fit_fmin")
    with c3:
        fmax = st.number_input("Max freq (Hz)", min_value=0.0, value=0.0, format="%.6g", key="fit_fmax")

    model_name = st.selectbox("Model", MODEL_OPTIONS, index=0, key="fit_model")

    uc1, uc2, uc3, uc4 = st.columns(4)
    with uc1:
        unc_mode = st.selectbox("Uncertainty", UNCERTAINTY_OPTIONS, index=0, key="fit_unc_mode")
    with uc2:
        unc_n = st.number_input("N samples/steps", min_value=1, value=80, step=1, key="fit_unc_n")
    with uc3:
        ci_level = st.number_input("CI (%)", min_value=50.0, max_value=99.9, value=95.0, step=0.5, key="fit_ci")
    with uc4:
        cvnn_ntrain = st.number_input("CVNN train N", min_value=200, value=3000, step=100, key="fit_cvnn_train")

    cvnn_epochs = st.number_input("CVNN epochs", min_value=50, value=800, step=50, key="fit_cvnn_epochs")

    uploaded_files = st.file_uploader(
        "Upload raw SIP CSV(s) or mean-DF CSV(s)",
        type=["csv", "txt"],
        accept_multiple_files=True,
        key="fit_uploader",
    )

    if uploaded_files:
        st.write(f"Loaded files: {len(uploaded_files)}")
        st.dataframe(pd.DataFrame({"filename": [f.name for f in uploaded_files]}), use_container_width=True, height=180)

    if not st.button("Fit Uploaded Files", type="primary", key="fit_button"):
        return

    if not uploaded_files:
        st.error("Upload at least one file to fit.")
        return

    fmin_val = fmin if fmin > 0 else None
    fmax_val = fmax if fmax > 0 else None
    if fmin_val is not None and fmax_val is not None and fmin_val > fmax_val:
        st.error("Min frequency must be <= max frequency.")
        return

    with st.spinner("Running fits..."):
        fit_results, plot_rows, skipped = _fit_uploaded_files(
            uploaded_files,
            model_name=model_name,
            gf=float(gf),
            fmin=fmin_val,
            fmax=fmax_val,
            unc_mode=unc_mode,
            unc_n=int(unc_n),
            ci_level=float(ci_level),
            cvnn_ntrain=int(cvnn_ntrain),
            cvnn_epochs=int(cvnn_epochs),
        )

    if skipped:
        st.warning("Some files failed:\n\n" + "\n".join(skipped))
    if not fit_results:
        st.error("No files could be fitted.")
        return

    fig = _build_fit_plot(model_name, plot_rows)
    st.pyplot(fig, use_container_width=True)

    df = _results_dataframe(fit_results)
    st.subheader("Fit Summary")
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download fit results CSV", data=csv_bytes, file_name="sip_fit_results.csv", mime="text/csv", key="dl_fit_csv"
    )

    png_buffer = BytesIO()
    fig.savefig(png_buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    st.download_button(
        "Download fit plot PNG",
        data=png_buffer.getvalue(),
        file_name="sip_fit_plot.png",
        mime="image/png",
        key="dl_fit_png",
    )


def main() -> None:
    st.set_page_config(layout="wide", page_title="SIP Analysis Web App")
    st.title("SIP Analysis Web App")

    tab_plot, tab_fit = st.tabs(["Plot and Compare", "Model Fitting"])
    with tab_plot:
        _render_plot_compare_tab()
    with tab_fit:
        _render_fit_tab()


if __name__ == "__main__":
    main()
