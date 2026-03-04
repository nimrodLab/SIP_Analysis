#!/usr/bin/env python3
"""Streamlit web app for SIP analysis and model fitting."""

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


def main() -> None:
    st.set_page_config(layout="wide", page_title="SIP Analysis Web App")
    st.title("SIP Analysis Web App")
    st.caption("Upload SIP CSV files, run relaxation-model fitting, review 2x2 fit plots, and export results.")

    with st.sidebar:
        st.header("Fit Configuration")
        gf = st.number_input("Geometric factor (1/cm)", min_value=1e-12, value=1.0, format="%.6g")
        fmin = st.number_input("Min freq (Hz)", min_value=0.0, value=0.0, format="%.6g")
        fmax = st.number_input("Max freq (Hz)", min_value=0.0, value=0.0, format="%.6g")
        model_name = st.selectbox("Model", MODEL_OPTIONS, index=0)
        unc_mode = st.selectbox("Uncertainty", UNCERTAINTY_OPTIONS, index=0)
        unc_n = st.number_input("N samples/steps", min_value=1, value=80, step=1)
        ci_level = st.number_input("CI (%)", min_value=50.0, max_value=99.9, value=95.0, step=0.5)
        cvnn_ntrain = st.number_input("CVNN train N", min_value=200, value=3000, step=100)
        cvnn_epochs = st.number_input("CVNN epochs", min_value=50, value=800, step=50)
        run_fit = st.button("Fit Uploaded Files", type="primary")

    uploaded_files = st.file_uploader(
        "Upload raw SIP CSV(s) or mean-DF CSV(s)",
        type=["csv", "txt"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload at least one file to begin.")
        return

    st.write(f"Loaded files: {len(uploaded_files)}")
    st.dataframe(pd.DataFrame({"filename": [f.name for f in uploaded_files]}), use_container_width=True, height=180)

    if not run_fit:
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
    st.download_button("Download fit results CSV", data=csv_bytes, file_name="sip_fit_results.csv", mime="text/csv")

    png_buffer = BytesIO()
    fig.savefig(png_buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    st.download_button("Download fit plot PNG", data=png_buffer.getvalue(), file_name="sip_fit_plot.png", mime="image/png")


if __name__ == "__main__":
    main()
