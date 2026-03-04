from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover
    least_squares = None

try:
    from sklearn.neural_network import MLPRegressor
except Exception:  # pragma: no cover
    MLPRegressor = None

try:
    import emcee
except Exception:  # pragma: no cover
    emcee = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

CVNN_REPO_DIR = Path(__file__).resolve().parent.parent / "clberube-cvnn-induced-polarization-fb3656b"


def sigma_to_rho(sigma_in_uS_cm: np.ndarray, sigma_q_uS_cm: np.ndarray) -> np.ndarray:
    sigma = (np.asarray(sigma_in_uS_cm, dtype=float) + 1j * np.asarray(sigma_q_uS_cm, dtype=float)) * 1e-6
    sigma = np.where(np.abs(sigma) < 1e-18, 1e-18 + 0j, sigma)
    return 1.0 / sigma


def rho_to_sigma_uS_cm(rho_ohm_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rho = np.asarray(rho_ohm_cm, dtype=np.complex128)
    rho = np.where(np.abs(rho) < 1e-18, 1e-18 + 0j, rho)
    sigma = 1.0 / rho * 1e6
    return np.real(sigma), np.imag(sigma)


def _model_spec(model_name: str) -> tuple[int, bool]:
    specs = {
        "Cole-Cole": (1, False),
        "Double Cole-Cole": (2, False),
        "Havriliak-Negami": (1, True),
        "Double Havriliak-Negami": (2, True),
    }
    if model_name not in specs:
        raise ValueError(f"Unsupported model: {model_name}")
    return specs[model_name]


def _rho_hn_spectrum_from_params(
    frequency_hz: np.ndarray,
    params: np.ndarray,
    n_terms: int,
    beta_free: bool,
) -> np.ndarray:
    omega = 2.0 * np.pi * np.asarray(frequency_hz, dtype=float)
    r0 = float(np.exp(params[0]))
    rho = np.full_like(omega, r0, dtype=np.complex128)
    idx = 1
    total_m = 0.0
    for _ in range(n_terms):
        m = float(params[idx])
        tau = float(np.exp(params[idx + 1]))
        alpha = float(params[idx + 2])
        idx += 3
        beta = float(params[idx]) if beta_free else 1.0
        if beta_free:
            idx += 1
        total_m += m
        term = (1.0 - (1.0 / np.power(1.0 + np.power(1j * omega * tau, alpha), beta)))
        rho = rho - r0 * m * term
    if total_m >= 0.995:
        rho = rho * (1.0 + (total_m - 0.995) * 10.0)
    return rho


def fit_relaxation_model_with_params(
    model_name: str,
    frequency_hz: np.ndarray,
    sigma_in: np.ndarray,
    sigma_q: np.ndarray,
    initial_params: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, int, bool]:
    if least_squares is None:
        raise RuntimeError("scipy is required for model fitting. Install scipy to use this tab.")

    n_terms, beta_free = _model_spec(model_name)
    n_params = 1 + n_terms * (4 if beta_free else 3)
    n_points = len(frequency_hz)
    if n_points < 8 or (2 * n_points) < (3 * n_params):
        raise ValueError(
            f"Not enough points for {model_name} fit: got {n_points}, need at least "
            f"{max(8, (3 * n_params + 1) // 2)}."
        )

    y_in = np.asarray(sigma_in, dtype=float)
    y_q = np.asarray(sigma_q, dtype=float)
    y_rho = sigma_to_rho(y_in, y_q)
    rho_re = np.real(y_rho)
    rho_im = np.imag(y_rho)
    scale_re = max(float(np.nanstd(rho_re)), 1e-9)
    scale_im = max(float(np.nanstd(rho_im)), 1e-9)

    fmin = float(np.nanmin(frequency_hz))
    fmax = float(np.nanmax(frequency_hz))
    r0_guess = max(1e-9, float(np.nanmedian(np.clip(rho_re[: max(2, len(rho_re) // 6)], 1e-9, np.inf))))
    peak_idx = int(np.nanargmax(np.abs(rho_im))) if np.isfinite(rho_im).any() else max(0, n_points // 2)
    tau0 = 1.0 / (2.0 * np.pi * float(np.clip(frequency_hz[peak_idx], fmin, fmax)))
    tau_min = 1.0 / (2.0 * np.pi * fmax * 20.0)
    tau_max = 1.0 / (2.0 * np.pi * fmin / 20.0)

    p0 = [np.log(r0_guess)]
    for i in range(n_terms):
        shift = (i - (n_terms - 1) / 2.0) * 0.8
        p0.extend([0.2 / n_terms, np.log(np.clip(tau0 * np.exp(shift), tau_min, tau_max)), 0.6])
        if beta_free:
            p0.append(0.8)
    p0 = np.asarray(p0, dtype=float)

    lb = [np.log(1e-9)]
    ub = [np.log(1e9)]
    for _ in range(n_terms):
        lb.extend([0.0, np.log(tau_min), 0.05])
        ub.extend([0.95, np.log(tau_max), 1.0])
        if beta_free:
            lb.append(0.05)
            ub.append(1.0)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    if initial_params is not None and len(initial_params) == len(p0):
        p0 = np.clip(np.asarray(initial_params, dtype=float), lb + 1e-10, ub - 1e-10)

    def residuals(p: np.ndarray) -> np.ndarray:
        rho_pred = _rho_hn_spectrum_from_params(frequency_hz, p, n_terms=n_terms, beta_free=beta_free)
        stride = 4 if beta_free else 3
        penalty = max(0.0, float(np.sum(p[1 : 1 + n_terms * stride : stride])) - 0.98)
        return np.concatenate(
            [
                (np.real(rho_pred) - rho_re) / scale_re,
                (np.imag(rho_pred) - rho_im) / scale_im,
                np.array([penalty * 50.0]),
            ]
        )

    fit = least_squares(residuals, p0, bounds=(lb, ub), loss="soft_l1", f_scale=1.0, max_nfev=6000)
    rho_pred = _rho_hn_spectrum_from_params(frequency_hz, fit.x, n_terms=n_terms, beta_free=beta_free)
    pred_in, pred_q = rho_to_sigma_uS_cm(rho_pred)
    rmse = float(np.sqrt(np.mean((pred_in - y_in) ** 2 + (pred_q - y_q) ** 2)))
    return pred_in, pred_q, rmse, fit.x, n_terms, beta_free


def fit_relaxation_model(
    model_name: str,
    frequency_hz: np.ndarray,
    sigma_in: np.ndarray,
    sigma_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    pred_in, pred_q, rmse, _, _, _ = fit_relaxation_model_with_params(
        model_name=model_name,
        frequency_hz=frequency_hz,
        sigma_in=sigma_in,
        sigma_q=sigma_q,
    )
    return pred_in, pred_q, rmse


def _predict_peaks_from_curve(frequency_hz: np.ndarray, sigma_q: np.ndarray) -> tuple[float, float, float]:
    x_dense = np.logspace(np.log10(float(np.min(frequency_hz))), np.log10(float(np.max(frequency_hz))), 700)
    y_dense = np.interp(np.log10(x_dense), np.log10(frequency_hz), sigma_q)
    idx = int(np.nanargmax(y_dense))
    f_peak = float(x_dense[idx])
    sigma_peak = float(y_dense[idx])
    tau_peak = 1.0 / (2.0 * np.pi * f_peak)
    return f_peak, sigma_peak, tau_peak


def _ci_from_samples(values: list[float], ci_level: float) -> tuple[float, float] | None:
    if not values:
        return None
    alpha = max(0.0, min(49.0, (100.0 - ci_level) / 2.0))
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, alpha)), float(np.percentile(arr, 100.0 - alpha))


def bootstrap_uncertainty_for_fit(
    model_name: str,
    frequency_hz: np.ndarray,
    sigma_in: np.ndarray,
    sigma_q: np.ndarray,
    n_boot: int = 80,
    ci_level: float = 95.0,
) -> dict[str, tuple[float, float]]:
    if model_name == "Neural Network (experimental)":
        raise ValueError("Bootstrap CI is disabled for neural-network fits in this version.")

    rng = np.random.default_rng(42)
    n = len(frequency_hz)
    f_samples: list[float] = []
    s_samples: list[float] = []
    t_samples: list[float] = []
    rmse_samples: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f_b = frequency_hz[idx]
        in_b = sigma_in[idx]
        q_b = sigma_q[idx]
        work = pd.DataFrame({"f": f_b, "in": in_b, "q": q_b}).groupby("f", as_index=False).mean().sort_values("f")
        f_b = work["f"].to_numpy(dtype=float)
        in_b = work["in"].to_numpy(dtype=float)
        q_b = work["q"].to_numpy(dtype=float)
        if len(f_b) < 8:
            continue
        try:
            _, pred_q_b, rmse_b = fit_relaxation_model(model_name, f_b, in_b, q_b)
            fpk, spk, tpk = _predict_peaks_from_curve(f_b, pred_q_b)
        except Exception:
            continue
        f_samples.append(fpk)
        s_samples.append(spk)
        t_samples.append(tpk)
        rmse_samples.append(rmse_b)

    if len(f_samples) < max(12, n_boot // 4):
        raise ValueError("Bootstrap CI failed: too few successful resampled fits.")

    out: dict[str, tuple[float, float]] = {}
    for key, arr in [
        ("f_peak_hz_ci", f_samples),
        ("sigma_q_peak_uS_cm_ci", s_samples),
        ("tau_peak_s_ci", t_samples),
        ("rmse_ci", rmse_samples),
    ]:
        ci = _ci_from_samples(arr, ci_level=ci_level)
        if ci is not None:
            out[key] = ci
    return out


def mcmc_uncertainty_for_fit(
    model_name: str,
    frequency_hz: np.ndarray,
    sigma_in: np.ndarray,
    sigma_q: np.ndarray,
    best_params: np.ndarray,
    n_terms: int,
    beta_free: bool,
    n_steps: int = 1200,
    ci_level: float = 95.0,
) -> dict[str, tuple[float, float]]:
    if emcee is None:
        raise RuntimeError("MCMC requires emcee. Install emcee to enable MCMC uncertainty.")
    if model_name == "Neural Network (experimental)":
        raise ValueError("MCMC uncertainty is not supported for neural-network fits.")

    y_in = np.asarray(sigma_in, dtype=float)
    y_q = np.asarray(sigma_q, dtype=float)
    y_rho = sigma_to_rho(y_in, y_q)
    scale_re = max(float(np.nanstd(np.real(y_rho))), 1e-9)
    scale_im = max(float(np.nanstd(np.imag(y_rho))), 1e-9)

    fmin = float(np.nanmin(frequency_hz))
    fmax = float(np.nanmax(frequency_hz))
    tau_min = 1.0 / (2.0 * np.pi * fmax * 20.0)
    tau_max = 1.0 / (2.0 * np.pi * fmin / 20.0)
    lb = [np.log(1e-9)]
    ub = [np.log(1e9)]
    for _ in range(n_terms):
        lb.extend([0.0, np.log(tau_min), 0.05])
        ub.extend([0.95, np.log(tau_max), 1.0])
        if beta_free:
            lb.append(0.05)
            ub.append(1.0)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    def log_prob(theta: np.ndarray) -> float:
        if np.any(theta <= lb) or np.any(theta >= ub):
            return -np.inf
        rho_pred = _rho_hn_spectrum_from_params(frequency_hz, theta, n_terms=n_terms, beta_free=beta_free)
        r = np.concatenate(
            [
                (np.real(rho_pred) - np.real(y_rho)) / scale_re,
                (np.imag(rho_pred) - np.imag(y_rho)) / scale_im,
            ]
        )
        return float(-0.5 * np.sum(r * r))

    ndim = len(best_params)
    nwalkers = max(24, 2 * ndim + 6)
    spread = np.maximum(np.abs(best_params), 1.0) * 1e-3
    p0 = best_params[None, :] + spread[None, :] * np.random.default_rng(42).standard_normal((nwalkers, ndim))
    p0 = np.clip(p0, lb + 1e-8, ub - 1e-8)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(p0, int(max(400, n_steps)), progress=False)
    burn = int(max(100, n_steps // 2))
    thin = max(1, n_steps // 250)
    samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
    if len(samples) == 0:
        raise RuntimeError("MCMC produced no samples after burn-in.")

    if len(samples) > 600:
        idx = np.random.default_rng(42).choice(len(samples), size=600, replace=False)
        samples = samples[idx]

    f_samples: list[float] = []
    s_samples: list[float] = []
    t_samples: list[float] = []
    for theta in samples:
        rho_pred = _rho_hn_spectrum_from_params(frequency_hz, theta, n_terms=n_terms, beta_free=beta_free)
        _, pred_q = rho_to_sigma_uS_cm(rho_pred)
        fpk, spk, tpk = _predict_peaks_from_curve(frequency_hz, pred_q)
        f_samples.append(fpk)
        s_samples.append(spk)
        t_samples.append(tpk)

    out: dict[str, tuple[float, float]] = {}
    for key, arr in [
        ("f_peak_hz_ci", f_samples),
        ("sigma_q_peak_uS_cm_ci", s_samples),
        ("tau_peak_s_ci", t_samples),
    ]:
        ci = _ci_from_samples(arr, ci_level=ci_level)
        if ci is not None:
            out[key] = ci
    return out


def fit_neural_network(
    frequency_hz: np.ndarray,
    sigma_in: np.ndarray,
    sigma_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    if MLPRegressor is None:
        raise RuntimeError("scikit-learn is required for neural-network fitting. Install scikit-learn to use this.")

    x_log = np.log10(np.asarray(frequency_hz, dtype=float))
    x_center = float(np.mean(x_log))
    x_scale = max(float(np.std(x_log)), 1e-6)
    x = np.column_stack([((x_log - x_center) / x_scale), np.linspace(-1.0, 1.0, len(x_log))])
    y_in = np.asarray(sigma_in, dtype=float)
    y_q = np.asarray(sigma_q, dtype=float)
    y = np.column_stack([y_in, y_q])
    y_std = np.maximum(np.std(y, axis=0), 1e-6)
    y_mean = np.mean(y, axis=0)
    y_scaled = (y - y_mean) / y_std

    model = MLPRegressor(
        hidden_layer_sizes=(40, 40, 40),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-4,
        batch_size=min(50, len(x)),
        max_iter=10000,
        random_state=42,
    )
    noise_scale = np.maximum(np.abs(y_scaled), 1.0) * 0.01
    rng = np.random.default_rng(42)
    y_noisy = y_scaled + rng.normal(0.0, noise_scale)
    model.fit(x, y_noisy)
    pred_scaled = model.predict(x)
    pred = pred_scaled * y_std + y_mean
    pred_in = pred[:, 0]
    pred_q = pred[:, 1]
    rmse = float(np.sqrt(np.mean((pred_in - y_in) ** 2 + (pred_q - y_q) ** 2)))
    return pred_in, pred_q, rmse


def _load_paper_parestim_models():
    if not CVNN_REPO_DIR.exists():
        raise FileNotFoundError(f"CVNN repo not found: {CVNN_REPO_DIR}")
    model_file = CVNN_REPO_DIR / "parestim" / "models.py"
    if not model_file.exists():
        raise FileNotFoundError(f"Missing paper model file: {model_file}")
    spec = importlib.util.spec_from_file_location("paper_parestim_models", model_file)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load paper CVNN model module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _generate_cvnn_training_set(
    frequency_hz: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
):
    if torch is None:
        raise RuntimeError("PyTorch is required for paper CVNN mode.")
    module = _load_paper_parestim_models()
    ColeCole = module.ColeCole
    max_modes = 2
    f = torch.tensor(frequency_hz, dtype=torch.float32)
    w = 2 * torch.pi * f
    logtau_min = float(torch.log10(1.0 / w.max()).item() - 1.0)
    logtau_max = float(torch.log10(1.0 / w.min()).item() + 1.0)

    X = torch.empty(n_samples, len(frequency_hz), dtype=torch.complex64)
    y = torch.empty(n_samples, 3 * max_modes, dtype=torch.float32)
    for i in range(n_samples):
        r0 = torch.tensor([1.0], dtype=torch.float32)
        while True:
            m = rng.uniform(0.0, 1.0, size=max_modes).astype(np.float32)
            m_sum = float(np.sum(m))
            if 0 < m_sum < 1:
                break
        m = torch.tensor(m, dtype=torch.float32)
        tau = torch.tensor([1.0, 1.0], dtype=torch.float32)
        while float(abs((tau[0] / tau[1]).item())) < 100.0:
            vals = 10 ** rng.uniform(logtau_min, logtau_max, size=max_modes)
            vals = np.sort(vals)[::-1].copy()
            tau = torch.tensor(vals, dtype=torch.float32)
        c = torch.tensor(rng.uniform(0.1, 1.0, size=max_modes), dtype=torch.float32)
        Z = ColeCole(w, r0, m, tau, c)
        Z = Z + Z * torch.randn_like(Z) * 0.01
        tau_n = (torch.log10(tau) - logtau_min) / (logtau_max - logtau_min)
        X[i] = Z
        y[i] = torch.cat((m, tau_n, c))

    return X, y, logtau_min, logtau_max


def fit_cvnn_paper_local(
    frequency_hz: np.ndarray,
    sigma_in: np.ndarray,
    sigma_q: np.ndarray,
    n_train: int = 3000,
    n_epoch: int = 800,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("PyTorch is required for paper CVNN mode.")
    module = _load_paper_parestim_models()
    cMLP = module.cMLP
    ColeCole = module.ColeCole

    freq = np.asarray(frequency_hz, dtype=float)
    if np.any(freq <= 0) or len(freq) < 10:
        raise ValueError("Paper CVNN mode requires at least 10 positive frequency points.")

    f_grid = np.logspace(np.log10(float(freq.min())), np.log10(float(freq.max())), 32)
    sig_c = np.asarray(sigma_in, dtype=float) + 1j * np.asarray(sigma_q, dtype=float)
    sig_c = np.where(np.abs(sig_c) < 1e-12, 1e-12 + 0j, sig_c)
    rho = 1.0 / sig_c
    rho_grid = np.interp(np.log10(f_grid), np.log10(freq), rho.real) + 1j * np.interp(
        np.log10(f_grid), np.log10(freq), rho.imag
    )
    r0 = float(np.median(np.clip(rho_grid.real[: max(2, len(rho_grid) // 6)], 1e-9, np.inf)))
    rho_norm = rho_grid / r0

    rng = np.random.default_rng(42)
    X_train, y_train, logtau_min, logtau_max = _generate_cvnn_training_set(f_grid, int(n_train), rng)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=min(50, len(dataset)), shuffle=True)

    model = cMLP(
        input_size=32,
        hidden_size=[40, 40, 40],
        output_size=6,
        bias=False,
        dropout_p=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(int(n_epoch)):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            yhat = model(xb)
            loss = criterion(yhat, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    x_eval = torch.tensor(rho_norm[np.newaxis, :], dtype=torch.complex64)
    with torch.no_grad():
        y_hat = model(x_eval)[0].detach().cpu().numpy()

    m_hat = np.clip(y_hat[:2], 0.0, 0.99)
    tau_n_hat = np.clip(y_hat[2:4], 0.0, 1.0)
    c_hat = np.clip(y_hat[4:6], 0.05, 1.0)
    tau_hat = 10 ** (tau_n_hat * (logtau_max - logtau_min) + logtau_min)

    _ = ColeCole(
        torch.tensor(2 * np.pi * f_grid, dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        torch.tensor(m_hat, dtype=torch.float32),
        torch.tensor(tau_hat, dtype=torch.float32),
        torch.tensor(c_hat, dtype=torch.float32),
    ).detach().cpu().numpy()

    p0 = np.array(
        [
            np.log(max(r0, 1e-9)),
            float(np.clip(m_hat[0], 0.0, 0.9)),
            np.log(max(float(tau_hat[0]), 1e-12)),
            float(np.clip(c_hat[0], 0.05, 1.0)),
            float(np.clip(m_hat[1], 0.0, 0.9)),
            np.log(max(float(tau_hat[1]), 1e-12)),
            float(np.clip(c_hat[1], 0.05, 1.0)),
        ],
        dtype=float,
    )
    pred_in, pred_q, rmse, best_p, _, _ = fit_relaxation_model_with_params(
        "Double Cole-Cole",
        frequency_hz=freq,
        sigma_in=np.asarray(sigma_in, dtype=float),
        sigma_q=np.asarray(sigma_q, dtype=float),
        initial_params=p0,
    )
    return pred_in, pred_q, rmse, best_p


def summarize_relaxation_params(
    model_name: str,
    params: np.ndarray,
    n_terms: int,
    beta_free: bool,
) -> str:
    rho0 = float(np.exp(params[0]))
    idx_sum = 1
    m_total = 0.0
    for _ in range(n_terms):
        m_total += float(params[idx_sum])
        idx_sum += 4 if beta_free else 3
    m_total = float(np.clip(m_total, 0.0, 0.98))
    sigma_dc_uS_cm = 1e6 / max(rho0, 1e-20)
    sigma_inf_uS_cm = 1e6 / max(rho0 * max(1e-6, 1.0 - m_total), 1e-20)

    parts = [f"sigma_dc={sigma_dc_uS_cm:.4g}uS/cm", f"sigma_inf={sigma_inf_uS_cm:.4g}uS/cm"]
    idx = 1
    for k in range(n_terms):
        m = float(params[idx])
        tau = float(np.exp(params[idx + 1]))
        alpha = float(params[idx + 2])
        idx += 3
        beta = float(params[idx]) if beta_free else 1.0
        if beta_free:
            idx += 1
        label = f"k{k+1}"
        if "Cole-Cole" in model_name:
            parts.append(f"{label}:m={m:.3g},tau={tau:.3g}s,c={alpha:.3g}")
        else:
            parts.append(f"{label}:d={m:.3g},tau={tau:.3g}s,a={alpha:.3g},b={beta:.3g}")
    return "; ".join(parts)


def conductivity_endpoints_from_params(params: np.ndarray, n_terms: int, beta_free: bool) -> tuple[float, float]:
    rho0 = float(np.exp(params[0]))
    idx = 1
    m_total = 0.0
    for _ in range(n_terms):
        m_total += float(params[idx])
        idx += 4 if beta_free else 3
    m_total = float(np.clip(m_total, 0.0, 0.98))
    sigma_dc_uS_cm = 1e6 / max(rho0, 1e-20)
    sigma_inf_uS_cm = 1e6 / max(rho0 * max(1e-6, 1.0 - m_total), 1e-20)
    return sigma_dc_uS_cm, sigma_inf_uS_cm
