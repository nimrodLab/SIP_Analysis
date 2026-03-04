# SIP Data Analyzer: Program and Fitting Guide

## 1) What this program does
This GUI supports three workflows:

- **Analyze SIP**: load raw SIP files, compute conductivity, plot spectra, and compute mean ± std.
- **Compare Mean DFs**: overlay previously saved mean dataframes.
- **Fit Models**: fit physical relaxation models (or NN modes) to selected spectra and summarize peak metrics + uncertainty.

The app accepts:

- Raw SIP files (instrument exports with frequency/impedance/phase)
- Mean dataframes saved by the app (`frequency_hz`, `sigma_in_phase_mean_uS_cm`, `sigma_quadrature_mean_uS_cm`)

## 2) Data conventions and units
Internally for fitting:

- Measured conductivity is treated as complex conductivity:
  - `sigma*(f) = sigma'(f) + i sigma''(f)` in **uS/cm**.
- Physical model fitting is performed in the **complex resistivity domain**:
  - `rho*(f) = 1 / sigma*(f)`.

This is important because Cole-Cole/HN formulations are classically defined in resistivity (or equivalent normalized forms).

Reported conductivity parameters:

- `sigma_dc (uS/cm)` from fitted `rho0`: `sigma_dc = 1e6 / rho0`
- `sigma_inf (uS/cm)` from fitted chargeability sum `m_total`: `sigma_inf = 1e6 / (rho0 * (1 - m_total))`

So you can interpret parameters directly in conductivity units while preserving physically consistent fitting.

## 3) Fitting options in the GUI
In **Fit Models**, each selected input is fitted independently with one selected model:

- `Cole-Cole`
- `Double Cole-Cole`
- `Havriliak-Negami`
- `Double Havriliak-Negami`
- `Neural Network (experimental)`
- `CVNN (paper repo, local train)`

### 3.1 Cole-Cole and Double Cole-Cole
For each mode `k`, parameters are:

- `m_k`: chargeability-like amplitude (dimensionless)
- `tau_k`: relaxation time (seconds)
- `c_k`: Cole-Cole exponent (dimensionless)

Double mode uses two terms to represent multi-process relaxation.

### 3.2 Havriliak-Negami and Double HN
For each mode `k`, parameters are:

- `d_k`: mode amplitude (dimensionless)
- `tau_k`: relaxation time (seconds)
- `a_k`: asymmetry exponent
- `b_k`: broadening exponent

Double mode allows two relaxation families.

### 3.3 Neural Network (experimental)
Learns curve shape directly and predicts `sigma'`, `sigma''` fit curves.
It is useful for flexible approximation but does not provide physically interpretable mode parameters by default.

### 3.4 CVNN (paper repo, local train)
Uses the local paper repository (`parestim` CVNN) as initializer and then performs robust physical refinement.
This yields physically interpretable parameters and stable curve fits.

## 4) What is reported in fit results
For each fitted dataset:

- `name`: file/DF label
- `model`: selected model
- `params`: compact parameter summary (including conductivity endpoints)
- `sigma_dc (uS/cm)`
- `sigma_inf (uS/cm)`
- `RMSE`: fit error on `(sigma', sigma'')`
- `f_peak (Hz)`: peak frequency of fitted `sigma''`
- `sigma''_peak (uS/cm)`
- `tau_peak (s) = 1/(2*pi*f_peak)`
- optional uncertainty intervals (if enabled)

## 5) Uncertainty and statistics
The app supports:

- `None`
- `Bootstrap CI`
- `MCMC (experimental)`

### 5.1 Bootstrap CI (non-parametric)
How it works:

1. Resample data points with replacement.
2. Refit the selected physical model on each resample.
3. Recompute derived metrics each time (`RMSE`, `f_peak`, `sigma''_peak`, `tau_peak`).
4. Report percentile intervals (e.g. 95% CI).

Interpretation:

- Narrow CI => stable estimate under data resampling.
- Wide CI => fit is sensitive to data perturbation (possible non-identifiability, insufficient range, or model mismatch).

Practical note:

- Bootstrap quantifies **resampling variability**, not full Bayesian posterior uncertainty.

### 5.2 MCMC (experimental)
How it works:

1. Starts from deterministic fitted parameters.
2. Samples parameter posterior with bounded priors and likelihood from residuals.
3. Propagates sampled parameters to derived metrics.
4. Reports posterior CI for `f_peak`, `sigma''_peak`, `tau_peak`.

Interpretation:

- Better for parameter-coupling and multimodality awareness than pure bootstrap.
- If CIs collapse at boundaries (e.g., `f_peak` at max measured frequency), this indicates edge-limited identifiability.

Dependencies:

- Requires `emcee`.

## 6) Reading model quality correctly
Use a combination of:

- **Visual fit quality** in all subplots (`sigma'`, `sigma''`, phase).
- **RMSE** (lower is better, but scale-dependent).
- **R² on sigma' and sigma''** (diagnostic, can differ between channels).
- **Parameter plausibility** (tau range, exponents within physical expectations).
- **Uncertainty width** (narrow vs broad CI).

A model can have low RMSE but still be physically implausible (over-flexibility, edge effects). Always inspect parameter values and CIs.

## 7) Exporting results
Use **Save Fit Results CSV** to export:

- all scalar metrics
- parameter summary and conductivity endpoints
- CI low/high columns for each uncertainty metric

Use **Save Fit Plot PNG** to export figure output.

## 8) Recommended workflow
1. Start with `Double Cole-Cole` and `Double Havriliak-Negami`.
2. Compare fit quality and parameter realism.
3. Enable `Bootstrap CI` first.
4. Use `MCMC` when parameter uncertainty/coupling is critical.
5. Export CSV and archive together with the fit figure.

## 9) Citation note
If using the CVNN paper workflow in publications, cite:

- https://doi.org/10.1093/gji/ggaf348
- https://doi.org/10.5281/zenodo.16950274
