#
# Author: Charles L. Bérubé
# Created on: Tue Sep 10 2024
#
# Copyright (c) 2024 CL Bérubé JL Gagnon & S Gagnon
#

import os

import numpy as np
import torch

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from plotlib import colors

plt.style.use("../seg.mplstyle")


# Quelques définitions de base
device = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # utilise le gpu si possible, sinon cpu


n_data = int(1e4)  # utiliser le bon nombre selon le n_data utilisé dans 01_gen_data.py

exp = "CVNN"

data_dir = f"./data/{exp}"  # relatif à ce .py file
fig_dir = "./figures"
os.makedirs(fig_dir, exist_ok=True)


X = torch.load(f"{data_dir}/X(n={n_data}).pt", weights_only=True)
y = torch.load(f"{data_dir}/y(n={n_data}).pt", weights_only=True)


n_curves = 5


X_py = X[(y == 0)[:, 0]].numpy()
X_gp = X[(y == 1)[:, 0]].numpy()

np.random.seed(8)
idx_py = np.random.choice(len(X_py), n_curves)
idx_gp = np.random.choice(len(X_gp), n_curves)

plot_py = X_py[idx_py].T
plot_gp = X_gp[idx_gp].T


plot_py = np.quantile(X_py.real, 0.5, 0) + 1j * np.quantile(X_py.imag, 0.5, 0)
plot_gp = np.quantile(X_gp.real, 0.5, 0) + 1j * np.quantile(X_gp.imag, 0.5, 0)

n_freq = 32  # nombre de fréquences
fmin_log10 = 2  # 10^2 (100 Hz)
fmax_log10 = 6  # 10^6 (1 MHz)
f = torch.logspace(fmin_log10, fmax_log10, n_freq)

kwargs = dict(
    ls="none",
    ms=3,
    alpha=1.0,
    mew=0.7,
    capsize=1.5,
    capthick=0.7,
    elinewidth=0,
)
colors = dict(py="k", gp="0.7")

mfc = dict(py="w", gp=colors["gp"])

fig, ax = plt.subplots()
error = np.array(
    list(
        zip(
            plot_py.real - np.quantile(X_py.real, 0.16, 0),
            np.quantile(X_py.real, 0.84, 0) - plot_py.real,
        )
    )
).T
ax.errorbar(
    f,
    plot_py.real,
    error,
    color=colors["py"],
    mfc=mfc["py"],
    marker="^",
    label=r"$\sigma'_\mathrm{p}$",
    **kwargs,
)
error = np.array(
    list(
        zip(
            plot_gp.real - np.quantile(X_gp.real, 0.16, 0),
            np.quantile(X_gp.real, 0.84, 0) - plot_gp.real,
        )
    )
).T
ax.errorbar(
    f,
    plot_gp.real,
    error,
    color=colors["gp"],
    mfc=mfc["gp"],
    marker="^",
    label=r"$\sigma'_\mathrm{g}$",
    **kwargs,
)
error = np.array(
    list(
        zip(
            plot_py.imag - np.quantile(X_py.imag, 0.16, 0),
            np.quantile(X_py.imag, 0.84, 0) - plot_py.imag,
        )
    )
).T
ax.errorbar(
    f,
    plot_py.imag,
    error,
    color=colors["py"],
    mfc=mfc["py"],
    marker="v",
    label=r"$\sigma''_\mathrm{p}$",
    **kwargs,
)
error = np.array(
    list(
        zip(
            plot_gp.imag - np.quantile(X_gp.imag, 0.16, 0),
            np.quantile(X_gp.imag, 0.84, 0) - plot_gp.imag,
        )
    )
).T
ax.errorbar(
    f,
    plot_gp.imag,
    error,
    color=colors["gp"],
    mfc=mfc["gp"],
    marker="v",
    label=r"$\sigma''_\mathrm{g}$",
    **kwargs,
)
ax.set_xscale("log")

# remove duplicates
handles, labels = plt.gca().get_legend_handles_labels()
newLabels, newHandles = [], []
for handle, label in zip(handles, labels):
    if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)

ax.legend(
    newHandles,
    newLabels,
    ncols=2,
    loc="upper left",
)
ax.set_xlabel(r"$f$ (Hz)")
ax.set_ylabel(r"$\sigma_\mathrm{eff}$ (S/m)")

plt.savefig(f"{fig_dir}/classip-data.pdf")
plt.savefig(f"{fig_dir}/classip-data.png", dpi=300)
plt.savefig(f"{fig_dir}/classip-data.svg")
