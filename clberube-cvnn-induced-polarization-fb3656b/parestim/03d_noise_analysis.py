#
# Author: Charles L. Bérubé
# Created on: Tue Sep 10 2024
#
# Copyright (c) 2024 CL Bérubé JL Gagnon & S Gagnon
#

import os
import copy

import torch
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from models import cMLP, rMLP, ColeCole
from utilities import denormalize
from train_mlp import architectures, batch_size, n_data, n_repeat
from plotlib import colors


plt.style.use("../seg.mplstyle")


# Quelques définitions de base
device = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # utilise le gpu si possible, sinon cpu

noise_levels = torch.linspace(0, 5, 25)
n_epoch = int(1e4)
n_repeat = 10

# Préparer les résultats
results = copy.deepcopy(architectures)
for j, arc in enumerate(results.keys()):
    for i, exp in enumerate(results[arc].keys()):
        results[arc][exp] = {}
        results[arc][exp]["y"] = np.empty((n_repeat, len(noise_levels), 6))
        results[arc][exp]["Z"] = np.empty((n_repeat, len(noise_levels), 32, 2))


labels = [r"$m_1$", r"$m_2$", r"$\tau_1$", r"$\tau_2$", r"$c_1$", r"$c_2$"]


fig_dir = f"./figures"
os.makedirs(fig_dir, exist_ok=True)

experiments = architectures[arc]


for k, noise in enumerate(noise_levels):
    for j, arc in enumerate(architectures.keys()):
        for i, exp in enumerate(experiments.keys()):

            wt_dir = f"./weights/{arc}/{exp}"
            data_dir = f"./data/{exp}"
            res_dir = f"./results/{arc}/{exp}"

            X = torch.load(f"{data_dir}/X(n={n_data})-noiseless.pt", weights_only=True)
            y = torch.load(f"{data_dir}/y(n={n_data})-noiseless.pt", weights_only=True)

            X += X * torch.empty(X.shape).normal_(0, noise / 100)

            input_size = X.shape[-1]
            output_size = y.shape[-1]

            # Paramètres du réseau de neurones
            model_params = dict(
                input_size=input_size,
                hidden_size=None,
                output_size=output_size,
                bias=False,
                dropout_p=0.0,
            )

            for n in range(0, n_repeat):
                print(f"Noise {k}, Network {exp}, Architecture {arc}, Repeat {n + 1}")

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=None
                )

                # Instanciation du réseau de neurones
                if X_train.dtype == torch.complex64:
                    model_params["hidden_size"] = architectures[arc][exp]
                    model = cMLP(**model_params)

                elif X_train.dtype == torch.float32:
                    model_params["hidden_size"] = architectures[arc][exp]
                    model = rMLP(**model_params)

                # Initialisation des poids
                model.load_state_dict(
                    torch.load(f"{wt_dir}/weights-best-r{n + 1}.pt", weights_only=True)
                )
                model.eval()

                if X_test.dtype == torch.complex64:

                    y_hat = model(X_test)

                elif X_test.dtype == torch.float32:
                    y_hat = model(X_test)

                max_modes = y_hat.shape[1] // 3

                if X_test.dtype == torch.complex64:
                    Z_test = X_test[:, :]
                    f = torch.logspace(-3, 4, input_size)

                elif X_test.dtype == torch.float32:
                    Z_test = torch.complex(
                        X_test[:, : input_size // 2], X_test[:, input_size // 2 :]
                    )
                    f = torch.logspace(-3, 4, input_size // 2)

                w = 2 * torch.pi * f

                m_hat = y_hat[:, :max_modes]
                tau_hat_n = y_hat[:, max_modes:-max_modes]
                c_hat = y_hat[:, -max_modes:]

                tau_hat = 10 ** denormalize(
                    tau_hat_n,
                    (1 / w.max()).log10(),
                    (1 / w.min()).log10(),
                    0,
                    1,
                )

                m_test = y_test[:, :max_modes]
                tau_test_n = y_test[:, max_modes:-max_modes]
                c_test = y_test[:, -max_modes:]

                tau_test = 10 ** denormalize(
                    tau_test_n,
                    (1 / w.max()).log10(),
                    (1 / w.min()).log10(),
                    0,
                    1,
                )

                Z_hat = ColeCole(w, 1.0, m_hat, tau_hat, c_hat)

                freq = f
                _, indices = torch.sort(freq)
                freq = freq[indices]
                Z_hat = Z_hat[indices]
                Z_test = Z_test[indices]

                y_rmse = ((y_hat - y_test) ** 2).mean(0).sqrt()
                y_range = y_test.max(0).values - y_test.min(0).values
                y_err = (y_rmse / y_range).detach().numpy()

                Z_rmse = (
                    ((torch.view_as_real(Z_hat) - torch.view_as_real(Z_test)) ** 2)
                    .mean(0)
                    .sqrt()
                )
                Z_range = (
                    torch.view_as_real(Z_test).max(0).values
                    - torch.view_as_real(Z_test).min(0).values
                )
                Z_err = (Z_rmse / Z_range).detach().numpy()

                results[arc][exp]["y"][n][k] = 100 * y_err
                results[arc][exp]["Z"][n][k] = 100 * Z_err


par_avg = {}
par_l16 = {}
par_h84 = {}
par_std = {}
fig, ax = plt.subplots()
linestyles = [":", "--", "-"]
for j, arc in enumerate(results.keys()):
    for i, exp in enumerate(experiments):
        val = np.array(results[arc][exp]["y"])
        par_avg[exp] = val.mean(0).mean(-1)

        par_l16[exp] = np.quantile(val, 0.16, axis=0).mean(-1)
        par_h84[exp] = np.quantile(val, 0.84, axis=0).mean(-1)

        ax.plot(
            noise_levels,
            par_avg[exp],
            ls=linestyles[j],
            color=colors[exp],
            label=exp + "-" + arc,
        )
        ax.fill_between(
            noise_levels,
            par_l16[exp],
            par_h84[exp],
            color=colors[exp],
            alpha=0.1,
        )
handles, labels = plt.gca().get_legend_handles_labels()
order = [0, 2, 4, 1, 3, 5]
ax.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc="upper left",
    ncol=2,
)

ax.set_xlabel("Validation noise factor")
ax.set_ylabel("Total parameter error (\%)")
plt.savefig(f"{fig_dir}/noise-errors.pdf")
plt.savefig(f"{fig_dir}/noise-errors.svg")
plt.savefig(f"{fig_dir}/noise-errors.png", dpi=300)
