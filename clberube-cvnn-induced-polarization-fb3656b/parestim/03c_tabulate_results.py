#
# Author: Charles L. Bérubé
# Created on: Tue Sep 10 2024
#
# Copyright (c) 2024 CL Bérubé JL Gagnon & S Gagnon
#

import os
import copy

import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy import stats

from models import cMLP, rMLP, ColeCole
from utilities import str_with_err
from utilities import denormalize
from train_mlp import architectures, batch_size, n_data, n_repeat
from plotlib import colors


plt.style.use("../seg.mplstyle")


# Quelques définitions de base
device = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # utilise le gpu si possible, sinon cpu

# Préparer les résultats
results = copy.deepcopy(architectures)
for j, arc in enumerate(results.keys()):
    for i, exp in enumerate(results[arc].keys()):
        results[arc][exp] = {}
        results[arc][exp]["y"] = np.empty((n_repeat, 6))
        results[arc][exp]["Z"] = np.empty((n_repeat, 32, 2))


labels = [r"$m_1$", r"$m_2$", r"$\tau_1$", r"$\tau_2$", r"$c_1$", r"$c_2$"]


for j, arc in enumerate(architectures.keys()):

    fig_dir = f"./figures/{arc}"
    os.makedirs(fig_dir, exist_ok=True)

    experiments = architectures[arc]

    for i, exp in enumerate(experiments.keys()):
        print(f"Network type: {exp}")
        wt_dir = f"./weights/{arc}/{exp}"
        res_dir = f"./results/{arc}/{exp}"

        for n in range(0, n_repeat):

            X_test = torch.load(
                f"{res_dir}/X_test(n={n_data})-best-r{n + 1}.pt", weights_only=True
            )

            y_test = torch.load(
                f"{res_dir}/y_test(n={n_data})-best-r{n + 1}.pt", weights_only=True
            )

            input_size = X_test.shape[-1]
            output_size = y_test.shape[-1]

            # Paramètres du réseau de neurones
            model_params = dict(
                input_size=input_size,
                hidden_size=architectures[arc][exp],
                output_size=output_size,
                bias=False,
                dropout_p=0.0,
            )

            dataset_test = TensorDataset(X_test, y_test)

            dataloader = {}
            dataloader["test"] = DataLoader(dataset_test, batch_size, shuffle=True)

            # Instanciation du réseau de neurones
            if X_test.dtype == torch.complex64:
                model = cMLP(**model_params)

            elif X_test.dtype == torch.float32:
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

            results[arc][exp]["y"][n] = 100 * y_err
            results[arc][exp]["Z"][n] = 100 * Z_err

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars
    multiplier = 0.5

    par_avg = {}
    par_std = {}
    for i, exp in enumerate(experiments):
        val = np.array(results[arc][exp]["y"])
        par_avg[exp] = val.mean(0)
        par_std[exp] = val.std(0)

    fig, ax = plt.subplots()
    for key in par_avg.keys():
        offset = width * multiplier
        rects = ax.bar(
            x + offset,
            par_avg[key],
            width,
            label=key + "-" + arc,
            yerr=par_std[key],
            color=colors[key],
            alpha=0.8,
            error_kw=dict(alpha=1.0),
        )
        # ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.set_ylabel("Error (\%)")
    ax.set_xticks(x + width, labels)
    if arc == "A":
        ax.set_ylim([0, 19.5])
    ax.legend(ncols=2)

    plt.savefig(f"{fig_dir}/parameter-errors.pdf")
    plt.savefig(f"{fig_dir}/parameter-errors.svg")
    plt.savefig(f"{fig_dir}/parameter-errors.png", dpi=300)


df_Z = pd.DataFrame()
for j, arc in enumerate(results.keys()):
    for i, exp in enumerate(results[arc].keys()):
        df_Z.loc[exp + "-" + arc, r"$\rho$"] = str_with_err(
            results[arc][exp]["Z"].mean(1).mean(1).mean(0),
            results[arc][exp]["Z"].mean(1).mean(1).std(0),
        )
        df_Z.loc[exp + "-" + arc, r"$\rho'$"] = str_with_err(
            results[arc][exp]["Z"].mean(1)[0].mean(0),
            results[arc][exp]["Z"].mean(1)[0].std(0),
        )
        df_Z.loc[exp + "-" + arc, r"$\rho''$"] = str_with_err(
            results[arc][exp]["Z"].mean(1)[1].mean(0),
            results[arc][exp]["Z"].mean(1)[1].std(0),
        )

df_Z = df_Z.iloc[[0, 2, 4, 1, 3, 5]]

diff = np.empty((3, 10, 2))
for j, arc in enumerate(results.keys()):
    diff[j] = results[arc]["CVNN"]["Z"].mean(1) - results[arc]["RVNN"]["Z"].mean(1)
    ttest = stats.ttest_1samp(diff[j], popmean=0)
    df_Z.loc[f"$p$-{arc}", [r"$\rho'$", r"$\rho''$"]] = [
        f"${p:.1e}".replace("e-0", r"\times 10^{-") + "}$" for p in ttest.pvalue
    ]

diff = np.empty((3, 10))
for j, arc in enumerate(results.keys()):
    diff[j] = results[arc]["CVNN"]["Z"].mean(1).mean(1) - results[arc]["RVNN"][
        "Z"
    ].mean(1).mean(1)
    ttest = stats.ttest_1samp(diff[j], popmean=0)
    df_Z.loc[f"$p$-{arc}", [r"$\rho$"]] = (
        f"${ttest.pvalue:.1e}".replace("e-0", r"\times 10^{-") + "}$"
    )


print(df_Z.to_latex(column_format="lrrr"))


df_y = pd.DataFrame()
for j, arc in enumerate(results.keys()):
    for i, exp in enumerate(results[arc].keys()):
        for k, label in enumerate(labels):
            df_y.loc[exp + "-" + arc, label] = str_with_err(
                results[arc][exp]["y"].mean(0)[k],
                results[arc][exp]["y"].std(0)[k],
            )
        df_y.loc[exp + "-" + arc, "All"] = str_with_err(
            results[arc][exp]["y"].mean(1).mean(0),
            results[arc][exp]["y"].mean(1).std(0),
        )

df_y = df_y.iloc[[0, 2, 4, 1, 3, 5]]


diff = np.empty((3, 10, 6))
for j, arc in enumerate(results.keys()):
    diff[j] = results[arc]["CVNN"]["y"] - results[arc]["RVNN"]["y"]
    ttest = stats.ttest_1samp(diff[j], popmean=0)
    df_y.loc[f"$p$-{arc}", df_y.keys()[:-1]] = [
        f"${p:.1e}".replace("e-0", r"\times 10^{-").replace("e-1", r"\times 10^{-1")
        + "}$"
        for p in ttest.pvalue
    ]

diff = np.empty((3, 10))
for j, arc in enumerate(results.keys()):
    diff[j] = results[arc]["CVNN"]["y"].mean(1) - results[arc]["RVNN"]["y"].mean(1)
    ttest = stats.ttest_1samp(diff[j], popmean=0)
    df_y.loc[f"$p$-{arc}", df_y.keys()[-1:]] = (
        f"${ttest.pvalue:.1e}".replace("e-0", r"\times 10^{-") + "}$"
    )


print(df_y.to_latex(column_format="lrrrrrrr"))
