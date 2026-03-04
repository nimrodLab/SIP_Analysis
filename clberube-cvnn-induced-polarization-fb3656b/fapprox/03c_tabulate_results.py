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

from models import cMLP, rMLP
from utilities import str_with_err
from pnp import max_u
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


labels = [
    r"$\delta n_1'$",
    r"$\delta n_1''$",
    r"$\delta n_2'$",
    r"$\delta n_2''$",
    r"$\delta U'$",
    r"$\delta U''$",
]
x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0.5

par_avg = {}
par_std = {}


for j, arc in enumerate(architectures.keys()):

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

            y_hat = model(X_test)

            if X_test.dtype == torch.float32:
                y_hat = y_hat * max_u
                y_rmse = ((y_hat - y_test) ** 2).mean(0).sqrt()
                y_range = y_test.max(0).values - y_test.min(0).values
                y_err = (y_rmse / y_range).detach().numpy()

            elif X_test.dtype == torch.complex64:
                y_hat = torch.view_as_complex(
                    torch.view_as_real(y_hat)
                    * torch.view_as_real(torch.view_as_complex(max_u.reshape(-1, 2)))
                )
                y_rmse = (
                    ((torch.view_as_real(y_hat) - torch.view_as_real(y_test)) ** 2)
                    .mean(0)
                    .sqrt()
                )
                y_range = (
                    torch.view_as_real(y_test).max(0).values
                    - torch.view_as_real(y_test).min(0).values
                )
                y_err = (y_rmse / y_range).detach().numpy().reshape(2 * output_size)

            results[arc][exp]["y"][n] = 100 * y_err

            # for i, exp in enumerate(experiments):
            val = np.array(results[arc][exp]["y"])
            par_avg[exp + "-" + arc] = val.mean(0)
            par_std[exp + "-" + arc] = val.std(0)


for j, arc in enumerate(architectures.keys()):
    fig_dir = f"./figures/{arc}"
    os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots()
    multiplier = 0.0
    for i, exp in enumerate(experiments.keys()):
        offset = width * multiplier
        rects = ax.bar(
            x + offset,
            par_avg[exp + "-" + arc],
            width,
            color=colors[exp],
            label=exp + "-" + arc,
            yerr=np.array(par_std[exp + "-" + arc]),
            alpha=1.0,
            error_kw=dict(alpha=1.0),
        )
        multiplier += 1
    ax.set_ylabel("Error (\%)")
    ax.set_xticks(x + width, labels)
    ax.legend(
        loc="upper left",
        ncol=2,
    )
    plt.savefig(f"{fig_dir}/pnp-errors.pdf")
    plt.savefig(f"{fig_dir}/pnp-errors.svg")
    plt.savefig(f"{fig_dir}/pnp-errors.png", dpi=300)


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
