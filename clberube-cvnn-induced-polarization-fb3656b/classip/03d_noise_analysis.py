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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from models import cMLP, rMLP
from train_mlp import architectures, n_repeat, n_epoch, n_data
from plotlib import colors


plt.style.use("../seg.mplstyle")


fig_dir = "./figures"
os.makedirs(fig_dir, exist_ok=True)

noise_levels = torch.linspace(0, 5, 25)

fpr = copy.deepcopy(architectures)
tpr = copy.deepcopy(architectures)
thresholds = copy.deepcopy(architectures)
auc = copy.deepcopy(architectures)

# Préparer les résultats
results = copy.deepcopy(architectures)
for j, arc in enumerate(results.keys()):
    for i, exp in enumerate(results[arc].keys()):
        results[arc][exp] = {}
        results[arc][exp]["y"] = np.empty((n_repeat, len(noise_levels), 6))
        results[arc][exp]["Z"] = np.empty((n_repeat, len(noise_levels), 32, 2))
        results[arc][exp]["auc"] = np.empty((n_repeat, len(noise_levels), 1))

for k, noise in enumerate(noise_levels):

    for j, arc in enumerate(architectures.keys()):
        experiments = architectures[arc]

        for i, exp in enumerate(experiments):

            print(f"Network type: {exp}")

            wt_dir = f"./weights/{arc}/{exp}"
            data_dir = f"./data/{exp}"
            res_dir = f"./results/{arc}/{exp}"

            auc[arc][exp] = list(range(n_repeat))
            fpr[arc][exp] = list(range(n_repeat))
            tpr[arc][exp] = list(range(n_repeat))
            thresholds[arc][exp] = list(range(n_repeat))

            for n in range(n_repeat):
                print(f"Noise {k}, Network {exp}, Architecture {arc}, Repeat {n + 1}")

                X = torch.load(
                    f"{data_dir}/X(n={n_data})-noiseless.pt", weights_only=True
                )
                y = torch.load(
                    f"{data_dir}/y(n={n_data})-noiseless.pt", weights_only=True
                )

                X += X * torch.empty(X.shape).normal_(0, noise / 100)

                input_size = X.shape[-1]
                output_size = y.shape[-1]

                # Paramètres du réseau de neurones
                model_params = dict(
                    input_size=input_size,
                    hidden_size=architectures[arc][exp],
                    output_size=output_size,
                    bias=False,
                    dropout_p=0.0,
                )

                # Instanciation du réseau de neurones
                if X.dtype == torch.complex64:
                    model = cMLP(**model_params)

                elif X.dtype == torch.float32:
                    model = rMLP(**model_params)

                # Initialisation des poids
                model.load_state_dict(
                    torch.load(f"{wt_dir}/weights-best-r{n + 1}.pt", weights_only=True)
                )
                model.eval()

                y_hat = model(X).detach()

                fpr[arc][exp][n], tpr[arc][exp][n], thresholds[arc][exp][n] = roc_curve(
                    y.numpy(), y_hat.numpy(), drop_intermediate=False
                )
                auc[arc][exp][n] = roc_auc_score(y.numpy(), y_hat.numpy())

                results[arc][exp]["auc"][n][k] = auc[arc][exp][n]


par_avg = {}
par_l16 = {}
par_h84 = {}
par_std = {}
fig, ax = plt.subplots()
linestyles = [":", "--", "-"]
for j, arc in enumerate(results.keys()):
    for i, exp in enumerate(experiments):
        val = np.array(results[arc][exp]["auc"])
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
    loc="lower left",
    ncol=2,
)

ax.set_xlabel("Validation noise factor")
ax.set_ylabel("ROC AUC")
plt.savefig(f"{fig_dir}/classip-noise-errors.pdf")
plt.savefig(f"{fig_dir}/classip-noise-errors.svg")
plt.savefig(f"{fig_dir}/classip-noise-errors.png", dpi=300)
