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
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy import stats

from models import cMLP, rMLP
from train_mlp import architectures, n_repeat, n_epoch, n_data
from plotlib import colors


plt.style.use("../seg.mplstyle")


res_dir = "./results"
fig_dir = "./figures"
os.makedirs(fig_dir, exist_ok=True)


fpr = copy.deepcopy(architectures)
tpr = copy.deepcopy(architectures)
thresholds = copy.deepcopy(architectures)
auc = copy.deepcopy(architectures)

for j, arc in enumerate(architectures.keys()):
    experiments = architectures[arc]

    y_hat = {e: np.empty((n_repeat, n_data // 5)) for e in experiments}

    for i, exp in enumerate(experiments):
        print(f"Network type: {exp}")
        res_dir = f"./results/{arc}/{exp}"
        wt_dir = f"./weights/{arc}/{exp}"

        auc[arc][exp] = list(range(n_repeat))
        fpr[arc][exp] = list(range(n_repeat))
        tpr[arc][exp] = list(range(n_repeat))
        thresholds[arc][exp] = list(range(n_repeat))

        for n in range(n_repeat):
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

            y_hat = model(X_test).detach()

            fpr[arc][exp][n], tpr[arc][exp][n], thresholds[arc][exp][n] = roc_curve(
                y_test.numpy(), y_hat.numpy(), drop_intermediate=False
            )
            auc[arc][exp][n] = roc_auc_score(y_test.numpy(), y_hat.numpy())

        print(
            arc, exp, f"{np.mean(auc[arc][exp]):.3f} $\pm$ {np.std(auc[arc][exp]):.3f}"
        )


for j, arc in enumerate(["A", "B", "C"]):
    diff = np.array(auc[arc]["CVNN"]) - np.array(auc[arc]["RVNN"])
    ttest = stats.ttest_1samp(diff, popmean=0)
    print(exp, arc, ttest)


fig, ax = plt.subplots()
xnew = np.linspace(0, 1, 1000)
linestyles = [":", "--", "-"]

for j, arc in enumerate(["A", "B", "C"]):
    experiments = architectures[arc]
    for i, exp in enumerate(experiments):
        for n in range(n_repeat):
            f_ = interpolate.interp1d(fpr[arc][exp][n], tpr[arc][exp][n])
            tpr[arc][exp][n] = f_(xnew)

        ax.plot(
            xnew,
            np.mean(tpr[arc][exp], 0),
            color=colors[exp],
            ls=linestyles[j],
            label=exp + "-" + arc,
        )

        ax.fill_between(
            xnew,
            np.quantile(tpr[arc][exp], 0.16, 0),
            np.quantile(tpr[arc][exp], 0.84, 0),
            color=colors[exp],
            alpha=0.1,
        )

ax.plot([0, 1], [0, 1], color="k", ls="-.", lw=0.5, label="Random")
handles, labels = plt.gca().get_legend_handles_labels()
order = [
    -1,
    0,
    2,
    4,
    1,
    3,
    5,
]

ax.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc="lower right",
    ncol=1,
)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([1e-3, 1e-0])
ax.set_ylim([1e-3, 1e-0])

plt.savefig(f"{fig_dir}/ROC.pdf")
plt.savefig(f"{fig_dir}/ROC.png", dpi=300)
plt.savefig(f"{fig_dir}/ROC.svg")
