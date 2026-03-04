#
# Author: Charles L. Bérubé
# Created on: Tue Sep 10 2024
#
# Copyright (c) 2024 CL Bérubé JL Gagnon & S Gagnon
#

import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from train_mlp import architectures, n_repeat, n_epoch
from plotlib import colors


plt.style.use("../seg.mplstyle")


res_dir = "./results"
fig_dir = "./figures"
os.makedirs(fig_dir, exist_ok=True)


epochs = range(1, n_epoch + 1)
fig, ax = plt.subplots()
linestyles = [":", "--", "-"]
skip = 1
for j, arc in enumerate(["A", "B", "C"]):
    experiments = architectures[arc]

    losses = {e: np.empty((n_repeat, n_epoch)) for e in experiments}

    for i, e in enumerate(experiments):
        print(f"Network type: {e}")
        res_dir = f"./results/{arc}/{e}"

        for n in range(n_repeat):
            losses[e][n] = torch.load(
                f"{res_dir}/losses-best-r{n + 1}.pt", weights_only=True
            )["valid"]

    for i, e in enumerate(experiments):

        ax.plot(
            epochs[::skip],
            losses[e].mean(0)[::skip],
            ls=linestyles[j],
            color=colors[e],
            label=f"{e}-{arc}",
        )
        print(e, arc, losses[e].mean(0)[-1])

        ax.fill_between(
            epochs[::skip],
            np.quantile(losses[e], 0.16, axis=0)[::skip],
            np.quantile(losses[e], 0.84, axis=0)[::skip],
            color=colors[e],
            alpha=0.1,
        )

ax.set_ylabel("Classification loss")
handles, labels = plt.gca().get_legend_handles_labels()
order = [0, 2, 4, 1, 3, 5]
ax.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc="upper right",
    ncol=2,
)

ax.set_xlabel("Epoch")
plt.savefig(f"{fig_dir}/classip-learning-curves.pdf")
plt.savefig(f"{fig_dir}/classip-learning-curves.svg")
plt.savefig(f"{fig_dir}/classip-learning-curves.png", dpi=300)
