#
# Author: Charles L. Bérubé
# Created on: Tue Sep 10 2024
#
# Copyright (c) 2024 CL Bérubé JL Gagnon & S Gagnon
#

import os

import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from models import rMLP, cMLP
from plotlib import plot_pnp_imshow
from pnp import generate_data, a, L, max_u
from train_mlp import architectures, n_data, n_repeat


plt.style.use("../seg.mplstyle")


res_dir = "./results"


# Extend ScalarFormatter
class MyScalarFormatter(ScalarFormatter):
    # Override '_set_format' with your own
    def _set_format(self):
        self.format = "%.2f"  # Show 2 decimals


# Les réseaux à tester
n_data = int(1e4)

x, y = generate_data(n_data, 0)

arc = "C"
experiments = architectures[arc]

fig_dir = f"./figures/{arc}"
os.makedirs(fig_dir, exist_ok=True)


for i, e in enumerate(experiments):
    print(f"Network type: {e}")

    model_params = dict(
        input_size=2,
        hidden_size=architectures[arc][e],
        output_size=None,
        bias=False,
        dropout_p=0.0,
    )

    wt_dir = f"./weights/{arc}/{e}"

    if e == "RVNN":
        X_test = torch.stack([x, y], -1)
        model_params["output_size"] = 6
        model = rMLP(**model_params)
        y_hat = torch.empty(n_repeat, n_data, 6, dtype=torch.float32)

    elif e == "CVNN":
        X_test = torch.stack([x, y], -1) + 1j * torch.stack([x, y], -1)
        model_params["output_size"] = 3
        model = cMLP(**model_params)
        y_hat = torch.empty(n_repeat, n_data, 3, dtype=torch.complex64)

    for n in range(n_repeat):

        model.load_state_dict(
            torch.load(f"{wt_dir}/weights-best-r{n + 1}.pt", weights_only=True)
        )
        model.train()

        if e == "RVNN":
            y_hat[n] = model(X_test) * max_u

        elif e == "CVNN":
            y_hat[n] = torch.view_as_complex(
                torch.view_as_real(model(X_test))
                * torch.view_as_real(torch.view_as_complex(max_u.reshape(-1, 2)))
            )

    vr = [2, 2, 2]
    vi = [2, 2, 5]
    if e == "RVNN":
        y_hat_plot = y_hat.mean(0).detach().numpy()
        plot_pnp_imshow(
            x,
            y,
            y_hat_plot,
            a.numpy(),
            L.numpy(),
            fname=f"{fig_dir}/{e}-approx-pnp",
            vr=vr,
            vi=vi,
        )

    elif e == "CVNN":
        y_hat_plot = torch.view_as_real(y_hat.mean(0)).flatten(1).detach().numpy()
        plot_pnp_imshow(
            x,
            y,
            y_hat_plot,
            a.numpy(),
            L.numpy(),
            fname=f"{fig_dir}/{e}-approx-pnp",
            vr=vr,
            vi=vi,
        )
