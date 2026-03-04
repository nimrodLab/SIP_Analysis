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
from matplotlib.ticker import ScalarFormatter

from utilities import denormalize, restore_minor_ticks_log_plot
from models import rMLP, cMLP, ColeCole
from train_mlp import architectures, hidden_size, batch_size, n_data, n_repeat
from plotlib import colors


plt.style.use("../seg.mplstyle")


res_dir = "./results"


# Extend ScalarFormatter
class MyScalarFormatter(ScalarFormatter):
    # Override '_set_format' with your own
    def _set_format(self):
        self.format = "%.2f"  # Show 2 decimals


# Les réseaux à tester

fig, axs = plt.subplots(2, 1, sharex=True)


arc = "A"
experiments = architectures[arc]

fig_dir = f"./figures/{arc}"
os.makedirs(fig_dir, exist_ok=True)


for i, e in enumerate(experiments):
    print(f"Network type: {e}")

    y_hat = torch.empty(n_repeat, 6)

    model_params = dict(
        input_size=None,
        hidden_size=architectures[arc][e],
        output_size=6,
        bias=False,
        dropout_p=0.0,
    )

    wt_dir = f"./weights/{arc}/{e}"

    if e == "RVNN":
        model_params["input_size"] = 64
        model = rMLP(**model_params)

    elif e == "CVNN":
        model_params["input_size"] = 32
        model = cMLP(**model_params)

    y_test = torch.tensor([[0.5, 0.4, 0.1, 0.7, 0.8, 0.4]])
    max_modes = y_test.shape[1] // 3

    f = torch.logspace(-3, 4, 32)
    w = 2 * torch.pi * f

    m_test = y_test[:, :max_modes]
    tau_test_n = y_test[:, max_modes:-max_modes]
    c_test = y_test[:, -max_modes:]

    tau_test = 10 ** denormalize(
        tau_test_n,
        (1 / w.max()).log10(),
        (1 / w.min()).log10(),
        0.0,
        1,
    )

    Z_test = ColeCole(w, 1.0, m_test, tau_test, c_test)
    Z_test += Z_test * torch.empty_like(Z_test).normal_(0, 0.01)

    for n in range(0, n_repeat):

        model.load_state_dict(
            torch.load(f"{wt_dir}/weights-best-r{n + 1}.pt", weights_only=True)
        )
        model.eval()

        if e == "RVNN":
            y_hat[n] = model(torch.cat((Z_test.real, Z_test.imag), -1))

        elif e == "CVNN":

            y_hat[n] = model(Z_test)

    m_hat = y_hat[:, :max_modes]
    tau_hat_n = y_hat[:, max_modes:-max_modes]
    c_hat = y_hat[:, -max_modes:]

    tau_hat = 10 ** denormalize(
        tau_hat_n,
        (1 / w.max()).log10(),
        (1 / w.min()).log10(),
        0.0,
        1,
    )

    Z_hat = ColeCole(w, 1.0, m_hat, tau_hat, c_hat)

    freq = f
    _, indices = torch.sort(freq)
    freq = freq[indices].detach().numpy()
    Z_hat = Z_hat[:, indices].detach().numpy()
    Z_test = Z_test[:, indices].detach().numpy()

    c = colors[e]
    ax = axs[i]
    ax.plot(freq, Z_hat.real.mean(0), ls="-", color=c, alpha=1.0, label=e + "-" + arc)

    ax.fill_between(
        freq,
        np.quantile(Z_hat.real, 0.025, 0),
        np.quantile(Z_hat.real, 0.975, 0),
        color=c,
        alpha=0.1,
    )
    ax.plot(freq, Z_test.real[0], "^k", ms=1, label=r"$\rho'$")
    ax.set_xscale("log")

    ax.plot(freq, -Z_hat.imag.mean(0), ls="-", color=c, alpha=1.0)

    ax.fill_between(
        freq,
        np.quantile(-Z_hat.imag, 0.025, 0),
        np.quantile(-Z_hat.imag, 0.975, 0),
        color=c,
        alpha=0.1,
    )
    ax.plot(freq, -Z_test.imag[0], "vk", ms=1, label=r"$-\rho''$")
    ax.set_xscale("log")

    custom_formatter = MyScalarFormatter(useMathText=True)

    ax.set_ylabel(r"$\rho_\mathrm{CC}/\rho_0$")
    ax.set_yscale("log")
    ax.set_ylim([6e-3, 2e0])

    ax.legend(loc="lower right", ncol=3, columnspacing=1.0)
    restore_minor_ticks_log_plot(ax, axis="x")

    axs[-1].set_xlabel("$f$ (Hz)")

plt.savefig(f"{fig_dir}/curve-fit.pdf")
plt.savefig(f"{fig_dir}/curve-fit.png", dpi=300)
