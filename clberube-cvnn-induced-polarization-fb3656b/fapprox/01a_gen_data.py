#
# Author: Charles L. Bérubé
# Created on: Tue Sep 10 2024
#
# Copyright (c) 2024 CL Bérubé JL Gagnon & S Gagnon
#

import os

import torch
import matplotlib.pyplot as plt

from pnp import generate_data, solution_PNP, a, L
from plotlib import plot_pnp_imshow


# Style des graphiques
plt.style.use("../seg.mplstyle")


fig_dir = f"./figures"
os.makedirs(fig_dir, exist_ok=True)

# Faire les calculs sur le CPU
device = "cpu"
torch.set_default_device(device)

# Nombre de data à générer
n_data = int(1e4)

# Les réseaux à tester
experiments = ["RVNN", "CVNN"]

vr = [2, 2, 2]
vi = [2, 2, 5]
x, y = generate_data(n_data, 0)
u = solution_PNP(x, y)
u += u * torch.empty_like(u).normal_(0.0, 0.01)

for i, e in enumerate(experiments):

    print(f"Network type: {e}")

    # Enregistre les données d'entraînement sur le disque
    data_dir = f"./data/{e}"
    os.makedirs(data_dir, exist_ok=True)

    # Tenseurs input et output pour le réseau de neurones

    if e == "CVNN":
        input = torch.empty(n_data, 2, dtype=torch.complex64)
        output = torch.empty(n_data, 3, dtype=torch.complex64)
        input[:, 0] = x + 1j * x
        input[:, 1] = y + 1j * y

    elif e == "RVNN":
        input = torch.empty(n_data, 2, dtype=torch.float32)
        output = torch.empty(n_data, 6, dtype=torch.float32)
        input[:, 0] = x
        input[:, 1] = y

    if e == "CVNN":
        output = u
        sc = torch.view_as_real(output).flatten(1)
        plot_pnp_imshow(
            x[:1000],
            y[:1000],
            sc[:1000],
            a.numpy(),
            L.numpy(),
            fname=f"{fig_dir}/{e}-analytical-sol-PNP",
            vr=vr,
            vi=vi,
        )

    elif e == "RVNN":
        output = torch.view_as_real(u).flatten(1)
        sr = output
        plot_pnp_imshow(
            x[:1000],
            y[:1000],
            sr[:1000],
            a.numpy(),
            L.numpy(),
            fname=f"{fig_dir}/{e}-analytical-sol-PNP",
            vr=vr,
            vi=vi,
        )

    if device != "cpu":
        input = input.cpu()
        output = output.cpu()
        torch.set_default_device("cpu")

    print(output.dtype)
    torch.save(input, f"{data_dir}/X(n={n_data:.0f})-latest.pt")
    torch.save(output, f"{data_dir}/y(n={n_data:.0f})-latest.pt")
