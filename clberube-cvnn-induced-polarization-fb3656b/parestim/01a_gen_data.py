#
# Author: Charles L. Bérubé
# Created on: Tue Sep 10 2024
#
# Copyright (c) 2024 CL Bérubé JL Gagnon & S Gagnon
#

import os

from tqdm.auto import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.quasirandom import SobolEngine

from utilities import normalize
from models import ColeCole


# Style des graphiques
plt.style.use("../seg.mplstyle")

# Faire les calculs sur le CPU
device = "cpu"
torch.set_default_device(device)

# Nombre de data à générer
n_data = int(1e4)

# Autres paramètres pour simuler les données Cole-Cole
max_modes = 2  # nombre de modes Cole-Cole maximum (1, 2, 3, ... modes)
n_freq = 32  # nombre de fréquences
fmin_log10 = -2  # 10^-2 (1 mHz)
fmax_log10 = 4  # 10^4 (10 kHz)

# Les réseaux à tester

# Enregistre les données d'entraînement sur le disque
data_dir_real = f"./data/RVNN"
data_dir_complex = f"./data/CVNN"

os.makedirs(data_dir_real, exist_ok=True)
os.makedirs(data_dir_complex, exist_ok=True)

# Tenseurs input et output pour le réseau de neurones
input_real = torch.empty(n_data, 2 * n_freq, dtype=torch.float32)
input_complex = torch.empty(n_data, n_freq, dtype=torch.complex64)

output_real = torch.empty(n_data, 3 * max_modes, dtype=torch.float32)
output_complex = torch.empty(n_data, 3 * max_modes, dtype=torch.float32)

# Échantillonnage des fréquences aléatoires entre fmin et fmax
sampler = SobolEngine(dimension=n_freq, scramble=True)


def generate_m():
    while True:
        t = torch.rand(2)  # values in [0, 1)
        total = t.sum().item()
        if 0 < total < 1:
            return t


for i in tqdm(range(n_data)):

    # Comme la série de Sobol est normalisée (0, 1) je la dénormalise
    # pour obtenir les vraies fréquences (10^f_log10) nécessaires
    # à calculer le modèle Cole-Cole
    f = torch.logspace(fmin_log10, fmax_log10, n_freq)
    # Fréquence angulaire
    w = 2 * torch.pi * f
    logtau_min = (1 / w.max()).log10() - 1
    logtau_max = (1 / w.min()).log10() + 1

    # Choix du nombre de modes Cole-Cole (1 ou 2)
    num_modes = torch.tensor(max_modes)

    # Le paramètre r0 est fixé à 1.0
    r0 = torch.tensor([1.0])

    # Le choix aléatoire du paramètre m
    # Note que je divise par le nombre de modes parce que
    # la somme de m1, m2, m3... doit être inférieure à 1
    m = torch.zeros(max_modes)
    idx = np.random.choice(max_modes, num_modes.item(), replace=False)
    m[idx] = generate_m()

    tau = torch.tensor([1, 1])
    while (tau[0] / tau[1]).abs() < 100:
        tau = (
            10
            ** torch.empty(max_modes)
            .uniform_(logtau_min, logtau_max)
            .sort(descending=True)
            .values
        )

    # Calcul de la réponse PP
    c = torch.empty(max_modes).uniform_(0.1, 1.0)

    Z = ColeCole(w, r0, m, tau, c)
    Z += Z * torch.empty_like(Z).normal_(0, 0.01)

    # Je normalise les fréquences pour que ça soit compris entre 0 et 1
    # (on dénormalise plus tard si nécessaire)
    f = normalize(f.log10(), fmin_log10, fmax_log10, 0, 1)

    tau_n = normalize(tau.log10(), logtau_min, logtau_max, 0, 1)

    # Les inputs du réseau de neurones sont (fréquence, résistivité complexe)
    # donc input.shape = (n_data, 2*n_freq) et input.dtype = torch.complex64
    input_real[i] = torch.cat((Z.real, Z.imag))
    input_complex[i] = torch.cat((Z,))

    # Les outputs du réseau de neurones sont (type de relaxation, )
    # donc output.shape = (n_data, 1) et output.dtype = torch.float32
    output_real[i] = torch.cat((m, tau_n, c))
    output_complex[i] = torch.cat((m, tau_n, c))


if device != "cpu":
    input_real = input_real.cpu()
    input_complex = input_complex.cpu()
    output_real = output_real.cpu()
    output_complex = output_complex.cpu()
    torch.set_default_device("cpu")


torch.save(input_real, f"{data_dir_real}/X(n={n_data:.0f})-latest.pt")
torch.save(output_real, f"{data_dir_real}/y(n={n_data:.0f})-latest.pt")

torch.save(input_complex, f"{data_dir_complex}/X(n={n_data:.0f})-latest.pt")
torch.save(output_complex, f"{data_dir_complex}/y(n={n_data:.0f})-latest.pt")
