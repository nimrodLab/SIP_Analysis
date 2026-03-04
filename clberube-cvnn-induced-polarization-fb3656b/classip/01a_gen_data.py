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
from scipy.stats import qmc

from torch_spherical_model import forward_spherical


# Faire les calculs sur le CPU
device = "cpu"
torch.set_default_device(device)

# Nombre de data à générer
n_data = int(1e4)
# Type de data à générer --> remplacer par 'sph vs sph' ou 'sph vs sheet'
# type_data = "rod vs rod"

# Autres paramètres pour simuler les données PPIP
epsilon_0 = 8.854e-12  # permittivité du vide (F/m)
n_freq = 32  # nombre de fréquences
fmin_log10 = 2  # 10^2 (100 Hz)
fmax_log10 = 6  # 10^6 (1 MHz)


# Enregistre les données d'entraînement sur le disque
data_dir_real = f"./data/RVNN"
data_dir_complex = f"./data/CVNN"
os.makedirs(data_dir_real, exist_ok=True)
os.makedirs(data_dir_complex, exist_ok=True)

# Tenseurs input et output pour le réseau de neurones
# input = torch.empty(n_data, 2 * n_freq, dtype=torch.complex64)
input_real = torch.empty(n_data, 2 * n_freq, dtype=torch.float32)
input_complex = torch.empty(n_data, 1 * n_freq, dtype=torch.complex64)
output_real = torch.empty(n_data, 1, dtype=torch.float32)
output_complex = torch.empty(n_data, 1, dtype=torch.float32)

# Échantillonnage Latin hypercube des 5 paramètres aléatoires
sampler_LHS = qmc.LatinHypercube(d=5, scramble=True)
# Ordre : a_i, phi_i, D_i, sigma_i, esp_i
# En log10 pour l'échantillonnage LHS selon la littérature

# Pyrite sphérique
min_param_pyrite = [
    np.log10(1e-6),
    np.log10(0.01),
    np.log10(2.5e-4),
    np.log10(0.5),
    np.log10(11 * epsilon_0),
]
max_param_pyrite = [
    np.log10(1e-3),
    np.log10(0.20),
    np.log10(7.5e-4),
    np.log10(4630),
    np.log10(13 * epsilon_0),
]
# Graphite sphérique
min_param_graphite = [
    np.log10(1e-6),
    np.log10(0.01),
    np.log10(7.5e-5),
    np.log10(2300),
    np.log10(14 * epsilon_0),
]
max_param_graphite = [
    np.log10(1e-3),
    np.log10(0.20),
    np.log10(2.5e-4),
    np.log10(21520),
    np.log10(16 * epsilon_0),
]

for i in tqdm(range(n_data)):

    # Comme la série de Sobol est normalisée (0, 1) je la dénormalise
    # pour obtenir les vraies fréquences (10^f_log10) nécessaires

    f = torch.logspace(fmin_log10, fmax_log10, n_freq)

    # Fréquence angulaire
    w = 2 * np.pi * f

    # Choix des paramètres aléatoires du modèle PPIP pour la classification binaire (0: Pyrite, 1: Graphite)
    mineral_type = torch.randint(0, 2, (1,))

    # Pyrite
    if mineral_type == 0:
        # Sampling
        sample = sampler_LHS.random(n=1)
        sample_scaled = torch.tensor(
            qmc.scale(sample, min_param_pyrite, max_param_pyrite)
        )

        # Définition mixture géologique
        mixture = {  # en tenseur pytorch et en log10
            "log_a_i": sample_scaled[0][0],  # rayon caractéristique inclusion (m)
            "log_phi_i": sample_scaled[0][1],  # fraction minéral inclusion (%)
            "log_D_i": sample_scaled[0][2],  # Coefficient diffusion inclusion (m^2/s)
            "log_sigma_i": sample_scaled[0][3],  # conductivité inclusion (S/m)
            "log_epsilon_i": sample_scaled[0][4],  # permittivité relative inclusion (-)
            "log_D_h": torch.tensor(
                np.log10(1e-9)
            ),  # Coefficient diffusion milieu (m^2/s)
            "log_sigma_h": torch.tensor(np.log10(0.1)),  # conductivité milieu (S/m)
            "log_epsilon_h": torch.tensor(
                np.log10(80 * epsilon_0)
            ),  # permittivité relatve milieu (-)
        }
        K_eff_py = forward_spherical(w, **mixture)
        K_eff = K_eff_py.squeeze(1)
        K_eff = K_eff / 0.1

    # Graphite
    elif mineral_type == 1:
        # Sampling
        sample = sampler_LHS.random(n=1)
        sample_scaled = torch.tensor(
            qmc.scale(sample, min_param_graphite, max_param_graphite)
        )

        # Définition mixture géologique
        mixture = {  # en log10
            "log_a_i": sample_scaled[0][0],  # rayon caractéristique inclusion (m)
            "log_phi_i": sample_scaled[0][1],  # fraction minéral inclusion (%)
            "log_D_i": sample_scaled[0][2],  # Coefficient diffusion inclusion (m^2/s)
            "log_sigma_i": sample_scaled[0][3],  # conductivité inclusion (S/m)
            "log_epsilon_i": sample_scaled[0][4],  # permittivité relative inclusion (-)
            "log_D_h": torch.tensor(
                np.log10(1e-9)
            ),  # Coefficient diffusion milieu (m^2/s)
            "log_sigma_h": torch.tensor(np.log10(0.1)),  # conductivité milieu (S/m)
            "log_epsilon_h": torch.tensor(
                np.log10(80 * epsilon_0)
            ),  # permittivité relatve milieu (-)
        }
        K_eff_g = forward_spherical(w, **mixture)
        K_eff = K_eff_g.squeeze(1)
        K_eff = K_eff / 0.1

    K_eff += K_eff * torch.empty_like(K_eff).normal_(0, 0.01)

    # Les inputs du réseau de neurones sont (fréquence, conductivité complexe)
    # donc input.shape = (n_data, 1*n_freq) et input.dtype = torch.complex64
    input_real[i] = torch.cat((K_eff.real, K_eff.imag))
    input_complex[i] = torch.cat((K_eff,))

    # Les outputs du réseau de neurones sont (type de relaxation, )
    # donc output.shape = (n_data, 1) et output.dtype = torch.float32
    output_real[i] = torch.cat((mineral_type,))
    output_complex[i] = torch.cat((mineral_type,))


if device != "cpu":
    input_real = input_real.cpu()
    input_complex = input_complex.cpu()
    output_real = output_real.cpu()
    output_complex = output_complex.cpu()
    torch.set_default_device("cpu")


torch.save(input_real, f"{data_dir_real}/X(n={n_data:.0f}).pt")
torch.save(input_complex, f"{data_dir_complex}/X(n={n_data:.0f}).pt")
torch.save(output_real, f"{data_dir_real}/y(n={n_data:.0f}).pt")
torch.save(output_complex, f"{data_dir_complex}/y(n={n_data:.0f}).pt")
