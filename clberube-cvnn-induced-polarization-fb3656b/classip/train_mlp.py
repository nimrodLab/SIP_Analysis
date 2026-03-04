#
# Author: Charles L. Bérubé
# Created on: Tue Sep 10 2024
#
# Copyright (c) 2024 CL Bérubé JL Gagnon & S Gagnon
#

import os
import math

import torch
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from models import cMLP, rMLP
from utilities import train
from utilities import xavier_real, xavier_complex


plt.style.use("../seg.mplstyle")


# Quelques définitions de base
device = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # utilise le gpu si possible, sinon cpu

save_as_best = False

batch_size = int(50)  # nombre d'exemples par étape d'optimisation
n_epoch = int(1e2)  # nombre de répétitions sur tous les data
n_data = int(1e4)  # utiliser le bon nombre selon le n_data utilisé dans 01_gen_data.py
n_repeat = int(1e1)  # nombre de répétition des expériences


# Les réseaux à tester
architectures = {
    "A": {"CVNN": 1 * [4]},
    "B": {"CVNN": 2 * [8]},
    "C": {"CVNN": 4 * [16]},
}


for arc, hidden_size in architectures.items():
    # Référence :
    # https://hal.science/hal-03529898v1/document

    NC = [32] + hidden_size["CVNN"] + [1]
    N0 = 2 * NC[0]  # RVNN has 2x the inputs (split real-imag)
    NL = NC[-1]  # output of CVNN is real (Cole-Cole regression)

    b = N0 * NC[1] + NC[-2] * NL
    a = 0
    for i in range(1, len(NC) - 2):
        a += NC[i] * NC[i + 1]

    beta = 2 * NC[0] * NC[1] + 2 * NC[-2] * NC[-1]
    tp = 2 * a + beta

    if len(hidden_size["CVNN"]) > 1:
        r = (-b + math.sqrt(b**2 - 4 * a * -tp)) / (2 * a)
    else:
        r = 2 * (NC[0] + NC[-1]) / (N0 + NC[-1])

    print(NC)
    print(arc, "r =", r, "tp =", tp)
    architectures[arc]["RVNN"] = [round(r * i) for i in hidden_size["CVNN"]]


if __name__ == "__main__":

    for arc in architectures.keys():

        for exp in architectures[arc].keys():

            data_dir = f"./data/{exp}"  # relatif à ce .py file
            wt_dir = f"./weights/{arc}/{exp}"
            res_dir = f"./results/{arc}/{exp}"

            # Créer des dossiers s'ils n'existent pas déjà
            os.makedirs(wt_dir, exist_ok=True)
            os.makedirs(res_dir, exist_ok=True)

            X = torch.load(f"{data_dir}/X(n={n_data}).pt", weights_only=True)
            y = torch.load(f"{data_dir}/y(n={n_data}).pt", weights_only=True)

            input_size = X.shape[-1]
            output_size = y.shape[-1]

            # Paramètres du réseau de neurones
            model_params = dict(
                input_size=input_size,
                hidden_size=None,
                output_size=output_size,
                bias=False,
                dropout_p=0.0,
            )

            for n in range(1, n_repeat + 1):
                print(f"Network {exp}, Architecture {arc}, Repeat {n}")

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=None
                )

                dataset_train = TensorDataset(X_train, y_train)
                dataset_test = TensorDataset(X_test, y_test)

                dataloader = {}
                dataloader["train"] = DataLoader(
                    dataset_train, batch_size, shuffle=True
                )
                dataloader["valid"] = DataLoader(dataset_test, batch_size, shuffle=True)

                # Instanciation du réseau de neurones
                if X_train.dtype == torch.complex64:
                    model_params["hidden_size"] = architectures[arc][exp]
                    model = cMLP(**model_params)
                    model.apply(xavier_complex)

                elif X_train.dtype == torch.float32:
                    model_params["hidden_size"] = architectures[arc][exp]
                    model = rMLP(**model_params)
                    model.apply(xavier_real)

                # Boucle d'entraînement (voir utilities.train)
                losses = train(
                    model=model,  # le modèle à entraîner
                    lr=1e-4,  # learning rate maximal
                    n_epoch=n_epoch,  # nombre d'itérations sur le dataset complet
                    train_loader=dataloader["train"],
                    valid_loader=dataloader["valid"],
                    device=device,  # cpu ou gpu
                    # verbose=(n_epoch // 10 or 1),  # imprime un suivi
                    verbose=0,  # imprime un suivi
                )

                torch.save(model.state_dict(), f"{wt_dir}/weights-r{n}.pt")
                torch.save(losses, f"{res_dir}/losses-r{n}.pt")
                torch.save(X_test, f"{res_dir}/X_test(n={n_data})-r{n}.pt")
                torch.save(y_test, f"{res_dir}/y_test(n={n_data})-r{n}.pt")

                if save_as_best:
                    torch.save(model.state_dict(), f"{wt_dir}/weights-best-r{n}.pt")
                    torch.save(losses, f"{res_dir}/losses-best-r{n}.pt")
                    torch.save(X_test, f"{res_dir}/X_test(n={n_data})-best-r{n}.pt")
                    torch.save(y_test, f"{res_dir}/y_test(n={n_data})-best-r{n}.pt")

        print()
