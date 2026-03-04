#
# Author: Charles L. Bérubé
# Created on: Tue Sep 10 2024
#
# Copyright (c) 2024 CL Bérubé JL Gagnon & S Gagnon
#

import math
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import matplotlib as mpl

from pnp import max_u


def truncate(n, decimals=0):
    multiplier = 10**decimals
    return int(n * multiplier) / multiplier


def to_latex_scientific_notation(mean, std, maxint=2):
    exponent_mean = int(np.floor(np.log10(abs(mean))))
    exponent_std = int(np.floor(np.log10(abs(std))))
    precision = abs(exponent_mean - exponent_std)
    coefficient_mean = round(mean / 10**exponent_mean, precision)
    coefficient_std = round(std / 10**exponent_mean, precision)
    if -maxint <= exponent_mean <= 0:
        return f"${truncate(mean, -exponent_std)} \\pm {truncate(std, -exponent_std)}$"
    elif 0 <= exponent_mean <= maxint and exponent_std >= 0:
        return f"${round(truncate(mean, -exponent_std))} \\pm {round(truncate(std, -exponent_std))}$"
    elif 0 <= exponent_mean <= maxint:
        return f"${truncate(mean, -exponent_std)} \\pm {truncate(std, -exponent_std)}$"
    else:
        if precision == 0:
            return (
                f"$({round(coefficient_mean)} \\pm {round(coefficient_std)}) \\cdot 10^{{{exponent_mean}}}$"
                if exponent_mean != 0
                else f"{round(coefficient_mean)} \\pm {round(coefficient_std)}$"
            )
        else:
            return (
                f"$({coefficient_mean} \\pm {coefficient_std}) \\cdot 10^{{{exponent_mean}}}$"
                if exponent_mean != 0
                else f"{coefficient_mean} \\pm {coefficient_std}$"
            )


def str_with_err(value, error):
    if error > 0:
        digits = -int(math.floor(math.log10(error)))
    else:
        digits = 0
    if digits < 0:
        digits = 0
    err10digits = math.floor(error * 10**digits)
    return "${0:.{2}f} \pm {1:.{2}f}$".format(value, error, digits)


def p_metric(y_pred, y_true, eps=1e-12):
    # Number of significant digits
    return np.log10(np.abs((y_pred - y_true) / y_true) + eps)


def r_metric(y_pred, y_true):
    # Should be centered around 0 and narrow
    return np.log10(np.abs(y_pred / y_true))


def mape(y_pred, y_true):
    return 100 * np.abs((y_pred - y_true) / y_true)


def normalize(x, xmin, xmax, ymin, ymax):
    # x mapped from xmin, xmax to ymin, ymax
    return (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin


def denormalize(x, xmin, xmax, ymin, ymax):
    # x mapped from ymin, ymax back to xmin, xmax
    return (x - ymin) * (xmax - xmin) / (ymax - ymin) + xmin


def weights_init(m):
    if m._get_name().__contains__("Linear"):
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
        if m.bias:
            torch.nn.init.zeros_(m.bias)


def xavier_real(m):
    if m._get_name().__contains__("Linear"):
        gain = 5 / 3
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias:
            torch.nn.init.zeros_(m.bias)


def xavier_complex(m):
    gain = 5 / (3 * torch.tensor(2.0).sqrt())
    if m._get_name().__contains__("Linear"):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias:
            torch.nn.init.zeros_(m.bias)


def restore_minor_ticks_log_plot(ax, n_subticks=9, axis="both"):
    """For axes with a logrithmic scale where the span (max-min) exceeds
    10 orders of magnitude, matplotlib will not set logarithmic minor ticks.
    If you don't like this, call this function to restore minor ticks.

    Args:
        ax:
        n_subticks: Number of Should be either 4 or 9.

    Returns:
        None
    """
    if ax is None:
        ax = plt.gca()
    # Method from SO user importanceofbeingernest at
    # https://stackoverflow.com/a/44079725/5972175
    locmaj = mpl.ticker.LogLocator(base=10, numticks=1000)
    locmin = mpl.ticker.LogLocator(
        base=10.0, subs=np.linspace(0, 1.0, n_subticks + 2)[1:-1], numticks=1000
    )

    if axis == "x" or axis == "both":
        ax.xaxis.set_major_locator(locmaj)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if axis == "y" or axis == "both":
        ax.yaxis.set_major_locator(locmaj)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def split_train_test(dataset, test_split=0.2, random_seed=None):
    """_summary_

    Args:
        dataset (_type_): PyTorch dataset à splitter
        test_split (float, optional): fraction pour le test data. Defaults to 0.2.
        random_seed (_type_, optional): pour répétabilité. Defaults to None.

    Returns:
        pytorch sampler: indices des tenseurs de train et test
    """
    shuffle_dataset = True
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    return train_sampler, test_sampler


class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return self.mse(torch.log(pred + 1), torch.log(actual + 1))


class ComplexCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(ComplexCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets):
        real_loss = nn.CrossEntropyLoss()(inputs.real, targets)
        if torch.is_complex(inputs):
            imag_loss = nn.CrossEntropyLoss()(inputs.imag, targets)
            return (real_loss + imag_loss) / 2
        else:
            return real_loss


def train(
    model,
    lr,
    n_epoch,
    train_loader=None,
    valid_loader=None,
    verbose=1,
    device=None,
    train_sampler=None,
    valid_sampler=None,
    **scheduler_kwargs,
):
    """Ma fonction de base pour entraîner des modèles PyTorch

    Args:
        model (_type_): modèle PyTorch
        train_loader (dataloader): training data
        verbose (_type_): fréquence des prints
        lr (_type_): learning rate
        n_epoch (_type_): nombre d'epoch d'entrainement
        device (_type_, optional): cpu ou gpu pytorch object. Defaults to None.
        valid_loader (_type_, optional): validation data. Defaults to None.

    Returns:
        dict: dictionnaire contenant les loss pour chaque epoch
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_losses = ["train"]  # parfois il y a plusieurs loss à minimiser
    valid_losses = ["valid"]  # et plusieurs loss de validation à inspecter

    history = {k: torch.zeros(n_epoch) for k in train_losses}
    history.update({k: torch.zeros(n_epoch) for k in valid_losses})

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.MSELoss()  # on minimise l'erreur quadratique

    start_time = timer()
    model = model.to(device)

    for e in range(n_epoch):
        running_loss = {k: 0 for k in train_losses}  # reset running losses
        model.train()
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            y_hat = model(X)

            if y.dtype == torch.complex64:

                y_n = torch.view_as_complex(
                    torch.view_as_real(y) / max_u.reshape(-1, 2)
                )

                loss = (0.5 * (y_hat - y_n) * (y_hat - y_n).conj()).abs().mean()

            elif y.dtype == torch.float32:
                loss = criterion(y_hat, y / max_u)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss["train"] += loss.mean().item()

        for k in train_losses:
            history[k][e] = running_loss[k] / len(train_loader.sampler)
            # history[k][e] = loss.item()

        verbose_str = f"Epoch: {(e + 1):.0f}, " f"Loss: {history['train'][e]:.1e}"

        if valid_loader:
            # La loop de validation se fait seulement si un
            # dataloader de validation est fourni à train()
            model.eval()
            running_loss = {k: 0 for k in valid_losses}  # reset running losses
            for X, y in valid_loader:
                X = X.to(device)
                y = y.to(device)

                y_hat = model(X)

                if y.dtype == torch.complex64:

                    y_n = torch.view_as_complex(
                        torch.view_as_real(y) / max_u.reshape(-1, 2)
                    )

                    loss = (0.5 * (y_hat - y_n) * (y_hat - y_n).conj()).abs().mean()

                elif y.dtype == torch.float32:
                    loss = criterion(y_hat, y / max_u)

                running_loss["valid"] += loss.mean().item()

            for k in valid_losses:
                history[k][e] = running_loss[k] / len(valid_loader.sampler)

            verbose_str += f", Valid: {history['valid'][e]:.1e}"

        verbose_str += f", LR: {optimizer.param_groups[0]['lr']:.1e}"

        # Imprime les loss
        if verbose:
            if (e + 1) % verbose == 0:
                print(verbose_str)

    end_time = timer()
    training_time = end_time - start_time
    history["time"] = training_time

    if verbose:
        print(f"Training time: {training_time / 60:.2f} m")

    return history
