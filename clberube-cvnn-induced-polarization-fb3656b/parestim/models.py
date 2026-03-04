#
# Author: Charles L. Bérubé
# Created on: Tue Sep 10 2024
#
# Copyright (c) 2024 CL Bérubé JL Gagnon & S Gagnon
#


import torch
from torch.nn.functional import (
    gelu,
    leaky_relu,
    elu,
    softmax,
    tanh,
    silu,
    relu,
    sigmoid,
    dropout,
    hardtanh,
)
import torch.nn.functional as F

import torch.nn as nn


def ColeCole(w, r, m, tau, c):
    """Modèle Cole-Cole multiple

    Args:
        w (array): Vecteur des fréquences angulaires
        r (float): résistivité DC
        m (float): chargeabilité
        tau (float): temps de relaxation
        c (float): exposant Cole-Cole

    Returns:
        complex: l'impédance Cole-Cole
    """
    # J'arrange les dimensions pour calculer de façon vectorielle
    w = w.unsqueeze(-1)
    m = m.unsqueeze(-2)
    c = c.unsqueeze(-2)
    tau = tau.unsqueeze(-2)

    # La somme est utilisée au cas où on additionne plusieurs modes Cole-Cole
    # Le nombre de modes est déterminé par max_modes et num_modes.
    z = torch.sum(m * (1.0 - 1.0 / (1.0 + ((1j * w * tau)) ** c)), -1)
    z = r * (1 - z)
    return z


# Jean-Luc : Il faudra vérifier ces fonctions d'activation
def complexRelu(inp):
    return relu(inp.abs()) * torch.sgn(inp)


def complexSilu(inp):
    return silu(inp.abs()) * torch.sgn(inp)


def complexLeakyRelu(inp):
    return torch.complex(leaky_relu(inp.real), leaky_relu(inp.imag))


def complexSoftmax(inp):
    return torch.complex(softmax(inp.real), softmax(inp.imag))


def complexElu(inp):
    return torch.complex(elu(inp.real), elu(inp.imag))


def complexGelu(inp):
    return torch.complex(gelu(inp.real), gelu(inp.imag))


def complexTanh(inp):
    return torch.complex(tanh(inp.real), tanh(inp.imag))


def complexSigmoid(inp):
    return torch.complex(sigmoid(inp.real), sigmoid(inp.imag))


def complexCardioid(inp):
    return ((1 + torch.cos(inp.angle())) * inp) / 2


def complexDropout(inp, p=0.5, training=True):
    """
    copy from https://github.com/wavefrontshaping/complexPyTorch
    """
    mask = torch.ones(*inp.shape, dtype=torch.float32, device=inp.device)
    mask = dropout(mask, p, training) * 1 / (1 - p)
    mask.type(inp.dtype)
    return mask * inp


class cTanh(nn.Module):
    @staticmethod
    def forward(inp):
        return complexTanh(inp)


class cSiLU(nn.Module):
    @staticmethod
    def forward(inp):
        return complexSilu(inp)


class cReLU(nn.Module):
    @staticmethod
    def forward(inp):
        return complexRelu(inp)


class cSigmoid(nn.Module):
    @staticmethod
    def forward(inp):
        return complexSigmoid(inp)


class cCardioid(nn.Module):
    @staticmethod
    def forward(inp):
        return complexCardioid(inp)


class cDropout(nn.Module):
    """
    copy from https://github.com/wavefrontshaping/complexPyTorch
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, inp):
        if self.training:
            return complexDropout(inp, self.p)
        else:
            return inp


def zReLU(z):
    """
    zReLU activation function for complex inputs.
    Zeroes out all elements where the real part is negative.

    Args:
        z (torch.Tensor): Complex-valued input tensor

    Returns:
        torch.Tensor: Activated complex-valued tensor
    """
    mask = torch.logical_and(z.real > 0, z.imag > 0)
    return z * mask


def phase_amplitude_sigmoid(z):
    return sigmoid(torch.abs(z)) * torch.exp(1.0j * torch.angle(z))


class modReLU(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.0))
        self.b.requiresGrad = True

    def forward(self, z):
        return relu(torch.abs(z) + self.b) * torch.exp(1.0j * torch.angle(z))


def cardioid_activation(z):
    """
    Cardioid activation for complex-valued input.
    Applies the activation: (1 + cos(arg(z))) / 2 * z
    where arg(z) is the phase of the complex number.

    Args:
        z (torch.Tensor): Complex-valued tensor (dtype=torch.complex64 or complex128)

    Returns:
        torch.Tensor: Activated complex-valued tensor
    """
    phase = torch.angle(z)  # Computes arg(z)
    scale = 0.5 * (1 + torch.cos(phase))  # Compute (1 + cos(arg(z))) / 2
    return scale * z


class cLinear(nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.cfloat)
        )
        nn.init.xavier_uniform_(self.weight)

        self.bias = None

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(1, out_features, dtype=torch.cfloat), requires_grad=bias
            )
            nn.init.xavier_uniform_(self.bias)

    def forward(self, inp):
        if not inp.dtype == torch.cfloat:
            inp = torch.complex(inp, torch.zeros_like(inp))
        return F.linear(inp, self.weight, self.bias)


class cMLP(nn.Module):
    """
    Complex Multilayer Perceptron
    """

    def __init__(self, input_size, hidden_size, output_size, bias, dropout_p=0.5):
        super().__init__()
        self.input_layer = cLinear(input_size, hidden_size[0], bias=bias)
        self.hidden_layers = nn.ModuleList(
            [
                cLinear(hidden_size[i], hidden_size[i + 1], bias=bias)
                for i in range(len(hidden_size) - 1)
            ]
        )
        self.output_layer = cLinear(hidden_size[-1], output_size, bias=bias)
        self.activation = cCardioid()
        self.dropout = cDropout(p=dropout_p)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.dropout(x)
        for i in range(len(self.hidden_layers)):
            x = self.activation(self.hidden_layers[i](x))
            x = self.dropout(x)
        output = self.output_layer(x)

        return output.abs()


class rMLP(nn.Module):
    """
    Real Multilayer Perceptron
    """

    def __init__(self, input_size, hidden_size, output_size, bias, dropout_p=0.5):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size[0], bias=bias)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(hidden_size[i], hidden_size[i + 1], bias=bias)
                for i in range(len(hidden_size) - 1)
            ]
        )
        self.output_layer = nn.Linear(hidden_size[-1], output_size, bias=bias)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.dropout(x)
        for i in range(len(self.hidden_layers)):
            x = self.activation(self.hidden_layers[i](x))
            x = self.dropout(x)
        output = self.output_layer(x)

        return output
