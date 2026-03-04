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
    silu,
    softmax,
    tanh,
    relu,
    sigmoid,
    dropout,
    hardtanh,
)
import torch.nn.functional as F

import torch.nn as nn


# Jean-Luc : Il faudra vérifier ces fonctions d'activation
def complexRelu(inp):
    return torch.complex(relu(inp.real), relu(inp.imag))


def complexLeakyRelu(inp):
    return torch.complex(leaky_relu(inp.real), leaky_relu(inp.imag))


def complexSoftmax(inp):
    return torch.complex(softmax(inp.real), softmax(inp.imag))


def complexElu(inp):
    return torch.complex(elu(inp.real), elu(inp.imag))


def complexGelu(inp):
    return torch.complex(gelu(inp.real), gelu(inp.imag))


def complexTanh(inp):
    return torch.tanh(inp.abs()) * torch.sgn(inp)


def complexSiLU(inp):
    return torch.complex(silu(inp.real), silu(inp.imag))


def complexSigmoid(inp):
    return torch.complex(sigmoid(inp.real), sigmoid(inp.imag))


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
        return complexSiLU(inp)


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
        self.activation = cTanh()
        self.dropout = cDropout(p=dropout_p)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.dropout(x)
        for i in range(len(self.hidden_layers)):
            x = self.activation(self.hidden_layers[i](x))
            x = self.dropout(x)
        output = self.output_layer(x)
        return output  # fonctionne bien


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
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.dropout(x)
        for i in range(len(self.hidden_layers)):
            x = self.activation(self.hidden_layers[i](x))
            x = self.dropout(x)
        output = self.output_layer(x)
        return output
