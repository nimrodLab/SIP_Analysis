#
# Author: Charles L. Bérubé
# Created on: Tue Sep 10 2024
#
# Copyright (c) 2024 CL Bérubé JL Gagnon & S Gagnon
#

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib.ticker import ScalarFormatter
import cmcrameri.cm as cmc


# Extend ScalarFormatter
class MyScalarFormatter(ScalarFormatter):
    # Override '_set_format' with your own
    def _set_format(self):
        self.format = "%.1f"  # Show 2 decimals


colors = {
    "RVNN": cmc.nuuk.colors[256 // 2],
    "CVNN": cmc.nuuk.colors[0],
}


def plot_learning_curves(losses, n_epoch):
    fig, ax = plt.subplots()
    ax.plot(
        range(1, n_epoch + 1),
        losses["train"],
        ls="-",
        color="0.0",
        lw=0.7,
        label="Training",
    )
    ax.plot(
        range(1, n_epoch + 1),
        losses["valid"],
        ls="--",
        color="0.7",
        lw=0.7,
        label="Validation",
    )
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.show()
    return fig, ax


def plot_roc_curve(y_true, y_score):

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots()
    plt.plot(fpr, tpr) # ROC curve = TPR vs FPR
    plt.plot([0, 1], [0, 1], ls=':')
    plt.title(f"ROC-AUC = {auc:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
    return fig, ax
