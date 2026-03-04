from __future__ import annotations

import matplotlib.pyplot as plt


def init_2x2_layout(title: str):
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(title)

    for ax in [axs[0, 0], axs[0, 1], axs[1, 0]]:
        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)

    axs[0, 0].set_ylabel("In-phase conductivity (uS/cm)")
    axs[0, 1].set_ylabel("Quadrature conductivity (uS/cm)")
    axs[1, 0].set_ylabel("-Phase (mrad)")
    axs[1, 0].set_xlabel("Frequency (Hz)")
    axs[1, 1].axis("off")
    return fig, axs
