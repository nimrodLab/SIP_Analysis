#
# Author: Charles L. Bérubé
# Created on: Fri Jun 02 2023
#
# Copyright (c) 2023 Charles L. Bérubé
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import griddata
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


def plot_pnp(x, y, u, a, L, title=None, fname=None, vr=None, vi=None):

    kwargs = dict(s=1, cmap=cmc.vik)
    fig, axs = plt.subplots(2, 3, figsize=(6, 3), sharex=True, sharey=True)
    ax = axs[0, 0]
    im = ax.scatter(x, y, c=1e6 * u[:, 0], vmin=-vr[0], vmax=vr[0], **kwargs)
    plt.colorbar(im, ax=ax, label=r"($\mu$mol/m$^3$)")
    ax.set_title(r"$\delta n_1'$")
    ax = axs[0, 1]
    im = ax.scatter(x, y, c=1e6 * u[:, 2], vmin=-vr[1], vmax=vr[1], **kwargs)
    plt.colorbar(im, ax=ax, label=r"($\mu$mol/m$^3$)")
    ax.set_title(r"$\delta n_2'$")
    ax = axs[0, 2]
    im = ax.scatter(x, y, c=1e7 * u[:, 4], vmin=-vr[2], vmax=vr[2], **kwargs)
    plt.colorbar(im, ax=ax, label=r"($\mu$V/10)")
    ax.set_title(r"$\delta U'$")

    ax = axs[1, 0]
    im = ax.scatter(x, y, c=1e6 * u[:, 1], vmin=-vi[0], vmax=vi[0], **kwargs)
    plt.colorbar(im, ax=ax, label=r"($\mu$mol/m$^3$)")
    ax.set_title(r"$\delta n_1''$")
    ax = axs[1, 1]
    im = ax.scatter(x, y, c=1e6 * u[:, 3], vmin=-vi[1], vmax=vi[1], **kwargs)
    plt.colorbar(im, ax=ax, label=r"($\mu$mol/m$^3$)")
    ax.set_title(r"$\delta n_2''$")
    ax = axs[1, 2]
    im = ax.scatter(x, y, c=1e8 * u[:, 5], vmin=-vi[2], vmax=vi[2], **kwargs)
    plt.colorbar(im, ax=ax, label=r"($\mu$V/100)")
    ax.set_title(r"$\delta U''$")

    for ax in axs.flat:
        ax.set_aspect("equal")
        ax.set_xlim([-2 * a / L, 2 * a / L])
        ax.set_ylim([-2 * a / L, 2 * a / L])
        ax.set_xticks([-2 * a / L, 0, 2 * a / L], ["$-2a$", 0, "$2a$"])
        ax.set_yticks([-2 * a / L, 0, 2 * a / L], ["$-2a$", 0, "$2a$"])

    if title:
        plt.suptitle(title)

    if fname:
        plt.savefig(f"{fname}.png")
        plt.savefig(f"{fname}.pdf")


def plot_pnp_contour(x, y, u, a, L, title=None, fname=None, vr=None, vi=None):
    # Create a grid spanning the domain (using the same limits as your axes)
    xi = np.linspace(-2 * a / L, 2 * a / L, 200)
    yi = np.linspace(-2 * a / L, 2 * a / L, 200)
    Xi, Yi = np.meshgrid(xi, yi)

    fig, axs = plt.subplots(2, 3, figsize=(6, 3), sharex=True, sharey=True)

    # Define each subplot's configuration: (row, col, values, vlim, title, colorbar label)
    plots = [
        (0, 0, 1e6 * u[:, 0], vr[0], r"$\delta n_1'$", r"($\mu$mol/m$^3$)"),
        (0, 1, 1e6 * u[:, 2], vr[1], r"$\delta n_2'$", r"($\mu$mol/m$^3$)"),
        (0, 2, 1e7 * u[:, 4], vr[2], r"$\delta U'$", r"($\mu$V/10)"),
        (1, 0, 1e6 * u[:, 1], vi[0], r"$\delta n_1''$", r"($\mu$mol/m$^3$)"),
        (1, 1, 1e6 * u[:, 3], vi[1], r"$\delta n_2''$", r"($\mu$mol/m$^3$)"),
        (1, 2, 1e8 * u[:, 5], vi[2], r"$\delta U''$", r"($\mu$V/100)"),
    ]

    for i, j, values, vlim, ax_title, cbar_label in plots:
        # Interpolate scattered data to the grid
        grid_values = griddata((x, y), values, (Xi, Yi), method="cubic")
        cs = axs[i, j].contourf(
            Xi, Yi, grid_values, levels=100, cmap=cmc.vik, vmin=-vlim, vmax=vlim
        )
        plt.colorbar(cs, ax=axs[i, j], label=cbar_label)
        axs[i, j].set_title(ax_title)
        axs[i, j].set_aspect("equal")
        axs[i, j].set_xlim([-2 * a / L, 2 * a / L])
        axs[i, j].set_ylim([-2 * a / L, 2 * a / L])
        axs[i, j].set_xticks([-2 * a / L, 0, 2 * a / L])
        axs[i, j].set_xticklabels(["$-2a$", "0", "$2a$"])
        axs[i, j].set_yticks([-2 * a / L, 0, 2 * a / L])
        axs[i, j].set_yticklabels(["$-2a$", "0", "$2a$"])

    if title:
        plt.suptitle(title)

    if fname:
        plt.savefig(f"{fname}.png")
        plt.savefig(f"{fname}.pdf")
    plt.show()


def plot_pnp_pcolormesh(x, y, u, a, L, title=None, fname=None, vr=None, vi=None):
    xi = np.linspace(-2 * a / L, 2 * a / L, 200)
    yi = np.linspace(-2 * a / L, 2 * a / L, 200)
    Xi, Yi = np.meshgrid(xi, yi)

    fig, axs = plt.subplots(2, 3, figsize=(6, 3), sharex=True, sharey=True)

    plots = [
        (0, 0, 1e6 * u[:, 0], vr[0], r"$\delta n_1'$", r"($\mu$mol/m$^3$)"),
        (0, 1, 1e6 * u[:, 2], vr[1], r"$\delta n_2'$", r"($\mu$mol/m$^3$)"),
        (0, 2, 1e7 * u[:, 4], vr[2], r"$\delta U'$", r"($\mu$V/10)"),
        (1, 0, 1e6 * u[:, 1], vi[0], r"$\delta n_1''$", r"($\mu$mol/m$^3$)"),
        (1, 1, 1e6 * u[:, 3], vi[1], r"$\delta n_2''$", r"($\mu$mol/m$^3$)"),
        (1, 2, 1e8 * u[:, 5], vi[2], r"$\delta U''$", r"($\mu$V/100)"),
    ]

    for i, j, values, vlim, ax_title, cbar_label in plots:
        grid_values = griddata((x, y), values, (Xi, Yi), method="cubic")
        R = np.sqrt(Xi**2 + Yi**2)
        grid_values_masked = np.ma.array(grid_values, mask=(R < a / L))

        cmap = cmc.vik.copy()
        cmap.set_bad("white")

        cs = axs[i, j].pcolormesh(
            Xi,
            Yi,
            grid_values_masked,
            cmap=cmap,
            shading="auto",
            vmin=-vlim,
            vmax=vlim,
            rasterized=True,
        )
        plt.colorbar(cs, ax=axs[i, j], label=cbar_label)
        axs[i, j].set_title(ax_title)

        axs[i, j].set_aspect("equal")
        axs[i, j].set_xlim([-2 * a / L, 2 * a / L])
        axs[i, j].set_ylim([-2 * a / L, 2 * a / L])
        axs[i, j].set_xticks([-2 * a / L, 0, 2 * a / L])
        axs[i, j].set_xticklabels(["$-2a$", "0", "$2a$"])
        axs[i, j].set_yticks([-2 * a / L, 0, 2 * a / L])
        axs[i, j].set_yticklabels(["$-2a$", "0", "$2a$"])

    if title:
        plt.suptitle(title)

    if fname:
        plt.savefig(f"{fname}.png")
        plt.savefig(f"{fname}.pdf")
    plt.show()


def plot_pnp_imshow(x, y, u, a, L, title=None, fname=None, vr=None, vi=None):
    xi = np.linspace(-2 * a / L, 2 * a / L, 200)
    yi = np.linspace(-2 * a / L, 2 * a / L, 200)
    Xi, Yi = np.meshgrid(xi, yi)

    fig, axs = plt.subplots(2, 3, figsize=(6, 3), sharex=True, sharey=True)

    plots = [
        (0, 0, 1e6 * u[:, 0], vr[0], r"$\delta n_1'$", r"($\mu$mol/m$^3$)"),
        (0, 1, 1e6 * u[:, 2], vr[1], r"$\delta n_2'$", r"($\mu$mol/m$^3$)"),
        (0, 2, 1e7 * u[:, 4], vr[2], r"$\delta U'$", r"($\mu$V/10)"),
        (1, 0, 1e6 * u[:, 1], vi[0], r"$\delta n_1''$", r"($\mu$mol/m$^3$)"),
        (1, 1, 1e6 * u[:, 3], vi[1], r"$\delta n_2''$", r"($\mu$mol/m$^3$)"),
        (1, 2, 1e8 * u[:, 5], vi[2], r"$\delta U''$", r"($\mu$V/100)"),
    ]

    for i, j, values, vlim, ax_title, cbar_label in plots:
        grid_values = griddata((x, y), values, (Xi, Yi), method="cubic")
        R = np.sqrt(Xi**2 + Yi**2)
        grid_values_masked = np.ma.array(grid_values, mask=(R < a / L))

        cmap = cmc.vik.copy()
        cmap.set_bad("white")

        cs = axs[i, j].imshow(
            grid_values_masked,
            cmap=cmap,
            vmin=-vlim,
            vmax=vlim,
            extent=[Xi.min(), Xi.max(), Yi.min(), Yi.max()],
            interpolation="none",
        )
        plt.colorbar(cs, ax=axs[i, j], label=cbar_label)
        axs[i, j].set_title(ax_title)

        axs[i, j].set_aspect("equal")
        axs[i, j].set_xlim([-2 * a / L, 2 * a / L])
        axs[i, j].set_ylim([-2 * a / L, 2 * a / L])
        axs[i, j].set_xticks([-2 * a / L, 0, 2 * a / L])
        axs[i, j].set_xticklabels(["$-2a$", "0", "$2a$"])
        axs[i, j].set_yticks([-2 * a / L, 0, 2 * a / L])
        axs[i, j].set_yticklabels(["$-2a$", "0", "$2a$"])

    if title:
        plt.suptitle(title)

    if fname:
        plt.savefig(f"{fname}.png", dpi=600)
        plt.savefig(f"{fname}.pdf", dpi=600)
        plt.savefig(f"{fname}.svg", dpi=600)
    plt.show()
