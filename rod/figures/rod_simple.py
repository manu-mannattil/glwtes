# -*- coding: utf-8 -*-

"""Plot results from the analysis of the simpler rod equations."""

import numpy as np
import matplotlib.pyplot as plt
import charu
from matplotlib.colors import LinearSegmentedColormap
from utils import *

BlueRed_data = ["C0", "white", "C3"]
BlueRed = LinearSegmentedColormap.from_list("BlueRed", BlueRed_data)

rc = {
    "charu.doc": "aps",
    "figure.figsize": [290 * charu.pt, 290 / 1.75 / charu.golden * charu.pt],
    "charu.tex": True,
    "charu.tex.font": "fourier",
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
}

# Physical parameters.
b = 0.1 # max curvature
a = 0.01 # min curvature
m = lambda x: b - (b - a) * sech(x)

@np.vectorize
def ratio(x, k):
    M = m(x)
    w2 = 0.5 * (k**4 + k**2 + M**2 + np.sqrt((k**4 - k**2 + M**2)**2 + 4 * k**2 * M**2))
    a = abs(k * M)
    b = abs(k**4 + M**2 - w2)

    return np.abs(a) / (np.abs(a) + np.abs(b))

def rescale(m):
    """Rescale an nd-array such that its elements are in [0, 1]."""
    a = m.flatten().min()
    b = m.flatten().max()
    return (m - a) / (b - a)

with plt.rc_context(rc):
    fig, axes = plt.subplots(1, 2)

    # Rays of flexural waves -----------------------------------------------

    ax = axes[0]

    N = 400
    x_max, k_max = 5, 0.2

    x = np.linspace(-(x_max + 0.01), (x_max + 0.01), N)
    k = np.linspace(-(k_max + 0.01), (k_max + 0.01), N)
    X, K = np.meshgrid(x, k)

    r = rescale(ratio(X, K))
    pcm = ax.pcolormesh(X, K, r, cmap=BlueRed, shading="nearest", rasterized=True)

    # Simpler-rays.
    @np.vectorize
    def H(x, k):
        M = m(x)
        return np.sqrt(0.5 * (k**2 + k**4 + M**2 + np.sqrt((k**2 - k**4 - M**2)**2 + 4 * k**2 * M**2)))

    W = H(X, K)

    # Plot the largest bound state.
    ax.contour(X, K, W, colors="k", linestyles="--", levels=[0.1])

    # Plot the bound orbits.
    name = "sech_bc_cc_b0.1_a0.01_N_2048"
    omega = np.loadtxt(f"../data/{name}/wkb.txt", unpack=True)[1]
    ax.contour(X, K, W, colors="k", linestyles="-", levels=omega[1::2])

    # Lastly, plot the "eyebrows".
    levels = [0.10440662, 0.12024195, 0.14398984, 0.17242652, 0.20359708, 0.2364232, 0.27030506]
    ax.contour(X, K, W, colors="k", levels=levels)

    # Now do the above with higher-order rays.
    # @np.vectorize
    # def H(x, k):
    #     M = m(x)
    #     return np.sqrt(0.5 * ((1 + k**2) *
    #                           (k**2 + M**2) + np.sqrt((1 + k**2)**2 * (k**2 + M**2)**2 - 4 *
    #                                                   (k**3 - k * M**2)**2)))
    # W = H(X, K)
    # ax.contour(X, K, W, colors="C1", linestyles="--", levels=levels)
    # ax.contour(X, K, W, colors="C1", linestyles="--", levels=[0.1])
    # ax.contour(X, K, W, colors="C1", linestyles="--", levels=omega[::3])

    ax.set_xlim(-5, 5)
    ax.set_xticks([-5, -2.5, 0, 2.5, 5])
    ax.set_ylim(-0.2, 0.2)
    ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    ax.set_xlabel(r"$\epsilon x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center")

    fig.colorbar(pcm, ax=ax, pad=0.064)
    ax.text(1.14,
            -0.225,
            r"$\mathscr{R}$",
            verticalalignment="bottom",
            horizontalalignment="right",
            transform=ax.transAxes)

    ax.text(0.05,
            0.85,
            r"\textbf{(a)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # WKB results ----------------------------------------------------------

    ax = axes[1]
    ax.set_aspect("equal")

    w_min = 0.01
    w_max = 0.1 + 0.01
    w_guide = np.linspace(w_min, w_max, 10)
    ticks = [0.02, 0.06, 0.1]
    size = 4.5

    ax.set_aspect("equal")
    ax.set_xlim(w_min, w_max)
    ax.set_ylim(w_min, w_max)
    ax.set_xlabel(r"$\omega$ (quantized)")
    ax.set_ylabel(r"$\omega$ (numerics)")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    name = "sech_bc_cc_b0.1_a0.01_N_2048"
    w_wkb = np.loadtxt(f"../data/{name}/wkb.txt", unpack=True)[1]
    w = np.loadtxt(f"../data/{name}/quantized.txt", unpack=True)[0]

    ax.plot(w_wkb, w, "C0o", markerfacecolor="none", markersize=size, markeredgewidth=0.5)
    ax.plot(w_guide, w_guide, color="#aaaaaa", zorder=-10)

    ax.text(0.10,
            0.85,
            r"\textbf{(b)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # Export ---------------------------------------------------------------

    plt.tight_layout(w_pad=-2)
    plt.savefig(
        "rod_simple.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )
