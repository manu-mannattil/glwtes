import numpy as np
import matplotlib.pyplot as plt
import charu
from utils import *

rc = {
    "charu.doc": "rspa",
    "figure.figsize": [300 * charu.pt, 300 * charu.pt],
    "charu.tex": True,
    "charu.tex.font": "fourier",
}

with plt.rc_context(rc):
    fig, axes = plt.subplots(1, 2)

    w_min = 0.0
    w_max = 0.1 + 0.02

    ticks = [0, 0.04, 0.08, 0.12]
    labelpos = (0.05, 0.85)

    # tanh -----------------------------------------------------------------

    ax = axes[0]

    ax.set_aspect("equal")
    ax.set_xlim(w_min, w_max)
    ax.set_ylim(w_min, w_max)
    ax.set_xlabel(r"$\omega$ (quantized)")
    ax.set_ylabel(r"$\omega$ (numerical)")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    name = "tanh_bc_cc_l_0.1_eps_0.01_b0.1_N_2048"

    w_wkb = np.loadtxt("../data/{}/wkb1.txt".format(name), unpack=True)[1]
    w = np.loadtxt("../data/{}/quantized1.txt".format(name), unpack=True)[0]
    ax.plot(w_wkb, w, "o", color="C3", markerfacecolor="none", markersize=3.2, markeredgewidth=0.5)

    w_wkb = np.loadtxt("../data/{}/wkb2.txt".format(name), unpack=True)[1]
    w = np.loadtxt("../data/{}/quantized2.txt".format(name), unpack=True)[0]
    ax.plot(w_wkb, w, "o", color="k", markerfacecolor="none", markersize=3.2, markeredgewidth=0.5)

    w_wkb = np.loadtxt("../data/{}/wkb3.txt".format(name), unpack=True)[1]
    w = np.loadtxt("../data/{}/quantized3.txt".format(name), unpack=True)[0]
    ax.plot(w_wkb, w, "o", color="C0", markerfacecolor="none", markersize=3.2, markeredgewidth=0.5)

    ax.plot([w_min, w_max], [w_min, w_max], color="#aaaaaa", zorder=-10, linewidth=0.5)

    ax.text(*labelpos,
            r"\textbf{(a)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # sech -----------------------------------------------------------------

    ax = axes[1]

    ax.set_aspect("equal")
    ax.set_xlim(w_min, w_max)
    ax.set_ylim(w_min, w_max)
    ax.set_xlabel(r"$\omega$ (quantized)")
    ax.set_ylabel(r"$\omega$ (numerical)")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    name = "sech_bc_cc_l_0.1_eps_0.01_b0.1_a0.01_N_2048"

    w_wkb = np.loadtxt("../data/{}/wkb1.txt".format(name), unpack=True)[1]
    w = np.loadtxt("../data/{}/quantized1.txt".format(name), unpack=True)[0]
    ax.plot(w_wkb, w, "o", color="C3", markerfacecolor="none", markersize=3.2, markeredgewidth=0.5)

    w_wkb = np.loadtxt("../data/{}/wkb2.txt".format(name), unpack=True)[1]
    w = np.loadtxt("../data/{}/quantized2.txt".format(name), unpack=True)[0]
    ax.plot(w_wkb, w, "o", color="k", markerfacecolor="none", markersize=3.2, markeredgewidth=0.5)

    w_wkb = np.loadtxt("../data/{}/wkb3.txt".format(name), unpack=True)[1]
    w = np.loadtxt("../data/{}/quantized3.txt".format(name), unpack=True)[0]
    ax.plot(w_wkb, w, "o", color="C0", markerfacecolor="none", markersize=3.2, markeredgewidth=0.5)

    ax.plot([w_min, w_max], [w_min, w_max], color="#aaaaaa", zorder=-10, linewidth=0.5)

    ax.text(*labelpos,
            r"\textbf{(b)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    plt.tight_layout(pad=1.75)
    plt.savefig(
        "shell_wkb_unannot.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        facecolor="none",
        pad_inches=0,
    )
