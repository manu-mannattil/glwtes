# -*- coding: utf-8 -*-
"""Plot numerical bound mode frequencies vs WKB results."""

import numpy as np
import matplotlib.pyplot as plt
import charu
from utils import *

rc = {
    "charu.doc": "rspa",
    "figure.figsize": [275 * charu.pt, 275 * charu.pt],
    "charu.tex": True,
    "charu.tex.font": "fourier",
}

with plt.rc_context(rc):
    fig, axes = plt.subplots(1, 2)

    labelpos = (0.08, 0.88)
    w_min = 0.01
    w_max = 0.1 + 0.01
    w_guide = np.linspace(w_min, w_max, 10)
    ticks = [0.02, 0.06, 0.1]
    size = 4.5

    # tanh type ------------------------------------------------------------

    ax = axes[0]

    ax.set_aspect("equal")
    ax.set_xlim(w_min, w_max)
    ax.set_ylim(w_min, w_max)
    ax.set_xlabel(r"$\omega$ (quantized)")
    ax.set_ylabel(r"$\omega$ (numerics)")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    name = "tanh_bc_cc_b0.1_N_2048"
    w_wkb = np.loadtxt(f"../data/{name}/wkb.txt", unpack=True)[1]
    w = np.loadtxt(f"../data/{name}/quantized.txt", unpack=True)[0]

    ax.plot(w_wkb, w, "C0o", markerfacecolor="none", markersize=size, markeredgewidth=0.5)
    ax.plot(w_guide, w_guide, color="#aaaaaa", zorder=-10)
    ax.text(*labelpos, r"\textbf{(a)}", transform=ax.transAxes)

    # sech type ------------------------------------------------------------

    ax = axes[1]

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
    ax.text(*labelpos, r"\textbf{(b)}", transform=ax.transAxes)

    plt.tight_layout(w_pad=2)
    plt.savefig(
        "rod_wkb.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        facecolor="none",
        pad_inches=0,
    )
