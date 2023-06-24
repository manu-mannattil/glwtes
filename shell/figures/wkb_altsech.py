import numpy as np
import matplotlib.pyplot as plt
import charu
from utils import *

rc = {
    "charu.doc": "rspa",
    "figure.figsize": [150 * charu.pt, 150 * charu.pt],
    "charu.tex": True,
    "charu.tex.font": "fourier",
}

with plt.rc_context(rc):
    fig, ax = plt.subplots()

    w_min = 0.03
    w_max = 0.065

    ticks = [0.03, 0.04, 0.05, 0.06]
    labelpos = (0.06, 0.89)

    ax.set_aspect("equal")
    ax.set_xlim(w_min, w_max)
    ax.set_ylim(w_min, w_max)
    ax.set_xlabel(r"$\omega$ (quantized)")
    ax.set_ylabel(r"$\omega$ (numerical)")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    name = "tanh_bc_cc_l_0.1_eps_0.01_b0.1_N_2048"
    name = "altsech_bc_cc_l_0.1_eps_0.01_b0.1_N_2048"

    w_wkb = np.loadtxt("../data/{}/wkb.txt".format(name), unpack=True)[1]
    w = np.loadtxt("../data/{}/quantized.txt".format(name), unpack=True)[0]
    ax.plot(w_wkb, w, "o", color="C0", markerfacecolor="none", markersize=4, markeredgewidth=0.5)

    ax.plot([w_min, w_max], [w_min, w_max], color="#aaaaaa", zorder=-10, linewidth=0.5)

    ax.text(*labelpos,
            r"\textbf{(c)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    plt.tight_layout(pad=1.75)
    plt.savefig(
        "shell_wkb_altsech.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        facecolor="none",
        pad_inches=0,
    )
