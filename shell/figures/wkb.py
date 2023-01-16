import numpy as np
import matplotlib.pyplot as plt
import charu
from utils import *

rc = {
    "charu.doc": "rspa",
    "figure.figsize": [360 * charu.pt, 360 / charu.golden * charu.pt],
    "charu.tex": True,
    "charu.tex.font": "fourier",
}

with plt.rc_context(rc):
    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    ax.set_aspect(1/charu.golden)

    name = "cc_10_tanh_sorted_l1.2_N2048"
    w_wkb = np.loadtxt("../data/{}/wkb.txt".format(name), unpack=True)[1]
    w, err = np.loadtxt("../data/{}/quantized.txt".format(name), unpack=True)
    N = len(w)

    ax.plot(w_wkb, w, "C0o", markerfacecolor="none", markersize=3.2, markeredgewidth=0.5)
    ax.plot(w_wkb, w_wkb, color="#aaaaaa", zorder=-10)
    ax.text(0.1, 0.8, r"\textbf{(a)}", transform=ax.transAxes)

    ax.set_xlabel(r"$\omega$ (quantized)")
    ax.set_ylabel(r"$\omega$ (numerical)", labelpad=0)

    ax = axes[1]
    ax.set_aspect(1/charu.golden)

    name = "cc_10_sech_zero_sorted_l1.2_N2048"
    w_wkb = np.loadtxt("../data/{}/wkb.txt".format(name), unpack=True)[1]
    w, err = np.loadtxt("../data/{}/quantized.txt".format(name), unpack=True)
    N = len(w)

    ax.plot(w_wkb, w, "C0o", markerfacecolor="none", markersize=3.2, markeredgewidth=0.5)
    ax.plot(w_wkb, w_wkb, color="#aaaaaa", zorder=-10)
    ax.text(0.1, 0.8, r"\textbf{(b)}", transform=ax.transAxes)

    ax.set_xlabel(r"$\omega$ (quantized)")
    ax.set_ylabel(r"$\omega$ (numerical)", labelpad=0)

    plt.tight_layout(pad=1.75)
    plt.savefig(
        "shell_wkb.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        facecolor="none",
        pad_inches=0,
    )
