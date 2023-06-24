# -*- coding: utf-8 -*-
"""Plot curvature profiles m(x) = bsech(x) and the corresponding shapes of the curve."""

import numpy as np
import matplotlib.pyplot as plt
import charu
from scipy.integrate import quad
from utils import *

rc = {
    "charu.doc": "rspa",
    "figure.figsize": [300 * charu.pt, 160 / charu.golden * charu.pt],
    "charu.tex": True,
    "charu.tex.font": "fourier",
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
}

b = 0.1 
eps = 0.01 # slowness parameter

def phi(x):
    # Find the angle that the curve makes with the x axis for
    # a sech-type curvature profile.
    return 2*b*np.arctan(np.tanh(x * eps / 2)) / eps

def point(x):
    return [quad(lambda s: np.cos(phi(s)), 0, x)[0], quad(lambda s: np.sin(phi(s)), 0, x)[0]]

with plt.rc_context(rc):
    fig, axes = plt.subplots(1, 2)
    labelpos = (0.06, 0.82)

    # curvature profile ---------------------------------------------

    ax = axes[0]

    ax.set_xlim(-3, 3)
    ax.set_xticks([-3, 0, 3])
    ax.set_ylim(-0.02, 0.12)
    ax.set_yticks([0, b])
    ax.set_xlabel(r"$\epsilon x$")
    ax.text(-0.4, 0.46, r"$m_3(x)$", transform=ax.transAxes)

    ax.plot([-5, 5], [0, 0], linestyle="--", color="#999999")
    ax.plot([0, 0], [-1, 1], linestyle="--", color="#999999")

    x = np.linspace(-3, 3, 500)
    ax.plot(x, b * sech(x), zorder=10)

    ax.text(*labelpos,
            r"\textbf{(a)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # curve profile ----------------------------------------------

    ax = axes[1]

    ax.set_xlim(-70, 70)
    ax.set_xticks([-50, 0, 50])
    ax.set_ylim(-30, 90)
    ax.set_yticks([0, 50])
    ax.set_xlabel(r"$X$")
    ax.text(-0.3, 0.46, r"$Z$", transform=ax.transAxes)

    ax.set_aspect("equal", adjustable="datalim")

    ax.plot([-200, 200], [0, 0], linestyle="--", color="#999999")
    ax.plot([0, 0], [-200, 200], linestyle="--", color="#999999")

    ax.text(*labelpos,
            r"\textbf{(b)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    xx = np.linspace(-3, 3, 1000)
    xx = xx / eps
    rr = np.array([point(x) for x in xx])

    ax.plot(rr[:, 0], rr[:, 1])

    # Export ---------------------------------------------------------------

    plt.tight_layout(w_pad=1)
    plt.savefig(
        "m3_sech_profile.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )

    plt.show()
