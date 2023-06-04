# -*- coding: utf-8 -*-
"""Plot the two curvature profiles and the corresponding shapes of the rod."""

import numpy as np
import matplotlib.pyplot as plt
import charu
from scipy.integrate import quad
from utils import *

rc = {
    "charu.doc": "rspa",
    "figure.figsize": [300 * charu.pt, 320 / charu.golden * charu.pt],
    "charu.tex": True,
    "charu.tex.font": "fourier",
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
}

b = 0.1 # max curvature (tanh and sech)
a = 0.01 # min curvature (sech only)
eps = 0.01 # slowness parameter

def phi1(x):
    # Find the angle that the curve makes with the x axis for
    # a tanh-type curvature profile.
    return b * np.log(np.cosh(x * eps)) / eps

def phi2(x):
    # Find the angle that the curve makes with the x axis for
    # a sech-type curvature profile.
    return b * x - 2 * (b - a) * np.arctan(np.tanh(x * eps / 2)) / eps

def point(x, profile="tanh"):
    # Find a point on the curve by integrating the tangent vector with
    # arc length.
    if profile == "tanh":
        phi = phi1
    else:
        phi = phi2

    return [quad(lambda s: np.cos(phi(s)), 0, x)[0], quad(lambda s: np.sin(phi(s)), 0, x)[0]]

with plt.rc_context(rc):
    fig, axes = plt.subplots(2, 2)
    labelpos = (0.06, 0.82)

    # tanh (curvature profile) ---------------------------------------------

    ax = axes[0, 0]

    ax.set_xlim(-3, 3)
    ax.set_xticks([-3, 0, 3])
    ax.set_ylim(-0.18, 0.18)
    ax.set_yticks([-b, 0, b])
    ax.set_xlabel(r"$\epsilon x$")
    ax.text(-0.4, 0.46, r"$m_1(x)$", transform=ax.transAxes)

    ax.plot([-5, 5], [0, 0], linestyle="--", color="#999999")
    ax.plot([0, 0], [-1, 1], linestyle="--", color="#999999")

    x = np.linspace(-3, 3, 500)
    ax.plot(x, b * np.tanh(x), zorder=10)

    ax.text(*labelpos,
            r"\textbf{(a)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # tanh (rod profile) ----------------------------------------------

    ax = axes[0, 1]

    ax.set_xlim(-70, 70)
    ax.set_xticks([-50, 0, 50])
    ax.set_ylim(-65, 65)
    ax.set_yticks([-50, 0, 50])
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
    rr = np.array([point(x, "tanh") for x in xx])

    ax.plot(rr[:, 0], rr[:, 1])

    # sech (curvature profile) ---------------------------------------------

    ax = axes[1, 0]

    ax.set_xlim(-3, 3)
    ax.set_xticks([-3, 0, 3])
    ax.set_ylim(-2 * a, 0.14)
    ax.set_yticks([a, 0.1])
    ax.set_xlabel(r"$\epsilon x$")
    ax.text(-0.4, 0.46, r"$m_2(x)$", transform=ax.transAxes)

    x = np.linspace(-5, 5, 500)
    ax.plot(x, b - (b - a) / np.cosh(x), zorder=10)

    ax.plot([-5, 5], [a, a], linestyle="--", color="#999999")
    ax.plot([0, 0], [-1, 1], linestyle="--", color="#999999")

    ax.text(*labelpos,
            r"\textbf{(c)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # sech (rod profile) ----------------------------------------------

    ax = axes[1, 1]

    ax.set_xlim(-90, 90)
    ax.set_xticks([-50, 0, 50])
    ax.set_ylim(-10, 80)
    ax.set_yticks([0, 50])
    ax.set_xlabel(r"$X$")
    ax.text(-0.3, 0.46, r"$Z$", transform=ax.transAxes)

    ax.set_aspect("equal", adjustable="datalim")

    ax.plot([-200, 200], [0, 0], linestyle="--", color="#999999")
    ax.plot([0, 0], [-200, 200], linestyle="--", color="#999999")

    ax.text(*labelpos,
            r"\textbf{(d)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    xx = np.linspace(-3, 3, 1000)
    xx = xx / eps
    rr = np.array([point(x, "sech") for x in xx])

    ax.plot(rr[:, 0], rr[:, 1])

    # Export ---------------------------------------------------------------

    plt.tight_layout(w_pad=1)
    plt.savefig(
        "rod_profile.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )

    plt.show()
