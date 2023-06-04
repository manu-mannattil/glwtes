# -*- coding: utf-8 -*-
"""Plot example bound states and corresponding rays."""

import numpy as np
import matplotlib.pyplot as plt
import charu
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import quad
from utils import *

b = 0.1 # max curvature (tanh and sech)
a = 0.01 # min curvature (sech only)
eps = 0.01 # slowness parameter

rc = {
    "charu.doc": "rspa",
    "figure.figsize": [300 * charu.pt, 345 / charu.golden * charu.pt],
    "charu.tex": True,
    "charu.tex.font": "fourier",
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
}

BlueRed_data = ["C0", "white", "C3"]
BlueRed = LinearSegmentedColormap.from_list("BlueRed", BlueRed_data)

def m1(x):
    return b * np.tanh(x)

def m2(x):
    return b - (b - a) * sech(x)

@np.vectorize
def ratio(x=0.1, k=0.1, mode=1, mfunc=m1):
    # Find the ratio of the transverse component (zeta) to the
    # longitudinal component (u) of the wave field.
    if mode == 1:
        sign = 1
    else:
        sign = -1

    m = mfunc(x)

    if m == 0:
        if mode == 1:
            r = 1.0
        else:
            r = 0.0
    else:
        w = np.sqrt(
            0.5 * (k**4 + k**2 + m**2 + sign * np.sqrt((k**4 - k**2 + m**2)**2 + 4 * k**2 * m**2)))
        a = abs(k * m)
        b = abs(k**4 + m**2 - w**2)

        r = a / (a + b)

    return r

@np.vectorize
def ratio_approx(x=0.1, k=0.1, mode=1, mfunc=m1):
    m = mfunc(x)

    if mode == 1:
        r = np.abs(m) / (np.abs(k) + np.abs(m))
    else:
        r = np.abs(k) / (np.abs(k) + np.abs(m))

    return r

def rescale(m):
    """Rescale an nd-array such that its elements are in [0, 1]."""
    a = m.flatten().min()
    b = m.flatten().max()
    return (m - a) / (b - a)

with np.load("../data/rays.npz", allow_pickle=True) as pack:
    sech1_eye = pack["sech1_eye"]
    sech1_eyebrows = pack["sech1_eyebrows"]
    sech1_eyelid = pack["sech1_eyelid"]
    sech1_mark = pack["sech1_mark"]

    tanh1_eye = pack["tanh1_eye"]
    tanh1_eyebrows = pack["tanh1_eyebrows"]
    tanh1_eyelid = pack["tanh1_eyelid"]
    tanh1_mark = pack["tanh1_mark"]

with plt.rc_context(rc):
    fig, axes = plt.subplots(2, 2)
    labelpos = (0.05, 0.85)
    omegapos = (0.63, 0.13)

    N = 300

    x = np.linspace(-5, 5, N)
    k = np.linspace(-0.4, 0.4, N)
    xx, kk = np.meshgrid(x, k)

    # tanh rays (localized) -----------------------------------------------------

    ax = axes[0, 0]

    ax.set_xlim(-5, 5)
    ax.set_xticks([-5, -2.5, 0, 2.5, 5])
    ax.set_ylim(-0.2, 0.2)
    ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    ax.set_xlabel(r"$\epsilon x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center")

    for r in tanh1_eye:
        ax.plot(r[0], r[1], "k-")

    for r in tanh1_mark:
        ax.plot(r[0], r[1], "k-")

    for r in tanh1_eyebrows:
        ax.plot(r[0], r[1], "k-")

    for r in tanh1_eyelid:
        ax.plot(r[0], r[1], color="k", linestyle="--", zorder=100)

    r = rescale(ratio(xx, kk, mode=1, mfunc=m1))
    pcm = ax.pcolormesh(xx, kk, r, cmap=BlueRed, shading="nearest", rasterized=True)

    ax.text(*labelpos,
            r"\textbf{(a)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # sech rays (localized) -----------------------------------------------------

    ax = axes[0, 1]

    ax.set_xlim(-5, 5)
    ax.set_xticks([-5, -2.5, 0, 2.5, 5])
    ax.set_ylim(-0.2, 0.2)
    ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    ax.set_xlabel(r"$\epsilon x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center", labelpad=0)

    for r in sech1_eye:
        ax.plot(r[0], r[1], "k-")

    for r in sech1_mark:
        ax.plot(r[0], r[1], "k-")

    for r in sech1_eyebrows:
        ax.plot(r[0], r[1], "k-")

    for r in sech1_eyelid:
        ax.plot(r[0], r[1], color="k", linestyle="--", zorder=100)

    r = rescale(ratio(xx, kk, mode=1, mfunc=m2))
    pcm = ax.pcolormesh(xx, kk, r, cmap=BlueRed, shading="nearest", rasterized=True)

    ax.text(*labelpos,
            r"\textbf{(b)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # tanh (localized) -----------------------------------------------------

    name = "tanh_bc_cc_b0.1_N_2048"

    pack = np.load("../data/{}.npz".format(name))
    x, evals, z, u = pack["x"], pack["evals"], pack["z"], pack["u"]

    ax = axes[1, 0]

    i = 187

    z, u = normalize([z[i], u[i]], eps*x)
    ax.plot(x * eps, z, "C3", label=r"$\zeta$")
    ax.plot(x * eps, u, "C0", label=r"$u$")
    ax.set_xlim((-5, 5))
    ax.set_xticks([-5, -2.5, 0, 2.5, 5])
    ax.set_ylim((-1.5, 1.5))
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel(r"$\epsilon x$")
    ax.set_ylabel(r"$\zeta, u$", labelpad=0)
    ax.text(*omegapos,
            r"$\omega = {:.4f}$".format(evals[i]),
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    xs = np.arctanh(evals[i] / b)
    ax.plot([xs, xs], [-5, 5], "#999999", linestyle="--", zorder=-100)
    ax.plot([-xs, -xs], [-5, 5], "#999999", linestyle="--", zorder=-100)

    ax.text(*labelpos,
            r"\textbf{(c)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    ax.legend(loc=(0.03, 0.015))

    # sech (localized) -----------------------------------------------------

    name = "sech_bc_cc_b0.1_a0.01_N_2048"

    pack = np.load("../data/{}.npz".format(name))
    x, evals, z, u = pack["x"], pack["evals"], pack["z"], pack["u"]

    ax = axes[1, 1]

    i = 161

    z, u = normalize([z[i], u[i]], eps*x)
    ax.plot(x * eps, z, "C3", label=r"$\zeta$")
    ax.plot(x * eps, u, "C0", label=r"$u$")
    ax.set_xlim((-5, 5))
    ax.set_xticks([-5, -2.5, 0, 2.5, 5])
    ax.set_ylim((-1.5, 1.5))
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel(r"$\epsilon x$")
    ax.set_ylabel(r"$\zeta, u$", labelpad=0)
    ax.text(*omegapos,
            r"$\omega = {:.4f}$".format(evals[i]),
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    xs = np.arccosh((b - a)/(b - evals[i]))
    ax.plot([xs, xs], [-5, 5], "#999999", linestyle="--", zorder=-100)
    ax.plot([-xs, -xs], [-5, 5], "#999999", linestyle="--", zorder=-100)

    ax.text(*labelpos,
            r"\textbf{(d)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    ax.legend(loc=(0.03, 0.015))

    # Export ---------------------------------------------------------------

    plt.tight_layout()
    plt.savefig(
        "rod_bound_inc.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )

    plt.show()