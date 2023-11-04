# -*- coding: utf-8 -*-
"""Plot example bound states and corresponding rays."""

import numpy as np
import matplotlib.pyplot as plt
import charu
from matplotlib.colors import LinearSegmentedColormap
from utils import *

b = 0.1 # max curvature (tanh and sech)
a = 0.01 # min curvature (sech only)
eps = 0.01 # slowness parameter

rc = {
    "charu.doc": "aps",
    "figure.figsize": [320 * charu.pt, 350 / charu.golden * charu.pt],
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

def H(x, k, mode=1, mfunc=m1):
    m = mfunc(x)
    if mode == 1:
        sign = 1
    else:
        sign = -1
    return np.sqrt(0.5 * ((1 + k**2) * (k**2 + m**2) +
                   sign * np.sqrt((k**2 - m**2)**2 * (1 - k**2)**2 + 4 * m**2 * k**2 *
                                  (1 + k**2)**2)))

@np.vectorize
def ratio(x, k, mode=1, mfunc=m1):
    # Find the ratio of the transverse component (zeta) to the
    # longitudinal component (u) of the wave field.
    if mode == 1:
        sign = 1
    else:
        sign = -1

    M = mfunc(x)

    if M == 0:
        if mode == 1:
            r = 1.0
        else:
            r = 0.0
    else:
        w2 = 0.5 * ((1 + k**2) *
                    (k**2 + M**2) + sign * np.sqrt((1 + k**2)**2 * (k**2 + M**2)**2 - 4 *
                                                   (k**3 - k * M**2)**2))
        a = k**4 + M**2 - w2
        b = k * M + k**3 * M

        return np.abs(b) / (np.abs(a) + np.abs(b))

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

    name = "ho_tanh_bc_cc_b0.1_N_2048"
    omega = np.loadtxt(f"../data/{name}/wkb.txt", unpack=True)[1]

    w = H(xx, kk)

    ax.contour(xx, kk, w, colors="k", linestyles="-", levels=omega[1::2])
    ax.contour(xx, kk, w, colors="k", linestyles="--", levels=[b])

    # Lastly, plot the "eyebrows".
    levels = [0.10440662, 0.12024195, 0.14398984, 0.17242652, 0.20359708, 0.2364232, 0.27030506]
    ax.contour(xx, kk, w, colors="k", levels=levels)

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

    name = "ho_sech_bc_cc_b0.1_a0.01_N_2048"
    omega = np.loadtxt(f"../data/{name}/wkb.txt", unpack=True)[1]

    w = H(xx, kk, mfunc=m2)

    ax.contour(xx, kk, w, colors="k", linestyles="-", levels=omega[1::2])
    ax.contour(xx, kk, w, colors="k", linestyles="--", levels=[b])

    # Lastly, plot the "eyebrows".
    levels = [0.10440662, 0.12024195, 0.14398984, 0.17242652, 0.20359708, 0.2364232, 0.27030506]
    ax.contour(xx, kk, w, colors="k", levels=levels)

    r = rescale(ratio(xx, kk, mode=1, mfunc=m2))
    pcm = ax.pcolormesh(xx, kk, r, cmap=BlueRed, shading="nearest", rasterized=True)

    ax.text(*labelpos,
            r"\textbf{(b)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # tanh (localized) -----------------------------------------------------

    name = "ho_tanh_bc_cc_b0.1_N_2048"

    pack = np.load("../data/{}.npz".format(name))
    x, evals, z, u = pack["x"], pack["evals"], pack["z"], pack["u"]

    ax = axes[1, 0]

    i = 209

    z, u = normalize([z[i], u[i]], eps * x)
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

    name = "ho_sech_bc_cc_b0.1_a0.01_N_2048"

    pack = np.load("../data/{}.npz".format(name))
    x, evals, z, u = pack["x"], pack["evals"], pack["z"], pack["u"]

    ax = axes[1, 1]

    i = 181

    z, u = normalize([z[i], u[i]], eps * x)
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

    xs = np.arccosh((b - a) / (b - evals[i]))
    ax.plot([xs, xs], [-5, 5], "#999999", linestyle="--", zorder=-100)
    ax.plot([-xs, -xs], [-5, 5], "#999999", linestyle="--", zorder=-100)

    ax.text(*labelpos,
            r"\textbf{(d)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    ax.legend(loc=(0.03, 0.015))

    # Export ---------------------------------------------------------------

    plt.tight_layout(h_pad=2, w_pad=3)
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
