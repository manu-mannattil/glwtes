# -*- coding: utf-8 -*-

from numba import njit
from utils import *
import charu
import matplotlib.pyplot as plt
import numpy as np

@njit(fastmath=True)
def _sech(x):
    return np.sqrt(1 - np.tanh(x)**2)

@njit(fastmath=True)
def H(x, k, c=1, sign=1):
    return 0.5 * (k**2 + k**4 + 64*(c - _sech(x))**2 + sign*np.sqrt((k**2 + k**4 + 64*(c - _sech(x))**2)**2 - 4*k**6))

N = 300
x = np.linspace(-6, 6, N)
k = np.linspace(-6, 6, N)
xx, kk = np.meshgrid(x, k)

rc = {
    "charu.doc": "rspa",
    "figure.figsize": [360 * charu.pt, 210 / charu.golden * charu.pt],
    "charu.tex": True,
    "charu.tex.font": "fourier",
}

with plt.rc_context(rc):
    fig, axes = plt.subplot_mosaic([["left", [["top"], ["bottom"]]]], layout="constrained")

    ax1, ax2, ax3 = axes["left"], axes["top"], axes["bottom"]

    ax1.set_xticks([-0.75, 0, 0.75])
    ax1.set_yticks([-5, -2.5, 0, 2.5, 5.0])
    ax1.set_xlim(-0.75, 0.75)
    ax1.set_ylim(-4, 4)

    zz = H(xx, kk)
    ax1.contour(xx, kk, zz, colors="C3", levels=[1], linestyles="--")

    zz = H(xx, kk, sign=-1)
    ax1.contour(xx, kk, zz, colors="C0", levels=[1], alpha=0.75, linestyles="--")

    zz = H(xx, kk, c=1.25)
    ax1.contour(xx, kk, zz, colors="C3", levels=[H(0, 1, c=1.25)])

    zz = H(xx, kk, c=1.25, sign=-1)
    ax1.contour(xx, kk, zz, colors="C0", levels=[H(0, 1, c=1.25)], alpha=0.75)

    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$k$", rotation=0, va="center", labelpad=0)
    ax1.text(0.03, 0.9, r"\textbf{(a)}", transform=ax1.transAxes)

    ax1.scatter([0, 0, 0], [1, 0, -1], color="k", zorder=100)

    # sech (localized) -----------------------------------------------------

    name = "cc_10_sech_zero_raw_2048"

    pack = np.load("../data/{}.npz".format(name))
    x, evals, z, u = pack["x"], pack["evals"], pack["z"], pack["u"]
    evals, z, u = sort_evals_modes(evals, z, u)

    i = 64

    ax2.plot(x, u[i], "C0")
    ax2.plot(x, z[i], "C3")
    ax2.set_xlim((-50, 50))
    ax2.set_ylim((-5, 5))
    ax2.set_xticks([])
    #ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$\zeta, u$", labelpad=0)
    ax2.text(0.75, 0.75, r"$\omega = {:.3f}$".format(evals[i]), transform=ax2.transAxes)
    ax2.text(0.03, 0.75, r"\textbf{(b)}", transform=ax2.transAxes)

    name = "cc_8_sech_nonzero_raw_2048"

    pack = np.load("../data/{}.npz".format(name))
    x, evals, z, u = pack["x"], pack["evals"], pack["z"], pack["u"]
    evals, z, u = sort_evals_modes(evals, z, u)

    i = 104

    ax3.plot(x, u[i], "C0")
    ax3.plot(x, z[i], "C3")
    ax3.set_xlim((-50, 50))
    ax3.set_ylim((-5, 5))
    ax3.set_xlabel(r"$x$")
    ax3.set_ylabel(r"$\zeta, u$", labelpad=0)
    ax3.text(0.75, 0.75, r"$\omega = {:.3f}$".format(evals[i]), transform=ax3.transAxes)
    ax3.text(0.03, 0.75, r"\textbf{(c)}", transform=ax3.transAxes)

    plt.tight_layout()
    plt.savefig(
        "degenerate.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        facecolor="none",
        pad_inches=0,
    )

    plt.show()
