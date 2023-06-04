# -*- coding: utf-8 -*-
"""Compare Poschl-Teller solutions and extensional bound states (sech)."""

import matplotlib.pyplot as plt
import numpy as np
import charu
import utils

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr, wl

session = WolframLanguageSession("/usr/local/bin/WolframKernel")
session.start()

def sol(x, n=0, b=0.1, a=0.01, chi=0.488428, eps=0.01):
    # Deformed parameters
    epsd = eps * chi
    bd = np.sqrt(b**2 - a**2)

    nu = (np.sqrt(1 + 4 * bd**2 / epsd**2) - 1) / 2
    tanhx = np.tanh(epsd * x)
    m = np.sqrt(b**2 - (b**2 - a**2) / np.cosh(epsd * x)**2)

    # Why use Mathematica to do this?  Because SciPy's Legengre
    # polynomials require integer (m, n) whereas we have rational (m, n).
    z = m * np.array(session.evaluate(wl.LegendreP(nu, -nu + n, list(tanhx))))

    expr = f"(Evaluate[D[LegendreP[{nu}, -{nu} + {n}, Tanh[{epsd} #]], #]])&"
    u = session.evaluate(wl.Map(wlexpr(expr), list(x)))

    return utils.normalize([z, u], eps * x)

rc = {
    "charu.doc": "rspa",
    "figure.figsize": [410 * charu.pt, 410 * charu.pt / charu.golden / 2.25],
    "charu.tex": True,
    "charu.tex.font": "fourier",
}

b = 0.1
a = 0.01
eps = 0.01

with plt.rc_context(rc):
    fig, axes = plt.subplots(1, 3)

    labelpos = (0.08, 0.83)
    w_min = 0.01
    w_max = 0.1 + 0.01
    w_guide = np.linspace(w_min, w_max, 10)
    ticks = [0.02, 0.06, 0.1]
    size = 3.7

    name = "sech_bc_cc_b0.1_a0.01_N_2048"
    pack = np.load("../data/{}.npz".format(name))
    x, evals, zz, uu = pack["x"], pack["evals"], pack["z"], pack["u"]

    # Mode shapes ----------------------------------------------------------

    ax = axes[0]
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xlabel(r"$\epsilon x$")
    ax.set_ylabel(r"$\zeta(\epsilon x)$", labelpad=0)
    ax.text(*labelpos, r"\textbf{(a)}", transform=ax.transAxes)

    alpha = np.linspace(0.2, 1, 5)[::-1]
    ii = [86, 113, 130, 142, 153]
    for n, i in enumerate(ii):
        z, u = zz[i], uu[i]
        z, u = utils.normalize([z, u], eps * x)
        ax.plot(x * eps, z, "w-", linewidth=2.25, zorder=100 * (5 - n))
        ax.plot(x * eps, z, "C3-", alpha=alpha[n], zorder=100 * (5 - n))

        z1, u1 = sol(x, n=n)
        ax.plot(x * eps, z1, "k--", alpha=alpha[n], dashes=(1.5, 2), zorder=100 * (5 - n))

    # Mode shapes ----------------------------------------------------------

    ax = axes[1]
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xlabel(r"$\epsilon x$")
    ax.set_ylabel(r"$u(\epsilon x)$", labelpad=0)
    ax.text(*labelpos, r"\textbf{(b)}", transform=ax.transAxes)

    alpha = np.linspace(0.2, 1, 5)[::-1]
    ii = [86, 113, 130, 142, 153]
    for n, i in enumerate(ii):
        z, u = zz[i], uu[i]
        z, u = utils.normalize([z, u], eps * x)
        ax.plot(x * eps, u, "w-", linewidth=2.25, zorder=100 * (5 - n))
        ax.plot(x * eps, u, "C0-", alpha=alpha[n], zorder=100 * (5 - n))

        z1, u1 = sol(x, n=n)
        ax.plot(x * eps, u1, "k--", alpha=alpha[n], dashes=(1.5, 2), zorder=100 * (5 - n))

    # Eigenfrequencies -----------------------------------------------------

    # Deformed parameters.
    chi = 0.488428
    epsd = eps * chi
    bd = np.sqrt(b**2 - a**2)

    nu = (np.sqrt(1 + 4 * bd**2 / epsd**2) - 1) / 2
    n = np.arange(0, int(nu) + 1)
    mu = n - nu

    w = b**2 - epsd**2 * (mu)**2
    w = np.sqrt(w)

    ax = axes[2]

    ax.set_aspect("equal")
    ax.set_xlim(w_min, w_max)
    ax.set_ylim(w_min, w_max)
    ax.set_xlabel(r"$\omega$ (quantized)")
    ax.set_ylabel(r"$\omega$ (numerics)")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.text(*labelpos, r"\textbf{(c)}", transform=ax.transAxes)

    name = "sech_bc_cc_b0.1_a0.01_N_2048"
    w_numerics = np.loadtxt(f"../data/{name}/quantized.txt", unpack=True)[0]
    w = w[:len(w_numerics)]

    ax.plot(w_guide, w_guide, color="#aaaaaa", zorder=-10)
    ax.plot(w, w_numerics, "C0o", markerfacecolor="none", markersize=size, markeredgewidth=0.5)

    plt.tight_layout(w_pad=0)
    plt.savefig(
        "rod_poschl_sech.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        facecolor="none",
        pad_inches=0,
    )
    plt.show()

session.stop()
