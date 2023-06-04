# -*- coding: utf-8 -*-
"""Compare Poschl-Teller solutions and extensional bound states (tanh)."""

import matplotlib.pyplot as plt
import numpy as np
import charu
import utils

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr, wl

session = WolframLanguageSession("/usr/local/bin/WolframKernel")
session.start()

def sol(x, n=0, b=0.1, eps=0.01):
    nu = (np.sqrt(1 + 4 * b**2 / eps**2) - 1) / 2
    tanhx = np.tanh(eps * x)

    # Why use Mathematica to do this?  Because SciPy's Legengre
    # polynomials require integer (m, n) whereas we have rational (m, n).
    z = b * tanhx * np.array(session.evaluate(wl.LegendreP(nu, -nu + n, list(tanhx))))

    expr = f"(Evaluate[D[LegendreP[{nu}, -{nu} + {n}, Tanh[{eps} #]], #]])&"
    u = session.evaluate(wl.Map(wlexpr(expr), list(x)))

    return utils.normalize([z, u], x * eps)

rc = {
    "charu.doc": "rspa",
    "figure.figsize": [410 * charu.pt, 410 * charu.pt / charu.golden / 2.25],
    "charu.tex": True,
    "charu.tex.font": "fourier",
}

b = 0.1
eps = 0.01

with plt.rc_context(rc):
    fig, axes = plt.subplots(1, 3)

    labelpos = (0.08, 0.83)
    w_min = 0.01
    w_max = 0.1 + 0.01
    w_guide = np.linspace(w_min, w_max, 10)
    ticks = [0.02, 0.06, 0.1]
    size = 3.7

    name = "tanh_bc_cc_b0.1_N_2048"
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
    ii = [105, 139, 157, 170, 179]
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
    ii = [105, 139, 157, 170, 179]
    for n, i in enumerate(ii):
        z, u = zz[i], uu[i]
        z, u = utils.normalize([z, u], eps * x)
        ax.plot(x * eps, u, "w-", linewidth=2.25, zorder=100 * (5 - n))
        ax.plot(x * eps, u, "C0-", alpha=alpha[n], zorder=100 * (5 - n))

        z1, u1 = sol(x, n=n)
        ax.plot(x * eps, u1, "k--", alpha=alpha[n], dashes=(1.5, 2), zorder=100 * (5 - n))

    # Eigenfrequencies -----------------------------------------------------

    nu = (np.sqrt(1 + 4 * b**2 / eps**2) - 1) / 2
    n = np.arange(0, int(nu) + 1)
    mu = n - nu

    w = b**2 - eps**2 * (mu)**2
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

    name = "tanh_bc_cc_b0.1_N_2048"
    w_numerics = np.loadtxt(f"../data/{name}/quantized.txt", unpack=True)[0]

    ax.plot(w_guide, w_guide, color="#aaaaaa", zorder=-10)
    ax.plot(w, w_numerics, "C0o", markerfacecolor="none", markersize=size, markeredgewidth=0.5)

    plt.tight_layout(w_pad=0)
    plt.savefig(
        "rod_poschl.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        facecolor="none",
        pad_inches=0,
    )
    plt.show()

session.stop()
