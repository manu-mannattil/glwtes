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
    fig, axes = plt.subplots(2, 2)

    # tanh (localized) -----------------------------------------------------

    name = "cc_10_tanh_sorted_l1.2_N2048"

    pack = np.load("../data/{}.npz".format(name))
    x, evals, z, u, v = pack["x"], pack["evals"], pack["z"], pack["u"], pack["v"]

    ax = axes[0, 0]

    i = 312

    ax.plot(x, z[i], "C3")
    ax.plot(x, u[i], "C0", zorder=-10)
    ax.plot(x, v[i], "#999999", zorder=-10)
    ax.set_xlim((-25, 25))
    ax.set_ylim((-5, 5))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\zeta, u, v$", labelpad=0)
    ax.text(0.7, 0.8, r"$\omega = {:.3f}$".format(evals[i]), transform=ax.transAxes)
    ax.text(0.05, 0.8, r"\textbf{(a)}", transform=ax.transAxes)

    # tanh (delocalized) ---------------------------------------------------

    ax = axes[0, 1]

    ax.plot(x, z[i + 1], "C3")
    ax.plot(x, u[i + 1], "C0", zorder=-10)
    ax.plot(x, v[i + 1], "#999999", zorder=-10)
    ax.set_xlim((-25, 25))
    ax.set_ylim((-5, 5))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\zeta, u, v$", labelpad=0)
    ax.text(0.7, 0.8, r"$\omega = {:.3f}$".format(evals[i + 1]), transform=ax.transAxes)
    ax.text(0.05, 0.8, r"\textbf{(b)}", transform=ax.transAxes)

    # sech (localized) -----------------------------------------------------

    name = "cc_10_sech_zero_sorted_l1.2_N2048"

    pack = np.load("../data/{}.npz".format(name))
    x, evals, z, u, v = pack["x"], pack["evals"], pack["z"], pack["u"], pack["v"]

    ax = axes[1, 0]

    i = 261

    ax.plot(x, z[i], "C3")
    ax.plot(x, u[i], "C0", zorder=-10)
    ax.plot(x, v[i], "#999999", zorder=-10)
    ax.set_xlim((-25, 25))
    ax.set_ylim((-2, 2))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\zeta, u, v$", labelpad=0)
    ax.text(0.7, 0.8, r"$\omega = {:.3f}$".format(evals[i]), transform=ax.transAxes)
    ax.text(0.05, 0.8, r"\textbf{(c)}", transform=ax.transAxes)

    # sech (delocalized) ---------------------------------------------------

    ax = axes[1, 1]

    ax.plot(x, z[i + 1], "C3")
    ax.plot(x, u[i + 1], "C0", zorder=-10)
    ax.plot(x, v[i + 1], "#999999", zorder=-10)
    ax.set_xlim((-25, 25))
    ax.set_ylim((-2, 2))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\zeta, u, v$", labelpad=0)
    ax.text(0.7, 0.8, r"$\omega = {:.3f}$".format(evals[i + 1]), transform=ax.transAxes)
    ax.text(0.05, 0.8, r"\textbf{(d)}", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(
        "shell_modes.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )
