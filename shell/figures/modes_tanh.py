# -*- coding: utf-8 -*-

"""Plot the example bound states of the shell (tanh type)."""

import numpy as np
import matplotlib.pyplot as plt
import charu
from utils import *
from matplotlib.lines import Line2D
from dispersion import *

rc = {
    "charu.doc": "rspa",
    "figure.figsize": [360 * charu.pt, 360 / charu.golden * charu.pt],
    "charu.tex": True,
    "charu.tex.font": "fourier",
}

eps = 0.01
b = 0.1
a = 0.01

def caustics(w, l=0.1, h=0.3, form="tanh", bowtie=False):
    """Find all caustics (including bowtie caustics)."""
    if bowtie:
        m = flex_find_m(w, l, h)
    else:
        m = np.sqrt((w**2 - l**2)/(w**2 - (1-h**2)*l**2)*(w**2 - l**4))

    if form == "tanh":
        return np.arctanh(m/b)
    else:
        return arcsech((b - m)/(b-a))

with plt.rc_context(rc):
    fig, axes = plt.subplots(3, 2)

    xmax = 5
    ymin, ymax = -1.25, 2.5
    wpos = 0.72, 0.74
    labelpos = 0.05, 0.75

    name = "tanh_bc_cc_l_0.1_eps_0.01_b0.1_N_2048"

    pack = np.load("../data/{}.npz".format(name))
    x, evals, z, u, v = pack["x"], pack["evals"], pack["z"], pack["u"], pack["v"]

    # flexural -------------------------------------------------------------

    ax = axes[0, 0]

    i = 0

    _z, _u, _v = normalize([z[i], u[i], v[i]], eps*x)
    ax.plot(x*eps, _z.real, "C3")
    ax.plot(x*eps, _u.real, "C0", zorder=-10)
    ax.plot(x*eps, _v.imag, "#333333", zorder=-10)
    ax.set_xlim((-xmax, xmax))
    ax.set_ylim((ymin, ymax))
    ax.set_ylabel(r"$\zeta, u, \Im(v)$")
    ax.text(*wpos, r"$\omega = {:.4f}$".format(evals[i]), transform=ax.transAxes)
    ax.text(*labelpos, r"\textbf{(a)}", transform=ax.transAxes)

    c = caustics(evals[i])
    ax.plot([-c, -c], [-ymax, ymax], "#999999", linestyle="--", zorder=-100)
    ax.plot([c, c], [-ymax, ymax], "#999999", linestyle="--", zorder=-100)

    # shear -------------------------------------------------------------

    ax = axes[1, 0]

    i = 153

    _z, _u, _v = normalize([z[i], u[i], v[i]], eps*x)
    ax.plot(x*eps, _z.real, "C3")
    ax.plot(x*eps, _u.real, "C0", zorder=-10)
    ax.plot(x*eps, _v.imag, "#333333", zorder=-10)
    ax.set_xlim((-xmax, xmax))
    ax.set_ylim((ymin, ymax))
    ax.set_ylabel(r"$\zeta, u, \Im(v)$")
    ax.text(*wpos, r"$\omega = {:.4f}$".format(evals[i]), transform=ax.transAxes)
    ax.text(*labelpos, r"\textbf{(c)}", transform=ax.transAxes)

    c = caustics(evals[i])
    ax.plot([-c, -c], [-ymax, ymax], "#999999", linestyle="--", zorder=-100)
    ax.plot([c, c], [-ymax, ymax], "#999999", linestyle="--", zorder=-100)

    # extensional -------------------------------------------------------------

    ax = axes[2, 0]

    i = 265

    _z, _u, _v = normalize([z[i], u[i], v[i]], eps*x)
    ax.plot(x*eps, _z.real, "C3")
    ax.plot(x*eps, _u.real, "C0", zorder=-10)
    ax.plot(x*eps, _v.imag, "#333333", zorder=-10)
    ax.set_xlim((-xmax, xmax))
    ax.set_ylim((ymin, ymax))
    ax.set_ylabel(r"$\zeta, u, \Im(v)$")
    ax.text(*wpos, r"$\omega = {:.4f}$".format(evals[i]), transform=ax.transAxes)
    ax.text(*labelpos, r"\textbf{(e)}", transform=ax.transAxes)

    c = caustics(evals[i])
    ax.plot([-c, -c], [-ymax, ymax], "#999999", linestyle="--", zorder=-100)
    ax.plot([c, c], [-ymax, ymax], "#999999", linestyle="--", zorder=-100)

    ax.set_xlabel(r"$x$")

    # new set on the right -------------------------------------------------

    ymin, ymax = -1.5, 2
    xmax = 7

    # flexural -------------------------------------------------------------

    ax = axes[0, 1]

    i = 8

    _z, _u, _v = normalize([z[i], u[i], v[i]], eps*x)
    ax.plot(x*eps, _z.real, "C3")
    ax.plot(x*eps, _u.real, "C0", zorder=-10)
    ax.plot(x*eps, _v.imag, "#333333", zorder=-10)
    ax.set_xlim((-xmax, xmax))
    ax.set_ylim((ymin, ymax))
    ax.set_ylabel(r"$\zeta, u, \Im(v)$")
    ax.text(*wpos, r"$\omega = {:.4f}$".format(evals[i]), transform=ax.transAxes)
    ax.text(*labelpos, r"\textbf{(b)}", transform=ax.transAxes)

    c = caustics(evals[i])
    ax.plot([-c, -c], [-ymax, ymax], "#999999", linestyle="--", zorder=-100)
    ax.plot([c, c], [-ymax, ymax], "#999999", linestyle="--", zorder=-100)

    c = caustics(evals[i], bowtie=True)
    ax.plot([-c, -c], [-ymax, ymax], "#999999", linestyle="--", zorder=-100)
    ax.plot([c, c], [-ymax, ymax], "#999999", linestyle="--", zorder=-100)

    # shear -------------------------------------------------------------

    ax = axes[1, 1]

    i = 180

    _z, _u, _v = normalize([z[i], u[i], v[i]], eps*x)
    ax.plot(x*eps, _z.real, "C3")
    ax.plot(x*eps, _u.real, "C0", zorder=-10)
    ax.plot(x*eps, _v.imag, "#333333", zorder=-10)
    ax.set_xlim((-xmax, xmax))
    ax.set_ylim((ymin, ymax))
    ax.set_ylabel(r"$\zeta, u, \Im(v)$")
    ax.text(*wpos, r"$\omega = {:.4f}$".format(evals[i]), transform=ax.transAxes)
    ax.text(*labelpos, r"\textbf{(d)}", transform=ax.transAxes)

    c = caustics(evals[i])
    ax.plot([-c, -c], [-ymax, ymax], "#999999", linestyle="--", zorder=-100)
    ax.plot([c, c], [-ymax, ymax], "#999999", linestyle="--", zorder=-100)

    # extensional -------------------------------------------------------------

    ax = axes[2, 1]

    i = 301

    _z, _u, _v = normalize([z[i], u[i], v[i]], eps*x)
    ax.plot(x*eps, _z.real, "C3")
    ax.plot(x*eps, _u.real, "C0", zorder=-10)
    ax.plot(x*eps, _v.imag, "#333333", zorder=-10)
    ax.set_xlim((-xmax, xmax))
    ax.set_ylim((ymin, ymax))
    ax.set_ylabel(r"$\zeta, u, \Im(v)$")
    ax.text(*wpos, r"$\omega = {:.4f}$".format(evals[i]), transform=ax.transAxes)
    ax.text(*labelpos, r"\textbf{(f)}", transform=ax.transAxes)

    c = caustics(evals[i])
    ax.plot([-c, -c], [-ymax, ymax], "#999999", linestyle="--", zorder=-100)
    ax.plot([c, c], [-ymax, ymax], "#999999", linestyle="--", zorder=-100)

    ax.set_xlabel(r"$x$")

    # Put a manual legend below the plots.
    colors = ["C3", "C0", "#333333"]
    lines = [Line2D([0], [0], color=c, linestyle="-") for c in colors]
    labels = [r"$\zeta$", r"$u$", r"$\Im(v)$"]
    fig.legend(lines, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.025), handlelength=2, handleheight=1)

    plt.tight_layout(w_pad=3)
    plt.savefig(
        "shell_modes_tanh.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
