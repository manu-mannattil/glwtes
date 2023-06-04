# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import charu
from dispersion import *

rc = {
    "charu.doc": "rspa",
    "figure.figsize": [380 * charu.pt, 380 * charu.pt / charu.golden / 2.25],
    "charu.tex": True,
    "charu.tex.font": "fourier",
    "axes.axisbelow": False,
}

with plt.rc_context(rc):
    fig, axes = plt.subplots(1, 3)

    l = 0.1 # transverse wave number
    h = 0.3 # Poisson's ratio
    m = 0.05 # curvature

    # pack = np.load("../data/dispersion_l{}_h_{}_m{}.npz".format(l, h, m))
    # k, w = pack["k"], pack["w"]

    k = np.linspace(0, 0.2, 300)
    w = np.array([omega(_, l, h, m) for _ in k])

    # Flexural ---------------------------------------------------------------

    ax = axes[0]

    # Move left y-axis to centre.
    ax.spines["left"].set_position("center")
    # Eliminate upper and right axes.
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # Plot zero curvature dispersion.
    q = np.linspace(-0.25, 0.25, 300)
    ax.plot(q, l**2 + q**2, "k--")

    # Pierce/Norris-Rebinsky dispersion.
    ax.plot(q, np.sqrt((l**2 + q**2)**2 + (1 - h**2) * m**2 * l**4 / (q**2 + l**2)**2), "C3--")

    # Numerical dispersion.
    ax.plot(k, w[:, 2], color="C0")
    ax.plot(-k[::-1], w[:, 2][::-1], color="C0")

    ax.minorticks_on()
    ax.set_ylim((l**2, 0.075))
    ax.set_xlim((-0.25, 0.25))
    ax.set_yticks([0.010, 0.060])
    ax.set_yticklabels(["0.01", "0.06"],
                       bbox={
                           "boxstyle": "square, pad=0.2",
                           "facecolor": "white",
                           "linewidth": 0,
                       })


    ax.text(1.05,
            -0.025,
            r"$k$",
            horizontalalignment="center",
            transform=ax.transAxes,
            verticalalignment="center")
    ax.text(0.5,
            1.1,
            r"$\omega(k)$",
            horizontalalignment="center",
            transform=ax.transAxes,
            verticalalignment="center")

    ax.text(0.10,
            0.15,
            r"\textbf{(a)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # Shear ------------------------------------------------------------------

    ax = axes[1]

    # Move left y-axis to centre.
    ax.spines["left"].set_position("center")
    # Eliminate upper and right axes.
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # Plot zero curvature dispersion.
    q = np.linspace(-0.25, 0.25, 300)
    ax.plot(q, np.sqrt(0.5 * (1 - h) * (l**2 + q**2)), "k--")

    # Norris-Rebinsky dispersion.
    ax.plot(
        q,
        np.sqrt(0.5 * (1 - h) * (l**2 + q**2) + 2 * (1 - h) * m**2 * q**2 * l**2 /
                (q**2 + l**2)**2),
        "C3--")

    # Numerical dispersion.
    ax.plot(k, w[:, 1], color="C0")
    ax.plot(-k[::-1], w[:, 1][::-1], color="C0")

    ax.minorticks_on()
    ax.set_ylim(np.sqrt(0.5*(1-h)*l**2), 0.08)
    ax.set_xlim((-0.1, 0.1))
    ax.set_yticks([0.059, 0.069, 0.079])
    ax.set_yticklabels(["0.059", "0.069", "0.079"],
                       bbox={
                           "boxstyle": "square, pad=0.2",
                           "facecolor": "white",
                           "linewidth": 0,
                       })

    ax.text(1.05,
            -0.025,
            r"$k$",
            horizontalalignment="center",
            transform=ax.transAxes,
            verticalalignment="center")
    ax.text(0.5,
            1.1,
            r"$\omega(k)$",
            horizontalalignment="center",
            transform=ax.transAxes,
            verticalalignment="center")

    ax.text(0.10,
            0.15,
            r"\textbf{(b)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # Extensional ------------------------------------------------------------

    ax = axes[2]

    # Move left y-axis to centre.
    ax.spines["left"].set_position("center")
    # Eliminate upper and right axes.
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # Plot zero curvature dispersion.
    q = np.linspace(-0.25, 0.25, 300)
    ax.plot(q, np.sqrt(l**2 + q**2), "k--")

    # Norris-Rebinsky dispersion.
    ax.plot(q, np.sqrt((l**2 + q**2) + m**2*(q**2 + h*l**2)**2/(q**2 + l**2)**2), "C3--")

    ax.plot(k, w[:, 0], color="C0")
    ax.plot(-k[::-1], w[:, 0][::-1], color="C0")

    ax.minorticks_on()
    ax.set_ylim((0.1, 0.11))
    ax.set_xlim((-0.05, 0.05))
    ax.set_xticks([-0.05, 0.00, 0.05])
    ax.set_yticks([0.1, 0.11])
    ax.set_yticklabels(["0.10", "0.11"],
                       bbox={
                           "boxstyle": "square, pad=0.2",
                           "facecolor": "white",
                           "linewidth": 0,
                       })

    ax.text(1.05,
            -0.025,
            r"$k$",
            horizontalalignment="center",
            transform=ax.transAxes,
            verticalalignment="center")
    ax.text(0.5,
            1.1,
            r"$\omega(k)$",
            horizontalalignment="center",
            transform=ax.transAxes,
            verticalalignment="center")

    ax.text(0.10,
            0.15,
            r"\textbf{(c)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # Export ---------------------------------------------------------------

    plt.tight_layout(h_pad=1.5)
    plt.savefig(
        "shell_disp.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
    )
    plt.show()
