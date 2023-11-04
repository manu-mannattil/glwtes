#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import charu

m = 0.05

def omega1(k):
    return 0.5 * ((1 + k**2) * (k**2 + m**2) +
                   np.sqrt((k**2 - m**2)**2 * (1 - k**2)**2 + 4 * m**2 * k**2 *
                                  (1 + k**2)**2))

def omega2(k):
    return 0.5 * ((1 + k**2) * (k**2 + m**2) -
                   np.sqrt((k**2 - m**2)**2 * (1 - k**2)**2 + 4 * m**2 * k**2 *
                                  (1 + k**2)**2))

rc = {
    "charu.doc": "aps",
    "figure.figsize": [525 * charu.pt, 525 / charu.golden / 3 * charu.pt],
    "charu.tex": True,
    "charu.tex.font": "fourier",
}

with plt.rc_context(rc):
    fig, axes = plt.subplots(1, 4)
    labelpos = (0.09, 0.18)

    # extensional (k >> m) -------------------------------------------------

    ax = axes[0]

    k = np.linspace(-0.125, 0.125, 1000)
    w = omega1(k)
    ax.plot(k, w)
    ax.set_ylim(0, w[-1])
    
    ax.plot(k, k**2 + m**2, "C3--")

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\omega^2$", rotation=None, va="center", labelpad=8)
    ax.ticklabel_format(axis="y", scilimits=(-2, 2))

    ax.text(0.5, 0.82, r"extensional $(k \gg m)$", ha="center", va="center", transform=ax.transAxes)

    ax.text(*labelpos,
            r"\textbf{(a)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # extensional (k << m) -------------------------------------------------

    ax = axes[1]

    k = np.linspace(-0.05, 0.05, 1000)
    w = omega1(k)
    ax.plot(k, w)

    ax.text(0.5, 0.82, r"extensional $(k \ll m)$", ha="center", va="center", transform=ax.transAxes)

    ax.plot(k, k**2 + m**2, "C3--")

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\omega^2$", rotation=None, va="center", labelpad=8)
    ax.ticklabel_format(axis="y", scilimits=(-2, 2))

    ax.text(*labelpos,
            r"\textbf{(b)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # Flexural (k >> m) ----------------------------------------------------

    ax = axes[2]

    k = np.linspace(-0.125, 0.125, 1000)
    w = omega2(k)
    ax.plot(k, w)
    ax.set_ylim(0, w[-1])

    ax.text(0.5, 0.82, r"flexural $(k \gg m)$", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\omega^2$", rotation=None, va="center", labelpad=8)
    ax.ticklabel_format(axis="y", scilimits=(-2, 2))

    ax.plot(k, k**4 - 3*k**2*m**2, "C3--")

    ax.text(*labelpos,
            r"\textbf{(c)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # Flexural (k << m) ----------------------------------------------------

    ax = axes[3]

    k = np.linspace(-0.063, 0.063, 1000)
    w = omega2(k)
    ax.plot(k, w)
    ax.set_ylim(0, w[-1])
    ax.set_xlim(-0.063, 0.063)
    
    ax.text(0.5, 0.82, r"flexural $(k \ll m)$", ha="center", va="center", transform=ax.transAxes)

    ax.plot(k, k**2*m**2 - 3*k**4, "C3--")

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\omega^2$", rotation=None, va="center", labelpad=8)
    ax.ticklabel_format(axis="y", scilimits=(-2, 2))

    ax.text(*labelpos,
            r"\textbf{(d)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=2))

    plt.tight_layout(w_pad=0.5)
    plt.savefig(
        "rod_dispersion.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        facecolor="none",
        pad_inches=0,
    )

    plt.show()
