# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import charu

rc = {
    "charu.doc": "rspa",
    "figure.figsize": [400 * charu.pt, 400 * charu.pt / charu.golden / 2.5],
    "charu.tex": True,
    "charu.tex.font": "fourier",
    "axes.axisbelow": False,
}

sqrt = lambda x: pow(x, 1 / 2)
cbrt = lambda x: pow(x, 1 / 3)

def omega(k, l=0.1, h=0.3, m=0.01):
    q = np.sqrt(k**2 + l**2)

    # Coefficients of the cubic.  We need to add 0j so that Python
    # treats these variables as complex.
    a = 0.5 * q**2 * ((3 - h) + 2 * q**2) + m**2 + 0j
    b = -0.5 * q**4 * ((3 - h) * q**2 + (1 - h)) - \
            0.5 * m**2 * (1 - h) * (q**2 + 2 * (1 + h) * l**2) + 0j
    c = 0.5 * (1 - h) * (q**8 + m**2 * l**4 * (1 - h**2)) + 0j

    # Common term
    d = cbrt(2 * a**3 + 9 * a * b + 27 * c +
             3 * sqrt(3) * sqrt(4 * a**3 * c - a**2 * b**2 + 18 * a * b * c - 4 * b**3 + 27 * c**2))

    # Quasi extensional
    w1 = a / 3 + cbrt(2) * (a**2 + 3 * b) / (3 * d) + d / (3 * cbrt(2))

    # Quasi shear
    w2 = a / 3 - (1 - 1j * sqrt(3)) * (a**2 + 3 * b) / (3 * cbrt(4) * d) \
            - (1 + 1j * sqrt(3)) * d / (6 * cbrt(2))

    # Quasi flexural
    w3 = a / 3 - (1 + 1j * sqrt(3)) * (a**2 + 3 * b) / (3 * cbrt(4) * d) \
            - (1 - 1j * sqrt(3)) * d / (6 * cbrt(2))

    w1, w2, w3 = sqrt(w1.real), sqrt(w2.real), sqrt(w3.real)

    return [w1, w2, w3]

l = 0.1
h = 0.3

N = 500
mm = np.linspace(0, 0.1, N)

# This is the value of curvature when shear and flexural waves have
# a mode-conversion point at k = 0.
m_critical = sqrt((1 + h)*l**2*(0.5*(1 - h) - l**2)/(2*h + 1)/(1 - h))

ww = np.array([omega(1e-8, m=m, l=l, h=h) for m in mm])

with plt.rc_context(rc):
    fig, axes = plt.subplots(1, 3)
    labelpos = (0.05, 0.8)

    # Quasi flexural -------------------------------------------------------

    ax = axes[0]

    ax.plot(mm, np.sqrt(l**4 + (1-h**2)*mm**2), "C0--")
    ax.plot(mm, ww[:, 2], "C0")
    ax.set_xlim(0, 0.1)
    ax.set_ylim(0, 0.1)
    ax.set_yticks([0, 0.05, 0.1])
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"$\omega_\mathsf{cut-on}$")
    ax.plot([m_critical, m_critical], [-1, 1], "#999999", linestyle="--", zorder=-10)

    ax.text(*labelpos,
            r"\textbf{(a)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # Quasi shear ----------------------------------------------------------

    ax = axes[1]

    ax.plot(mm, np.sqrt(0.5*(1 - h)*l**2)*np.ones(N), "C0", linestyle="--")
    ax.plot(mm, ww[:, 1], "C0")
    ax.set_xlim(0, 0.1)
    ax.set_ylim(0.05, 0.1)
    ax.set_yticks([0.05, 0.075, 0.1])
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"$\omega_\mathsf{cut-on}$")
    ax.plot([m_critical, m_critical], [-1, 1], "#999999", linestyle="--", zorder=-10)

    # The shear-wave gap asymptotes to a constant value as m -> infty.
    # ax.plot(mm, np.sqrt((1 - h**2)*l**2)*np.ones(N), "#999999", linestyle="--", zorder=-10)

    ax.text(*labelpos,
            r"\textbf{(b)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # Quasi extensional ----------------------------------------------------------

    ax = axes[2]

    ax.plot(mm, np.sqrt(l**2 + mm**2*h**2), "C0", linestyle="--")
    ax.plot(mm, ww[:, 0], "C0")
    ax.set_xlim(0, 0.1)
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"$\omega_\mathsf{cut-on}$")

    ax.text(*labelpos,
            r"\textbf{(c)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    plt.tight_layout(w_pad=-0.1)
    plt.savefig(
        "shell_gap_unannot.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
    )
    plt.show()
