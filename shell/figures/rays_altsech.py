# -*- coding: utf-8 -*-

"""Plot ray trajectories for curvature profile m3(x)."""

import numpy as np
import matplotlib.pyplot as plt
import charu
from matplotlib.colors import LinearSegmentedColormap, Normalize
from dispersion import omega
from scipy.linalg import null_space
from utils import *

# Physical parameters.
h = 0.3 # Poisson ratio
l = 0.1 # transverse wave number
b = 0.1 # max curvature
mfunc = lambda x: b*sech(x)

BlueRed_data = ["C0", "white", "C3"]
BlueRed = LinearSegmentedColormap.from_list("BlueRed", BlueRed_data)

rc = {
    "charu.doc": "rspa",
    "figure.figsize": [386 * charu.pt, 390 * charu.pt / charu.golden / 2.28],
    "charu.tex": True,
    "charu.tex.font": "fourier",
}

def D(k, l, m, w=0, h=0.3):
    """Dispersion matrix."""
    return np.array([[k**4 + 2 * k**2 * l**2 + l**4 + m**2 - w**2, -1j * k * m, -1j * l * h * m],
                     [1j * k * m, k**2 + 0.5 * l**2 * (1 - h) - w**2, 0.5 * k * l * (1 + h)],
                     [1j * l * h * m, 0.5 * k * l * (1 + h), l**2 + 0.5 * k**2 * (1 - h) - w**2]])

@np.vectorize
def ratio(x, k, l=0.1, h=0.3, mode=0, rcond=1e-10):
    m = mfunc(x)
    w = omega(k, l, m=m, h=h)[mode]
    d = D(k, l, m, w, h)

    z, u, v = np.abs(null_space(d, rcond=rcond).T[0])

    return z/(u + v + z)

with plt.rc_context(rc):
    x_max = 3
    k_max = 0.2

    N = 300
    x = np.linspace(-(x_max + 0.01), (x_max + 0.01), N)
    k = np.linspace(-(k_max + 0.01), (k_max + 0.01), N)
    X, K = np.meshgrid(x, k)

    fig, axes = plt.subplots(1, 3)
    labelpos = (0.05, 0.85)

    # Flexural -------------------------------------------------------------

    ax = axes[0]

    @np.vectorize
    def H(x, k):
        return omega(k, l, h, mfunc(x))[2]

    Z = H(X, K)

    # Eye.
    w = [0.037495493, 0.039611509, 0.042137019, 0.045070222, 0.048409679, 0.052103298, 0.055889582, 0.058781679]
    ax.contour(X, K, Z, colors="k", levels=w[::3])

    # Maximum omega for which we see a bound state, which is the double-well minimum.
    ax.contour(X, K, Z, levels=[0.03502325], linestyles="--", colors="k")

    # Left/right of eye.
    w = np.linspace(0.01, 0.033, 7)
    ax.contour(X, K, Z, colors="k", levels=w)

    r = ratio(X, K, mode=2)

    # Add two invisible points outside xlim, ylim where the ratio is
    # 0 and 1.  This is so that the colorbars represent the accurate
    # ratio, and not the rescaled one.
    r[0][0], r[1][0] = 0, 1

    pcm = ax.pcolormesh(X, K, r, cmap=BlueRed, shading="nearest", rasterized=True)

    ax.set_xlim(-x_max, x_max)
    ax.set_xticks([-3, 0, 3])
    ax.set_ylim(-k_max, k_max)
    ax.set_xlabel(r"$\epsilon x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center", labelpad=0)
    ax.text(*labelpos,
            r"\textbf{(a)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # Shear ----------------------------------------------------------------

    ax = axes[1]

    @np.vectorize
    def H(x, k):
        return omega(k, l, h, mfunc(x))[1]

    Z = H(X, K)

    # Saddle.
    w = [np.sqrt(0.5*(l**2 + l**4 + b**2 - np.sqrt((l**2 - l**4 - b**2)**2 + 4*h**2*l**2*b**2)))]
    ax.contour(X, K, Z, colors="k", levels=w, linestyles="--")

    # Others.
    ax.contour(X, K, Z, colors="k", levels=8)

    r = ratio(X, K, mode=1)

    # Add two invisible points outside xlim, ylim where the ratio is
    # 0 and 1.  This is so that the colorbars represent the accurate
    # ratio, and not the rescaled one.
    r[0][0], r[1][0] = 0, 1

    pcm = ax.pcolormesh(X, K, r, cmap=BlueRed, shading="nearest", rasterized=True)

    ax.set_xlim(-x_max, x_max)
    ax.set_xticks([-3, 0, 3])
    ax.set_ylim(-k_max, k_max)
    ax.set_xlabel(r"$\epsilon x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center", labelpad=0)
    ax.text(*labelpos,
            r"\textbf{(b)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # Extensional ----------------------------------------------------------

    ax = axes[2]

    @np.vectorize
    def H(x, k):
        return omega(k, l, h, mfunc(x))[0]

    Z = H(X, K)

    # Saddle.
    w = [np.sqrt(0.5*(l**2 + l**4 + b**2 + np.sqrt((l**2 - l**4 - b**2)**2 + 4*h**2*l**2*b**2)))]
    ax.contour(X, K, Z, colors="k", levels=w, linestyles="--")

    # Others.
    ax.contour(X, K, Z, colors="k", levels=8)

    # Left/right of saddle.
    w = np.linspace(0.1, w[0], 4)[:-1]
    ax.contour(X, K, Z, colors="k", levels=w)

    r = ratio(X, K, mode=0)

    # Add two invisible points outside xlim, ylim where the ratio is
    # 0 and 1.  This is so that the colorbars represent the accurate
    # ratio, and not the rescaled one.
    r[0][0], r[1][0] = 0, 1

    pcm = ax.pcolormesh(X, K, r, cmap=BlueRed, shading="nearest", rasterized=True)

    ax.set_xlim(-x_max, x_max)
    ax.set_xticks([-3, 0, 3])
    ax.set_ylim(-k_max, k_max)
    ax.set_xlabel(r"$\epsilon x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center", labelpad=0)
    ax.text(*labelpos,
            r"\textbf{(c)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # Export ---------------------------------------------------------------

    plt.tight_layout()
    plt.savefig(
        "shell_rays_altsech_inc.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
