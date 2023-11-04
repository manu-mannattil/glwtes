# -*- coding: utf-8 -*-

"""Plot rays for all three waves on a shell, with color-coded phase portraits."""

import numpy as np
import matplotlib.pyplot as plt
import charu
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.linalg import null_space
from dispersion import *
from utils import *

b = 0.1 # max curvature (tanh and sech)
a = 0.01 # min curvature (sech only)
eps = 0.01 # slowness parameter

BlueRed_data = ["C0", "white", "C3"]
BlueRed = LinearSegmentedColormap.from_list("BlueRed", BlueRed_data)

def m1(x):
    return b * np.tanh(x)

def m2(x):
    return b - (b - a) * sech(x)

def D(k, l, m, w=0, h=0.3):
    """Dispersion matrix."""
    return np.array([[k**4 + 2 * k**2 * l**2 + l**4 + m**2 - w**2, -1j * k * m, -1j * l * h * m],
                     [1j * k * m, k**2 + 0.5 * l**2 * (1 - h) - w**2, 0.5 * k * l * (1 + h)],
                     [1j * l * h * m, 0.5 * k * l * (1 + h), l**2 + 0.5 * k**2 * (1 - h) - w**2]])

@np.vectorize
def ratio(x, k, l=0.1, h=0.3, mfunc=m1, mode=0, rcond=1e-10):
    m = mfunc(x)
    w = omega(k, l, m=m, h=h)[mode]
    d = D(k, l, m, w, h)

    z, u, v = np.abs(null_space(d, rcond=rcond).T[0])

    return z/(u + v + z)

rc = {
    "charu.doc": "rspa",
    "figure.figsize": [420 * charu.pt, 772 * charu.pt / charu.golden / 2.28],
    "charu.tex": True,
    "charu.tex.font": "fourier",
}

with plt.rc_context(rc):
    fig, axes = plt.subplots(2, 3)

    x_max = 4
    k_max = 0.2
    labelpos = (0.05, 0.85)
    N = 300

    x = np.linspace(-(x_max + 0.1), (x_max + 0.1), N)
    k = np.linspace(-(k_max + 0.1), (k_max + 0.1), N)
    xx, kk = np.meshgrid(x, k)

    # tanh (flexural) ------------------------------------------------------ 

    ax = axes[0][0]

    with np.load("../data/tanh_flex_l0.1_h0.3.npz", allow_pickle=True) as pack:
        r_eye = pack["r_eye"]
        r_eyelid = pack["r_eyelid"]
        r_eyebrows = pack["r_eyebrows"]
        r_sides = pack["r_sides"]

    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-k_max, k_max)
    ax.set_xlabel(r"$\epsilon x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center", labelpad=0)

    for r in r_eye[::2]:
        ax.plot(r[0], r[1], "k-")

    for r in r_eyebrows:
        ax.plot(r[0], r[1], "k-")

    for r in r_eyelid:
        ax.plot(r[0], r[1], color="k", linestyle="--")

    for r in r_sides:
        ax.plot(r[0], r[1], "k-")

    r = ratio(xx, kk, mfunc=m1, mode=2)

    # Add two invisible points outside xlim, ylim where the ratio is
    # 0 and 1.  This is so that the colorbars represent the accurate
    # ratio, and not the rescaled one.
    r[0][0], r[1][0] = 0, 1

    pcm = ax.pcolormesh(xx, kk, r, cmap=BlueRed, shading="nearest", rasterized=True)

    ax.text(*labelpos,
            r"\textbf{(a)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # sech (flexural) ------------------------------------------------------ 

    ax = axes[1][0]

    with np.load("../data/sech_flex_l0.1_h0.3.npz", allow_pickle=True) as pack:
        r_eye = pack["r_eye"]
        r_eyelid = pack["r_eyelid"]
        r_eyebrows = pack["r_eyebrows"]
        r_sides = pack["r_sides"]

    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-k_max, k_max)
    ax.set_xlabel(r"$\epsilon x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center", labelpad=0)

    for r in r_eye[::2]:
        ax.plot(r[0], r[1], "k-")

    for r in r_eyebrows:
        ax.plot(r[0], r[1], "k-")

    for r in r_eyelid:
        ax.plot(r[0], r[1], color="k", linestyle="--")

    for r in r_sides:
        ax.plot(r[0], r[1], "k-")

    r = ratio(xx, kk, mfunc=m2, mode=2)

    # Add two invisible points outside xlim, ylim where the ratio is
    # 0 and 1.  This is so that the colorbars represent the accurate
    # ratio, and not the rescaled one.
    r[0][0], r[1][0] = 0, 1

    pcm = ax.pcolormesh(xx, kk, r, cmap=BlueRed, shading="nearest", rasterized=True)

    ax.text(*labelpos,
            r"\textbf{(d)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # tanh (shear) ------------------------------------------------------ 

    ax = axes[0][1]

    with np.load("../data/tanh_shear_l0.1_h0.3.npz", allow_pickle=True) as pack:
        r_eye = pack["r_eye"]
        r_eyelid = pack["r_eyelid"]
        r_eyebrows = pack["r_eyebrows"]

    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-k_max, k_max)
    ax.set_xlabel(r"$\epsilon x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center", labelpad=0)

    for r in r_eye[::2]:
        ax.plot(r[0], r[1], "k-")

    for r in r_eyebrows:
        ax.plot(r[0], r[1], "k-")

    for r in r_eyelid:
        ax.plot(r[0], r[1], color="k", linestyle="--")

    r = ratio(xx, kk, mfunc=m1, mode=1)

    # Add two invisible points outside xlim, ylim where the ratio is
    # 0 and 1.  This is so that the colorbars represent the accurate
    # ratio, and not the rescaled one.
    r[0][0], r[1][0] = 0, 1

    pcm = ax.pcolormesh(xx, kk, r, cmap=BlueRed, shading="nearest", rasterized=True)

    ax.text(*labelpos,
            r"\textbf{(b)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # sech (shear) ------------------------------------------------------ 

    ax = axes[1][1]

    with np.load("../data/sech_shear_l0.1_h0.3.npz", allow_pickle=True) as pack:
        r_eye = pack["r_eye"]
        r_eyelid = pack["r_eyelid"]
        r_eyebrows = pack["r_eyebrows"]

    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-k_max, k_max)
    ax.set_xlabel(r"$\epsilon x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center", labelpad=0)

    for r in r_eye[::2]:
        ax.plot(r[0], r[1], "k-")

    for r in r_eyebrows:
        ax.plot(r[0], r[1], "k-")

    for r in r_eyelid:
        ax.plot(r[0], r[1], color="k", linestyle="--")

    r = ratio(xx, kk, mfunc=m2, mode=1)

    # Add two invisible points outside xlim, ylim where the ratio is
    # 0 and 1.  This is so that the colorbars represent the accurate
    # ratio, and not the rescaled one.
    r[0][0], r[1][0] = 0, 1

    pcm = ax.pcolormesh(xx, kk, r, cmap=BlueRed, shading="nearest", rasterized=True)

    ax.text(*labelpos,
            r"\textbf{(e)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # tanh (ext) ------------------------------------------------------ 

    ax = axes[0][2]

    with np.load("../data/tanh_ext_l0.1_h0.3.npz", allow_pickle=True) as pack:
        r_eye = pack["r_eye"]
        r_eyelid = pack["r_eyelid"]
        r_eyebrows = pack["r_eyebrows"]

    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-k_max, k_max)
    ax.set_xlabel(r"$\epsilon x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center", labelpad=0)

    for r in r_eye[::2]:
        ax.plot(r[0], r[1], "k-")

    for r in r_eyebrows:
        ax.plot(r[0], r[1], "k-")

    for r in r_eyelid:
        ax.plot(r[0], r[1], color="k", linestyle="--")

    r = ratio(xx, kk, mfunc=m1, mode=0)

    # Add two invisible points outside xlim, ylim where the ratio is
    # 0 and 1.  This is so that the colorbars represent the accurate
    # ratio, and not the rescaled one.
    r[0][0], r[1][0] = 0, 1

    pcm = ax.pcolormesh(xx, kk, r, cmap=BlueRed, shading="nearest", rasterized=True)

    ax.text(*labelpos,
            r"\textbf{(c)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # sech (ext) ------------------------------------------------------ 

    ax = axes[1][2]

    with np.load("../data/sech_ext_l0.1_h0.3.npz", allow_pickle=True) as pack:
        r_eye = pack["r_eye"]
        r_eyelid = pack["r_eyelid"]
        r_eyebrows = pack["r_eyebrows"]

    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-k_max, k_max)
    ax.set_xlabel(r"$\epsilon x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center", labelpad=0)

    for r in r_eye[::2]:
        ax.plot(r[0], r[1], "k-")

    for r in r_eyebrows:
        ax.plot(r[0], r[1], "k-")

    for r in r_eyelid:
        ax.plot(r[0], r[1], color="k", linestyle="--")

    r = ratio(xx, kk, mfunc=m2, mode=0)

    # Add two invisible points outside xlim, ylim where the ratio is
    # 0 and 1.  This is so that the colorbars represent the accurate
    # ratio, and not the rescaled one.
    r[0][0], r[1][0] = 0, 1

    pcm = ax.pcolormesh(xx, kk, r, cmap=BlueRed, shading="nearest", rasterized=True)

    ax.text(*labelpos,
            r"\textbf{(f)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    plt.tight_layout(w_pad=3)
    plt.savefig(
        "shell_rays_inc.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
