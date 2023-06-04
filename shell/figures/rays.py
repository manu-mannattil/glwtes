import numpy as np
import matplotlib.pyplot as plt
import charu
from utils import *

rc = {
    "charu.doc": "rspa",
    "figure.figsize": [386 * charu.pt, 772 * charu.pt / charu.golden / 2.28],
    "charu.tex": True,
    "charu.tex.font": "fourier",
}

with plt.rc_context(rc):
    fig, axes = plt.subplots(2, 3)

    x_max = 4
    k_max = 0.2
    labelpos = (0.05, 0.85)

    # tanh (flexural) ------------------------------------------------------ 

    ax = axes[0][0]

    with np.load("../data/tanh_flex_l0.1_h0.3.npz", allow_pickle=True) as pack:
        r_eye = pack["r_eye"]
        r_eyelid = pack["r_eyelid"]
        r_eyebrows = pack["r_eyebrows"]
        r_sides = pack["r_sides"]

    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-k_max, k_max)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center", labelpad=0)

    for r in r_eye:
        ax.plot(r[0], r[1], "C3-")

    for r in r_eyebrows:
        ax.plot(r[0], r[1], "C0-")

    for r in r_eyelid:
        ax.plot(r[0], r[1], color="k", linestyle="--")

    for r in r_sides:
        ax.plot(r[0], r[1], "C0-")

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
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center", labelpad=0)

    for r in r_eye:
        ax.plot(r[0], r[1], "C3-")

    for r in r_eyebrows:
        ax.plot(r[0], r[1], "C0-")

    for r in r_eyelid:
        ax.plot(r[0], r[1], color="k", linestyle="--")

    for r in r_sides:
        ax.plot(r[0], r[1], "C0-")

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
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center", labelpad=0)

    for r in r_eye:
        ax.plot(r[0], r[1], "C3-")

    for r in r_eyebrows:
        ax.plot(r[0], r[1], "C0-")

    for r in r_eyelid:
        ax.plot(r[0], r[1], color="k", linestyle="--")

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
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center", labelpad=0)

    for r in r_eye:
        ax.plot(r[0], r[1], "C3-")

    for r in r_eyebrows:
        ax.plot(r[0], r[1], "C0-")

    for r in r_eyelid:
        ax.plot(r[0], r[1], color="k", linestyle="--")

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
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center", labelpad=0)

    for r in r_eye:
        ax.plot(r[0], r[1], "C3-")

    for r in r_eyebrows:
        ax.plot(r[0], r[1], "C0-")

    for r in r_eyelid:
        ax.plot(r[0], r[1], color="k", linestyle="--")

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
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center", labelpad=0)

    for r in r_eye:
        ax.plot(r[0], r[1], "C3-")

    for r in r_eyebrows:
        ax.plot(r[0], r[1], "C0-")

    for r in r_eyelid:
        ax.plot(r[0], r[1], color="k", linestyle="--")

    ax.text(*labelpos,
            r"\textbf{(f)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    plt.tight_layout()
    plt.savefig(
        "shell_rays.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
