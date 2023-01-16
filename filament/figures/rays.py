import numpy as np
import matplotlib.pyplot as plt
import charu
from utils import *

rc = {
    "charu.doc": "rspa",
    "figure.figsize": [360 * charu.pt, 380 / charu.golden * charu.pt],
    "charu.tex": True,
    "charu.tex.font": "fourier",
}

with np.load("../data/rays.npz", allow_pickle=True) as pack:
    tanh1_eye = pack["tanh1_eye"]
    tanh1_eyelid = pack["tanh1_eyelid"]
    tanh1_eyebrows = pack["tanh1_eyebrows"]
    sech1_eye = pack["sech1_eye"]
    sech1_eyelid = pack["sech1_eyelid"]
    sech1_eyebrows = pack["sech1_eyebrows"]
    tanh2_rays = pack["tanh2_rays"]
    sech2_rays = pack["sech2_rays"]

with plt.rc_context(rc):
    fig, axes = plt.subplots(2, 2)

    # tanh (localized) -----------------------------------------------------

    ax = axes[0, 0]

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center")

    for r in tanh1_eye[2:]:
        ax.plot(r[0], r[1], 'C3-')

    ax.text(0.1,
            0.8,
            r"\textbf{(a)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    for r in tanh1_eyebrows:
        ax.plot(r[0], r[1], "C0-")

    for r in tanh1_eyelid:
        ax.plot(r[0], r[1], color="k", linestyle="--", zorder=100)

    # tanh (delocalized) ---------------------------------------------------

    ax = axes[0, 1]

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center")

    for r in tanh2_rays:
        ax.plot(r[0], r[1], "C0-")

    ax.plot([-5, 5], [0, 0], "C0-")

    ax.text(0.1,
            0.8,
            r"\textbf{(b)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # sech (localized) -----------------------------------------------------

    ax = axes[1, 0]

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center")

    for r in sech1_eye:
        ax.plot(r[0], r[1], 'C3-')

    for r in sech1_eyelid:
        ax.plot(r[0], r[1], "k--")

    for r in sech1_eyebrows:
        ax.plot(r[0], r[1], "C0-")

    ax.text(0.1,
            0.8,
            r"\textbf{(c)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # sech (delocalized) ---------------------------------------------------

    ax = axes[1, 1]

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center")

    for r in sech2_rays:
        ax.plot(r[0], r[1], "C0-")

    ax.plot([-5, 5], [0, 0], "C0-")

    ax.text(0.1,
            0.8,
            r"\textbf{(d)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    plt.tight_layout(pad=2)
    plt.savefig(
        "filament_rays.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )
