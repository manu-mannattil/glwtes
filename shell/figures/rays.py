import numpy as np
import matplotlib.pyplot as plt
import charu
from utils import *

rc = {
    "charu.doc": "rspa",
    "figure.figsize": [386 * charu.pt, 386 * charu.pt / charu.golden / 2.28],
    "charu.tex": True,
    "charu.tex.font": "fourier",
}

with plt.rc_context(rc):
    fig, axes = plt.subplots(1, 3)

    # sech (flexural) ------------------------------------------------------ 

    with np.load("../data/sech_flex_l1.2_h0.3.npz", allow_pickle=True) as pack:
        r_eye = pack["r_eye"]
        r_eyelid = pack["r_eyelid"]
        r_eyebrows = pack["r_eyebrows"]

    ax = axes[0]

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center", labelpad=0)

    for r in r_eye:
        ax.plot(r[0], r[1], 'C3-')

    for r in r_eyebrows:
        ax.plot(r[0], r[1], "C0-")

    for r in r_eyelid:
        ax.plot(r[0], r[1], color="k", linestyle="--", zorder=100)

    ax.text(0.05,
            0.85,
            r"\textbf{(a)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # sech (membrane) ------------------------------------------------------

    ax = axes[1]

    with np.load("../data/sech_membrane_l1.2_h0.3.npz", allow_pickle=True) as pack:
        r_above   = pack["r_above"]  
        r_saddle  = pack["r_saddle"] 
        r_side    = pack["r_side"]   

    for r in r_above:
        ax.plot(r[0], r[1], 'C0-')

    for r in r_saddle:
        ax.plot(r[0], r[1], 'C0-')

    for r in r_side:
        ax.plot(r[0], r[1], 'C0-')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center", labelpad=0)

    ax.text(0.05,
            0.85,
            r"\textbf{(b)}",
            transform=ax.transAxes,
            backgroundcolor="w",
            bbox=dict(facecolor="w", edgecolor="w", pad=1))

    # sech (shear) ---------------------------------------------------------

    ax = axes[2]

    with np.load("../data/sech_shear_l1.2_h0.3.npz", allow_pickle=True) as pack:
        r_propagating = pack["r_propagating"]
        r_reflected = pack["r_reflected"]
        r_saddle = pack["r_saddle"]

    for r in r_propagating:
        ax.plot(r[0], r[1], 'C0-')

    for r in r_reflected:
        ax.plot(r[0], r[1], 'C0-')

    for r in r_saddle:
        ax.plot(r[0], r[1], 'C0-')
    
    # Entire x axis is also a ray.
    ax.plot([-5, 5], [0, 0], 'C0-')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$k$", rotation=0, va="center", labelpad=0)

    ax.text(0.05,
            0.85,
            r"\textbf{(c)}",
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
