# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import charu
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BlueRed_data = ["C0", "white", "C3"]
BlueRed = LinearSegmentedColormap.from_list("BlueRed", BlueRed_data)

rc = {
    "charu.doc": "rspa",
    "figure.figsize": [150*charu.pt, 150*charu.pt],
    "charu.tex": True,
    "charu.tex.font": "fourier",
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
}

with plt.rc_context(rc):
    fig, ax = plt.subplots()
    im = ax.imshow(np.linspace(0, 1, 100).reshape((10, 10)), cmap=BlueRed)
    fig.colorbar(im, ax=ax, location="bottom", ticks=[0, 0.25, 0.5, 0.75, 1.0])
    ax.remove()

    plt.tight_layout()
    plt.savefig(
        "colorbar.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )

    plt.show()
