# -*- coding: utf-8 -*-
"""Cover image for the repository"""

import matplotlib.pyplot as plt
import numpy as np
import charu
import utils

rc = {
    "charu.doc": "rspa",
    "figure.figsize": [200 * charu.pt, 200 * charu.pt / charu.golden],
    "charu.tex": True,
    "charu.tex.font": "fourier",
}

b = 0.1
eps = 0.01

with plt.rc_context(rc):
    fig, ax = plt.subplots()

    labelpos = (0.08, 0.83)
    w_min = 0.01
    w_max = 0.1 + 0.01
    w_guide = np.linspace(w_min, w_max, 10)
    ticks = [0.02, 0.06, 0.1]
    size = 3.7

    name = "tanh_bc_cc_b0.1_N_2048"
    pack = np.load("../data/{}.npz".format(name))
    x, evals, zz, uu = pack["x"], pack["evals"], pack["z"], pack["u"]

    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.25, 1.25)

    ii = [105, 139, 157, 170, 179, 187, 192, 197, 200, 202]
    ii = ii[:7]
    alpha = np.linspace(0, 1, len(ii))[::-1]
    for n, i in enumerate(ii):
        z, u = zz[i], uu[i]
        z, u = utils.normalize([z, u], eps * x)
        ax.plot(x * eps, u, "w-", linewidth=2.75, zorder=100 * (5 - n))
        ax.plot(x * eps, u, "C3-", linewidth=1, alpha=alpha[n], zorder=100 * (5 - n))

    plt.axis("off")
    plt.tight_layout(w_pad=0)
    plt.savefig(
        "repo.svg",
        transparent=True,
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
