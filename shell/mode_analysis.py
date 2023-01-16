# -*- coding: utf-8 -*-
"""Automated mode analysis to check for quantization."""

from utils import *
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_quant(name):
    """Detect quantized modes in data/name.npz."""
    with np.load("data/{}.npz".format(name)) as dat:
        x, evals, z, u, v = dat["x"], dat["evals"], dat["z"], dat["u"], dat["v"]

    os.makedirs("data/{}".format(name), exist_ok=True)

    quantized = []
    for i in range(800, 1001):
        plt.plot(x, u[i], "C0-")
        plt.plot(x, z[i], "C3-")
        plt.plot(x, v[i], "#888888", alpha=0.75)
        third_index, edge_index = quantindex(x, z[i], u[i], v[i])

        if (i < 800 and third_index < 0.15) or (i > 800 and edge_index < 0.02):
            quantized.append(i)
            s, c = "YES", "C3"
        else:
            s, c = "NO", "#999999"

        plt.title(r"mode $n$ = {}; eval = {:.6f}; quantized? {} [{:.3f}, {:.3f}]".format(
            i, evals[i], s, third_index, edge_index),
                  color=c)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$\zeta, u,$ and $v$")
        plt.savefig("data/{}/{:04d}.png".format(name, i), dpi=100)
        plt.clf()
        print("mode {} analyzed".format(i))

    np.savetxt("data/{}/automated.txt".format(name), quantized, fmt="%d")


def manual_to_freq(name):
    with np.load("data/{}.npz".format(name)) as dat:
        x, evals, z, u = dat["x"], dat["evals"], dat["z"], dat["u"]

    quant, quant_range = [], []
    with open("data/{}/manual.txt".format(name)) as f:
        for line in f.readlines():
            if line[0] == "#":
                continue

            ii = [int(x.strip()) for x in line.split(",")]
            ee = evals[ii]
            quant.append(np.mean(ee))
            quant_range.append(np.max(ee) - np.min(ee))

    pack = np.array([quant, quant_range]).T
    np.savetxt("data/{}/quantized.txt".format(name), pack, fmt="%.9f")

name = "cc_10_tanh_sorted_l1.2_N2048"
manual_to_freq(name)
detect_quant(name)
