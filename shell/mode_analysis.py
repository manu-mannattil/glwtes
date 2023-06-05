# -*- coding: utf-8 -*-
"""Automated mode analysis to check for quantization."""

from utils import *
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_quant(name):
    """Detect quantized modes in data/name.npz."""
    with np.load(f"data/{name}.npz") as dat:
        x, evals, z, u, v = dat["x"], dat["evals"], dat["z"], dat["u"], dat["v"]

    os.makedirs(f"data/{name}", exist_ok=True)

    quantized = []
    for i in range(0, 512):
        plt.plot(x, z[i].real, "C3-")
        plt.plot(x, u[i].real, "C0-")
        plt.plot(x, v[i].imag, "#888888", alpha=0.75)

        # plt.plot(x, u[i].imag, "C0")
        # plt.plot(x, z[i].imag, "C3")
        # plt.plot(x, v[i].real, "#888888", alpha=0.75)

        third_index, edge_index = quantindex(x, z[i], u[i], v[i])

        if third_index < 0.15 and edge_index < 0.02:
            quantized.append(i)
            s, c = "YES", "C3"
        else:
            s, c = "NO", "#999999"

        plt.title(
            f"mode $n$ = {i}; eval = {evals[i]:.6f}; quantized? {s} [{third_index:.3f}, {edge_index:.3f}]",
            color=c)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$\zeta, u,$ and $v$")
        plt.savefig(f"data/{name}/{i:04d}.png", dpi=100)
        plt.clf()
        print(f"mode {i} analyzed")

    np.savetxt(f"data/{name}/automated.txt", quantized, fmt="%d")

def manual_to_freq(name, index="manual", out="quantized"):
    with np.load(f"data/{name}.npz") as dat:
        x, evals, z, u = dat["x"], dat["evals"], dat["z"], dat["u"]

    quant, quant_range = [], []
    with open(f"data/{name}/{index}.txt") as f:
        for line in f.readlines():
            if line[0] == "#":
                continue

            ii = [int(x.strip()) for x in line.split(",")]
            ee = evals[ii]
            quant.append(np.mean(ee))
            quant_range.append(np.max(ee) - np.min(ee))

    pack = np.array([quant, quant_range]).T
    np.savetxt(f"data/{name}/{out}.txt", pack, fmt="%.9f")

N = 2**11
bc = "cc"
eps = 0.01
l = 0.1
b = 0.1
a = 0.01

# sech type ------------------------------------------------------------

form = "sech"
name = f"{form}_bc_{bc}_l_{l}_eps_{eps}_b{b}_a{a}_N_{N}"

# tanh type ------------------------------------------------------------

# form = "tanh"
# name = f"{form}_bc_{bc}_l_{l}_eps_{eps}_b{b}_N_{N}"

# detect_quant(name)

manual_to_freq(name, "manual1", "quantized1")
manual_to_freq(name, "manual2", "quantized2")
manual_to_freq(name, "manual3", "quantized3")
