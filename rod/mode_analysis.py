# -*- coding: utf-8 -*-
"""Automated mode analysis to check for quantization."""

from utils import *
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_quant(name):
    """Detect quantized modes in data/name.npz."""
    with np.load(f"data/{name}.npz") as dat:
        x, evals, z, u = dat["x"], dat["evals"], dat["z"], dat["u"]

    os.makedirs(f"data/{name}", exist_ok=True)

    quantized = []
    for i in range(500 + 1):
        plt.plot(x, u[i], "C0-")
        plt.plot(x, z[i], "C3-")
        third_index, edge_index = quantindex(x, z[i], u[i])
        zp = parity(z[i])[1]
        up = parity(u[i])[1]

        if third_index < 0.25:
            quantized.append(i)
            s, c = "YES", "C3"
        else:
            s, c = "NO", "#999999"

        plt.title(
            f"mode $n$ = {i}; eval = {evals[i]:.6f}\n"
            f"quantized? {s} [{third_index:.3f}, {edge_index:.3f}]; parity: z = {zp}, u = {up}",
            color=c)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$\zeta$ and $u$")
        plt.savefig(f"data/{name}/{i:03d}.png", dpi=100)
        plt.clf()
        print(f"mode {i} analyzed")

    np.savetxt(f"data/{name}/automated.txt", quantized, fmt="%d")

def manual_to_freq(name):
    with np.load(f"data/{name}.npz") as dat:
        x, evals, z, u = dat["x"], dat["evals"], dat["z"], dat["u"]

    quant, quant_range = [], []
    with open(f"data/{name}/manual.txt") as f:
        for line in f.readlines():
            if line[0] == "#":
                continue

            ii = [int(x.strip()) for x in line.split(",")]
            ee = evals[ii]
            quant.append(np.mean(ee))
            quant_range.append(np.max(ee) - np.min(ee))

    pack = np.array([quant, quant_range]).T
    np.savetxt(f"data/{name}/quantized.txt", pack, fmt="%.9f")

N = 2**11
bc = "cc"
b = 0.1
a = 0.01

# tanh type ------------------------------------------------------------

form = "tanh"
name = f"{form}_bc_{bc}_b{b}_N_{N}"

# sech type ----------------------------------------------------------

# form = "sech"
# name = f"{form}_bc_{bc}_b{b}_a{a}_N_{N}"

detect_quant(name)
manual_to_freq(name)
