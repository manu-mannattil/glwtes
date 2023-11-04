# -*- coding: utf-8 -*-
"""
Why is the code so complicated and ugly, when I could just make contour
plots?  The reason is that, although contour plots are sufficient for
tracing the rays in this particular example, contours made by Matplotlib
and other plotting programs aren't technically smooth/continuous curves,
and I like smooth curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from utils import *

def dH1(t, y):
    """RHS of Hamilton's equations for a curved rod (wave type 1)."""
    x, k = y

    X = np.sqrt((k**4 - k**2 + m(x)**2)**2 + 4 * k**2 * m(x)**2)
    Y = k**4 + k**2 + m(x)**2

    a = k * (2 * k**2 + 1 + ((2 * k**2 + 1) * Y - 6 * k**4) / X)
    b = -m(x) * dm(x) * (1 + Y / X)

    n = np.sqrt(a**2 + b**2)
    return [a/n, b/n]

def dH2(t, y):
    """RHS of Hamilton's equations for a curved rod (wave type 2)."""
    x, k = y

    X = np.sqrt((k**4 - k**2 + m(x)**2)**2 + 4 * k**2 * m(x)**2)
    Y = k**4 + k**2 + m(x)**2

    a = k * (2 * k**2 + 1 - ((2 * k**2 + 1) * Y - 6 * k**4) / X)
    b = -m(x) * dm(x) * (1 - Y / X)

    n = np.sqrt(a**2 + b**2)
    return [a/n, b/n]

def m(x):
    if form == "tanh":
        return b * math.tanh(x)
    else:
        return b - (b - a) / np.cosh(x)

def dm(x):
    if form == "tanh":
        return b / math.cosh(x)**2
    else:
        return (b - a) / math.cosh(x) * math.tanh(x)

def remove(x, a):
    """Remove first occurance of `a` from an array `x`."""
    i = np.where(x == a)[0][0]
    return np.concatenate((x[:i], x[i + 1:]))

max_step = 1e-3
b = 0.1 # max curvature
a = 0.01 # min curvature (sech-type)

# tanh (localized) -----------------------------------------------------

form = "tanh" # curvature form

rt = RayTracer(dH1, xlim=(-5, 5), klim=(-0.25, 0.25))

# Mark a ray for highlighting.
k_mark = 0.0892569
tanh1_mark = rt.trace([0], [k_mark], symmetry="xk", max_step=max_step, orbit_check=True)

# Find the rays that are localized (i.e., inside the "eye").
k = np.array([0.0308586, 0.052487, 0.0660218, 0.0759104, 0.083455, 0.0892569, 0.0936456, 0.0968088, 0.0988592, 0.0998787])
k = k[1:-1:2]
k = remove(k, k_mark)
x = np.zeros(len(k))
tanh1_eye = rt.trace(x, k, symmetry="xk", max_step=max_step, orbit_check=True)

# Find the eyelid.
k = [b]
x = [0]
tanh1_eyelid = rt.trace(x, k, symmetry="xk", max_step=max_step, orbit_check=False)

# Find the eyebrow.
omega = lambda k: np.sqrt(0.5 * (k**4 + k**2 + b**2 + np.sqrt(
    (k**4 - k**2 + b**2)**2 + 4 * k**2 * b**2)))
omega = np.vectorize(omega)
N = 7
k = np.linspace(0.03, 0.25, N)
w = omega(k)
k = w
x = np.zeros(N)
tanh1_eyebrows = rt.trace(x, k, symmetry="xk", max_step=max_step, orbit_check=False)

# tanh (delocalized) ---------------------------------------------------

form = "tanh" # curvature form

rt = RayTracer(dH2, xlim=(-5, 5), klim=(-0.5, 0.5))

# Mark a ray for highlighting.
w_mark = 0.09
k_mark = np.sqrt(w_mark)
tanh2_mark = rt.trace([0], [k_mark], symmetry="xk", max_step=max_step, orbit_check=False)

# Trace remaining rays.
N = 6
k1 = np.linspace(0, k_mark, N)[1:-1]
dk = k1[1] - k1[0]
k2 = np.arange(k_mark + dk, 0.5, dk)
k = np.concatenate((k1, k2))
x = np.zeros(len(k))

tanh2_rays = rt.trace(x, k, symmetry="xk", max_step=max_step, orbit_check=False)

# sech (localized) -----------------------------------------------------

form = "sech" # curvature form

rt = RayTracer(dH1, xlim=(-5, 5), klim=(-0.25, 0.25))

# Mark a ray for highlighting.
k_mark = 0.0664891
sech1_mark = rt.trace([0], [k_mark], symmetry="xk", max_step=max_step, orbit_check=True)

# Find the rays that are localized (i.e., inside the "eye").
dk = 0.5*(b - k_mark)
k1 = np.arange(0.01, k_mark, dk)[:-1]
k2 = [k_mark + dk]
k = np.concatenate((k1, k2))

k = np.array([0.0189872, 0.0337405, 0.044509, 0.0531225, 0.0603289, 0.0664891, 0.0718212, 0.0764696, 0.0805337, 0.0840838, 0.0871732, 0.0898469, 0.0921436, 0.094092, 0.0957118])
k = k[1::2]
k = remove(k, k_mark)
x = np.zeros(len(k))
sech1_eye = rt.trace(x, k, symmetry="xk", max_step=max_step, orbit_check=True)

# Find the eyelid.
k = [b]
x = [0]
sech1_eyelid = rt.trace(x, k, symmetry="xk", max_step=max_step, orbit_check=False)

# Find the eyebrow.
omega = lambda k: np.sqrt(0.5 * (k**4 + k**2 + b**2 + np.sqrt(
    (k**4 - k**2 + b**2)**2 + 4 * k**2 * b**2)))
omega = np.vectorize(omega)
N = 7
k = np.linspace(0.03, 0.25, N)
w = omega(k)
k = w
x = np.zeros(N)
sech1_eyebrows = rt.trace(x, k, symmetry="xk", max_step=max_step, orbit_check=False)

# sech (delocalized) ---------------------------------------------------

form = "sech" # curvature form

rt = RayTracer(dH2, xlim=(-5, 5), klim=(-0.5, 0.5))

# Mark a ray for highlighting.
w_mark = 0.066
k_mark = np.sqrt(w_mark)
sech2_mark = rt.trace([0], [k_mark], symmetry="xk", max_step=max_step, orbit_check=False)

# Trace remaining rays.
N = 5
k1 = np.linspace(0, k_mark, N)[1:-1]
dk = k1[1] - k1[0]
k2 = np.arange(k_mark + dk, 0.5, dk)
k = np.concatenate((k1, k2))
x = np.zeros(len(k))

sech2_rays = rt.trace(x, k, symmetry="xk", max_step=max_step, orbit_check=False)

np.savez("data/rays.npz",
         tanh1_eye=tanh1_eye,
         tanh1_eyelid=tanh1_eyelid,
         tanh1_eyebrows=tanh1_eyebrows,
         tanh1_mark=tanh1_mark,
         tanh2_rays=tanh2_rays,
         tanh2_mark=tanh2_mark,
         sech1_eye=sech1_eye,
         sech1_eyelid=sech1_eyelid,
         sech1_mark=sech1_mark,
         sech1_eyebrows=sech1_eyebrows,
         sech2_rays=sech2_rays,
         sech2_mark=sech2_mark)

# Test plotting --------------------------------------------------------

with np.load("data/rays.npz", allow_pickle=True) as pack:
    tanh1_eye = pack["tanh1_eye"]
    tanh1_eyebrows = pack["tanh1_eyebrows"]
    tanh1_eyelid = pack["tanh1_eyelid"]
    tanh1_mark = pack["tanh1_mark"]

    tanh2_mark = pack["tanh2_mark"]
    tanh2_rays = pack["tanh2_rays"]

    sech1_eye = pack["sech1_eye"]
    sech1_eyebrows = pack["sech1_eyebrows"]
    sech1_eyelid = pack["sech1_eyelid"]
    sech1_mark = pack["sech1_mark"]

    sech2_mark = pack["sech2_mark"]
    sech2_rays = pack["sech2_rays"]

fig, axes = plt.subplots(2, 2)

# ------------- tanh (localized) ------------- #

ax = axes[0, 0]
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-0.25, 0.25)
for r in tanh1_eye:
    ax.plot(r[0], r[1], 'C3-')

for r in tanh1_eyebrows:
    ax.plot(r[0], r[1], "C0-")

for r in tanh1_eyelid:
    ax.plot(r[0], r[1], color="k", linestyle="--", zorder=100)

for r in tanh1_mark:
    ax.plot(r[0], r[1], color="C1", linestyle="--", zorder=100)

# ------------- tanh (delocalized) ------------- #

ax = axes[0, 1]
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-0.4, 0.4)
for r in tanh2_rays:
    ax.plot(r[0], r[1], "C0-")

for r in tanh2_mark:
    ax.plot(r[0], r[1], "C1-")

ax.plot([-3.5, 3.5], [0, 0], "C0-")

# ------------- sech (localized) ------------- #

ax = axes[1, 0]
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-0.25, 0.25)
for r in sech1_eye:
    ax.plot(r[0], r[1], 'C3-')

for r in sech1_eyelid:
    ax.plot(r[0], r[1], "k--")

for r in sech1_eyebrows:
    ax.plot(r[0], r[1], "C0-")

for r in sech1_mark:
    ax.plot(r[0], r[1], color="C1", linestyle="--", zorder=100)

# ------------- sech (delocalized) ------------- #

ax = axes[1, 1]
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-0.4, 0.4)
for r in sech2_rays:
    ax.plot(r[0], r[1], "C0-")

for r in sech2_mark:
    ax.plot(r[0], r[1], "C1-")

ax.plot([-3.5, 3.5], [0, 0], "C0-")

plt.savefig("data/rays.png")
plt.show()
