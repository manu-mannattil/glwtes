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

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr

session = WolframLanguageSession("/usr/local/bin/WolframKernel")

def dH1(t, y):
    """RHS of Hamilton's equations for a curved filament (wave type 1)."""
    x, k = y

    X = np.sqrt((k**4 - k**2 + m(x)**2)**2 + 4 * k**2 * m(x)**2)
    Y = k**4 + k**2 + m(x)**2

    a = k * (2 * k**2 + 1 + ((2 * k**2 + 1) * Y - 6 * k**4) / X)
    b = -m(x) * dm(x) * (1 + Y / X)

    n = np.sqrt(a**2 + b**2)
    return [a/n, b/n]

def dH2(t, y):
    """RHS of Hamilton's equations for a curved filament (wave type 2)."""
    x, k = y

    X = np.sqrt((k**4 - k**2 + m(x)**2)**2 + 4 * k**2 * m(x)**2)
    Y = k**4 + k**2 + m(x)**2

    a = k * (2 * k**2 + 1 - ((2 * k**2 + 1) * Y - 6 * k**4) / X)
    b = -m(x) * dm(x) * (1 - Y / X)

    n = np.sqrt(a**2 + b**2)
    return [a/n, b/n]

def m(x):
    if form == "tanh":
        return m0 * math.tanh(x)
    else:
        return m0 / 1.25 * (1.25 - 1 / math.cosh(x))

def dm(x):
    if form == "tanh":
        return m0 / math.cosh(x)**2
    else:
        return m0 / 1.25 / math.cosh(x) * math.tanh(x)

max_step = 1e-3
m0 = 10 # max curvature

session.start()

# tanh (localized) -----------------------------------------------------

form = "tanh" # curvature form

rt = RayTracer(dH1, xlim=(-5, 5), klim=(-6, 6))

# Find the rays that are localized (i.e., inside the "eye").
N = 10
x = np.zeros(N)
k_max = np.sqrt(10)
k = np.linspace(0.8, k_max - 0.1, N)
tanh1_eye = rt.trace(x, k, symmetry="xk", max_step=max_step, orbit_check=True)

# Find the eyelid.
k = [k_max]
x = [0]
tanh1_eyelid = rt.trace(x, k, symmetry="xk", max_step=max_step, orbit_check=False)

# Find the eyebrow.
omega = lambda k: np.sqrt(0.5 * (k**4 + k**2 + m0**2 + np.sqrt(
    (k**4 - k**2 + m0**2)**2 + 4 * k**2 * m0**2)))
omega = np.vectorize(omega)
N = 16
k = np.linspace(0, 6, N + 1)[1:]
w = omega(k)
k = np.sqrt(w)
x = np.zeros(N)
tanh1_eyebrows = rt.trace(x, k, symmetry="xk", max_step=max_step, orbit_check=True)

# tanh (delocalized) ---------------------------------------------------

form = "tanh" # curvature form

rt = RayTracer(dH2, xlim=(-5, 5), klim=(-6, 6))

omega = lambda k: np.sqrt(0.5 * (k**4 + k**2 + m0**2 - np.sqrt(
    (k**4 - k**2 + m0**2)**2 + 4 * k**2 * m0**2)))
omega = np.vectorize(omega)
N = 13
k = np.linspace(0.5, 5, N)
omega2k = np.vectorize(lambda w: np.sqrt(w) if w < 1 else w)
w = omega(k)
k = omega2k(w)
x = np.zeros(N)
tanh2_rays = rt.trace(x, k, symmetry="xk", max_step=max_step, orbit_check=False)

# sech (localized) -----------------------------------------------------

form = "sech" # curvature form

rt = RayTracer(dH1, xlim=(-5, 5), klim=(-6, 6))

def omega2k(w, m):
    """Find k given values of frequency (w) and curvature (m)."""
    code = """w = {}; m = {};
    Select[k /.
        Solve[(k^4 + k^2 + m^2 + Sqrt[(k^4 - k^2 + m^2)^2 + 4*k^2*m^2])/2 -
               w^2 == 0, k] // N // Chop, (Element[#, Reals] && # > 0) &]
    """.format(w, m)

    return session.evaluate(wlexpr(code))[0]

# Find the rays that are localized (i.e., inside the "eye").
N = 10
x = np.zeros(N)
k_max = omega2k(m0, m(0))
k = np.linspace(0.8, k_max - 0.1, N)
sech1_eye = rt.trace(x, k, symmetry="xk", max_step=max_step, orbit_check=True)

# Find the eyelid.
x = [0]
k = [k_max]
sech1_eyelid = rt.trace(x, k, symmetry="xk", max_step=max_step, orbit_check=False)

# Find the eyebrows.
N = 11
k_min = sech1_eyelid[0][1][-1]
k = np.linspace(k_min, 5, N + 1)[1:]
omega = lambda k: np.sqrt(0.5 * (k**4 + k**2 + m0**2 + np.sqrt(
    (k**4 - k**2 + m0**2)**2 + 4 * k**2 * m0**2)))
omega = np.vectorize(omega)
w = omega(k)
k = [omega2k(_w, m(0)) for _w in w]
x = np.zeros(N)
sech1_eyebrows = rt.trace(x, k, symmetry="xk", max_step=max_step, orbit_check=False)

# sech (delocalized) ---------------------------------------------------

form = "sech" # curvature form

rt = RayTracer(dH2, xlim=(-5, 5), klim=(-6, 6))

omega = lambda k: np.sqrt(0.5 * (k**4 + k**2 + m0**2 - np.sqrt(
    (k**4 - k**2 + m0**2)**2 + 4 * k**2 * m0**2)))
omega = np.vectorize(omega)
N = 13
k = np.linspace(0.5, 5, N)
w = omega(k)

def omega2k(w, m):
    """Find k given values of frequency (w) and curvature (m)."""
    code = """w = {}; m = {};
    Select[k /.
        Solve[(k^4 + k^2 + m^2 - Sqrt[(k^4 - k^2 + m^2)^2 + 4*k^2*m^2])/2 -
               w^2 == 0, k] // N // Chop, (Element[#, Reals] && # > 0) &]
    """.format(w, m)

    return session.evaluate(wlexpr(code))[0]

k = [omega2k(_w, m(0)) for _w in w]
x = np.zeros(N)
sech2_rays = rt.trace(x, k, symmetry="xk", max_step=max_step, orbit_check=False)

session.terminate()

np.savez("data/rays.npz",
         tanh1_eye=tanh1_eye,
         tanh1_eyelid=tanh1_eyelid,
         tanh1_eyebrows=tanh1_eyebrows,
         tanh2_rays=tanh2_rays,
         sech1_eye=sech1_eye,
         sech1_eyelid=sech1_eyelid,
         sech1_eyebrows=sech1_eyebrows,
         sech2_rays=sech2_rays)

# Test plotting --------------------------------------------------------

with np.load("data/rays.npz", allow_pickle=True) as pack:
    tanh1_eye = pack["tanh1_eye"]
    tanh1_eyelid = pack["tanh1_eyelid"]
    tanh1_eyebrows = pack["tanh1_eyebrows"]
    tanh2_rays = pack["tanh2_rays"]
    sech1_eye = pack["sech1_eye"]
    sech1_eyelid = pack["sech1_eyelid"]
    sech1_eyebrows = pack["sech1_eyebrows"]
    sech2_rays = pack["sech2_rays"]

fig, axes = plt.subplots(2, 2)
ax = axes[0, 0]
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
for r in tanh1_eye:
    ax.plot(r[0], r[1], 'C3-')

for r in tanh1_eyebrows:
    ax.plot(r[0], r[1], "C0-")

for r in tanh1_eyelid:
    ax.plot(r[0], r[1], color="k", linestyle="--", zorder=100)

ax = axes[0, 1]
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
for r in tanh2_rays:
    ax.plot(r[0], r[1], "C0-")

ax = axes[1, 0]
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
for r in sech1_eye:
    ax.plot(r[0], r[1], 'C3-')

for r in sech1_eyelid:
    ax.plot(r[0], r[1], "k--")

for r in sech1_eyebrows:
    ax.plot(r[0], r[1], "C0-")

ax = axes[1, 1]
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
for r in sech2_rays:
    ax.plot(r[0], r[1], "C0-")

plt.show()
