# -*- coding: utf-8 -*-

"""Trace rays for flexural waves on a curved shell."""

import numpy as np
import matplotlib.pyplot as plt
import math
from utils import *

def dH(t, y, w):
    """RHS of Hamilton"s equations for a curved shell."""
    x, k = y

    w2 = w**2
    q2 = k**2 + l**2
    X = (w2 - q2**2)
    Y = (w2 - q2)
    Z = (w2 - 0.5 * (1 - h) * q2)

    a = k * (2 * Z * (2 * q2 * Y + X) + (1 - h) * (X * Y - m(x)**2 * w2))
    b = -(2 * m(x) * dm(x) * ((w2 - 0.5 * (1 - h) * l**2) * (w2 - (1 - h**2) * l**2) - 0.5 *
                              (1 - h) * k**2 * w2))

    # Normalize the RHS.  This doesn"t change the rays, but
    # produces smoother trajectories.
    n = np.sqrt(a**2 + b**2)
    return [a / n, b / n]

# Physical parameters.
h = 0.3 # Poisson ratio
l = 1.2 # transverse wave number
m0 = 10 # max curvature
form = "tanh" # curvature form

if form == "tanh":
    m = lambda x: m0 * math.tanh(x)
    dm = lambda x: m0 * (1 - math.tanh(x)**2)
else:
    m = lambda x: m0 * (1 - 1 / math.cosh(x))
    dm = lambda x: m0 / math.cosh(x) * math.tanh(x)

# Dispersion of polarization I on the k axis.
omega = np.vectorize(lambda k: k**2 + l**2 if k**2 + l**2 > 1 else np.sqrt(k**2 + l**2))

# What"s the largest k for which we see localization?  To find that
# first compute the largest frequency for which there"s localization.
w_max = np.sqrt(
    0.5 * (l**4 + l**2 + m0**2 + np.sqrt((l**4 - l**2 + m0**2)**2 + 4 * h**2 * m0**2 * l**2)))

# Now, find the largest k (i.e., on the k axis) corresponding to the
# above omega.
if w_max < 1:
    # w^2 = k^2 + l^2
    k_max = np.sqrt(w_max**2 - l**2)
else:
    # w^2 = (k^2 + l^2)^2
    k_max = np.sqrt(w_max - l**2)

rt = RayTracer(dH, xlim=(-5, 5), klim=(-6, 6))

# Find the rays that are localized (i.e., inside the "eye").
N = 7
x = np.zeros(N)
k = np.linspace(0.02, k_max, N + 1)[:-1]
w = omega(k).reshape((N, 1))
r_eye = rt.trace(x, k, w, symmetry="xk", max_step=1e-3, orbit_check=True)
for r in r_eye:
    plt.plot(r[0], r[1], "C3-")

# Now, find the rays that form the "eyelid".
r_eyelid = rt.trace([0], [k_max + 0.001], [[omega(k_max + 0.001)]],
                 symmetry="xk",
                 max_step=1e-3,
                 orbit_check=True)
for r in r_eyelid:
    plt.plot(r[0], r[1], "k--")

# Find the rays that are delocalized (i.e., above and below the eye).
N = 6
x = np.zeros(N)
k = np.linspace(k_max, 5, N + 1)[1:]
w = omega(k).reshape((N, 1))
r_eyebrows = rt.trace(x, k, w, symmetry="xk", max_step=1e-3, orbit_check=True)
for r in r_eyebrows:
    plt.plot(r[0], r[1], "C0-")

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel(r"position $x$")
plt.ylabel(r"wave number $k$")
plt.title(r"flexural waves")

name = f"{form}_flex_l{l}_h{h}"
np.savez(f"data/{name}.npz", r_eye=r_eye, r_eyelid=r_eyelid, r_eyebrows=r_eyebrows)
plt.savefig(f"data/{name}.png")

plt.show()
