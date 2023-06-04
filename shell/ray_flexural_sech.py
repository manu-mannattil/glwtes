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
l = 0.1 # transverse wave number
b = 0.1 # max curvature
a = 0.01 # min curvature

m = lambda x: b - (b - a) * sech(x)
dm = lambda x: (b - a) * np.tanh(x) * sech(x)

max_step = 1e-3

# Dispersion of polarization I on the k axis.
omega = np.vectorize(lambda k: k**2 + l**2)

# What's the largest k for which we see localization?  To find that
# look at the frequency for which we have a double-minima in the dispersion.
w_max = 0.035

# Now, find the largest k (i.e., on the k axis) corresponding to the
# above omega.
k_max = 0.158026

rt = RayTracer(dH, xlim=(-7, 7), klim=(-1, 1))

# Find the rays that are localized (i.e., inside the "eye").
w = np.array([
    0.010939861,
    0.013235236,
    0.015782963,
    0.018190801,
    0.020414220,
    0.022446344,
    0.024292061,
    0.025956742,
    0.027445048,
    0.028760802,
    0.029909261,
    0.030899401,
    0.031745216,
    0.032463685,
    0.033070297,
    0.033575409,
    0.033985700,
    0.034309825,
    0.034560888
])[::2]
w2k = lambda w: np.sqrt(w - l**2)
N = len(w)
k = w2k(w)
w = w.reshape((N, 1))
x = np.zeros(N)
r_eye = rt.trace(x, k, w, symmetry="xk", max_step=max_step, orbit_check=True)
for r in r_eye:
    plt.plot(r[0], r[1], "C3-")

# Now, find the rays that form the "eyelid".
r_eyelid = rt.trace([0], [k_max], [[w_max]], symmetry="xk", max_step=max_step, orbit_check=True)

for r in r_eyelid:
    plt.plot(r[0], r[1], "k--")

# To find the rays to the left and right of the eye, first compute the
# turning point on the x axis at w = w_max.
m_turn_1 = np.sqrt((w_max**2 - l**2) * (w_max**2 - l**4) / (w_max**2 - (1 - h**2) * l**2))
x_turn_1 = -arcsech((b - m_turn_1) / (b - a))

w_max = np.sqrt(0.5 * (1 - h) * l**2)
m_turn_2 = np.sqrt((w_max**2 - l**2) * (w_max**2 - l**4) / (w_max**2 - (1 - h**2) * l**2))
x_turn_2 = -arcsech((b - m_turn_2) / (b - a))

omega = lambda x: np.sqrt(0.5 * (l**2 + l**4 + m(x)**2 - np.sqrt(
    (l**2 - l**4 - m(x)**2)**2 + 4 * h**2 * l**2 * m(x)**2)))

N = 4
x = np.linspace(x_turn_2 + 0.01, x_turn_1 - 0.05, N)
k = np.zeros(N)
w = omega(x).reshape((N, 1))
r_sides = rt.trace(x, k, w, symmetry="xk", max_step=max_step, orbit_check=True)

for r in r_sides:
    plt.plot(r[0], r[1], "C0-")

# Find the rays that are delocalized (i.e., above and below the eye).
N = 4
x = np.zeros(N)
k = np.linspace(k_max, 0.25, N + 1)[1:]
w = omega(k).reshape((N, 1))
r_eyebrows = rt.trace(x, k, w, symmetry="xk", max_step=max_step, orbit_check=True)
for r in r_eyebrows:
    plt.plot(r[0], r[1], "C0-")

plt.xlim(-5, 5)
plt.ylim(-0.25, 0.25)
plt.xlabel(r"position $x$")
plt.ylabel(r"wave number $k$")
plt.title(r"flexural waves")

name = f"sech_flex_l{l}_h{h}"
np.savez(f"data/{name}.npz", r_eye=r_eye, r_eyelid=r_eyelid, r_eyebrows=r_eyebrows, r_sides=r_sides)
plt.savefig(f"data/{name}.png")

plt.show()
