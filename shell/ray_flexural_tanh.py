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
a = 0.01

m = lambda x: b * np.tanh(x)
dm = lambda x: b / np.cosh(x) ** 2

max_step = 1e-3

# Dispersion of polarization I on the k axis.
omega = np.vectorize(lambda k: k**2 + l**2)

# What's the largest k for which we see localization?  To find that
# look at the frequency for which we have a double-minima in the dispersion.
w_max = 0.03502325

# Now, find the largest k (i.e., on the k axis) corresponding to the
# above omega.
k_max = np.sqrt(w_max - l**2)

rt = RayTracer(dH, xlim=(-5, 5), klim=(-1, 1))

# Find the rays that are localized (i.e., inside the "eye").
w = np.array([0.014031939, 0.021584755, 0.025413825, 0.028434470, 0.030533730, 0.032158646, 0.033283388, 0.034087900, 0.034594909, 0.034899765])
w2k = lambda w: np.sqrt(w - l**2)
N = len(w)
k = w2k(w)
w = w.reshape((N, 1))
x = np.zeros(N)
r_eye = rt.trace(x, k, w, symmetry="xk", max_step=max_step, orbit_check=True)
for r in r_eye:
    plt.plot(r[0], r[1], "C3-")

# Now, find the rays that form the "eyelid".
r_eyelid = rt.trace([0], [k_max], [[omega(k_max)]],
                 symmetry="xk",
                 max_step=max_step,
                 orbit_check=True)
for r in r_eyelid:
    plt.plot(r[0], r[1], "k--")

# To find the rays to the left and right of the eye, first compute the
# turning point on the x axis at w = w_max.
m_turn_1 = np.sqrt((w_max**2 - l**2)*(w_max**2 - l**4)/(w_max**2 - (1-h**2)*l**2))
x_turn_1 = -np.arctanh(m_turn_1/b)

w_max = np.sqrt(0.5*(1-h)*l**2)
m_turn_2 = np.sqrt((w_max**2 - l**2)*(w_max**2 - l**4)/(w_max**2 - (1-h**2)*l**2))
x_turn_2 = -np.arctanh(m_turn_2/b)

omega = lambda x: np.sqrt(0.5*(l**2 + l**4 + m(x)**2 - np.sqrt((l**2 - l**4 - m(x)**2)**2 + 4*h**2*l**2*m(x)**2)))

N = 4
x = np.linspace(x_turn_2 + 0.01, x_turn_1 - 0.05, N)
k = np.zeros(N)
w = omega(x).reshape((N, 1))
r_sides = rt.trace(x, k, w,
                 symmetry="xk",
                 max_step=max_step,
                 orbit_check=True)

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
plt.title(r"flexural waves (tanh)")

name = f"tanh_flex_l{l}_h{h}"
np.savez(f"data/{name}.npz", r_eye=r_eye, r_eyelid=r_eyelid, r_eyebrows=r_eyebrows, r_sides=r_sides)
plt.savefig(f"data/{name}.png")

plt.show()
