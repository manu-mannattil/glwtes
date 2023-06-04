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

m = lambda x: b * np.tanh(x)
dm = lambda x: b / np.cosh(x) ** 2

max_step = 1e-3

# What's the largest k for which we see localization?
w_max = np.sqrt(0.5*(l**2 + l**4 + b**2 + np.sqrt((l**2 - l**4 - b**2)**2 + 4*h**2*l**2*b**2)))

# Now, find the largest k (i.e., on the k axis) corresponding to the
# above omega.
w2k = lambda w: np.sqrt(w**2 - l**2)
k_max = w2k(w_max)

rt = RayTracer(dH, xlim=(-5, 5), klim=(-1, 1))

# Find the rays that are localized (i.e., inside the "eye").
w = np.array([0.101516005, 0.104806505, 0.108025501, 0.110820271, 0.112923737])
N = len(w)
k = w2k(w)
w = w.reshape((N, 1))
x = np.zeros(N)
r_eye = rt.trace(x, k, w, symmetry="xk", max_step=max_step, orbit_check=True)
for r in r_eye:
    plt.plot(r[0], r[1], "C3-")

# Now, find the rays that form the "eyelid".
r_eyelid = rt.trace([0], [k_max], [[w_max]],
                 symmetry="xk",
                 max_step=max_step,
                 orbit_check=True)
for r in r_eyelid:
    plt.plot(r[0], r[1], "k--")

# Find the rays that are delocalized (i.e., above and below the eye).
N = 9
x = np.zeros(N)
k = np.linspace(k_max + 0.008, 0.25, N)
omega = lambda k: np.sqrt(l**2 + k**2)
w = omega(k).reshape((N, 1))
r_eyebrows = rt.trace(x, k, w, symmetry="xk", max_step=max_step, orbit_check=False)
for r in r_eyebrows:
    plt.plot(r[0], r[1], "C0-")

plt.xlim(-5, 5)
plt.ylim(-0.25, 0.25)
plt.xlabel(r"position $x$")
plt.ylabel(r"wave number $k$")
plt.title(r"extensional waves (tanh)")

name = f"tanh_ext_l{l}_h{h}"
np.savez(f"data/{name}.npz", r_eye=r_eye, r_eyelid=r_eyelid, r_eyebrows=r_eyebrows)
plt.savefig(f"data/{name}.png")

plt.show()
