# -*- coding: utf-8 -*-

"""Trace rays for shear waves on a curved shell."""

import numpy as np
import matplotlib.pyplot as plt
import math
from utils import *

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr

session = WolframLanguageSession("/usr/local/bin/WolframKernel")

def dH(t, y, w):
    """RHS of Hamilton's equations for a curved shell."""
    x, k = y

    w2 = w**2
    q2 = k**2 + l**2
    X = (w2 - q2**2)
    Y = (w2 - q2)
    Z = (w2 - 0.5 * (1 - h) * q2)

    a = k * (2 * Z * (2 * q2 * Y + X) + (1 - h) * (X * Y - m(x)**2 * w2))
    b = -(2 * m(x) * dm(x) * ((w2 - 0.5 * (1 - h) * l**2) * (w2 - (1 - h**2) * l**2) - 0.5 *
                              (1 - h) * k**2 * w2))

    # Normalize the RHS.  This doesn't change the rays, but
    # produces smoother trajectories.
    n = np.sqrt(a**2 + b**2)
    return [a / n, b / n]

session.start()

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

rt = RayTracer(dH, xlim=(-5, 5), klim=(-6, 6))

# Saddle point rays ----------------------------------------------------

# First trace the trajectory leaving the saddle point on the x axis.
k = [1e-3]
r = np.sqrt((1 + h) * l**2 * (2 * l**2 - (1 - h)) / (2 * (1 - h))) / m0
if form == "tanh":
    x = [np.arctanh(r)]
else:
    x = [np.arccosh(1 / (1 - r))]

w = np.array([np.sqrt(0.5 * (1 - h) * l**2)]).reshape((1, 1))
r_saddle = rt.trace(x, k, w, symmetry="xk", max_step=1e-3, orbit_check=False)

# Propagating rays -----------------------------------------------------

# Now trace trajectories "above" and "below" the saddle-point trajectory.
N = 8

# Find k_min -- the wave number of the saddle point ray at x = -5, where
# m(x) ~ m0.
code = f"""
h = {h}; l={l}; m = {m0};
L = ((-1 + h)k^2(-((2k^2 + (1 + h)l^2)(2k^4 + 4k^2l^2 + l^2(-1 + h + 2l^2))) - 2(-1 + h)l^2m^2))/8;
Select[k/.Solve[L == 0, k], (Element[#, Reals] && # > 0) &][[1]]
"""
k_min = session.evaluate(wlexpr(code))

# Start slightly above k_min to find the propagating rays.
k = np.linspace(k_min, 5, N + 1)[1:]
x = -5 * np.ones(N)
omega = np.vectorize(lambda k: np.sqrt((1 - h) * (k**2 + l**2) / 2))
w = omega(np.zeros(N)).reshape((N, 1))
r_propagating = rt.trace(x, k, w, symmetry="k", max_step=1e-3, orbit_check=False)

# Reflected rays -------------------------------------------------------

def omega(k):
    code = f"""
    L = m^2((w^2 - 1/2(1 - h)l^2)(w^2 - (1 - h^2)l^2) - 1/2(1 - h)k^2w^2) +
            - (w^2 - (k^2 + l^2)^2)(w^2 - (k^2 + l^2))(w^2 - 1/2(1 - h)(k^2 + l^2));

    (* We want to find omega < the propagation frequency. *)
    cutoff = Sqrt[0.5(1-{h}){l}^2];
    Select[w/.Solve[(L/.{{ m->{m0}, k->{k}, l->{l}, h->{h} }}) == 0, w], (# < cutoff && # > 0) &][[1]]
    """

    return session.evaluate(wlexpr(code))

w = [[omega(0.5 * k_min)]]
k = [0.5 * k_min]
x = [-5]
r_reflected = rt.trace(x, k, w, backwards=True, max_step=1e-3, orbit_check=False)

# Test plotting --------------------------------------------------------

name = f"{form}_shear_l{l}_h{h}"
np.savez(f"data/{name}.npz",
         r_propagating=r_propagating,
         r_reflected=r_reflected,
         r_saddle=r_saddle)

with np.load(f"data/{name}.npz", allow_pickle=True) as pack:
    r_propagating = pack["r_propagating"]
    r_reflected = pack["r_reflected"]
    r_saddle = pack["r_saddle"]

for r in r_saddle:
    plt.plot(r[0], r[1], "C3-")

for r in r_reflected:
    plt.plot(r[0], r[1], "C3-")

for r in r_propagating:
    plt.plot(r[0], r[1], "C0-")

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel(r"position $x$")
plt.ylabel(r"wave number $k$")
plt.title(r"shear waves")
plt.savefig(f"data/{name}.png")
plt.show()

session.terminate()
