# -*- coding: utf-8 -*-

"""Trace rays for membrane waves on a curved shell."""

import math
import matplotlib.pyplot as plt
import numpy as np
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

# Rays above and below the saddle --------------------------------------

# First trace the trajectories "above" and "below" the saddle.
N = 8
x = np.zeros(N)
k = np.linspace(0, 5, N + 1)[1:]
omega = np.vectorize(lambda k: np.sqrt(k**2 + l**2) if k**2 + l**2 > 1 else k**2 + l**2)
w = omega(k).reshape((N, 1))
r_above = rt.trace(x, k, w, symmetry="xk", max_step=1e-3, orbit_check=False)

# Now trace the saddle itself.
x = [0]
k = [1e-8]
w = [omega(k)]
r_saddle = rt.trace(x, k, w, symmetry="xk", max_step=1e-3, orbit_check=False)

# Rays on either side of the saddle ------------------------------------

# Now trace the rays "left" and "right" to the saddle.
x = [-5]
k = [0.18]

code = f"""
L = m^2((w^2 - 1/2(1 - h)l^2)(w^2 - (1 - h^2)l^2) - 1/2(1 - h)k^2w^2) +
    - (w^2 - (k^2 + l^2)^2)(w^2 - (k^2 + l^2))(w^2 - 1/2(1 - h)(k^2 + l^2));

(*  We need to find the cutoff frequency because the cutoff frequency
    for shear waves is lower.  Thus, assume that the ray grazes the
    x axis at x ~ -5, so k ~ 0 and m ~ m0.  The frequency we're
    interested in lies between the cutoff and l, which is the frequency
    for propagating membrane waves.  *)
cutoff = Sqrt[1/2(l^4 + l^2 + m^2 - Sqrt[(l^4 - l^2 + m^2)^2 + 4h^2l^2m^2])]/.{{ l->{l}, m->{m0}, h->{h} }};
Select[w/.Solve[(L/.{{ k->{k[0]}, l->{l}, m->{m0}, h->{h} }}) == 0, w], (# > cutoff && # < {l}) &]
"""
w = [session.evaluate(wlexpr(code))]

r_side = rt.trace(x, k, w, symmetry="x", backwards=True, max_step=1e-3, orbit_check=False)

name = f"{form}_membrane_l{l}_h{h}"
np.savez(f"data/{name}.npz",
         r_above=r_above,
         r_saddle=r_saddle,
         r_side=r_side)

# Test plotting --------------------------------------------------------

with np.load(f"data/{name}.npz", allow_pickle=True) as pack:
    r_above, r_saddle, r_side = pack["r_above"], pack["r_saddle"], pack["r_side"]

for r in r_above:
    plt.plot(r[0], r[1], "C0-")

for r in r_saddle:
    plt.plot(r[0], r[1], "C3-")

for r in r_side:
    plt.plot(r[0], r[1], "C3--")

plt.savefig(f"data/{name}.png")
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel(r"position $x$")
plt.ylabel(r"wave number $k$")
plt.title(r"membrane waves")
plt.show()

session.terminate()
