# -*- coding: utf-8 -*-
import dedalus.public as d3
import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from subprocess import run


run(r"""ffmpeg
    -f image2
    -framerate 24
    -i filament_%05d.png
    -c:v libx264
    -preset slow
    -crf 20
    -pix_fmt yuv420p
    altsech.mp4""".split())
exit()
# ------------- Slowness Parameters -------------

slowness = 0.1

# Use eps to include a "global" slowness in all derivates.
# This is like replacing all derivatives d/dx with eps*d/dx,
# eps being the "slowness" parameter.  Set the curvature
# slowness (mu) to 1 in that case.
eps, mu = slowness, 1

# Use mu to only include curvature slowness in the
# equations, i.e., curvature is a function of the form
# f(mu*x).  Set the global slowness (eps) to 1 in that case.
mu, eps = slowness, 1

# Warning: Irrespective of how slowness is set (i.e., via
# eps or mu), the computed eigenvalues must be the same!  If
# they aren't the same, there is a bug in the
# implementation.

# ------------- Physical Parameters -------------

# Parameters
L = 100 # interval length
L *= eps # rescale by global slowness
N = 2**11 # number of collocation points

K = 10 # max curvature
h = 1 # thickness

# ------------- Problem Setup -------------

# Bases
coord = d3.Coordinate("x")
dist = d3.Distributor(coord, dtype=np.float64)
basis = d3.Chebyshev(coord, size=N, bounds=(-L / 2, L / 2))

# Fields.
z = dist.Field(name="z", bases=basis) # zeta (transverse)
u = dist.Field(name="u", bases=basis) # u (longitudinal)
tz = dist.Field(name="tz", bases=basis) # zeta (transverse)
tu = dist.Field(name="tu", bases=basis) # u (longitudinal)
omega2 = dist.Field(name="omega2")
x = dist.local_grid(basis)

# Substitutions.
D = lambda f: d3.Differentiate(f, coord)
lift_basis = basis.derivative_basis(1)
lift = lambda f: d3.Lift(f, lift_basis, -1)

# Tau fields and higher derivatives.
tau_z1 = dist.Field(name="tau_z1")
z1 = D(z) + lift(tau_z1)
tau_z2 = dist.Field(name="tau_z2")
z2 = D(z1) + lift(tau_z2)
tau_z3 = dist.Field(name="tau_z3")
z3 = D(z2) + lift(tau_z3)
tau_z4 = dist.Field(name="tau_z4")
z4 = D(z3) + lift(tau_z4)
tau_u1 = dist.Field(name="tau_u1")
u1 = D(u) + lift(tau_u1)
tau_u2 = dist.Field(name="tau_u2")
u2 = D(u1) + lift(tau_u2)

# NCC fields for curvature, derivative of curvature, and
# square of curvature.
k = dist.Field(name="k", bases=basis)
dk = dist.Field(name="dk", bases=basis)
k2 = dist.Field(name="k2", bases=basis)

# Make use of only the curvature slowness here.
# k["g"] = K * np.tanh(mu * x)
# k2["g"] = K**2 * np.tanh(mu * x)**2
# dk["g"] = K * mu / np.cosh(mu * x)**2

k["g"] = K * (1.25 - 1/np.cosh(mu * x))
k2["g"] = K ** 2 * (1.25 - 1/np.cosh(mu * x)) ** 2
dk["g"] = K * mu / np.cosh(mu * x) * np.tanh(mu * x)

# k["g"] = K * (1 - 1/np.cosh(mu * x)**4)
# k2["g"] = K ** 2 * (1 - 1/np.cosh(mu * x)**4) ** 2
# dk["g"] = 4 * K * mu / np.cosh(mu * x) ** 4 * np.tanh(mu * x)

problem = d3.IVP([z, u, tz, tu, tau_z1, tau_z2, tau_z3, tau_z4, tau_u1, tau_u2],
                 namespace=locals())

# Only make use of the "global" (aka derivative) slowness here.
# Derivative expression, e.g., dm must also be multiplied by eps.
problem.add_equation("dt(z) - tz = 0")
problem.add_equation("dt(u) - tu = 0")
problem.add_equation("dt(tz) + h**2*eps**4*z4 + k2*z - eps*k*u1 = 0")
problem.add_equation("dt(tu) + eps*k*z1 + eps*dk*z - eps**2*u2 = 0")

# Boundary conditions zclamped-clamped).
problem.add_equation("z(x=-L/2) = 0")
problem.add_equation("z(x=L/2) = 0")
problem.add_equation("z1(x=-L/2) = 0")
problem.add_equation("z1(x=L/2) = 0")
problem.add_equation("u(x=-L/2) = 0")
problem.add_equation("u(x=L/2) = 0")

# Initial conditions ---------------------------------------------------

x = dist.local_grid(basis)

# Approximate a Gaussian that vanishes at the ends by computing
# the amplitudes of the first 10 modes using Mathematica:
#   Integrate[Sin[n x] Sin[x] Exp[-(x - \[Pi]/2)^2], {x, 0, \[Pi]}]/(\[Pi]/2)
# ampl = np.array([0.768907, 0., -0.222695, 0., 0.00726429, 0., -0.00167352, 0., -0.000839758, 0.])
# freqs = np.arange(1, 10 + 1)
# ip = np.sum([a * np.sin(f * x) for (a, f) in zip(ampl, freqs)], axis=0)

# Problem 6-12 from A. P. French, Vibrations and Waves (1971).
# ip = np.pi/2 - np.abs(x-np.pi/2)

# A random initial profile that vanishes at the end point and generated
# by a simple moving average.
# win = 10
# ip = np.random.random(N + win - 1)
# ip = np.cumsum(y)
# ip[win:] = ip[win:] - ip[:-win]
# ip = ip[win - 1:] / win
# ip = ip * (np.sin(x) ** 0.3)

# A perturbed odd function.
eps = 0.1
ip = np.exp(-eps*x)*np.sin(2*x)

z["g"] = 2*np.exp(-0.015*x**2)
u["g"] = 0.5*np.exp(-0.01*x**2)

tz["g"] = np.zeros(N)
tu["g"] = np.zeros(N)

# Solve ----------------------------------------------------------------

stop_time = 100
timestepper = d3.SBDF2
timestep = 1e-3

solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_time

while solver.proceed:
    solver.step(timestep)

    if solver.iteration % 10 == 0:
        fig, ax = plt.subplots()
        ax.set_xlim((-L/2, L/2))
        ax.set_ylim((-5, 5))
        ax.plot(x, u["g"], "C0", alpha=0.5)
        ax.plot(x, z["g"], "C3")

        plt.title("time = {:.3f}".format(solver.sim_time))
        plt.xlabel(r"$x$")
        plt.ylabel(r"$\zeta$ and $u$")
        plt.tight_layout()
        plt.savefig("filament_{:05d}.png".format(solver.iteration // 10), dpi=200)
        print("filament_{:05d}.png written".format(solver.iteration // 10))
        plt.close(fig)

run(r"""ffmpeg
    -f image2
    -framerate 24
    -i filament_%05d.png
    -c:v libx264
    -preset slow
    -crf 20
    -pix_fmt yuv420p
    altsech.mp4""".split())
