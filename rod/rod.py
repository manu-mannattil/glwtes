# -*- coding: utf-8 -*-

"""Solve the curved rod equations and find the eigenmodes."""

import dedalus.public as d3
import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import trapezoid
from utils import *

def rod_evp(L=2000,
            h=1,
            m=1,
            dm=0,
            bc="cc",
            N=2**7,
            slowness=0.01,
            slow_deriv=False,
            evals_only=False):

    # ------------- Slowness Parameters -------------

    if slow_deriv:
        # Use eps to include a "global" slowness in all derivates.
        # This is like replacing all derivatives d/dx with eps*d/dx,
        # eps being the "slowness" parameter.  Set the curvature
        # slowness (mu) to 1 in that case.
        eps, mu = slowness, 1
    else:
        # Use mu to only include curvature slowness in the
        # equations, i.e., curvature is a function of the form
        # f(mu*x).  Set the global slowness (eps) to 1 in that case.
        mu, eps = slowness, 1

    # NOTE: Irrespective of how slowness is set (i.e., via
    # eps or mu), the computed eigenvalues must be the same!  If
    # they aren't the same, there is a bug in the
    # implementation.

    # ------------- Problem Setup -------------

    # Bases
    coord = d3.Coordinate("x")
    dist = d3.Distributor(coord, dtype=np.float64)
    basis = d3.Chebyshev(coord, size=N, bounds=(-L / 2, L / 2))

    # Fields.
    z = dist.Field(name="z", bases=basis) # zeta (transverse)
    u = dist.Field(name="u", bases=basis) # u (longitudinal)
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

    # Make use of only the curvature slowness here.
    # NCC fields for curvature and derivative of curvature.
    if callable(m):
        mf = dist.Field(name="m", bases=basis)
        mf["g"] = m(mu * x)
        m_left, m_right = m(L / 2), m(-L / 2)
    else:
        mf, m_left, m_right = m, m, m

    if callable(dm):
        dmf = dist.Field(name="dm", bases=basis)
        dmf["g"] = mu * dm(mu * x)
    else:
        dmf = dm

    problem = d3.EVP([z, u, tau_z1, tau_z2, tau_z3, tau_z4, tau_u1, tau_u2],
                     eigenvalue=omega2,
                     namespace=locals())

    # Only make use of the "global" (aka derivative) slowness here.
    # Derivative expression, e.g., dm must also be multiplied by eps.
    problem.add_equation("h**2*eps**4*z4 + mf**2*z - eps*mf*u1 - omega2*z/(h**2) = 0")
    problem.add_equation("eps*mf*z1 + eps*dmf*z - eps**2*u2 - omega2*u/(h**2) = 0")

    # ------------ Boundary Conditions ------------ #

    if bc[0] == "c": # clamped
        problem.add_equation("z1(x=-L/2) = 0")
        problem.add_equation("z(x=-L/2) = 0")
        problem.add_equation("u(x=-L/2) = 0")
    elif bc[0] == "s": # simply supported
        problem.add_equation("z2(x=-L/2) = 0")
        problem.add_equation("z(x=-L/2) = 0")
        problem.add_equation("u(x=-L/2) = 0")
    else: # free
        problem.add_equation("z3(x=-L/2) = 0")
        problem.add_equation("z2(x=-L/2) = 0")
        problem.add_equation(f"u1(x=-L/2) - {m_left}*z(x=-L/2) = 0")

    if bc[1] == "c":
        problem.add_equation("z1(x=L/2) = 0")
        problem.add_equation("z(x=L/2) = 0")
        problem.add_equation("u(x=L/2) = 0")
    elif bc[1] == "s":
        problem.add_equation("z2(x=L/2) = 0")
        problem.add_equation("z(x=L/2) = 0")
        problem.add_equation("u(x=L/2) = 0")
    else:
        problem.add_equation("z3(x=L/2) = 0")
        problem.add_equation("z2(x=L/2) = 0")
        problem.add_equation(f"u1(x=L/2) - {m_left}*z(x=L/2) = 0")

    # ------------ Solve ------------ #

    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])

    evals = solver.eigenvalues
    evals_N = len(evals)

    # ------------ Return eigenvalues ------------ #

    if evals_only:
        return evals
    else:
        ug, zg = np.empty((evals_N, N), dtype=np.float64), np.empty((evals_N, N), dtype=np.float64)
        for i in range(evals_N):
            solver.set_state(i, solver.subsystems[0])
            zg[i] = solver.state[0]["g"]
            ug[i] = solver.state[1]["g"]

        x = dist.local_grid(basis)
        return x, evals, zg, ug

N = 2**11
bc = "cc"
b = 0.1
a = 0.01

# tanh type ------------------------------------------------------------

form = "tanh"
name = f"{form}_bc_{bc}_b{b}_N_{N}"
m = lambda x: b * np.tanh(x)
dm = lambda x: b / np.cosh(x)**2

# sech type ------------------------------------------------------------

# form = "sech"
# name = f"{form}_bc_{bc}_b{b}_a{a}_N_{N}"
# m = lambda x: b - (b - a) * sech(x)
# dm = lambda x: (b - a) * np.tanh(x) * sech(x)

# Run ------------------------------------------------------------------

x, evals, z, u = rod_evp(N=N, m=m, dm=dm, bc=bc)
evals, z, u = sort_evals_modes(evals, z, u)
np.savez(f"data/{name}.npz", x=x, evals=evals, z=z, u=u)
