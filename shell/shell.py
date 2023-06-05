# -*- coding: utf-8 -*-
"""Compute the eigenspectrum for a curved shell."""

import dedalus.public as d3
import numpy as np
import os

from scipy.integrate import trapezoid
from utils import *

from dedalus.tools.config import config

config["logging"]["stdout_level"] = "debug"

def shell_evp(
        L=2000, # length of the shell
        l=0.1, # transverse wave number
        h=0.3, # Poisson's ratio
        m=1, # curvature m(x)
        dm=0, # curvature derivature dm(x)/dx
        bc="cc", # boundary condition
        N=2**7, # collocation points
        eps=0.01): # slowness parameter

    # ------------- Problem Setup -------------

    # Bases
    coord = d3.Coordinate("x")
    dist = d3.Distributor(coord, dtype=np.complex128)
    basis = d3.Chebyshev(coord, size=N, bounds=(-L / 2, L / 2), dealias=3 / 2)

    # Fields.
    z = dist.Field(name="z", bases=basis) # zeta (transverse)
    u = dist.Field(name="u", bases=basis) # u (in plane)
    v = dist.Field(name="v", bases=basis) # v (in plane)
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
    tau_v1 = dist.Field(name="tau_v1")
    v1 = D(v) + lift(tau_v1)
    tau_v2 = dist.Field(name="tau_v2")
    v2 = D(v1) + lift(tau_v2)

    if callable(m):
        mf = dist.Field(name="m", bases=basis)
        mf["g"] = m(eps * x)
        m_left, m_right = m(L / 2), m(-L / 2)
    else:
        mf, m_left, m_right = m, m, m

    if callable(dm):
        dmf = dist.Field(name="dm", bases=basis)
        dmf["g"] = eps * dm(eps * x)
    else:
        dmf = dm

    problem = d3.EVP([z, u, v, tau_z1, tau_z2, tau_z3, tau_z4, tau_u1, tau_u2, tau_v1, tau_v2],
                     eigenvalue=omega2,
                     namespace=locals())

    problem.add_equation("z4 - 2*l**2*z2 + l**4*z + mf**2*z - mf*u1 - 1j*h*l*mf*v - omega2*z = 0")
    problem.add_equation("mf*z1 + dmf*z - u2 + 0.5*(1-h)*l**2*u - 1j*0.5*(1+h)*l*v1 - omega2*u = 0")
    problem.add_equation("1j*h*l*mf*z - 1j*0.5*(1+h)*l*u1 - 0.5*(1-h)*v2 + l**2*v - omega2*v = 0")

    # ------------ Boundary Conditions ------------ #

    if bc[0] == "c":
        problem.add_equation("z1(x=-L/2) = 0")
        problem.add_equation("z(x=-L/2) = 0")
        problem.add_equation("u(x=-L/2) = 0")
        problem.add_equation("v(x=-L/2) = 0")
    elif bc[0] == "s":
        problem.add_equation(f"z2(x=-L/2) - {h*l**2}*z(x=-L/2) = 0")
        problem.add_equation("z(x=-L/2) = 0")
        problem.add_equation("u(x=-L/2) = 0")
        problem.add_equation("v(x=-L/2) = 0")

    if bc[1] == "c":
        problem.add_equation("z1(x=L/2) = 0")
        problem.add_equation("z(x=L/2) = 0")
        problem.add_equation("u(x=L/2) = 0")
        problem.add_equation("v(x=L/2) = 0")
    elif bc[1] == "s":
        problem.add_equation(f"z2(x=L/2) - {h*l**2}*z(x=L/2) = 0")
        problem.add_equation("z(x=L/2) = 0")
        problem.add_equation("u(x=L/2) = 0")
        problem.add_equation("v(x=L/2) = 0")

    # ------------ Solve ------------ #

    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])

    evals = solver.eigenvalues
    evals_N = len(evals)

    ug = np.empty((evals_N, N), dtype=np.complex128)
    vg = np.empty((evals_N, N), dtype=np.complex128)
    zg = np.empty((evals_N, N), dtype=np.complex128)

    for i in range(evals_N):
        solver.set_state(i, solver.subsystems[0])
        zg[i] = solver.state[0]["g"]
        ug[i] = solver.state[1]["g"]
        vg[i] = solver.state[2]["g"]

        # Choose a phase convention such that z and u are real,
        # and v is complex.
        if zg[i].real.std() < zg[i].imag.std():
            zg[i] = 1j * zg[i]
            ug[i] = 1j * ug[i]

            # We could've not multiplied v by 1j -- that would have made
            # u, v, and z real.  But then the triplet (u, v, z) would no
            # longer be an eigenvector.
            vg[i] = 1j * vg[i]

    x = dist.local_grid(basis)
    return x, evals, zg, ug, vg

N = 2**11
bc = "cc"
eps = 0.01
l = 0.1
b = 0.1
a = 0.01

# sech type ------------------------------------------------------------

form = "sech"
m = lambda x: b - (b - a) * sech(x)
dm = lambda x: (b - a) * np.tanh(x) * sech(x)
name = f"{form}_bc_{bc}_l_{l}_eps_{eps}_b{b}_a{a}_N_{N}"

# tanh type ------------------------------------------------------------

# form = "tanh"
# m = lambda x: b * np.tanh(x)
# dm = lambda x: b / np.cosh(x) ** 2
# name = f"{form}_bc_{bc}_l_{l}_eps_{eps}_b{b}_N_{N}"

# gauss type ------------------------------------------------------------

# form = "gauss"
# m = lambda x: b * np.exp(-x**2)
# dm = lambda x: -2*b*x*np.exp(-x**2)
# name = f"{form}_bc_{bc}_l_{l}_eps_{eps}_b{b}_N_{N}"

# Run ------------------------------------------------------------------

x, evals, z, u, v = shell_evp(N=N, m=m, dm=dm, l=l, eps=eps, bc=bc)
evals, z, u, v = sort_evals_modes(evals, z, u, v)

np.savez(f"data/{name}", x=x, evals=evals, z=z, u=u, v=v)
