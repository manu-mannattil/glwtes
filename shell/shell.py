# -*- coding: utf-8 -*-

"""Compute the eigenspectrum for a curved shell."""

import dedalus.public as d3
import numpy as np
import os

from scipy.integrate import trapezoid
from utils import *

def shell_evp(
        L=100, # length of the shell
        l=1.2, # transverse wave number
        h=0.3, # Poisson's ratio
        m=1, # curvature m(x)
        dm=0, # curvature derivature dm(x)/dx
        bc="clamped-clamped", # boundary condition
        N=2**7, # collocation points
        eps=0.1, # slowness parameter
        evals_only=False, # return only eigenvalues
        clean=True): # clean eigenvalues

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

    problem.add_equation("z1(x=-L/2) = 0")
    problem.add_equation("z(x=-L/2) = 0")
    problem.add_equation("u(x=-L/2) = 0")
    problem.add_equation("v(x=-L/2) = 0")

    problem.add_equation("z1(x=L/2) = 0")
    problem.add_equation("z(x=L/2) = 0")
    problem.add_equation("u(x=L/2) = 0")
    problem.add_equation("v(x=L/2) = 0")

    # ------------ Solve ------------ #

    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])

    evals = solver.eigenvalues
    evals_N = len(evals)

    ug, zg, vg = np.empty((evals_N, N)), np.empty((evals_N, N)), np.empty((evals_N, N))
    for i in range(evals_N):
        solver.set_state(i, solver.subsystems[0])
        zg[i] = solver.state[0]["g"].real
        ug[i] = solver.state[1]["g"].real
        vg[i] = solver.state[2]["g"].real

    x = dist.local_grid(basis)
    return x, evals, zg, ug, vg

m = lambda x: 10 * (1 - 1 / np.cosh(x))
dm = lambda x: 10 * np.tanh(x) / np.cosh(x)
curv = "10_sech_zero"

m = lambda x: 10 * np.tanh(x)
dm = lambda x: 10 / np.cosh(x)**2
curv = "10_tanh"

l = 1.2
N = 2**11

name = "cc_{}_sorted_l{}_N{}".format(curv, l, N)

x, evals, z, u, v = shell_evp(N=N, m=m, dm=dm, l=l)
evals, z, u, v = sort_evals_modes(evals, z, u, v)
np.savez("data/{}.npz".format(name), x=x, evals=evals, z=z, u=u, v=v)
