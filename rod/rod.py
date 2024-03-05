# -*- coding: utf-8 -*-

"""Solve the curved rod equations and find the eigenmodes."""

import dedalus.public as d3
import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import trapezoid
from utils import *

def rod_evp(L=2000,  # interval of rod: [-L/2, L/2]
            m=1,     # curvature m
            dm=0,    # dm/dx^2
            d2m=0,   # d^2m/dx^2
            N=2**7,  # number of points
            bc="cc", # boundary condition
            mu=0.01, # epsilon in paper
            e=1,     # include extra terms (compared to simpler rod model)
            evals_only=False):

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

    # NOTE: We don't add a tau field for u3 as we don't have any extra
    # boundary conditions to use.  Just use D(u2) in the equations.

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
        dm_left, dm_right = dm(L / 2), dm(-L / 2)
    else:
        dmf = dm
        dm_left, dm_right = 0, 0

    if callable(d2m):
        d2mf = dist.Field(name="d2m", bases=basis)
        d2mf["g"] = mu ** 2 * d2m(mu * x)
    else:
        d2mf = d2m

    problem = d3.EVP([z, u, tau_z1, tau_z2, tau_z3, tau_z4, tau_u1, tau_u2],
                     eigenvalue=omega2,
                     namespace=locals())

    problem.add_equation("z4 + mf**2*z + e*mf*D(u2) - mf*u1 + 2*e*dmf*u2 + e*d2mf*u1 - omega2*z = 0")
    problem.add_equation("-e*mf*z3 + mf*z1 + dmf*z - e*dmf*z2 - (1 + e*mf**2)*u2 - 2*e*mf*dmf*u1 - omega2*u = 0")

    # ------------ Boundary Conditions ------------ #

    if bc[0] == "c": # clamped
        problem.add_equation("z1(x=-L/2) = 0")
        problem.add_equation("z(x=-L/2) = 0")
        problem.add_equation("u(x=-L/2) = 0")
    elif bc[0] == "s": # simply supported
        problem.add_equation(f"z2(x=-L/2) + {m_left}*u1(x=-L/2) = 0")
        problem.add_equation("z(x=-L/2) = 0")
        problem.add_equation("u(x=-L/2) = 0")
    else: # free
        problem.add_equation(f"z3(x=-L/2) + {dm_left}*u1(x=-L/2) + {m_left}*u2(x=-L/2) = 0")
        problem.add_equation(f"z2(x=-L/2) + {m_left}*u1(x=-L/2) = 0")
        problem.add_equation(f"u1(x=-L/2) - {m_left}*z(x=-L/2) + {m_left}**2*u1(x=-L/2) + {m_left}*z2(x=-L/2) = 0")

    if bc[1] == "c":
        problem.add_equation("z1(x=L/2) = 0")
        problem.add_equation("z(x=L/2) = 0")
        problem.add_equation("u(x=L/2) = 0")
    elif bc[1] == "s":
        problem.add_equation(f"z2(x=L/2) + {m_right}*u1(x=L/2) = 0")
        problem.add_equation("z(x=L/2) = 0")
        problem.add_equation("u(x=L/2) = 0")
    else:
        problem.add_equation(f"z3(x=L/2) + {dm_left}*u1(x=L/2) + {m_left}*u2(x=L/2) = 0")
        problem.add_equation(f"z2(x=L/2) + {m_right}*u1(x=L/2) = 0")
        problem.add_equation(f"u1(x=L/2) - {m_left}*z(x=L/2) + {m_left}**2*u1(x=L/2) + {m_left}*z2(x=L/2) = 0")

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
dm = lambda x: b * sech(x) ** 2
d2m = lambda x: -2*b*sech(x)**2*np.tanh(x)

# sech type ------------------------------------------------------------

# form = "sech"
# name = f"{form}_bc_{bc}_b{b}_a{a}_N_{N}"
# m = lambda x: b - (b - a) * sech(x)
# dm = lambda x: (b - a) * np.tanh(x) * sech(x)
# d2m = lambda x: (b - a) * (-sech(x)**3 + sech(x)*np.tanh(x)**2)

# Run ------------------------------------------------------------------

x, evals, z, u = rod_evp(N=N, m=m, dm=dm, d2m=d2m, bc=bc)
evals, z, u = sort_evals_modes(evals, z, u)
np.savez(f"data/{name}.npz", x=x, evals=evals, z=z, u=u)
