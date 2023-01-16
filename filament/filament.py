# -*- coding: utf-8 -*-

import dedalus.public as d3
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from utils import *

def filament_evp(L=100,
                 h=1,
                 m=1,
                 dm=0,
                 bc="clamped-clamped",
                 N=2**7,
                 slowness=0.1,
                 slow_deriv=False,
                 evals_only=False,
                 clean=True):

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
    dist = d3.Distributor(coord, dtype=np.complex128)
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

    left, right = bc.split("-")
    if left == "clamped":
        problem.add_equation("z1(x=-L/2) = 0")
        problem.add_equation("z(x=-L/2) = 0")
        problem.add_equation("u(x=-L/2) = 0")
    else:
        problem.add_equation("z3(x=-L/2) = 0")
        problem.add_equation("z2(x=-L/2) = 0")
        problem.add_equation("u1(x=-L/2) - {}*z(x=-L/2) = 0".format(m_left))

    if right == "clamped":
        problem.add_equation("z1(x=L/2) = 0")
        problem.add_equation("z(x=L/2) = 0")
        problem.add_equation("u(x=L/2) = 0")
    else:
        problem.add_equation("z3(x=L/2) = 0")
        problem.add_equation("z2(x=L/2) = 0")
        problem.add_equation("u1(x=L/2) - {}*z(x=L/2) = 0".format(m_right))

    # ------------ Solve ------------ #

    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])

    evals = solver.eigenvalues
    evals_N = len(evals)

    # ------------ Clean and Filter ------------ #

    if clean:
        # Just remove large eigenvalues.
        i1 = np.abs(evals) < 1e10
        evals = evals[i1]

        i2 = np.argsort(evals)
        evals = np.sqrt(evals[i2].real)

    if evals_only:
        return evals
    else:
        ug, zg = np.empty((evals_N, N)), np.empty((evals_N, N))
        for i in range(evals_N):
            solver.set_state(i, solver.subsystems[0])
            zg[i] = solver.state[0]["g"].real
            ug[i] = solver.state[1]["g"].real

        if clean:
            zg, ug = zg[i1], ug[i1]
            zg, ug = zg[i2], ug[i2]

        x = dist.local_grid(basis)
        return x, evals, zg, ug

def solve(self, clean=False, clean_factor=2):
    x, evals_low, z, u = self.solve_evp_N(self.N)

    if clean:
        evals_hi = self.solve_evp_N(int(clean_factor * self.N), evals_only=True)

        cleaner = FilterEigenvalues(evals_low, evals_hi)
        evals, indx = cleaner.clean()

        return x, np.sqrt(evals.real), z[indx], u[indx]
    else:
        # Just remove large eigenvalues.
        indx = np.abs(evals_low) < 1e10
        evals, z, u = evals_low[indx], z[indx], u[indx]
        indx = np.argsort(evals)
        evals = evals[indx]

    return x, np.sqrt(evals.real), z[indx], u[indx]

m = lambda x: 10 * (1 - 1 / np.cosh(x))
dm = lambda x: 10 * np.tanh(x) / np.cosh(x)
name = "ff_10_sech_zero_raw"

# m = lambda x: 10 * np.tanh(x)
# dm = lambda x: 10 / np.cosh(x)**2
# name = "ff_10_tanh_raw"

N = 2**11
x, evals_low, z, u, = filament_evp(N=N, m=m, dm=dm, clean=False, bc="free-free")
np.savez("data/{}_{}.npz".format(name, N), x=x, evals=evals_low, z=z, u=u)

# N = 2**8
# x, evals_low, z, u, = filament_evp(N=N, m=m, dm=dm, clean=False)
# np.savez("data/{}_{}.npz".format(name, N), x=x, evals=evals_low, z=z, u=u)

# cleaner = FilterEigenvalues(evals_low, evals_hi)
# evals, indx = cleaner.clean(drift_threshold=1e5)
# cleaner.plot()
# plt.show()
