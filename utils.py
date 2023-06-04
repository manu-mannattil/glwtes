# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_ivp, trapezoid

def sech(x):
    # Sech is not implemented in NumPy.  Computing it as 1/np.cosh(x) is
    # a bad idea since cosh is not bounded.
    return 2 * np.exp(-np.abs(x)) / (1 + np.exp(-2 * np.abs(x)))

def arcsech(x):
    # ArcSech isn't implemented either.
    if x == 0:
        return np.inf
    else:
        return np.arccosh(1/x)

class RayTracer:
    # - We're assuming that the Hamiltonian possesses reflection symmetry
    #   about the x and k axes and that all orbits are centered around the
    #   origin (0, 0).
    # - Closed-orbit "detection" will only work we're orbiting clockwise.
    #   Change the sign of the Hamilton's equations (or alternatively,
    #   the sign of t_max) if required to ensure this.
    def __init__(self, dH, xlim=(-1, 1), klim=(-1, 1)):
        self.dH = dH
        self.xlim = xlim
        self.klim = klim

        # State variables to check for closed orbit.
        self.x0, self.k0, self.flipped, self.closed = None, None, False, False

    def bounded(self, t, y, *params):
        # Bail out if the trajectory gets unbounded.
        if y[0] < self.xlim[0] or y[0] > self.xlim[1] or y[1] < self.klim[0] or y[1] > self.klim[1]:
            return False

        return True

    bounded.terminal = True
    bounded.direction = 1

    def proceed(self, t, y, *params):
        if not self.bounded(t, y, *params):
            return False

        # How do we detect a closed orbit?  First, compute the cross
        # product "c" between the current position (x, k) and the
        # initial position (x0, k0).  "c" will be positive in the
        # beginning when we're orbiting clockwise.  At some point, when
        # (x, k) becomes antiparallel to (x0, k0), c will vanish and then
        # will become negative.  Some time after that c will become zero
        # again once we make it back to (x0, k0).  The trick, therefore,
        # is to stop the integration when c is positive, provided it has
        # been negative once in the past.
        c = y[0] * self.k0 - self.x0 * y[1]
        if not self.flipped and c < 0:
            self.flipped = True
        elif self.flipped and c > 0:
            self.closed = True
            return False

        return True

    proceed.terminal = True
    proceed.direction = 1

    def integrate(self,
                  x0,
                  k0,
                  w=None,
                  t_max=np.inf,
                  max_step=1e-3,
                  backwards=False,
                  orbit_check=True):
        # We need to reset the closed-orbit checker each time.
        self.x0, self.k0, self.flipped, self.closed = x0, k0, False, False

        if backwards:
            t_max = -t_max

        if orbit_check:
            events = self.proceed
        else:
            events = self.bounded

        sol = solve_ivp(self.dH, [0, t_max], [x0, k0], max_step=max_step, events=events, args=w)

        # For reasons I don't understand (and don't want to), the last
        # two points on the trajectory returned by the integrator is the
        # same.  So delete the "extra" point.
        sol = np.delete(sol.y, -1, axis=1)

        # Close the orbit "manually" for aesthetics.
        if self.closed:
            sol[0][-1], sol[1][-1] = sol[0][0], sol[1][0]

        return sol

    def trace(self, x0, k0, w=None, symmetry="xk", max_points=1000, **kwargs):
        if len(x0) != len(k0):
            raise ValueError("x0 and k0 must have the same lengths.")
        if w is None:
            w = len(x0) * [None]
        elif w is not None and len(w) != len(x0):
            raise ValueError("w must have the same length as x0 and k0.")

        ray = []
        for _x0, _k0, _w in zip(x0, k0, w):
            sol = self.integrate(_x0, _k0, _w, **kwargs)

            x, k = sol[0], sol[1]

            # If we have too many points on the trajectory, downsample.
            n = len(x) // max_points
            if not self.closed and n > 1:
                x, k = x[::n], k[::n]

            ray.append([x, k])

            # We're on a closed orbit, so there's nothing more to be done.
            if self.closed:
                continue

            if symmetry in ("xk", "kx"):
                ray.append([-x, k])
                ray.append([x, -k])
                ray.append([-x, -k])
            elif symmetry == "x":
                ray.append([-x, k])
            elif symmetry == "k":
                ray.append([x, -k])

        return np.array(ray, dtype=object)

def normalize(psi, x, fix_sign=True, A=1):
    """Simple trapezoidal normalization of an N-component wavefunction."""
    psi = np.asarray(psi)
    psi2 = np.sum(psi.conj()*psi, axis=0)

    A = A/np.sqrt(trapezoid(psi2, x))
    psi = psi*A

    if fix_sign:
        # Round to 5 decimals before finding the max.
        # Odd functions would create an issue otherwise.
        _psi = np.round(psi.flatten(), 5)
        i = np.argmax(np.abs(_psi))
        sign = np.sign(_psi[i])
        psi *= sign

    return *psi,

def sort_evals_modes(evals, *modes):
    """Convert complex eigenvalues into real eigenvalues and sort modes accordingly."""
    evals = evals.real
    i1 = evals > 0
    evals = evals[i1]
    i2 = np.argsort(evals)
    evals = evals[i2]

    modes_sorted = []
    for m in modes:
        new_m = m[i1]
        new_m = new_m[i2]
        modes_sorted.append(new_m)

    return np.sqrt(evals), *modes_sorted

def quantindex(x, *modes):
    """Return two indices in [0, 1] to tell us whether the modes are quantized."""
    # To check if a mode f(x) is quantized we use a crude method:
    # divide the x interval into three parts, and compute integral
    # f(x)^2 for the left and right intervals.  Compare these values
    # with the integral of f(x)^2 for the entire x interval.
    N = x.size // 3

    # But the above method fails for larger modes, which come close to
    # the domain boundaries.  So define a new index, which computes
    # similar integrals as above, but only for edges.
    M = x.size // 10

    third_index = 0
    edge_index = 0

    for f in modes:
        left_int = trapezoid(f[:N]**2, x[:N])
        right_int = trapezoid(f[2 * N:]**2, x[2 * N:])
        full_int = trapezoid(f**2, x)
        left_edge_int = trapezoid(f[:M]**2, x[:M])
        right_edge_int = trapezoid(f[-M:]**2, x[-M:])

        if full_int > 0:
            # The RHS has a value of 2/3 for a nearly constant mode.
            third_index += left_int / full_int + right_int / full_int

            # The RHS has a value of 1/5 for a nearly constant mode.
            edge_index += left_edge_int / full_int + right_edge_int / full_int

    third_index = min(3 / 2 * third_index / len(modes), 1)
    edge_index = min(5 * edge_index / len(modes), 1)

    return third_index, edge_index

def parity(psi):
    """Check if the wave field psi is of definite parity."""
    N = len(psi)
    if N % 2 == 0:
        left, right = psi[:N//2], psi[N//2:][::-1]
    else:
        left, right = psi[:(N-1)//2], psi[(N+1)//2:][::-1]

    odd = np.std(0.5*(left - right))
    even = np.std(0.5*(left + right))
    std = np.std(psi)

    if odd > even and np.round(odd) == np.round(std):
        return -1, "odd"
    elif even > odd and np.round(even) == np.round(std):
        return 0, "even"
    else:
        return None, "none"
