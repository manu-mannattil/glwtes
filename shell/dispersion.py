# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize

sqrt = lambda x: pow(x, 1 / 2)
cbrt = lambda x: pow(x, 1 / 3)

def omega(k, l=0.1, h=0.3, m=0.01):
    q = sqrt(k**2 + l**2)

    # Coefficients of the cubic.  We need to add 0j so that Python
    # treats these variables as complex.
    a = 0.5 * q**2 * ((3 - h) + 2 * q**2) + m**2 + 0j
    b = -0.5 * q**4 * ((3 - h) * q**2 + (1 - h)) - \
            0.5 * m**2 * (1 - h) * (q**2 + 2 * (1 + h) * l**2) + 0j
    c = 0.5 * (1 - h) * (q**8 + m**2 * l**4 * (1 - h**2)) + 0j

    # Common term
    d = cbrt(2 * a**3 + 9 * a * b + 27 * c +
             3 * sqrt(3) * sqrt(4 * a**3 * c - a**2 * b**2 + 18 * a * b * c - 4 * b**3 + 27 * c**2))

    # Quasi extensional
    w1 = a / 3 + cbrt(2) * (a**2 + 3 * b) / (3 * d) + d / (3 * cbrt(2))

    # Quasi shear
    w2 = a / 3 - (1 - 1j * sqrt(3)) * (a**2 + 3 * b) / (3 * cbrt(4) * d) \
            - (1 + 1j * sqrt(3)) * d / (6 * cbrt(2))

    # Quasi flexural
    w3 = a / 3 - (1 + 1j * sqrt(3)) * (a**2 + 3 * b) / (3 * cbrt(4) * d) \
            - (1 - 1j * sqrt(3)) * d / (6 * cbrt(2))

    w1, w2, w3 = sqrt(w1.real), sqrt(w2.real), sqrt(w3.real)

    # Extensional, shear, flexural.
    return [w1, w2, w3]

def flex_omega_min(l=0.1, h=0.3, m=0.02):
    """Find the double-well minimum in the flexural dispersion."""
    fun = lambda k: omega(k, l, h, m)[2]
    guess = np.sqrt(l * sqrt(m) * pow(1 - h**2, 1 / 4) - l**2)
    bounds = ((guess - guess / 2, guess + guess / 2), )
    res = minimize(fun, guess, bounds=bounds)

    if isinstance(res.fun, np.ndarray):
        return res.fun[0], res.x[0]
    else:
        return res.fun, res.x[0]

def flex_find_m(w, l=0.1, h=0.3, N=10, Niter=10):
    """Find the curvature for which the double-well minimum is w."""
    m_min = sqrt(l**4 / (1 - h**2))
    m_max = 0.2

    for n in range(Niter):
        mm = np.linspace(m_min, m_max, N)
        ww = np.array([flex_omega_min(l, h, _)[0] for _ in mm])
        i = np.searchsorted(ww, w)

        m_min = mm[i - 1]
        m_max = mm[i + 1]

    return 0.5 * (m_min + m_max)
