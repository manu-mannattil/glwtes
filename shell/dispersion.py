# -*- coding: utf-8 -*-

"""Dispersion curves for a curved shell.

This script finds the dispersion curves for a curved shell
of given curvature.
"""

import numpy as np

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr

def disp_curve(k, l, h, m):
    code = """k = {}; l = {}; h = {}; m = {};
    L = -((-k^2 - l^2 + w^2)*(-1/2*((1 - h)*(k^2 + l^2)) + w^2)*(-(k^2 + l^2)^2 + w^2)) +
            (-1/2*((1 - h)*k^2*w^2) + (-1/2*((1 - h)*l^2) + w^2)*(-((1 - h^2)*l^2) + w^2))*m^2;
    w /. Solve[L == 0 && w > 0, w]
    """.format(k, l, h, m)

    return np.array(session.evaluate(wlexpr(code)))

l = 1.2 # transverse wave number
h = 0.3 # Poisson's ratio
m = 10  # curvature

session = WolframLanguageSession("/usr/local/bin/WolframKernel")
session.start()

kk = np.linspace(0, 10, 200)
ww = np.array([disp_curve(k, l=l, h=h, m=m) for k in kk])

session.terminate()

np.savez("data/dispersion_l{}_h_{}_m{}.npz".format(l, h, m), k=kk, w=ww)
