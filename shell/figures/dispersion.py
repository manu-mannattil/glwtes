# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

l = 1.2 # transverse wave number
h = 0.3 # Poisson"s ratio
m = 10  # curvature

pack = np.load("../data/dispersion_l{}_h_{}_m{}.npz".format(l, h, m))
k, w = pack["k"], pack["w"]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Move left y-axis to centre.
ax.spines["left"].set_position("center")

# Eliminate upper and right axes.
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")

# Show ticks in the left and lower axes only.
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")

ax.plot(k, w[:, 0], color="k")
ax.plot(k, w[:, 1], color="C0")
ax.plot(k, w[:, 2], color="C3")
ax.plot(k, np.sqrt(10**2+k**4), color="C3")
ax.plot(-k[::-1], w[:, 0][::-1], color="k")
ax.plot(-k[::-1], w[:, 1][::-1], color="C0")
ax.plot(-k[::-1], w[:, 2][::-1], color="C3")
ax.set_ylim((0,30))
ax.set_xlim((-7.5,7.5))

inset = ax.inset_axes([0, 0.3, 0.42, 0.4])
# sub region of the original image
x1, x2, y1, y2 = -2, 2, 0.5, 2.0
inset.plot(k, w[:, 0], color="#888888")
inset.plot(k, w[:, 1], color="C0")
inset.plot(-k[::-1], w[:, 0][::-1], color="#888888")
inset.plot(-k[::-1], w[:, 1][::-1], color="C0")
inset.set_xlim(x1, x2)
inset.set_ylim(y1, y2)
#inset.set_xticklabels([])
#inset.set_yticklabels([])

ax.indicate_inset_zoom(inset, edgecolor="black", linestyle="--")

plt.show()
