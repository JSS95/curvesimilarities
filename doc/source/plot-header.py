import matplotlib.pyplot as plt
import numpy as np

from curvesimilarities import ifd_owp
from curvesimilarities.util import curve_matching, parameter_space

P = [[0, 0], [2, 2], [4, 2], [4, 4], [2, 1], [5, 1], [7, 2]]
Q = [[2, 0], [1, 3], [5, 3], [5, 2], [7, 3]]
_, path = ifd_owp(P, Q, 0.1, "squared_euclidean")
weight, p, q, p_vert, q_vert = parameter_space(P, Q, 200, 100)
pairs = curve_matching(P, Q, path, 50)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(*np.array(P).T)
axes[0].plot(*np.array(Q).T)
axes[0].plot(*pairs, "--", color="gray")

axes[0].set_title("Curve space")
for sp in axes[0].spines:
    axes[0].spines[sp].set_visible(False)
axes[0].xaxis.set_visible(False)
axes[0].yaxis.set_visible(False)

axes[1].pcolormesh(p, q, weight.T)
axes[1].vlines(p_vert, 0, q[-1], "k", "--")
axes[1].hlines(q_vert, 0, p[-1], "k", "--")
axes[1].plot(*path.T, "k")

axes[1].set_title("Parameter space")
axes[1].xaxis.set_visible(False)
axes[1].yaxis.set_visible(False)

fig.tight_layout()
