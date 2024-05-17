"""Discrete Frechet distance."""

import numpy as np
from numba import njit
from scipy.spatial.distance import cdist

__all__ = [
    "dfd",
]


def dfd(P, Q):
    """Discrete Frechet distance."""
    if len(P) == 0 or len(Q) == 0:
        raise ValueError("Vertices must not be empty.")
    dist = cdist(P, Q)
    return _dfd(dist)[-1, -1]


@njit
def _dfd(dist):
    # Eiter, T., & Mannila, H. (1994)
    p, q = dist.shape
    ret = np.empty((p, q), dtype=np.float_)

    ret[0, 0] = dist[0, 0]

    for i in range(1, p):
        ret[i, 0] = max(ret[i - 1, 0], dist[i, 0])

    for j in range(1, q):
        ret[0, j] = max(ret[0, j - 1], dist[0, j])

    for i in range(1, p):
        for i in range(1, q):
            ret[i, j] = max(
                min(ret[i - 1, j], ret[i, j - 1], ret[i - 1, j - 1]),
                dist[i, j],
            )

    return ret
