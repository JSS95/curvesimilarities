"""Discrete Frechet distance."""

import numpy as np
from numba import njit
from scipy.spatial.distance import cdist

__all__ = [
    "dfd",
]


def dfd(P, Q):
    """Discrete Frechet distance.

    Parameters
    ----------
    P : array_like
        An :math:`p` by :math:`n` array of :math:`p` verticess in an
        :math:`n`-dimensional space.
    Q : array_like
        An :math:`q` by :math:`n` array of :math:`q` verticess in an
        :math:`n`-dimensional space.

    Returns
    -------
    dist : double
        The discrete Frechet distance between P and Q.

    Raises
    ------
    ValueError
        An exception is thrown if empty array is passed.

    Notes
    -----
    This function implements Eiter and Mannila's algorithm [#]_.

    References
    ----------
    .. [#] Eiter, T., & Mannila, H. (1994). Computing discrete Fréchet distance.

    Examples
    --------
    >>> dfd([[0, 0], [1, 1], [2, 0]], [[0, 1], [2, -4]])
    4.0
    """
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
        for j in range(1, q):
            ret[i, j] = max(
                min(ret[i - 1, j], ret[i, j - 1], ret[i - 1, j - 1]),
                dist[i, j],
            )

    return ret
