"""Frechet distance and its variants."""

import numpy as np
from numba import njit
from scipy.spatial.distance import cdist

__all__ = [
    "fd",
    "dfd",
]


def fd(P, Q):
    """(Continuous) Frechet distance.

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
        The (continuous) Frechet distance between P and Q.

    Raises
    ------
    ValueError
        An exception is thrown if empty array is passed.

    Notes
    -----
    This function implements Alt and Godau's algorithm [#]_.

    References
    ----------
    .. [#] Alt, H., & Godau, M. (1995). Computing the Fréchet distance between
       two polygonal curves. International Journal of Computational Geometry &
       Applications, 5(01n02), 75-91.
    """
    if len(P) == 0 or len(Q) == 0:
        raise ValueError("Vertices must not be empty.")


def _free_interval(pt0, pt1, pt2, eps):
    NAN = np.float_(np.nan)
    # resulting interval is always in [0, 1] or is [nan, nan].
    coeff1 = pt1 - pt0
    coeff2 = pt0 - pt2
    a = np.dot(coeff1, coeff1)
    c = np.dot(coeff2, coeff2) - eps**2
    if a == 0:  # degenerate case
        if c > 0:
            interval = [NAN, NAN]
        else:
            interval = [np.float_(0), np.float_(1)]
        return interval
    b = 2 * np.dot(coeff1, coeff2)
    D = b**2 - 4 * a * c
    if D < 0:
        interval = [NAN, NAN]
    else:
        start = np.max([(-b - D**0.5) / 2 / a, 0])
        end = np.min([(-b + D**0.5) / 2 / a, 1])
        if start > 1 or end < 0:
            start = end = NAN
        interval = [start, end]
    return interval


def _decision_problem(P, Q, eps):
    # Early abandon
    first = P[0] - Q[0]
    if np.dot(first, first) > eps**2:
        return False

    # Decide reachablilty
    B = np.empty((len(P) - 1, len(Q), 2), dtype=np.float_)
    start, end = _free_interval(P[0], P[1], Q[0], eps)
    if start == 0:
        B[0, 0] = [start, end]
    else:
        B[0, 0] = [np.nan, np.nan]
    for i in range(1, len(P) - 1):
        _, prev_end = B[i - 1, 0]
        if prev_end == 1:
            start, end = _free_interval(P[i], P[i + 1], Q[0], eps)
            if start == 0:
                B[i, 0] = [start, end]
                continue
        B[i, 0] = [np.nan, np.nan]

    L = np.empty((len(P), len(Q) - 1, 2), dtype=np.float_)
    start, end = _free_interval(Q[0], Q[1], P[0], eps)
    if start == 0:
        L[0, 0] = [start, end]
    else:
        L[0, 0] = [np.nan, np.nan]
    for j in range(1, len(Q) - 1):
        _, prev_end = L[0, j - 1]
        if prev_end == 1:
            start, end = _free_interval(Q[j], Q[j + 1], P[0], eps)
            if start == 0:
                L[0, j] = [start, end]
                continue
        L[0, j] = [np.nan, np.nan]

    for i in range(len(P) - 1):
        for j in range(len(Q) - 1):
            prevL_start, _ = L[i, j]
            prevB_start, _ = B[i, j]
            L_start, L_end = _free_interval(P[i], P[i + 1], Q[j + 1], eps)
            B_start, B_end = _free_interval(Q[j], Q[j + 1], P[i + 1], eps)

            if not np.isnan(prevB_start):
                L[i + 1, j] = [L_start, L_end]
            elif prevL_start <= L_end:
                L[i + 1, j] = [max(prevL_start, L_start), L_end]
            else:
                L[i + 1, j] = [np.nan, np.nan]

            if not np.isnan(prevL_start):
                B[i, j + 1] = [B_start, B_end]
            elif prevB_start <= B_end:
                B[i, j + 1] = [max(prevB_start, B_start), B_end]
            else:
                B[i, j + 1] = [np.nan, np.nan]

    return L[-1, -1, 1] == 1


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
