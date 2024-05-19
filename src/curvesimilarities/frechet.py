"""Fréchet distance and its variants."""

import sys

import numpy as np
from numba import njit
from scipy.spatial.distance import cdist

__all__ = [
    "fd",
    "dfd",
]


EPSILON = np.float_(sys.float_info.epsilon)


def fd(P, Q):
    r"""(Continuous) Fréchet distance between two open polygonal curves.

    Let :math:`f: [0, 1] \to \Omega` and :math:`g: [0, 1] \to \Omega` be curves
    where :math:`\Omega` is a metric space. The Fréchet distance between
    :math:`f` and :math:`g` is defined as

    .. math::

        \inf_{\alpha, \beta} \max_{t \in [0, 1]}
        \lVert f(\alpha(t)) - g(\beta(t)) \rVert

    where :math:`\alpha, \beta: [0, 1] \to [0, 1]` are continuous non-decreasing
    surjections and :math:`\lVert \cdot \rVert` is the underlying metric, which
    is the Euclidean metric in this implementation.

    Parameters
    ----------
    P : array_like
        A :math:`p` by :math:`n` array of :math:`p` vertices in an
        :math:`n`-dimensional space.
    Q : array_like
        A :math:`q` by :math:`n` array of :math:`q` vertices in an
        :math:`n`-dimensional space.

    Returns
    -------
    dist : double
        The (continuous) Fréchet distance between P and Q.

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

    Examples
    --------
    >>> fd([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]])
    1.0...
    """
    if len(P) == 0 or len(Q) == 0:
        raise ValueError("Vertices must not be empty.")
    P = np.asarray(P, dtype=np.float_)
    Q = np.asarray(Q, dtype=np.float_)
    return _fd(P, Q)


@njit
def _free_interval(A, B, C, eps):
    NAN = np.float_(np.nan)
    # resulting interval is always in [0, 1] or is [nan, nan].
    coeff1 = B - A
    coeff2 = A - C
    a = np.dot(coeff1, coeff1)
    c = np.dot(coeff2, coeff2) - eps**2
    if a == 0:  # degenerate case
        if c > 0:
            interval = [NAN, NAN]
        else:
            interval = [np.float_(0), np.float_(1)]
        return interval
    b = 2 * np.dot(coeff1, coeff2)
    Det = b**2 - 4 * a * c
    if Det < 0:
        interval = [NAN, NAN]
    else:
        start = max((-b - Det**0.5) / 2 / a, np.float_(0))
        end = min((-b + Det**0.5) / 2 / a, np.float_(1))
        if start > 1 or end < 0:
            start = end = NAN
        interval = [start, end]
    return interval


@njit
def _decision_problem(P, Q, eps):
    """Algorithm 1 of Alt & Godau (1995)."""
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
            L_start, L_end = _free_interval(Q[j], Q[j + 1], P[i + 1], eps)
            B_start, B_end = _free_interval(P[i], P[i + 1], Q[j + 1], eps)

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

    return L[-1, -1, 1] == 1 or B[-1, -1, 1] == 1


@njit
def _critical_b(A, B, C):
    v = B - A
    w = C - A
    vv = np.dot(v, v)
    if vv == 0:
        return np.linalg.norm(w)
    t = np.dot(v, w) / vv
    if t < 0:
        dist = np.linalg.norm(w)
    elif t > 1:
        dist = np.linalg.norm(C - B)
    else:
        dist = np.linalg.norm(t * v - w)
    return dist


@njit
def _fd(P, Q):
    """Algorithm 3 of Alt & Godau (1995)."""
    crit_a = max(np.linalg.norm(P[0] - Q[0]), np.linalg.norm(P[-1] - Q[-1]))

    crit_b = np.empty(2 * len(P) * len(Q) - len(P) - len(Q) + 1, dtype=np.float_)
    count = 0
    for i in range(len(P) - 1):
        for j in range(len(Q)):
            dist = _critical_b(P[i], P[i + 1], Q[j])
            if dist > crit_a:
                crit_b[count] = dist
                count += 1
    for i in range(len(P)):
        for j in range(len(Q) - 1):
            dist = _critical_b(Q[j], Q[j + 1], P[i])
            if dist > crit_a:
                crit_b[count] = dist
                count += 1
    crit_b[count] = crit_a
    crit_b = np.sort(crit_b[: count + 1])

    # binary search
    start, end = 0, count
    while end - start > 1:
        mid = (start + end) // 2
        mid_reachable = _decision_problem(P, Q, crit_b[mid])
        if mid_reachable:
            end = mid
        else:
            start = mid

    # parametric search
    e1, e2 = crit_b[start], crit_b[end]
    tol = max(e1 / 100, EPSILON)
    while e2 - e1 > tol:
        mid = (e1 + e2) / 2
        mid_reachable = _decision_problem(P, Q, mid)
        if mid_reachable:
            e2 = mid
        else:
            e1 = mid

    return e2


def dfd(P, Q):
    r"""Discrete Fréchet distance between two open polygonal curves.

    Let :math:`\{P_0, P_1, ..., P_n\}` and :math:`\{Q_0, Q_1, ..., Q_m\}` be
    polyline vertices in metris space. The discrete Fréchet distance between
    two polylines is defined as

    .. math::

        \min_{C} \max_{(i, j) \in C} \lVert P_i - Q_j \rVert

    where :math:`C` is a nondecreasing coupling over
    :math:`\{0, ..., n\} \times \{0, ..., m\}`, starting from :math:`(0, 0)` and
    ending with :math:`(n, m)`.

    Parameters
    ----------
    P : array_like
        An :math:`p` by :math:`n` array of :math:`p` vertices in an
        :math:`n`-dimensional space.
    Q : array_like
        An :math:`q` by :math:`n` array of :math:`q` vertices in an
        :math:`n`-dimensional space.

    Returns
    -------
    dist : double
        The discrete Fréchet distance between P and Q.

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
