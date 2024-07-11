"""Continuous and discrete Fréchet distances."""

import numpy as np
from numba import njit
from scipy.spatial.distance import cdist

from .util import sanitize_vertices

__all__ = [
    "fd",
    "dfd",
    "dfd_idxs",
]


EPSILON = np.finfo(np.float64).eps
NAN = np.float64(np.nan)


@sanitize_vertices(owp=False)
def fd(P, Q, *, rel_tol=0.0, abs_tol=float(EPSILON)):
    r"""(Continuous) Fréchet distance between two open polygonal curves.

    Let :math:`f: [0, 1] \to \Omega` and :math:`g: [0, 1] \to \Omega` be curves
    where :math:`\Omega` is a metric space. The Fréchet distance between
    :math:`f` and :math:`g` is defined as

    .. math::

        \inf_{\alpha, \beta} \max_{t \in [0, 1]}
        \lVert f(\alpha(t)) - g(\beta(t)) \rVert,

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
    rel_tol, abs_tol : double
        Relative and absolute tolerances for parametric search of the Fréchet distance.
        The search is terminated if the upper boundary ``a`` and the lower boundary
        ``b`` satisfy: ``a - b <= max(rel_tol * a, abs_tol)``.
        If both values are zero, parametric search is disabled and analytic solution is
        returned.

    Returns
    -------
    dist : double
        The (continuous) Fréchet distance between *P* and *Q*, NaN if any vertice
        is empty.

    Raises
    ------
    ValueError
        If *P* and *Q* are not 2-dimensional arrays with same number of columns.

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
    return float(_fd(P, Q, rel_tol, abs_tol))


@njit(cache=True)
def _free_interval(A, B, C, eps):
    # resulting interval is always in [0, 1] or is [nan, nan].
    coeff1 = B - A
    coeff2 = A - C
    a = np.dot(coeff1, coeff1)
    c = np.dot(coeff2, coeff2) - eps**2
    if a == 0:  # degenerate case
        if c > 0:
            interval = [NAN, NAN]
        else:
            interval = [np.float64(0), np.float64(1)]
        return interval
    b = 2 * np.dot(coeff1, coeff2)
    Det = b**2 - 4 * a * c
    if Det < 0:
        interval = [NAN, NAN]
    else:
        start = max((-b - Det**0.5) / 2 / a, np.float64(0))
        end = min((-b + Det**0.5) / 2 / a, np.float64(1))
        if start > 1 or end < 0:
            start = end = NAN
        interval = [start, end]
    return interval


@njit(cache=True)
def _decision_problem(P, Q, eps):
    """Algorithm 1 of Alt & Godau (1995)."""
    # Decide reachablilty
    B = np.empty((len(P) - 1, len(Q), 2), dtype=np.float64)
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

    L = np.empty((len(P), len(Q) - 1, 2), dtype=np.float64)
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


@njit(cache=True)
def _critical_b(A, B, P):
    v = B - A
    w = P - A
    vv = np.dot(v, v)
    if vv == 0:
        return np.linalg.norm(w)
    t = np.dot(v, w) / vv
    if t < 0:
        dist = np.linalg.norm(w)
    elif t > 1:
        dist = np.linalg.norm(P - B)
    else:
        dist = np.linalg.norm(t * v - w)
    return dist


@njit(cache=True)
def _critical_c(A, B, P1, P2):
    M = (P1 + P2) / 2
    AB = B - A
    MA = A - M
    PP = P2 - P1

    a = np.dot(AB, PP)
    b = np.dot(MA, PP)
    if a == 0:
        ret = NAN
    else:
        t = -b / a
        if t < 0 or t > 1:
            ret = NAN
        else:
            MT = AB * t + MA
            ret = np.linalg.norm(MT)
    return ret


@njit(cache=True)
def _fd(P, Q, rel_tol, abs_tol):
    """Algorithm 3 of Alt & Godau (1995)."""
    LP, LQ = len(P), len(Q)
    ANALYTIC = rel_tol == 0 and abs_tol == 0

    MAX_A = max(np.linalg.norm(P[0] - Q[0]), np.linalg.norm(P[-1] - Q[-1]))
    crit_a = np.array((MAX_A,))

    crit_b = np.empty(LP * (LQ - 1) + (LP - 1) * LQ, dtype=np.float64)
    count = 0
    for i in range(LP - 1):
        for j in range(LQ):
            dist = _critical_b(P[i], P[i + 1], Q[j])
            if dist > MAX_A:
                crit_b[count] = dist
                count += 1
    for i in range(LP):
        for j in range(LQ - 1):
            dist = _critical_b(Q[j], Q[j + 1], P[i])
            if dist > MAX_A:
                crit_b[count] = dist
                count += 1
    crit_b = crit_b[:count]

    if ANALYTIC:
        crit_c = np.empty(
            int(LP * (LP - 1) * (LQ - 1) / 2 + LQ * (LQ - 1) * (LP - 1) / 2),
            dtype=np.float64,
        )
        count = 0
        for i in range(LP):
            for j in range(i + 1, LP):
                for k in range(LQ - 1):
                    dist = _critical_c(Q[k], Q[k + 1], P[i], P[j])
                    if dist > MAX_A:
                        crit_c[count] = dist
                        count += 1
        for i in range(LQ):
            for j in range(i + 1, LQ):
                for k in range(LP - 1):
                    dist = _critical_c(P[k], P[k + 1], Q[i], Q[j])
                    if dist > MAX_A:
                        crit_c[count] = dist
                        count += 1
        crit_c = crit_c[:count]

        crit = np.sort(np.concatenate((crit_a, crit_b, crit_c)))
    else:
        crit = np.sort(np.concatenate((crit_a, crit_b)))

    # binary search
    start, end = 0, len(crit) - 1
    if _decision_problem(P, Q, crit[start]):
        end = start
    while end - start > 1:
        mid = (start + end) // 2
        mid_reachable = _decision_problem(P, Q, crit[mid])
        if mid_reachable:
            end = mid
        else:
            start = mid

    if ANALYTIC:
        ret = crit[end]
    else:
        # parametric search
        e1, e2 = crit[start], crit[end]
        while e2 - e1 > max(rel_tol * e2, abs_tol):
            mid = (e1 + e2) / 2
            if (mid - e1 < EPSILON) or (e2 - mid < EPSILON):
                break
            mid_reachable = _decision_problem(P, Q, mid)
            if mid_reachable:
                e2 = mid
            else:
                e1 = mid
        ret = e2
    return ret


@sanitize_vertices(owp=False)
def dfd(P, Q):
    r"""Discrete Fréchet distance between two two ordered sets of points.

    Let :math:`\{P_0, P_1, ..., P_n\}` and :math:`\{Q_0, Q_1, ..., Q_m\}` be ordered
    sets of points in metric space. The discrete Fréchet distance between two sets is
    defined as

    .. math::

        \min_{C} \max_{(i, j) \in C} \lVert P_i - Q_j \rVert,

    where :math:`C` is a nondecreasing coupling over
    :math:`\{0, ..., n\} \times \{0, ..., m\}`, starting from :math:`(0, 0)` and
    ending with :math:`(n, m)`. :math:`\lVert \cdot \rVert` is the underlying
    metric, which is the Euclidean metric in this implementation.

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
        The discrete Fréchet distance between *P* and *Q*, NaN if any vertice
        is empty.

    Raises
    ------
    ValueError
        If *P* and *Q* are not 2-dimensional arrays with same number of columns.

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
    dist = cdist(P, Q)
    return float(_dfd_ca(dist)[-1, -1])


def dfd_idxs(P, Q):
    """Discrete Fréchet distance and its indices in curve space.

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
    d : double
        The discrete Fréchet distance between *P* and *Q*, NaN if any vertice
        is empty.
    index_1 : int
        Index of point contributing to discrete Fréchet distance in *P*.
    index_2 : int
        Index of point contributing to discrete Fréchet distance in *Q*.
    """
    dist = cdist(P, Q)
    ca = _dfd_ca(dist)
    index_1, index_2 = _dfd_idxs(ca)
    return float(ca[-1, -1]), int(index_1), int(index_2)


@njit(cache=True)
def _dfd_ca(dist):
    # Eiter, T., & Mannila, H. (1994)
    p, q = dist.shape
    ret = np.empty((p, q), dtype=np.float64)

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


@njit(cache=True)
def _dfd_idxs(ca):
    p, q = ca.shape
    i, j = p - 1, q - 1

    while i > 0 or j > 0:
        current = ca[i, j]
        LEFT = np.inf if i == 0 else ca[i - 1, j]
        DOWN = np.inf if j == 0 else ca[i, j - 1]
        DIAG = np.inf if (i == 0 or j == 0) else ca[i - 1, j - 1]
        prev = min(LEFT, DOWN, DIAG)
        if current > prev:
            break
        elif current == LEFT:
            i -= 1
        elif current == DOWN:
            j -= 1
        else:
            i -= 1
            j -= 1
    return (i, j)
