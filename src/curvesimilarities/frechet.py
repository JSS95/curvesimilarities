"""Continuous and discrete Fréchet distances."""

import numpy as np
from numba import njit

from ._algorithms.dfd import _dfd_ca, _dfd_ca_1d, _dfd_idxs
from ._algorithms.fd import _fd, _reachable_boundaries_1d
from ._algorithms.lcfm import _computeLCFM, _significant_events
from .util import index2arclength

__all__ = [
    "fd",
    "decision_problem",
    "significant_events",
    "fd_matching",
    "dfd",
    "dfd_idxs",
]


EPSILON = np.finfo(np.float64).eps
NAN = np.float64(np.nan)


@njit(cache=True)
def fd(P, Q, rel_tol=0.0, abs_tol=float(EPSILON)):
    r"""(Continuous) Fréchet distance between two open polylines.

    Let :math:`f: [0, 1] \to \Omega` and :math:`g: [0, 1] \to \Omega` be curves
    in a metric space :math:`\Omega`. The Fréchet distance between
    :math:`f` and :math:`g` is defined as

    .. math::

        \inf_{\alpha, \beta} \max_{t \in [0, 1]}
        \lVert f(\alpha(t)) - g(\beta(t)) \rVert,

    where :math:`\alpha, \beta: [0, 1] \to [0, 1]` are continuous non-decreasing
    surjections and :math:`\lVert \cdot \rVert` is the underlying metric.
    The Euclidean metric is used in this function.

    Parameters
    ----------
    P : array_like
        A :math:`p` by :math:`n` array of :math:`p` vertices of a polyline in an
        :math:`n`-dimensional space.
    Q : array_like
        A :math:`q` by :math:`n` array of :math:`q` vertices of a polyline in an
        :math:`n`-dimensional space.
    rel_tol, abs_tol : double
        Relative and absolute tolerances for parametric search of the Fréchet distance.

    Returns
    -------
    dist : double
        The (continuous) Fréchet distance between *P* and *Q*, NaN if any vertex
        is empty.

    See Also
    --------
    decision_problem
    significant_events
    fd_matching

    Notes
    -----
    This function implements Alt and Godau's algorithm [1]_.

    The parametric search for the Fréchet distance terminates if the upper boundary
    ``a`` and the lower boundary ``b`` satisfy: ``a - b <= max(rel_tol * a, abs_tol)``.

    References
    ----------
    .. [1] Alt, H., & Godau, M. (1995). Computing the Fréchet distance between
       two polygonal curves. International Journal of Computational Geometry &
       Applications, 5(01n02), 75-91.

    Examples
    --------
    >>> P, Q = [[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]]
    >>> fd(np.asarray(P), np.asarray(Q))
    1.0...
    """
    return _fd(P.astype(np.float64), Q.astype(np.float64), rel_tol, abs_tol)


@njit(cache=True)
def decision_problem(P, Q, epsilon):
    """Decision problem of the (continuous) Fréchet distance.

    Parameters
    ----------
    P : array_like
        A :math:`p` by :math:`n` array of :math:`p` vertices of a polyline in an
        :math:`n`-dimensional space.
    Q : array_like
        A :math:`q` by :math:`n` array of :math:`q` vertices of a polyline in an
        :math:`n`-dimensional space.
    epsilon : double
        Minimum distance to be checked.

    Returns
    -------
    bool
        True if *epsilon* is greater than or equal to the Fréchet distance between
        *P* and *Q*, false otherwise.
    """
    if len(P.shape) != 2:
        raise ValueError("P must be a 2-dimensional array.")
    if len(Q.shape) != 2:
        raise ValueError("Q must be a 2-dimensional array.")
    if P.shape[1] != Q.shape[1]:
        raise ValueError("P and Q must have the same number of columns.")

    B, L = _reachable_boundaries_1d(P.astype(np.float64), Q.astype(np.float64), epsilon)
    if B[-1, 1] == 1 or L[-1, 1] == 1:
        ret = True
    else:
        ret = False
    return ret


@njit(cache=True)
def significant_events(
    P,
    Q,
    param="arc-length",
    rel_tol=0.0,
    abs_tol=float(EPSILON),
    event_rel_tol=0.0,
    event_abs_tol=float(EPSILON),
):
    """Return significant events of the (continuous) Fréchet distance [1]_ .

    A significant event is a matching which determines the Fréchet distance.

    Parameters
    ----------
    P : array_like
        A :math:`p` by :math:`n` array of :math:`p` vertices of a polyline in an
        :math:`n`-dimensional space.
    Q : array_like
        A :math:`q` by :math:`n` array of :math:`q` vertices of a polyline in an
        :math:`n`-dimensional space.
    param : {'arc-length', 'vertex'}
        Parametrization of matching.
    rel_tol, abs_tol : double
        Relative and absolute tolerances for parametric search of the feasible distance.
    event_rel_tol, event_abs_tol : double
        Relative and absolute tolerances to determine realizing events.

    Returns
    -------
    events : ndarray
        :math:`N` significant events in a :math:`(N, 2, 2)`-shaped array of parameters.
        The second axis is the starting and ending points of the event.
        The last axis is the parameters of *P* and *Q*.
    errors : ndarray
        Difference between the Fréchet distance and the values of the events.

    Notes
    -----
    This function implements Buchin et al.'s algorithm [1]_, except that backtracking is
    used to extract significant events. Plus, the "type-A" events are included.

    The parametric search for the feasible distance terminates if the upper boundary
    ``a`` and the lower boundary ``b`` satisfy: ``a - b <= max(rel_tol * a, abs_tol)``.

    An event is considered to be realizing if its value `e` and the realizing distnace
    `d` satisfy: ``e - d <= max(rel_tol * d, abs_tol)``.

    References
    ----------
    .. [1] Buchin, K., et al. "Locally correct Fréchet matchings."
       Computational Geometry 76 (2019): 1-18.

    Examples
    --------
    >>> from curvesimilarities.frechet import significant_events
    >>> from curvesimilarities.util import parameter_space
    >>> P = np.array([[0, 0], [2, 2], [4, 2], [4, 4], [2, 1], [5, 1], [7, 2]])
    >>> Q = np.array([[2, 0], [1, 3], [5, 3], [5, 2], [7, 3]])
    >>> events, _ = significant_events(P, Q)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> weight, p, q, _, _ = parameter_space(P, Q, 200, 100)
    >>> plt.pcolormesh(p, q, weight.T < fd(P, Q), cmap="gray")  # doctest: +SKIP
    >>> plt.plot(*events.transpose(2, 1, 0), "o-")  # doctest: +SKIP
    """
    P, Q = P.astype(np.float64), Q.astype(np.float64)
    p, q = len(P), len(Q)

    events = np.empty((p * q, 2, 2), dtype=np.float64)
    errors = np.empty(p * q, dtype=np.float64)
    count = 0
    if p == 0 or q == 0:
        return events, errors

    A0 = np.linalg.norm(P[0] - Q[0])
    A1 = np.linalg.norm(P[-1] - Q[-1])
    if p <= 2 and q <= 2:
        d = max(A0, A1)
    else:
        eps, BE, LE, BE_val, LE_val, _, _ = _significant_events(
            P, Q, rel_tol, abs_tol, event_rel_tol, event_abs_tol
        )
        d = max(eps, A0, A1)
        if abs(d - eps) <= max(event_rel_tol * max(abs(eps), abs(d)), event_abs_tol):
            # BE and LE are significant events (if exist).
            for i in range(len(BE)):
                it, j0, j1 = BE[i]
                events[count, 0] = [it, j0]
                events[count, 1] = [it, j1]
                errors[count] = abs(d - BE_val[i])
                count += 1
            for j in range(len(LE)):
                i0, i1, jt = LE[j]
                events[count, 0] = [i0, jt]
                events[count, 1] = [i1, jt]
                errors[count] = abs(d - LE_val[i])
                count += 1
    if abs(d - A0) <= max(event_rel_tol * max(abs(A0), abs(d)), event_abs_tol):
        # A0 is a significant event.
        events[count, 0] = [0, 0]
        events[count, 1] = [0, 0]
        errors[count] = abs(d - A0)
        count += 1
    if abs(d - A1) <= max(event_rel_tol * max(abs(A1), abs(d)), event_abs_tol):
        # A1 is a significant event.
        events[count, 0] = [p - 1, q - 1]
        events[count, 1] = [p - 1, q - 1]
        errors[count] = abs(d - A1)
        count += 1
    events = events[:count]
    errors = errors[:count]

    if param == "arc-length":
        events = np.stack(
            (
                index2arclength(P, events[:, :, 0].copy()),
                index2arclength(Q, events[:, :, 1].copy()),
            )
        ).transpose(1, 2, 0)
    elif param == "vertex":
        pass
    else:
        raise ValueError("Unknown option for parametrization.")

    return events, errors


@njit(cache=True)
def fd_matching(
    P,
    Q,
    param="arc-length",
    rel_tol=0.0,
    abs_tol=float(EPSILON),
    event_rel_tol=0.0,
    event_abs_tol=float(EPSILON),
):
    """Locally correct Fréchet matching [1]_.

    A locally correct Fréchet matching is a Fréchet matching between two curves whose
    any sub-matching is still a Fréchet matching.

    Parameters
    ----------
    P : array_like
        A :math:`p` by :math:`n` array of :math:`p` vertices of a polyline in an
        :math:`n`-dimensional space.
    Q : array_like
        A :math:`q` by :math:`n` array of :math:`q` vertices of a polyline in an
        :math:`n`-dimensional space.
    param : {'arc-length', 'vertex'}
        Parametrization of matching.
    rel_tol, abs_tol : double
        Relative and absolute tolerances for parametric search of the Fréchet distance.
    event_rel_tol, event_abs_tol : double
        Relative and absolute tolerances to determine realizing events.

    Returns
    -------
    dist : double
        The (continuous) Fréchet distance between *P* and *Q*, NaN if any vertex
        is empty.
    matching : ndarray
        Vertices of a locally correct Fréchet matching in parameter space.

    Notes
    -----
    This function implements Buchin et al.'s algorithm [1]_, except that backtracking is
    used to extract significant events.

    The parametric search for the feasible distance terminates if the upper boundary
    ``a`` and the lower boundary ``b`` satisfy: ``a - b <= max(rel_tol * a, abs_tol)``.

    An event is considered to be realizing if its value `e` and the realizing distnace
    `d` satisfy: ``e - d <= max(rel_tol * d, abs_tol)``.

    References
    ----------
    .. [1] Buchin, Kevin, et al. "Locally correct Fréchet matchings."
       Computational Geometry 76 (2019): 1-18.

    Examples
    --------
    >>> from curvesimilarities.frechet import fd_matching
    >>> from curvesimilarities.util import parameter_space
    >>> P = np.array([[0, 0], [2, 2], [4, 2], [4, 4], [2, 1], [5, 1], [7, 2]])
    >>> Q = np.array([[2, 0], [1, 3], [5, 3], [5, 2], [7, 3]])
    >>> dist, path = fd_matching(P, Q)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> weight, p, q, _, _ = parameter_space(P, Q, 200, 100)
    >>> plt.pcolormesh(p, q, weight.T < dist, cmap="gray")  # doctest: +SKIP
    >>> plt.plot(*path.T, "--")  # doctest: +SKIP
    """
    P, Q = P.astype(np.float64), Q.astype(np.float64)
    eps, matching = _computeLCFM(P, Q, rel_tol, abs_tol, event_rel_tol, event_abs_tol)
    if not np.isnan(eps):
        dist = max(eps, np.linalg.norm(P[0] - Q[0]), np.linalg.norm(P[-1] - Q[-1]))
    else:
        dist = max(np.linalg.norm(P[0] - Q[0]), np.linalg.norm(P[-1] - Q[-1]))
    if param == "arc-length":
        matching = np.stack(
            (
                index2arclength(P, matching[:, 0].copy()),
                index2arclength(Q, matching[:, 1].copy()),
            )
        ).T
    elif param == "vertex":
        pass
    else:
        raise ValueError("Unknown option for parametrization.")
    return dist, matching


@njit(cache=True)
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
    metric, which is the Euclidean metric in this function.

    Parameters
    ----------
    P : ndarray
        A :math:`p` by :math:`n` array of :math:`p` points in an :math:`n`-dimensional
        space.
    Q : ndarray
        A :math:`q` by :math:`n` array of :math:`q` points in an :math:`n`-dimensional
        space.

    Returns
    -------
    dist : double
        The discrete Fréchet distance between *P* and *Q*, NaN if any array of points
        is empty.

    Notes
    -----
    This function implements Eiter and Mannila's algorithm [1]_.

    References
    ----------
    .. [1] Eiter, T., & Mannila, H. (1994). Computing discrete Fréchet distance.

    Examples
    --------
    >>> P, Q = [[0, 0], [1, 1], [2, 0]], [[0, 1], [2, -4]]
    >>> dfd(np.asarray(P), np.asarray(Q))
    4.0
    """
    ca = _dfd_ca_1d(P.astype(np.float64), Q.astype(np.float64))
    if ca.size == 0:
        ret = NAN
    else:
        ret = ca[-1]
    return ret


@njit(cache=True)
def dfd_idxs(P, Q):
    """Discrete Fréchet distance and its indices in curve space.

    Parameters
    ----------
    P : ndarray
        An :math:`p` by :math:`n` array of :math:`p` points in an
        :math:`n`-dimensional space.
    Q : ndarray
        An :math:`q` by :math:`n` array of :math:`q` points in an
        :math:`n`-dimensional space.

    Returns
    -------
    d : double
        The discrete Fréchet distance between *P* and *Q*, NaN if any array of points
        is empty.
    index_1 : int
        Index of point contributing to discrete Fréchet distance in *P*.
    index_2 : int
        Index of point contributing to discrete Fréchet distance in *Q*.

    Examples
    --------
    >>> P = np.array([[0, 0], [2, 2], [4, 2], [4, 4], [2, 1], [5, 1], [7, 2]])
    >>> Q = np.array([[2, 0], [1, 3], [5, 3], [5, 2], [7, 3]])
    >>> from curvesimilarities.util import sample_polyline
    >>> P_len = np.sum(np.linalg.norm(np.diff(P, axis=0), axis=-1))
    >>> P_pts = sample_polyline(P, np.linspace(P_len, 0, 30))
    >>> Q_len = np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=-1))
    >>> Q_pts = sample_polyline(Q, np.linspace(Q_len, 0, 30))
    >>> _, idx0, idx1 = dfd_idxs(P_pts, Q_pts)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> plt.plot(*P_pts.T, "x"); plt.plot(*Q_pts.T, "x")  # doctest: +SKIP
    >>> plt.plot(*np.array([P_pts[idx0], Q_pts[idx1]]).T, "--")  # doctest: +SKIP
    """
    ca = _dfd_ca(P.astype(np.float64), Q.astype(np.float64))
    if ca.size == 0:
        ret = NAN, -1, -1
    else:
        index_1, index_2 = _dfd_idxs(ca)
        ret = ca[-1, -1], int(index_1), int(index_2)
    return ret
