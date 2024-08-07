"""Continuous and discrete Fréchet distances."""

import numpy as np
from numba import njit

from ._algorithms.dfd import _dfd_ca, _dfd_ca_1d, _dfd_idxs
from ._algorithms.fd import _fd, _fd_params, _reachable_boundaries_1d
from ._algorithms.lcfm import _computeLCFM

__all__ = [
    "decision_problem",
    "fd",
    "fd_matching",
    "fd_params",
    "dfd",
    "dfd_idxs",
]


EPSILON = np.finfo(np.float64).eps
NAN = np.float64(np.nan)


@njit(cache=True)
def decision_problem(P, Q, epsilon):
    """Decision problem of the (continuous) Fréchet distance.

    Parameters
    ----------
    P : array_like
        A :math:`p` by :math:`n` array of :math:`p` vertices in an
        :math:`n`-dimensional space.
    Q : array_like
        A :math:`q` by :math:`n` array of :math:`q` vertices in an
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

    P, Q = P.astype(np.float64), Q.astype(np.float64)
    B, L = _reachable_boundaries_1d(P, Q, epsilon)
    if B[-1, 1] == 1 or L[-1, 1] == 1:
        ret = True
    else:
        ret = False
    return ret


@njit(cache=True)
def fd(P, Q, rel_tol=0.0, abs_tol=float(EPSILON)):
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
    >>> P, Q = [[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]]
    >>> fd(np.asarray(P), np.asarray(Q))
    1.0...
    """
    return _fd(P, Q, rel_tol, abs_tol)


@njit(cache=True)
def fd_matching(
    P,
    Q,
    rel_tol=0.0,
    abs_tol=float(EPSILON),
    event_rel_tol=0.0,
    event_abs_tol=float(EPSILON),
):
    """Locally correct Fréchet matching [1]_.

    A locally correct Fréchet matching is a matching between two curves whose any
    sub-matching is still a Fréchet matching.

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
    event_rel_tol, event_abs_tol : double
        Relative and absolute tolerances to determine realizing events.

    Returns
    -------
    dist : double
        The (continuous) Fréchet distance between *P* and *Q*.
    matching : ndarray
        Vertices of a locally correct Fréchet matching in parameter space.

    References
    ----------
    .. [1] Buchin, Kevin, et al. "Locally correct Fréchet matchings."
       Computational Geometry 76 (2019): 1-18.
    """
    P, Q = P.astype(np.float64), Q.astype(np.float64)
    eps, matching = _computeLCFM(P, Q, rel_tol, abs_tol, event_rel_tol, event_abs_tol)
    if len(P) > 2 or len(Q) > 2:
        dist = max(eps, np.linalg.norm(P[0] - Q[0]), np.linalg.norm(P[-1] - Q[-1]))
    else:
        dist = eps
    return dist, matching


@njit(cache=True)
def fd_params(P, Q, rel_tol=0.0, abs_tol=float(EPSILON)):
    """(Continuous) Fréchet distance and its parameters in curve space.

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

    Returns
    -------
    dist : double
        The (continuous) Fréchet distance between *P* and *Q*, NaN if any vertice
        is empty.
    param : ndarray
        Parameters of critical points contributing to Fréchet distance.

    Notes
    -----
    The resulting parameters adopt arc-length parametrization [#]_.

    References
    ----------
    .. [#] https://en.wikipedia.org/wiki/Differentiable_curve
           #Length_and_natural_parametrization

    Examples
    --------
    >>> P = np.array([[0, 0], [2, 2], [4, 2], [4, 4], [2, 1], [5, 1], [7, 2]])
    >>> Q = np.array([[2, 0], [1, 3], [5, 3], [5, 2], [7, 3]])
    >>> _, params = fd_params(P, Q)
    >>> from curvesimilarities.util import sample_polyline
    >>> pts = [sample_polyline(P, params[:, 0]), sample_polyline(Q, params[:, 1])]
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> plt.plot(*P.T); plt.plot(*Q.T)  # doctest: +SKIP
    >>> plt.plot(*np.array(pts).transpose(2, 0, 1), "--", color="k")  # doctest: +SKIP
    """
    return _fd_params(P, Q, rel_tol, abs_tol)


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
    metric, which is the Euclidean metric in this implementation.

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
    dist : double
        The discrete Fréchet distance between *P* and *Q*, NaN if any array of points
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
    >>> P, Q = [[0, 0], [1, 1], [2, 0]], [[0, 1], [2, -4]]
    >>> dfd(np.asarray(P), np.asarray(Q))
    4.0
    """
    ca = _dfd_ca_1d(P, Q)
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
    ca = _dfd_ca(P, Q)
    if ca.size == 0:
        ret = NAN, -1, -1
    else:
        index_1, index_2 = _dfd_idxs(ca)
        ret = ca[-1, -1], int(index_1), int(index_2)
    return ret
