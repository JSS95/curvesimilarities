"""Continuous and discrete Fréchet distances."""

import numpy as np
from numba import njit

from ._algorithms.dfd import _dfd_ca, _dfd_ca_1d, _dfd_idxs
from ._algorithms.fd import _fd

__all__ = [
    "fd",
    "dfd",
    "dfd_idxs",
]


EPSILON = np.finfo(np.float64).eps
NAN = np.float64(np.nan)


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
    >>> P, Q = [[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]]
    >>> fd(np.asarray(P), np.asarray(Q))
    1.0...
    """
    return _fd(P, Q, rel_tol, abs_tol)


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
        An :math:`p` by :math:`n` array of :math:`p` vertices in an
        :math:`n`-dimensional space.
    Q : ndarray
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
        An :math:`p` by :math:`n` array of :math:`p` vertices in an
        :math:`n`-dimensional space.
    Q : ndarray
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
    ca = _dfd_ca(P, Q)
    if ca.size == 0:
        ret = NAN, -1, -1
    else:
        index_1, index_2 = _dfd_idxs(ca)
        ret = ca[-1, -1], int(index_1), int(index_2)
    return ret
