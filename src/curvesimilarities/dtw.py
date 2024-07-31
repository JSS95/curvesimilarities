"""Dynamic time warping distance.

This module implements only the basic algorithm. If you need advanced features, use
dedicated packages such as `dtw-python
<https://pypi.org/project/dtw-python/>`_.
"""

import numpy as np
from numba import njit

from ._algorithms.dtw import _dtw_acm, _dtw_acm_1d, _dtw_owp

__all__ = [
    "dtw",
    "dtw_owp",
]


NAN = np.float64(np.nan)


@njit(cache=True)
def dtw(P, Q, dist="euclidean"):
    r"""Dynamic time warping distance between two ordered sets of points.

    Let :math:`\{P_0, P_1, ..., P_n\}` and :math:`\{Q_0, Q_1, ..., Q_m\}` be ordered
    sets of points in metric space. The dynamic time warping distance between
    two sets is defined as

    .. math::

        \min_{C} \sum_{(i, j) \in C} dist\left(P_i, Q_j\right),

    where :math:`C` is a nondecreasing coupling over
    :math:`\{0, ..., n\} \times \{0, ..., m\}`, starting from :math:`(0, 0)` and
    ending with :math:`(n, m)`.

    Parameters
    ----------
    P : ndarray
        A :math:`p` by :math:`n` array of :math:`p` points in an
        :math:`n`-dimensional space.
    Q : ndarray
        A :math:`q` by :math:`n` array of :math:`q` points in an
        :math:`n`-dimensional space.
    dist : {'euclidean', 'squared_euclidean'}
        Type of :math:`dist`. Refer to the Notes section for more information.

    Returns
    -------
    double
        The dynamic time warping distance between *P* and *Q*, NaN if any
        array of points is empty.

    Raises
    ------
    ValueError
        If *P* and *Q* are not 2-dimensional arrays with same number of columns.

    See Also
    --------
    dtw_owp : Dynamic time warping distance with optimal warping path.

    Notes
    -----
    This function implements the algorithm described by Senin [#]_.

    The following functions are available for :math:`dist`:

    1. Euclidean distance
        .. math::

            dist\left(p, q\right) = \lVert p - q \rVert_2

    2. Squared Euclidean distance
        .. math::

            dist\left(p, q\right) = \lVert p - q \rVert_2^2

    References
    ----------
    .. [#] Senin, P. (2008). Dynamic time warping algorithm review. Information
        and Computer Science Department University of Hawaii at Manoa Honolulu,
        USA, 855(1-23), 40.

    Examples
    --------
    >>> P = np.linspace([0, 0], [1, 0], 10)
    >>> Q = np.linspace([0, 1], [1, 1], 20)
    >>> dtw(P, Q)
    20.0...
    """
    acm = _dtw_acm_1d(P, Q, dist)
    if acm.size == 0:
        ret = NAN
    else:
        ret = acm[-1]
    return ret


@njit(cache=True)
def dtw_owp(P, Q, dist="euclidean"):
    """Dynamic time warping distance and its optimal warping path.

    Parameters
    ----------
    P : ndarray
        A :math:`p` by :math:`n` array of :math:`p` points in an
        :math:`n`-dimensional space.
    Q : ndarray
        A :math:`q` by :math:`n` array of :math:`q` points in an
        :math:`n`-dimensional space.
    dist : {'euclidean', 'squared_euclidean'}
        Type of :math:`dist`. Refer to :func:`dtw`.

    Returns
    -------
    dtw : double
        The dynamic time warping distance between *P* and *Q*, NaN if any
        array of points is empty.
    owp : ndarray
        Indices of *P* and *Q* for optimal warping path, empty if any array of points
        empty.

    Raises
    ------
    ValueError
        If *P* and *Q* are not 2-dimensional arrays with same number of columns.

    Examples
    --------
    >>> P = np.array([[0, 0], [2, 2], [4, 2], [4, 4], [2, 1], [5, 1], [7, 2]])
    >>> Q = np.array([[2, 0], [1, 3], [5, 3], [5, 2], [7, 3]])
    >>> from curvesimilarities.util import sample_polyline
    >>> P_len = np.sum(np.linalg.norm(np.diff(P, axis=0), axis=-1))
    >>> P_pts = sample_polyline(P, np.linspace(P_len, 0, 30))
    >>> Q_len = np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=-1))
    >>> Q_pts = sample_polyline(Q, np.linspace(Q_len, 0, 30))
    >>> _, owp = dtw_owp(P_pts, Q_pts)
    >>> lines = np.array([P_pts[owp[:, 0]], Q_pts[owp[:, 1]]])
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> plt.plot(*P_pts.T, "x"); plt.plot(*Q_pts.T, "x")  # doctest: +SKIP
    >>> plt.plot(*lines.transpose(2, 0, 1), "--", color="gray")  # doctest: +SKIP
    """
    acm = _dtw_acm(P, Q, dist)
    if acm.size == 0:
        ret = NAN, np.empty((0, 2), dtype=np.int_)
    else:
        ret = acm[-1, -1], _dtw_owp(acm)
    return ret
