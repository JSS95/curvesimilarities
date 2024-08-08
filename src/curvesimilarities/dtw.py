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

        \min_{C} \sum_{(i, j) \in C} \lVert P_i - Q_j \rVert,

    where :math:`C` is a nondecreasing coupling over
    :math:`\{0, ..., n\} \times \{0, ..., m\}`, starting from :math:`(0, 0)` and
    ending with :math:`(n, m)`. :math:`\lVert \cdot \rVert` is the underlying
    metric.

    Parameters
    ----------
    P : ndarray
        A :math:`p` by :math:`n` array of :math:`p` points in an :math:`n`-dimensional
        space.
    Q : ndarray
        A :math:`q` by :math:`n` array of :math:`q` points in an :math:`n`-dimensional
        space.
    dist : {'euclidean', 'squared_euclidean'}
        Type of underlying metric. Refer to the Notes section for more information.

    Returns
    -------
    double
        The dynamic time warping distance between *P* and *Q*, NaN if any
        array of points is empty.

    See Also
    --------
    dtw_owp

    Notes
    -----
    This function implements the algorithm described by Senin [1]_.

    The following functions are available for :math:`\lVert \cdot \rVert`:

    1. Euclidean distance
        .. math::

            \lVert p - q \rVert = \lVert p - q \rVert_2

    2. Squared Euclidean distance
        .. math::

            \lVert p - q \rVert = \lVert p - q \rVert_2^2

    References
    ----------
    .. [1] Senin, P. (2008). Dynamic time warping algorithm review. Information
        and Computer Science Department University of Hawaii at Manoa Honolulu,
        USA, 855(1-23), 40.

    Examples
    --------
    >>> P = np.linspace([0, 0], [1, 0], 10)
    >>> Q = np.linspace([0, 1], [1, 1], 20)
    >>> dtw(P, Q)
    20.0...
    """
    acm = _dtw_acm_1d(P.astype(np.float64), Q.astype(np.float64), dist)
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
        A :math:`p` by :math:`n` array of :math:`p` points in an :math:`n`-dimensional
        space.
    Q : ndarray
        A :math:`q` by :math:`n` array of :math:`q` points in an :math:`n`-dimensional
        space.
    dist : {'euclidean', 'squared_euclidean'}
        Type of underlying metric. Refer to :func:`dtw`.

    Returns
    -------
    dtw : double
        The dynamic time warping distance between *P* and *Q*, NaN if any
        array of points is empty.
    owp : ndarray
        Indices of *P* and *Q* for optimal warping path, empty if any array of points
        empty.

    Examples
    --------
    >>> P = np.array([[0, 0], [2, 2], [4, 2], [4, 4], [2, 1], [5, 1], [7, 2]])
    >>> Q = np.array([[2, 0], [1, 3], [5, 3], [5, 2], [7, 3]])
    >>> _, owp = dtw_owp(P, Q)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> x, y = np.meshgrid(np.arange(len(P)), np.arange(len(Q)))
    >>> plt.plot(*np.vstack([x.ravel(), y.ravel()]), "x")  # doctest: +SKIP
    >>> plt.plot(*owp.T, "o")  # doctest: +SKIP
    >>> plt.axis("equal")  # doctest: +SKIP
    """
    acm = _dtw_acm(P.astype(np.float64), Q.astype(np.float64), dist)
    if acm.size == 0:
        ret = NAN, np.empty((0, 2), dtype=np.int_)
    else:
        ret = acm[-1, -1], _dtw_owp(acm)
    return ret
