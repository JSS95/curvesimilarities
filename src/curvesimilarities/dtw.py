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
        A :math:`p` by :math:`n` array of :math:`p` vertices in an
        :math:`n`-dimensional space.
    Q : ndarray
        A :math:`q` by :math:`n` array of :math:`q` vertices in an
        :math:`n`-dimensional space.
    dist : {'euclidean', 'squared_euclidean'}
        Type of :math:`dist`. Refer to the Notes section for more information.

    Returns
    -------
    double
        The dynamic time warping distance between *P* and *Q*, NaN if any vertice
        is empty.

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
        A :math:`p` by :math:`n` array of :math:`p` vertices in an
        :math:`n`-dimensional space.
    Q : ndarray
        A :math:`q` by :math:`n` array of :math:`q` vertices in an
        :math:`n`-dimensional space.
    dist : {'euclidean', 'squared_euclidean'}
        Type of :math:`dist`. Refer to :func:`dtw`.

    Returns
    -------
    dtw : double
        The dynamic time warping distance between *P* and *Q*, NaN if any vertice
        is empty.
    owp : ndarray
        Indices of *P* and *Q* for optimal warping path, empty if any vertice is
        empty.

    Raises
    ------
    ValueError
        If *P* and *Q* are not 2-dimensional arrays with same number of columns.

    Examples
    --------
    >>> P = np.linspace([0, 0], [1, 0], 10)
    >>> Q = np.linspace([0, 1], [1, 1], 20)
    >>> dist, path = dtw_owp(P, Q)
    >>> dist / len(path)  # averaged dynamic time warping
    1.00...
    >>> import matplotlib.pyplot as plt #doctest: +SKIP
    >>> plt.plot(*path.T, "x")  #doctest: +SKIP
    """
    acm = _dtw_acm(P, Q, dist)
    if acm.size == 0:
        ret = NAN, np.empty((0, 2), dtype=np.int_)
    else:
        ret = acm[-1, -1], _dtw_owp(acm)
    return ret
