"""Dynamic time warping.

This module implements only the basic algorithm. If you need advanced features, use
dedicated packages such as `dtw-python
<https://pypi.org/project/dtw-python/>`_.
"""

import numpy as np
from numba import njit
from scipy.spatial.distance import cdist

__all__ = [
    "dtw",
    "dtw_owp",
]


def dtw(P, Q):
    r"""Dynamic time warping distance.

    Let :math:`\{P_0, P_1, ..., P_n\}` and :math:`\{Q_0, Q_1, ..., Q_m\}` be
    polyline vertices in metric space. The dynamic time warping distance between
    two polylines is defined as

    .. math::

        \min_{C} \sum_{(i, j) \in C} \lVert P_i - Q_j \rVert,

    where :math:`C` is a nondecreasing coupling over
    :math:`\{0, ..., n\} \times \{0, ..., m\}`, starting from :math:`(0, 0)` and
    ending with :math:`(n, m)`. :math:`\lVert \cdot \rVert` is the underlying
    metric, which is the Euclidean metric in this implementation.

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
        The dynamic time warping distance between P and Q.

    See Also
    --------
    dtw_owp : Dynamic time warping distance with optimal warping path.

    Notes
    -----
    This function implements the algorithm described by Senin [#]_.

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
    if len(P) == 0 or len(Q) == 0:
        return np.nan
    dist = cdist(P, Q)
    return _dtw_acm(dist)[-1, -1]


def dtw_owp(P, Q):
    """Dynamic time warping distance and its optimal warping path.

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
        The dynamic time warping distance between P and Q.
    owp : ndarray
        Indices of P and Q for optimal warping path.

    Examples
    --------
    .. plot::
        :include-source:

        >>> P = np.linspace([0, 0], [1, 0], 10)
        >>> Q = np.linspace([0, 1], [1, 1], 20)
        >>> dist, path = dtw_owp(P, Q)
        >>> dist / len(path)  # averaged dynamic time warping
        1.00...
        >>> import matplotlib.pyplot as plt #doctest: +SKIP
        >>> plt.plot(*path.T, "x")  #doctest: +SKIP
    """
    if len(P) == 0 or len(Q) == 0:
        return np.nan
    dist = cdist(P, Q)
    acm = _dtw_acm(dist)
    return acm[-1, -1], _dtw_owp(acm)


@njit(cache=True)
def _dtw_acm(cm):
    """Accumulated cost matrix for dynamic time warping."""
    p, q = cm.shape
    ret = np.empty((p, q), dtype=np.float_)

    ret[0, 0] = cm[0, 0]
    for i in range(1, p):
        ret[i, 0] = ret[i - 1, 0] + cm[i, 0]
    for j in range(1, q):
        ret[0, j] = ret[0, j - 1] + cm[0, j]
    for i in range(1, p):
        for j in range(1, q):
            ret[i, j] = min(ret[i - 1, j], ret[i, j - 1], ret[i - 1, j - 1]) + cm[i, j]
    return ret


@njit(cache=True)
def _dtw_owp(acm):
    p, q = acm.shape
    path = np.empty((p + q - 1, 2), dtype=np.int32)
    path_len = np.int32(0)

    i, j = p - 1, q - 1
    path[path_len] = [i, j]
    path_len += np.int32(1)

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            d = min(acm[i - 1, j], acm[i, j - 1], acm[i - 1, j - 1])
            if acm[i - 1, j] == d:
                i -= 1
            elif acm[i, j - 1] == d:
                j -= 1
            else:
                i -= 1
                j -= 1

        path[path_len] = [i, j]
        path_len += np.int32(1)
    return path[-(len(path) - path_len + 1) :: -1, :]
