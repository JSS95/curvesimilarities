"""Dynamic time warping.

This module implements only the basic algorithms.
If you need advanced features, use dedicated package such as
`dtw-python <https://pypi.org/project/dtw-python/>`_ instead.
"""

import numpy as np
from numba import njit
from scipy.spatial.distance import cdist

__all__ = [
    "dtw",
    "dtw_acm",
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

    Raises
    ------
    ValueError
        An exception is thrown if empty array is passed.

    See Also
    --------
    dtw_owp : Optimal warping path.

    Examples
    --------
    >>> dtw([[0, 0], [1, 1], [2, 0]], [[0, 1], [2, -4], [3, 2]])
    8.23...
    """
    if len(P) == 0 or len(Q) == 0:
        raise ValueError("Vertices must not be empty.")
    dist = cdist(P, Q)
    return dtw_acm(dist)[-1, -1]


@njit(cache=True)
def dtw_acm(cm):
    """Accumulated cost matrix for dynamic time warping.

    Parameters
    ----------
    cm : ndarray
        2D local cost matrix.

    Returns
    -------
    acm : ndarray
        2D accumulated cost matrix.

    Notes
    -----
    This function implements the algorithm described Senin [#]_.

    References
    ----------
    .. [#] Senin, P. (2008). Dynamic time warping algorithm review. Information
        and Computer Science Department University of Hawaii at Manoa Honolulu,
        USA, 855(1-23), 40.

    Examples
    --------
    .. plot::
        :include-source:

        >>> from scipy.spatial.distance import cdist
        >>> t, s = np.linspace(0, 2 * np.pi, 100), np.linspace(0, 2, 200)
        >>> P = np.asarray([t, np.sin(t)]).T
        >>> Q = np.asarray([np.zeros(len(s)), s]).T
        >>> cm = cdist(P, Q)
        >>> acm = dtw_acm(cm)
        >>> import matplotlib.pyplot as plt #doctest: +SKIP
        >>> plt.pcolormesh(acm.T)  #doctest: +SKIP
    """
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
def dtw_owp(acm):
    """Optimal warping path for dynamic time warping.

    Parameters
    ----------
    acm : ndarray
        Accumulated cost matrix.

    Returns
    -------
    owp : ndarray
        Indices of optimal warping path.

    See Also
    --------
    dtw_acm : Accumulated cost matrix.

    Notes
    -----
    This function implements the algorithm described Senin [#]_.

    References
    ----------
    .. [#] Senin, P. (2008). Dynamic time warping algorithm review. Information
        and Computer Science Department University of Hawaii at Manoa Honolulu,
        USA, 855(1-23), 40.

    Examples
    --------
    .. plot::
        :include-source:

        >>> from scipy.spatial.distance import cdist
        >>> t, s = np.linspace(0, 2 * np.pi, 100), np.linspace(0, 2, 200)
        >>> P = np.asarray([t, np.sin(t)]).T
        >>> Q = np.asarray([np.zeros(len(s)), s]).T
        >>> cm = cdist(P, Q)
        >>> acm = dtw_acm(cm)
        >>> owp = dtw_owp(acm)
        >>> import matplotlib.pyplot as plt #doctest: +SKIP
        >>> plt.pcolormesh(acm.T)  #doctest: +SKIP
        >>> plt.plot(owp[:, 0], owp[:, 1], color="r")  #doctest: +SKIP
    """
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
