"""Integral Fréchet distance."""

import numpy as np
from numba import njit

from ._algorithms.ifd import _ifd_acm, _ifd_acm_1d, _ifd_owp

__all__ = [
    "ifd",
    "ifd_owp",
]


NAN = np.float64(np.nan)
EPSILON = np.finfo(np.float64).eps


@njit(cache=True)
def ifd(P, Q, delta, dist="euclidean"):
    r"""Integral Fréchet distance between two open polygonal curves.

    Let :math:`f, g: [0, 1] \to \Omega` be curves defined in a metric space
    :math:`\Omega`. Let :math:`\alpha, \beta: [0, 1] \to [0, 1]` be continuous
    non-decreasing surjections, and define :math:`\pi: [0, 1] \to [0, 1] \times
    [0, 1]` such that :math:`\pi(t) = \left(\alpha(t), \beta(t)\right)`.
    The integral Fréchet distance between :math:`f` and :math:`g` is defined as

    .. math::

        \inf_{\pi} \int_0^1
        dist\left(\pi(t)\right) \cdot
        \lVert \pi'(t) \rVert_1
        \mathrm{d}t,

    where :math:`dist\left(\pi(t)\right)` is a distance between
    :math:`f\left(\alpha(t)\right)` and :math:`g\left(\beta(t)\right)` and
    :math:`\lVert \cdot \rVert_1` is the Manhattan norm.

    Parameters
    ----------
    P : ndarray
        A :math:`p` by :math:`n` array of :math:`p` vertices in an
        :math:`n`-dimensional space.
    Q : ndarray
        A :math:`q` by :math:`n` array of :math:`q` vertices in an
        :math:`n`-dimensional space.
    delta : double
        Maximum length of edges between Steiner points.
        Refer to the Reference section for more information.
    dist : {'euclidean', 'squared_euclidean'}
        Type of :math:`dist`. Refer to the Notes section for more information.

    Returns
    -------
    double
        The integral Fréchet distance between *P* and *Q*, NaN if any vertice
        is empty or both vertices consist of a single point.

    Raises
    ------
    ValueError
        If *P* and *Q* are not 2-dimensional arrays with same number of columns.

    See Also
    --------
    ifd_owp : Integral Fréchet distance with optimal warping path.

    Notes
    -----
    This function implements the algorithm of Brankovic et al [#]_.

    The following functions are available for :math:`dist`:

    1. Euclidean distance
        .. math::

            dist\left(p, q\right) = \lVert p - q \rVert_2

        .. note::

            This distance is not implemented yet.

    2. Squared Euclidean distance
        .. math::

            dist\left(p, q\right) = \lVert p - q \rVert_2^2

    References
    ----------
    .. [#] Brankovic, M., et al. "(k, l)-Medians Clustering of Trajectories Using
       Continuous Dynamic Time Warping." Proceedings of the 28th International
       Conference on Advances in Geographic Information Systems. 2020.

    Examples
    --------
    >>> P, Q = [[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]]
    >>> ifd(np.asarray(P), np.asarray(Q), 0.1, "squared_euclidean")
    2.0
    """
    B, L = _ifd_acm_1d(P, Q, delta, dist)
    if len(B) == 0 or len(L) == 0:
        ret = NAN
    else:
        ret = L[-1]
    return ret


@njit(cache=True)
def ifd_owp(P, Q, delta, dist="euclidean"):
    """Integral Fréchet distance and its optimal warping path.

    Parameters
    ----------
    P : ndarray
        A :math:`p` by :math:`n` array of :math:`p` vertices in an
        :math:`n`-dimensional space.
    Q : ndarray
        A :math:`q` by :math:`n` array of :math:`q` vertices in an
        :math:`n`-dimensional space.
    delta : double
        Maximum length of edges between Steiner points. Refer to :func:`ifd`.
    dist : {'euclidean', 'squared_euclidean'}
        Type of :math:`dist`. Refer to :func:`ifd`.

    Returns
    -------
    ifd : double
        The integral Fréchet distance between *P* and *Q*, NaN if any vertice
        is empty or both vertices consist of a single point.
    owp : ndarray
        Optimal warping path, empty if any vertice is empty or both vertices
        consist of a single point.

    Raises
    ------
    ValueError
        If *P* and *Q* are not 2-dimensional arrays with same number of columns.

    Examples
    --------
    >>> P = np.array([[0, 0], [2, 2], [4, 2], [4, 4], [2, 1], [5, 1], [7, 2]])
    >>> Q = np.array([[2, 0], [1, 3], [5, 3], [5, 2], [7, 3]])
    >>> _, path = ifd_owp(P, Q, 0.1, "squared_euclidean")
    >>> from curvesimilarities.util import matching_pairs
    >>> pairs = matching_pairs(P, Q, path, 100)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> plt.plot(*P.T); plt.plot(*Q.T)  # doctest: +SKIP
    >>> plt.plot(*pairs, "--", color="gray")  # doctest: +SKIP
    """
    B, L = _ifd_acm(P, Q, delta, dist)
    if len(B) == 0 or len(L) == 0:
        ifd = NAN
    else:
        ifd = L[-1, -1]
    path = _ifd_owp(P, Q, B, L, delta, dist)
    return ifd, path[::-1]
