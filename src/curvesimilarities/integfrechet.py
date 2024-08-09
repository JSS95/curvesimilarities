"""Integral Fréchet distance."""

import numpy as np
from numba import njit

from ._algorithms.ifd import _ifd_acm, _ifd_acm_1d, _ifd_owp
from .util import index2arclength

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
    P : array_like
        A :math:`p` by :math:`n` array of :math:`p` vertices of a polyline in an
        :math:`n`-dimensional space.
    Q : array_like
        A :math:`q` by :math:`n` array of :math:`q` vertices of a polyline in an
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

    See Also
    --------
    ifd_owp

    Notes
    -----
    This function implements the algorithm of Brankovic et al [1]_.

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
    .. [1] Brankovic, M., et al. "(k, l)-Medians Clustering of Trajectories Using
       Continuous Dynamic Time Warping." Proceedings of the 28th International
       Conference on Advances in Geographic Information Systems. 2020.

    Examples
    --------
    >>> P, Q = [[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]]
    >>> ifd(np.asarray(P), np.asarray(Q), 0.1, "squared_euclidean")
    2.0
    """
    B, L = _ifd_acm_1d(P.astype(np.float64), Q.astype(np.float64), delta, dist)
    if len(B) == 0 or len(L) == 0:
        ret = NAN
    else:
        ret = L[-1]
    return ret


@njit(cache=True)
def ifd_owp(P, Q, delta, dist="euclidean", param="arc-length"):
    """Integral Fréchet distance and its optimal warping path.

    Parameters
    ----------
    P : array_like
        A :math:`p` by :math:`n` array of :math:`p` vertices of a polyline in an
        :math:`n`-dimensional space.
    Q : array_like
        A :math:`q` by :math:`n` array of :math:`q` vertices of a polyline in an
        :math:`n`-dimensional space.
    delta : double
        Maximum length of edges between Steiner points. Refer to :func:`ifd`.
    dist : {'euclidean', 'squared_euclidean'}
        Type of :math:`dist`. Refer to :func:`ifd`.
    param : {'arc-length', 'vertex'}
        Parametrization of matching.

    Returns
    -------
    ifd : double
        The integral Fréchet distance between *P* and *Q*, NaN if any vertice
        is empty or both vertices consist of a single point.
    owp : ndarray
        Optimal warping path, empty if any vertice is empty or both vertices
        consist of a single point.

    Examples
    --------
    >>> from curvesimilarities.util import parameter_space
    >>> P = np.array([[0, 0], [2, 2], [4, 2], [4, 4], [2, 1], [5, 1], [7, 2]])
    >>> Q = np.array([[2, 0], [1, 3], [5, 3], [5, 2], [7, 3]])
    >>> _, path = ifd_owp(P, Q, 0.1, "squared_euclidean")
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> weight, p, q, _, _ = parameter_space(P, Q, 200, 100)
    >>> plt.pcolormesh(p, q, weight.T, cmap="gray")  # doctest: +SKIP
    >>> plt.plot(*path.T, "--")  # doctest: +SKIP
    """
    P, Q = P.astype(np.float64), Q.astype(np.float64)
    B, L = _ifd_acm(P, Q, delta, dist)
    if len(B) == 0 or len(L) == 0:
        ifd = NAN
    else:
        ifd = L[-1, -1]
    path = _ifd_owp(P, Q, B, L, delta, dist)[::-1]

    if param == "arc-length":
        path = np.stack(
            (
                index2arclength(P, path[:, 0].copy()),
                index2arclength(Q, path[:, 1].copy()),
            )
        ).T
    elif param == "vertex":
        pass
    else:
        raise ValueError("Unknown option for parametrization.")

    return ifd, path
