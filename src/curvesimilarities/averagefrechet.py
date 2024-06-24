"""Average Fréchet distance and its variants."""

import numpy as np
from numba import njit

from .integfrechet import (
    _ifd,
    _ifd_owp,
    _line_point_square_integrate,
    _sample_ifd_pts,
    sanitize_vertices_ifd,
)
from .util import sanitize_vertices

__all__ = [
    "qafd",
    "qafd_owp",
]


@njit(cache=True)
def qafd_degenerate(curve, point):
    ret = 0
    length = 0
    for i in range(len(curve) - 1):
        a, b = curve[i], curve[i + 1]
        ret += _line_point_square_integrate(a, b, point)
        length += np.linalg.norm(b - a)
    return np.sqrt(ret / length)


@sanitize_vertices(owp=False)
@sanitize_vertices_ifd(qafd_degenerate, owp=False)
def qafd(P, Q, delta):
    r"""Quadratic average Fréchet distance between two open polygonal curves.

    The quadratic average Fréchet distance is defined as

    .. math::

        \inf_{\pi}
        \sqrt{
            \frac{
                \int_0^1
                \delta\left(\pi(t)\right)^2 \cdot
                \lVert \pi'(t) \rVert
                \mathrm{d}t
            }{
                \int_0^1
                \lVert \pi'(t) \rVert
                \mathrm{d}t
            }
        },

    where :math:`\delta` is the Euclidean distance and :math:`\pi` is the continuous
    nondecreasing path in the parameter space. We use the Manhattan norm for
    :math:`\lVert \cdot \rVert`, so the formula can be reduced to

    .. math::

        \frac{1}{\sqrt{\lvert f \rvert + \lvert g \rvert}}
        \inf_{\pi}
        \sqrt{
        \int_0^1
        \delta\left(\pi(t)\right)^2 \cdot
        \lVert \pi'(t) \rVert
        \mathrm{d}t
        },

    where :math:`\lvert \cdot \rvert` denotes the length of a polygonal curve.

    Parameters
    ----------
    P : array_like
        A :math:`p` by :math:`n` array of :math:`p` vertices in an
        :math:`n`-dimensional space.
    Q : array_like
        A :math:`q` by :math:`n` array of :math:`q` vertices in an
        :math:`n`-dimensional space.
    delta : double
        Maximum length of edges between Steiner points.

    Returns
    -------
    dist : double
        The quadratic average Fréchet distance between *P* and *Q*, NaN if any
        vertice is empty or both vertices consist of a single point.

    Raises
    ------
    ValueError
        If *P* and *Q* are not 2-dimensional arrays with same number of columns.

    See Also
    --------
    afd : Average Fréchet distance.
    qafd_owp : Quadratic average Fréchet distance with optimal warping path.

    Examples
    --------
    >>> qafd([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]], 0.1)
    1.0
    """
    (
        P_edge_len,
        P_subedges_num,
        P_pts,
        Q_edge_len,
        Q_subedges_num,
        Q_pts,
    ) = _sample_ifd_pts(P, Q, delta)
    square_ifd = _ifd(
        P_edge_len,
        P_subedges_num,
        P_pts,
        Q_edge_len,
        Q_subedges_num,
        Q_pts,
    )
    return float(np.sqrt(square_ifd / (np.sum(P_edge_len) + np.sum(Q_edge_len))))


@sanitize_vertices(owp=True)
@sanitize_vertices_ifd(qafd_degenerate, owp=True)
def qafd_owp(P, Q, delta):
    """Quadratic average Fréchet distance and its optimal warping path.

    Parameters
    ----------
    P : array_like
        A :math:`p` by :math:`n` array of :math:`p` vertices in an
        :math:`n`-dimensional space.
    Q : array_like
        A :math:`q` by :math:`n` array of :math:`q` vertices in an
        :math:`n`-dimensional space.
    delta : double
        Maximum length of edges between Steiner points.

    Returns
    -------
    dist : double
        The quadratic average Fréchet distance between *P* and *Q*, NaN if any
        vertice is empty or both vertices consist of a single point.
    owp : ndarray
        Optimal warping path, empty if any vertice is empty or both vertices
        consist of a single point.

    Raises
    ------
    ValueError
        If *P* and *Q* are not 2-dimensional arrays with same number of columns.

    Examples
    --------
    >>> dist, path = qafd_owp([[0, 0], [0.5, 0], [1, 0]], [[0.5, 1], [1.5, 1]], 0.1)
    >>> import matplotlib.pyplot as plt #doctest: +SKIP
    >>> plt.plot(*path.T)  #doctest: +SKIP
    """
    (
        P_edge_len,
        P_subedges_num,
        P_pts,
        Q_edge_len,
        Q_subedges_num,
        Q_pts,
    ) = _sample_ifd_pts(P, Q, delta)
    dist, path, count = _ifd_owp(
        P_edge_len,
        P_subedges_num,
        P_pts,
        Q_edge_len,
        Q_subedges_num,
        Q_pts,
    )
    owp = path[:count]
    return float(np.sqrt(dist / np.sum(owp[-1]))), owp
