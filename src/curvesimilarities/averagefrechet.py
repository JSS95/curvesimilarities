"""Average Fréchet distance and its variants."""

import numpy as np
from numba import njit

from .integfrechet import (
    _ifd,
    _ifd_owp,
    _line_line_integrate,
    _line_point_integrate,
    _refine_path,
    _sample_pts,
    ifd_owp,
)
from .util import sanitize_vertices

__all__ = [
    "afd",
    "afd_owp",
    "qafd",
    "qafd_owp",
]


EPSILON = np.finfo(np.float_).eps


@sanitize_vertices(owp=False)
def afd(P, Q, delta):
    r"""Average Fréchet distance between two open polygonal curves.

    The average Fréchet distance is defined as [#]_

    .. math::

        \inf_{\pi} \frac{
            \int_0^1
            \delta\left(\pi(t)\right) \cdot
            \lVert \pi'(t) \rVert
            \mathrm{d}t
        }{
            \int_0^1
            \lVert \pi'(t) \rVert
            \mathrm{d}t
        },

    with symbols explained in :func:`~.ifd`. We use the Manhattan norm for
    :math:`\lVert \cdot \rVert`, so the formula can be reduced to

    .. math::

        \frac{1}{\lvert f \rvert + \lvert g \rvert}
        \inf_{\pi} \int_0^1
        \delta\left(\pi(t)\right) \cdot
        \lVert \pi'(t) \rVert
        \mathrm{d}t,

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
        The average Fréchet distance between *P* and *Q*, NaN if any vertice
        is empty.

    Raises
    ------
    ValueError
        If *P* and *Q* are not 2-dimensional arrays with same number of columns.

    See Also
    --------
    ifd : Integral Fréchet distance.
    afd_owp : Average Fréchet distance with optimal warping path.

    Notes
    -----
    Using this function is marginally faster than calling :func:`~.ifd` and then
    dividing.

    References
    ----------
    .. [#] Buchin, M. E. On the computability of the Fréchet distance between
       triangulated surfaces. Diss. 2007.

    Examples
    --------
    >>> afd([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]], 0.1)
    1.0
    """
    if len(P) < 2 or len(Q) < 2:
        return np.nan

    # No need to add Steiner points if the other polyline is just a line segment.
    if len(Q) == 2:
        P_edge_len = np.linalg.norm(np.diff(P, axis=0), axis=-1)
        P_subedges_num = np.ones(len(P) - 1, dtype=np.int_)
        P_pts = P
    else:
        P_edge_len, P_subedges_num, P_pts = _sample_pts(P, delta)
    if len(P) == 2:
        Q_edge_len = np.linalg.norm(np.diff(Q, axis=0), axis=-1)
        Q_subedges_num = np.ones(len(Q) - 1, dtype=np.int_)
        Q_pts = Q
    else:
        Q_edge_len, Q_subedges_num, Q_pts = _sample_pts(Q, delta)
    ifd = _ifd(
        P_edge_len,
        P_subedges_num,
        P_pts,
        Q_edge_len,
        Q_subedges_num,
        Q_pts,
        _line_point_integrate,
        _line_line_integrate,
    )
    return ifd / (np.sum(P_edge_len) + np.sum(Q_edge_len))


@sanitize_vertices(owp=True)
def afd_owp(P, Q, delta):
    """Average Fréchet distance and its optimal warping path.

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
        The average Fréchet distance between *P* and *Q*, NaN if any vertice
        is empty.
    owp : ndarray
        Optimal warping path, empty if any vertice is empty.

    Raises
    ------
    ValueError
        If *P* and *Q* are not 2-dimensional arrays with same number of columns.

    Examples
    --------
    .. plot::
        :include-source:

        >>> dist, path = afd_owp([[0, 0], [0.5, 0], [1, 0]], [[0.5, 1], [1.5, 1]], 0.1)
        >>> import matplotlib.pyplot as plt #doctest: +SKIP
        >>> plt.plot(*path.T)  #doctest: +SKIP
    """
    dist, path = ifd_owp(P, Q, delta)
    return dist / np.sum(path[-1]), path


@sanitize_vertices(owp=False)
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

    with symbols explained in :func:`~.ifd`. We use the Manhattan norm for
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
        vertice is empty.

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
    if len(P) < 2 or len(Q) < 2:
        return np.nan

    # No need to add Steiner points if the other polyline is just a line segment.
    if len(Q) == 2:
        P_edge_len = np.linalg.norm(np.diff(P, axis=0), axis=-1)
        P_subedges_num = np.ones(len(P) - 1, dtype=np.int_)
        P_pts = P
    else:
        P_edge_len, P_subedges_num, P_pts = _sample_pts(P, delta)
    if len(P) == 2:
        Q_edge_len = np.linalg.norm(np.diff(Q, axis=0), axis=-1)
        Q_subedges_num = np.ones(len(Q) - 1, dtype=np.int_)
        Q_pts = Q
    else:
        Q_edge_len, Q_subedges_num, Q_pts = _sample_pts(Q, delta)
    square_ifd = _ifd(
        P_edge_len,
        P_subedges_num,
        P_pts,
        Q_edge_len,
        Q_subedges_num,
        Q_pts,
        _line_point_square_integrate,
        _line_line_square_integrate,
    )
    return np.sqrt(square_ifd / (np.sum(P_edge_len) + np.sum(Q_edge_len)))


@sanitize_vertices(owp=True)
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
        vertice is empty.
    owp : ndarray
        Optimal warping path, empty if any vertice is empty.

    Raises
    ------
    ValueError
        If *P* and *Q* are not 2-dimensional arrays with same number of columns.

    Examples
    --------
    .. plot::
        :include-source:

        >>> dist, path = qafd_owp([[0, 0], [0.5, 0], [1, 0]], [[0.5, 1], [1.5, 1]], 0.1)
        >>> import matplotlib.pyplot as plt #doctest: +SKIP
        >>> plt.plot(*path.T)  #doctest: +SKIP
    """
    if len(P) < 2 or len(Q) < 2:
        return np.nan, np.empty((0, 2), dtype=np.float_)

    if len(Q) == 2:
        P_edge_len = np.linalg.norm(np.diff(P, axis=0), axis=-1)
        P_subedges_num = np.ones(len(P) - 1, dtype=np.int_)
        P_pts = P
    else:
        P_edge_len, P_subedges_num, P_pts = _sample_pts(P, delta)
    if len(P) == 2:
        Q_edge_len = np.linalg.norm(np.diff(Q, axis=0), axis=-1)
        Q_subedges_num = np.ones(len(Q) - 1, dtype=np.int_)
        Q_pts = Q
    else:
        Q_edge_len, Q_subedges_num, Q_pts = _sample_pts(Q, delta)
        _, Q_subedges_num, Q_pts = _sample_pts(Q, delta)
    dist, path = _ifd_owp(
        P_edge_len,
        P_subedges_num,
        P_pts,
        Q_edge_len,
        Q_subedges_num,
        Q_pts,
        _line_point_square_integrate,
        _line_line_square_integrate,
    )
    return np.sqrt(dist / np.sum(path[-1])), _refine_path(path)


@njit(cache=True)
def _line_point_square_integrate(a, b, p):
    r"""Analytic integration from AP to BP (squared).

    .. math::
        \int_0^1 \lVert (A - P) + (B - A) t \rVert^2 \cdot \lVert (B - A) \rVert dt
    """
    # integrate (A*t**2 + B*t + C) * sqrt(A) dt over t [0, 1]
    # where A = dot(ab, ab), B = 2 * dot(pa, ab) and C = dot(pa, pa).
    ab = b - a
    pa = a - p
    A = np.dot(ab, ab)
    B = 2 * np.dot(ab, pa)
    C = np.dot(pa, pa)
    return (A / 3 + B / 2 + C) * np.sqrt(A)


@njit(cache=True)
def _line_line_square_integrate(a, b, c, d):
    r"""Analytic integration from AC to BD (squared).

    .. math::
        \int_0^1 \lVert (A - C) + (B - A + C - D)t \rVert^2 \cdot
        \left( \lVert B - A \rVert + \lVert D - C \rVert \right) dt
    """
    # integrate (A*t**2 + B*t + C) * (sqrt(D) + sqrt(E)) dt over t [0, 1]
    # where A = dot(vu, vu), B = 2 * dot(vu, w), C = dot(w, w), D = dot(u, u),
    # and E = dot(v, v); where u = b - a, v = d - c and w = a - c.
    u = b - a
    v = d - c
    w = a - c
    vu = u - v
    A = np.dot(vu, vu)
    B = 2 * np.dot(vu, w)
    C = np.dot(w, w)
    D = np.dot(u, u)
    E = np.dot(v, v)
    return (A / 3 + B / 2 + C) * (np.sqrt(D) + np.sqrt(E))
