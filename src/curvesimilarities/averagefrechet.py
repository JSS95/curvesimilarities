"""Average Fréchet distance."""

import numpy as np

from .integfrechet import _ifd, _sample_pts

__all__ = [
    "afd",
]


def afd(P, Q, delta):
    r"""Average Fréchet distance between two open polygonal curves.

    Average Fréchet distance is defined as [#]_

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

    with symbols explained in :func:`~.ifd`. We choose the Manhattan norm as
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
        The average Fréchet distance between P and Q.

    Notes
    -----
    Using this function is marginally faster than calling :func:`~.ifd` and then
    dividing.

    References
    ----------
    .. [#] Buchin, M. E. On the computability of the Fréchet distance between
       triangulated surfaces. Diss. 2007.

    See Also
    --------
    ifd : Integral Fréchet distance.

    Examples
    --------
    >>> afd([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]], 0.1)
    1.0
    """
    P = np.asarray(P, dtype=np.float_)
    Q = np.asarray(Q, dtype=np.float_)

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
    ifd = _ifd(P_edge_len, P_subedges_num, P_pts, Q_edge_len, Q_subedges_num, Q_pts)
    return ifd / (np.sum(P_edge_len) + np.sum(Q_edge_len))
