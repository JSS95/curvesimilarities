"""Utility functions."""

import numpy as np
from numba import njit
from scipy.spatial.distance import cdist

__all__ = [
    "parameter_space",
    "matching_pairs",
    "sample_polyline",
]


def parameter_space(P, Q, p_num, q_num):
    r"""Parameter space betwee two polylines.

    Parameters
    ----------
    P : array_like
        A :math:`p` by :math:`n` array of :math:`p` vertices of a polyline in an
        :math:`n`-dimensional space.
    Q : array_like
        A :math:`q` by :math:`n` array of :math:`q` vertices of a polyline in an
        :math:`n`-dimensional space.
    p_num, q_num : int
        Number of sample points in `P` and `Q`, respectively.

    Returns
    -------
    weight : ndarray
        A `p_num` by `q_num` array containing the distance between the points in
        `P` and `Q`.
    p_coord, q_coord : ndarray
        Axis coordinates for the parameter space.
    p_vert, q_vert : ndarray
        Coordinates for the vertices of polylines.

    Examples
    --------
    Curve space:

    .. plot::
        :context: close-figs

        >>> P = np.array([[0, 0], [2, 2], [4, 2], [4, 4], [2, 1], [5, 1], [7, 2]])
        >>> Q = np.array([[2, 0], [1, 3], [5, 3], [5, 2], [7, 3]])
        >>> import matplotlib.pyplot as plt  # doctest: +SKIP
        >>> plt.plot(*P.T)  # doctest: +SKIP
        >>> plt.plot(*Q.T)  # doctest: +SKIP

    Parameter space with vertices as dashed lines:

    .. plot::
        :context: close-figs

        >>> weight, p, q, p_vert, q_vert = parameter_space(P, Q, 200, 100)
        >>> plt.pcolormesh(p, q, weight.T)  # doctest: +SKIP
        >>> plt.vlines(p_vert, 0, q[-1], "k", "--")  # doctest: +SKIP
        >>> plt.hlines(q_vert, 0, p[-1], "k", "--")  # doctest: +SKIP
    """
    p_vert = np.insert(np.cumsum(np.linalg.norm((np.diff(P, axis=0)), axis=-1)), 0, 0)
    p_coord = np.linspace(0, p_vert[-1], p_num)
    P_pts = sample_polyline(P, p_coord)
    q_vert = np.insert(np.cumsum(np.linalg.norm((np.diff(Q, axis=0)), axis=-1)), 0, 0)
    q_coord = np.linspace(0, q_vert[-1], q_num)
    Q_pts = sample_polyline(Q, q_coord)
    return cdist(P_pts, Q_pts), p_coord, q_coord, p_vert, q_vert


def matching_pairs(P, Q, path, sample_num):
    """Sample a continuous path in parameter space to matching pairs in curve space.

    Parameters
    ----------
    P : array_like
        A :math:`p` by :math:`n` array of :math:`p` vertices in an
        :math:`n`-dimensional space.
    Q : array_like
        A :math:`q` by :math:`n` array of :math:`q` vertices in an
        :math:`n`-dimensional space.
    path : ndarray
        A :math:`N` by :math:`2` array of :math:`N` vertices of polyline in
        parameter space.
    sample_num : int
        Number of sample points to be uniformly taken from `path`.

    Returns
    -------
    ndarray
        A :math:`n` by :math:`2` by `sample_num` array of point pairs in curve space.

    Examples
    --------
    >>> from curvesimilarities import ifd_owp
    >>> from curvesimilarities.util import matching_pairs
    >>> P = np.array([[0, 0], [2, 2], [4, 2], [4, 4], [2, 1], [5, 1], [7, 2]])
    >>> Q = np.array([[2, 0], [1, 3], [5, 3], [5, 2], [7, 3]])
    >>> _, path = ifd_owp(P, Q, 0.1, "squared_euclidean")
    >>> pairs = matching_pairs(P, Q, path, 50)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> plt.plot(*P.T); plt.plot(*Q.T)  # doctest: +SKIP
    >>> plt.plot(*pairs, "--", color="gray")  # doctest: +SKIP
    """
    path_len = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=-1))
    path_pts = sample_polyline(path, np.linspace(0, path_len, sample_num))
    P_pts = sample_polyline(P, path_pts[:, 0])
    Q_pts = sample_polyline(Q, path_pts[:, 1])
    return np.stack([P_pts, Q_pts]).transpose(2, 0, 1)


def sample_polyline(vert, param):
    """Sample points from a polyline.

    Parameters
    ----------
    vert : array_like
        A :math:`p` by :math:`n` array of :math:`p` vertices of a polyline in an
        :math:`n`-dimensional space.
    param : array_like
        An 1-D array of :math:`q` parameters for sampled points, using arc-length
        parametrization.

    Returns
    -------
    array_like
        A :math:`q` by :math:`n` array of sampled points.

    Notes
    -----
    Parameters smaller than :math:`0` or larger than the total arc length are clipped to
    the nearest valid value.
    """
    vert = np.asarray(vert)

    seg_vec = np.diff(vert, axis=0)
    seg_len = np.linalg.norm(seg_vec, axis=-1)
    vert_param = np.insert(np.cumsum(seg_len), 0, 0)
    param = np.clip(param, vert_param[0], vert_param[-1])

    pt_vert_idx = np.clip(np.searchsorted(vert_param, param) - 1, 0, len(vert) - 2)
    t = param - vert_param[pt_vert_idx]
    seg_unitvec = seg_vec / seg_len[..., np.newaxis]
    return vert[pt_vert_idx] + t[..., np.newaxis] * seg_unitvec[pt_vert_idx]


@njit(cache=True)
def index2arclength(curve, param):
    """Convert index based parameters of a curve to arc length based parameters.

    Parameters
    ----------
    curve : array_like
        A :math:`p` by :math:`n` array of :math:`p` vertices in an
        :math:`n`-dimensional space.
    param : ndarray
        Index based parameters.
    """
    orig_shape = param.shape
    param = param.reshape(-1)

    len_cumsum = np.empty(len(curve), dtype=np.float64)
    len_cumsum[0] = 0
    for i in range(1, len(curve)):
        len_cumsum[i] = len_cumsum[i - 1] + np.linalg.norm(curve[i] - curve[i - 1])

    ret = np.empty_like(param, dtype=np.float64)
    for i in range(len(param)):
        n = int(param[i])
        t = param[i] - n
        if t == 0:
            ret[i] = len_cumsum[n]
        else:
            ret[i] = len_cumsum[n] + t * (len_cumsum[n + 1] - len_cumsum[n])
    return ret.reshape(orig_shape)
