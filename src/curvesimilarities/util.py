"""Utility functions."""

import functools

import numpy as np

__all__ = [
    "sample_polyline",
    "refine_polyline",
]


def sanitize_vertices(owp):
    """Decorator to sanitize the vertices."""

    def decorator(func):

        @functools.wraps(func)
        def wrapper(P, Q, *args, **kwargs):
            P = np.asarray(P, dtype=np.float64)
            Q = np.asarray(Q, dtype=np.float64)

            if len(P.shape) != 2:
                raise ValueError("P must be a 2-dimensional array.")
            if len(Q.shape) != 2:
                raise ValueError("Q must be a 2-dimensional array.")
            if P.shape[1] != Q.shape[1]:
                raise ValueError("P and Q must have the same number of columns.")

            if P.size == 0 or Q.size == 0:
                if owp:
                    return np.float64(np.nan), np.empty((0, 2), dtype=np.int_)
                else:
                    return np.float64(np.nan)
            return func(P, Q, *args, **kwargs)

        return wrapper

    return decorator


def sample_polyline(vert, param):
    """Sample points from a polyline.

    Parameters
    ----------
    vert : array_like
        A :math:`p` by :math:`n` array of :math:`p` vertices in an
        :math:`n`-dimensional space.
    param : array_like
        An 1-D array of :math:`q` parameters for sampled points.
        Natural parametrization is used, i.e., the polygonal curve
        is parametrized by its arc length.
        Parameters smaller than :math:`0` or larger than the total
        arc length are clipped to the nearest valid value.

    Returns
    -------
    array_like
        A :math:`q` by :math:`n` array of sampled points.

    Examples
    --------
    >>> vert = [[0, 0], [0.5, 0.5], [1, 0]]
    >>> param = np.linspace(0, 2, 10)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> plt.plot(*np.array(vert).T)  # doctest: +SKIP
    >>> plt.plot(*sample_polyline(vert, param).T, "x")  # doctest: +SKIP
    """
    vert = np.array(vert)

    seg_vec = np.diff(vert, axis=0)
    seg_len = np.linalg.norm(seg_vec, axis=-1)
    vert_param = np.insert(np.cumsum(seg_len), 0, 0)
    param = np.clip(param, vert_param[0], vert_param[-1])

    pt_vert_idx = np.clip(np.searchsorted(vert_param, param) - 1, 0, len(vert) - 2)
    t = param - vert_param[pt_vert_idx]
    seg_unitvec = seg_vec / seg_len[..., np.newaxis]
    return vert[pt_vert_idx] + t[..., np.newaxis] * seg_unitvec[pt_vert_idx]


def refine_polyline(vert):
    r"""Remove degenerate vertices from a polyline.

    Consecutive duplicate vertices, and three or more vertices on a same line
    are removed.

    Parameters
    ----------
    vert : array_like
        A :math:`p` by :math:`n` array of :math:`p` vertices in an
        :math:`n`-dimensional space.

    Returns
    -------
    array_like
        A :math:`q` by :math:`n` array of :math:`q \leq p` vertices in an
        :math:`n`-dimensional space.

    Examples
    --------
    >>> vert = [[0, 0], [0, 0], [1, 0], [2, 0]]
    >>> refine_polyline(vert)
    array([[0, 0],
           [2, 0]])
    """
    ret = np.empty_like(vert)
    ret[0] = vert[0]
    count = 1

    for i in range(1, len(vert)):
        current = vert[i]
        prev = ret[count - 1]
        if np.all(prev == current):
            continue
        if count >= 2:
            vec = current - prev
            prev_vec = prev - ret[count - 2]
            if np.dot(vec, prev_vec) == np.linalg.norm(vec) * np.linalg.norm(prev_vec):
                ret[count - 1] = current
                continue
        ret[count] = current
        count += 1
    return ret[:count]
