"""Integral Fréchet distance."""

import functools

import numpy as np
from numba import njit
from numba.np.extensions import cross2d

from .util import sanitize_vertices

__all__ = [
    "ifd",
    "ifd_owp",
]


EPSILON = np.finfo(np.float_).eps


def sanitize_vertices_ifd(degenerate_fallback, owp):
    """Decorator to sanitize the vertices for IFD and its variants.

    The integral-based Fréchet distance variants must return NaN for two
    degenerate curves with zero-lengths to avoid returning false zero-value.
    Use this decorator with :func:`sanitize_vertices`.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(P, Q, *args, **kwargs):
            if len(P) != 1 and len(Q) != 1:
                return func(P, Q, *args, **kwargs)
            elif len(P) == 1 and len(Q) == 1:
                if owp:
                    return np.float_(np.nan), np.empty((0, 2), dtype=np.float_)
                else:
                    return np.float_(np.nan)

            # degenerate cases
            if len(P) == 1:
                curve, point, curve_idx = Q, P[0], 1
            else:
                curve, point, curve_idx = P, Q[0], 0

            dist = degenerate_fallback(curve, point)
            if owp:
                path = np.zeros((2, 2), dtype=np.float_)
                curve_len = np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=-1))
                path[1, curve_idx] = curve_len
                return dist, path
            else:
                return dist

        return wrapper

    return decorator


@njit(cache=True)
def ifd_degenerate(curve, point):
    ret = 0
    for i in range(len(curve) - 1):
        a, b = curve[i], curve[i + 1]
        ret += _line_point_integrate(a, b, point)
    return ret


@sanitize_vertices(owp=False)
@sanitize_vertices_ifd(ifd_degenerate, owp=False)
def ifd(P, Q, delta):
    r"""Integral Fréchet distance between two open polygonal curves.

    Let :math:`f, g: [0, 1] \to \Omega` be curves defined in a metric space
    :math:`\Omega`. Let :math:`\alpha, \beta: [0, 1] \to [0, 1]` be continuous
    non-decreasing surjections, and define :math:`\pi: [0, 1] \to [0, 1] \times
    [0, 1]` such that :math:`\pi(t) = \left(\alpha(t), \beta(t)\right)`.
    The integral Fréchet distance between :math:`f` and :math:`g` is defined as

    .. math::

        \inf_{\pi} \int_0^1
        \delta\left(\pi(t)\right) \cdot
        \lVert \pi'(t) \rVert
        \mathrm{d}t,

    where :math:`\delta\left(\pi(t)\right)` is a distance between
    :math:`f\left(\alpha(t)\right)` and :math:`g\left(\beta(t)\right)` in
    :math:`\Omega` and :math:`\lVert \cdot \rVert` is a norm in :math:`[0, 1]
    \times [0, 1]`. In this implementation, we use the Euclidean distance
    for :math:`\delta` and the Manhattan norm for :math:`\lVert \cdot \rVert`.

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

    References
    ----------
    .. [#] Brankovic, M., et al. "(k, l)-Medians Clustering of Trajectories Using
       Continuous Dynamic Time Warping." Proceedings of the 28th International
       Conference on Advances in Geographic Information Systems. 2020.

    Examples
    --------
    >>> ifd([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]], 0.1)
    2.0
    """
    return _ifd(
        *_sample_ifd_pts(P, Q, delta), _line_point_integrate, _line_line_integrate
    )


@njit(cache=True)
def _ifd(
    P_edge_len,
    P_subedges_num,
    P_pts,
    Q_edge_len,
    Q_subedges_num,
    Q_pts,
    line_point_cost,
    line_line_cost,
):
    NP = len(P_subedges_num) + 1
    P_vert_indices = np.empty(NP, dtype=np.int_)
    P_vert_indices[0] = 0
    P_vert_indices[1:] = np.cumsum(P_subedges_num)

    NQ = len(Q_subedges_num) + 1
    Q_vert_indices = np.empty(NQ, dtype=np.int_)
    Q_vert_indices[0] = 0
    Q_vert_indices[1:] = np.cumsum(Q_subedges_num)

    # Cost containers; elements will be updated.
    P_costs = np.empty(len(P_pts), dtype=np.float_)
    P_costs[0] = 0
    Q_costs = np.empty(len(Q_pts), dtype=np.float_)
    Q_costs[0] = 0

    # TODO: parallelize this i-loop.
    # Must ensure that cell (i - 1, j) is computed before (i, j).
    p0 = P_costs[:1]  # will be updated during i-loop when j == 0
    for i in range(NP - 1):
        p_pts = P_pts[P_vert_indices[i] : P_vert_indices[i + 1] + 1]

        q0 = Q_costs[:1]  # will be updated during j-loop
        for j in range(NQ - 1):
            q_pts = Q_pts[Q_vert_indices[j] : Q_vert_indices[j + 1] + 1]

            if j == 0:
                p_costs = np.concatenate(
                    (p0, P_costs[P_vert_indices[i] + 1 : P_vert_indices[i + 1] + 1])
                )
            else:
                p_costs = P_costs[P_vert_indices[i] : P_vert_indices[i + 1] + 1].copy()
            q_costs = np.concatenate(
                (q0, Q_costs[Q_vert_indices[j] + 1 : Q_vert_indices[j + 1] + 1])
            )

            p1, q1 = _cell_owcs(
                P_edge_len[i],
                p_pts,
                p_costs,
                P_costs[P_vert_indices[i] : P_vert_indices[i + 1] + 1],
                Q_edge_len[j],
                q_pts,
                q_costs,
                Q_costs[Q_vert_indices[j] : Q_vert_indices[j + 1] + 1],
                i == 0,
                j == 0,
                i == NP - 2,
                j == NQ - 2,
                line_point_cost,
                line_line_cost,
            )

            # store for the next loops
            if j == 0:
                p0 = p1
            q0 = q1

    return Q_costs[-1]


@njit(cache=True)
def _cell_owcs(
    L1,
    p_pts,
    p_costs,
    p_costs_out,
    L2,
    q_pts,
    q_costs,
    q_costs_out,
    p_is_initial,
    q_is_initial,
    p_is_last,
    q_is_last,
    line_point_cost,
    line_line_cost,
):
    """Optimal warping costs for all points in an cell."""
    # p_costs = lower boundary, p_costs_out = upper boundary,
    # q_costs = left boundary, q_costs_out = right boundary of the cell.
    P1, Q1, u, v, b, delta_P, delta_Q = _cell_info(p_pts, L1, q_pts, L2)

    # Will be reused for each border point (t) to find best starting point (s).
    p_cost_candidates = np.empty(len(p_pts), dtype=np.float_)
    q_cost_candidates = np.empty(len(q_pts), dtype=np.float_)

    s = np.empty((2,), dtype=np.float_)
    t = np.empty((2,), dtype=np.float_)

    # compute upper boundary
    t[1] = L2
    if q_is_last:  # No need steiner points on upper boundary. Just check corner point.
        start_idx = len(p_pts) - 1
    else:
        start_idx = 0
    for i in range(start_idx, len(p_pts)):  # Fill p_costs_out[i]
        t[0] = delta_P * i

        s[0] = 0
        if p_is_initial:  # No steiner points on left boundary; just check [0, 0]
            q_end_idx = 1
        else:
            q_end_idx = len(q_pts)
        for j in range(0, q_end_idx):
            s[1] = delta_Q * j
            _, cost = _st_owp(P1, u, Q1, v, b, s, t, line_point_cost, line_line_cost)
            q_cost_candidates[j] = q_costs[j] + cost

        s[1] = 0
        if q_is_initial:  # No steiner points on bottom boundary; just check [0, 0]
            p_end_idx = 1
        else:
            p_end_idx = i + 1
        p_cost_candidates[0] = q_cost_candidates[0]  # cost from [0, 0] already known.
        for i_ in range(1, p_end_idx):  # let bottom border points be (s). (to right)
            s[0] = delta_P * i_
            _, cost = _st_owp(P1, u, Q1, v, b, s, t, line_point_cost, line_line_cost)
            p_cost_candidates[i_] = p_costs[i_] + cost

        p_costs_out[i] = min(
            np.min(p_cost_candidates[:p_end_idx]), np.min(q_cost_candidates[:q_end_idx])
        )

    # compute right boundary
    t[0] = L1
    if p_is_last:  # No need steiner points on right boundary. Just check corner point.
        start_idx = len(q_pts) - 1
    elif q_is_initial:
        start_idx = 0
    else:  # LR corner already computed by the lower cell
        start_idx = 1
    # Don't need to compute the last j (already done by P loop just above)
    for j in range(start_idx, len(q_pts) - 1):
        t[1] = delta_Q * j

        s[1] = 0
        if q_is_initial:  # No steiner points on bottom boundary; just check [0, 0]
            p_end_idx = 1
        else:
            p_end_idx = len(p_pts)
        for i in range(0, p_end_idx):
            s[0] = delta_P * i
            _, cost = _st_owp(P1, u, Q1, v, b, s, t, line_point_cost, line_line_cost)
            p_cost_candidates[i] = p_costs[i] + cost

        s[0] = 0
        if p_is_initial:  # No steiner points on left boundary; just check [0, 0]
            q_end_idx = 1
        else:
            q_end_idx = j + 1
        q_cost_candidates[0] = p_cost_candidates[0]  # cost from [0, 0] already known.
        for j_ in range(1, q_end_idx):  # cost from [0, 0] already known.
            s[1] = delta_Q * j_
            _, cost = _st_owp(P1, u, Q1, v, b, s, t, line_point_cost, line_line_cost)
            q_cost_candidates[j_] = q_costs[j_] + cost

        q_costs_out[j] = min(
            np.min(p_cost_candidates[:p_end_idx]), np.min(q_cost_candidates[:q_end_idx])
        )
    q_costs_out[-1] = p_costs_out[-1]

    # Lower-right corner and upper-left corner of cells.
    return q_costs_out[:1], p_costs_out[:1]


@sanitize_vertices(owp=True)
@sanitize_vertices_ifd(ifd_degenerate, owp=True)
def ifd_owp(P, Q, delta):
    """Integral Fréchet distance and its optimal warping path.

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
    .. plot::
        :include-source:

        >>> dist, path = ifd_owp([[0, 0], [0.5, 0], [1, 0]], [[0.5, 1], [1.5, 1]], 0.1)
        >>> import matplotlib.pyplot as plt #doctest: +SKIP
        >>> plt.plot(*path.T)  #doctest: +SKIP
    """
    dist, owp = _ifd_owp(
        *_sample_ifd_pts(P, Q, delta), _line_point_integrate, _line_line_integrate
    )
    return dist, _refine_path(owp)


@njit(cache=True)
def _ifd_owp(
    P_edge_len,
    P_subedges_num,
    P_pts,
    Q_edge_len,
    Q_subedges_num,
    Q_pts,
    line_point_cost,
    line_line_cost,
):
    # Same as _ifd(), but stores paths so needs more memory.
    NP = len(P_subedges_num) + 1
    P_vert_indices = np.empty(NP, dtype=np.int_)
    P_vert_indices[0] = 0
    P_vert_indices[1:] = np.cumsum(P_subedges_num)

    NQ = len(Q_subedges_num) + 1
    Q_vert_indices = np.empty(NQ, dtype=np.int_)
    Q_vert_indices[0] = 0
    Q_vert_indices[1:] = np.cumsum(Q_subedges_num)

    # Cost containers; elements will be updated.
    P_costs = np.empty(len(P_pts), dtype=np.float_)
    P_costs[0] = 0
    Q_costs = np.empty(len(Q_pts), dtype=np.float_)
    Q_costs[0] = 0

    # Path containers; elements will be updated.
    # Path passes (NP + NQ - 3) cells, and has 4 vertices in each cell ([s, cs, ct, t]
    # or [s, c', c', t]). (NP + NQ - 4) vertices overlap.
    MAX_PATH_VERT_NUM = (NP + NQ - 3) * 4 - (NP + NQ - 4)
    P_paths = np.empty((len(P_pts), MAX_PATH_VERT_NUM, 2), dtype=np.float_)
    P_paths[0, 0] = [0, 0]
    Q_paths = np.empty((len(Q_pts), MAX_PATH_VERT_NUM, 2), dtype=np.float_)
    Q_paths[0, 0] = [0, 0]

    # TODO: parallelize this i-loop.
    p0_cost = P_costs[:1]
    p0_path = P_paths[:1, :1]
    for i in range(NP - 1):
        p_pts = P_pts[P_vert_indices[i] : P_vert_indices[i + 1] + 1]

        q0_cost = Q_costs[:1]
        q0_path = Q_paths[:1, : 3 * i + 1]
        for j in range(NQ - 1):
            q_pts = Q_pts[Q_vert_indices[j] : Q_vert_indices[j + 1] + 1]

            pc = 3 * (i + j) + 1  # path count
            if j == 0:
                p_costs = np.concatenate(
                    (
                        p0_cost,
                        P_costs[P_vert_indices[i] + 1 : P_vert_indices[i + 1] + 1],
                    )
                )
                p_paths = np.concatenate(
                    (
                        p0_path,
                        P_paths[P_vert_indices[i] + 1 : P_vert_indices[i + 1] + 1, :pc],
                    )
                )
            else:
                p_costs = P_costs[P_vert_indices[i] : P_vert_indices[i + 1] + 1].copy()
                p_paths = P_paths[
                    P_vert_indices[i] : P_vert_indices[i + 1] + 1, :pc
                ].copy()
            q_costs = np.concatenate(
                (q0_cost, Q_costs[Q_vert_indices[j] + 1 : Q_vert_indices[j + 1] + 1])
            )
            q_paths = np.concatenate(
                (
                    q0_path,
                    Q_paths[Q_vert_indices[j] + 1 : Q_vert_indices[j + 1] + 1, :pc],
                )
            )

            p1_cost, p1_path, q1_cost, q1_path = _cell_owps(
                P_edge_len[i],
                p_pts,
                p_costs,
                P_costs[P_vert_indices[i] : P_vert_indices[i + 1] + 1],
                p_paths,
                P_paths[P_vert_indices[i] : P_vert_indices[i + 1] + 1, : pc + 3],
                Q_edge_len[j],
                q_pts,
                q_costs,
                Q_costs[Q_vert_indices[j] : Q_vert_indices[j + 1] + 1],
                q_paths,
                Q_paths[Q_vert_indices[j] : Q_vert_indices[j + 1] + 1, : pc + 3],
                i == 0,
                j == 0,
                i == NP - 2,
                j == NQ - 2,
                line_point_cost,
                line_line_cost,
            )

            if j == 0:
                p0_cost = p1_cost
                p0_path = p1_path
            q0_cost = q1_cost
            q0_path = q1_path

    return Q_costs[-1], Q_paths[-1]


@njit(cache=True)
def _cell_owps(
    L1,
    p_pts,
    p_costs,
    p_costs_out,
    p_paths,
    p_paths_out,
    L2,
    q_pts,
    q_costs,
    q_costs_out,
    q_paths,
    q_paths_out,
    p_is_initial,
    q_is_initial,
    p_is_last,
    q_is_last,
    line_point_cost,
    line_line_cost,
):
    """Optimal warping paths and their costs for all points in an cell."""
    P1, Q1, u, v, b, delta_P, delta_Q = _cell_info(p_pts, L1, q_pts, L2)

    p_cost_candidates = np.empty(len(p_pts), dtype=np.float_)
    q_cost_candidates = np.empty(len(q_pts), dtype=np.float_)
    p_path_candidates = np.empty((len(p_pts), 3, 2), dtype=np.float_)
    q_path_candidates = np.empty((len(q_pts), 3, 2), dtype=np.float_)

    s = np.empty((2,), dtype=np.float_)
    t = np.empty((2,), dtype=np.float_)

    # compute upper boundary
    t[1] = L2
    if q_is_last:  # No need steiner points on upper boundary. Just check corner point.
        start_idx = len(p_pts) - 1
    else:
        start_idx = 0
    for i in range(start_idx, len(p_pts)):  # Fill p_costs_out[i]
        t[0] = delta_P * i

        s[0] = 0
        if p_is_initial:  # No steiner points on left boundary; just check [0, 0]
            q_end_idx = 1
        else:
            q_end_idx = len(q_pts)
        for j in range(0, q_end_idx):
            s[1] = delta_Q * j
            vert, cost = _st_owp(P1, u, Q1, v, b, s, t, line_point_cost, line_line_cost)
            q_cost_candidates[j] = q_costs[j] + cost
            q_path_candidates[j] = vert[1:] - vert[0]

        s[1] = 0
        if q_is_initial:  # No steiner points on bottom boundary; just check [0, 0]
            p_end_idx = 1
        else:
            p_end_idx = i + 1
        p_cost_candidates[0] = q_cost_candidates[0]  # cost from [0, 0] already known.
        p_path_candidates[0] = q_path_candidates[0]  # path from [0, 0] already known.
        for i_ in range(1, p_end_idx):  # let bottom border points be (s). (to right)
            s[0] = delta_P * i_
            vert, cost = _st_owp(P1, u, Q1, v, b, s, t, line_point_cost, line_line_cost)
            p_cost_candidates[i_] = p_costs[i_] + cost
            p_path_candidates[i_] = vert[1:] - vert[0]

        p_min_idx = np.argmin(p_cost_candidates[:p_end_idx])
        q_min_idx = np.argmin(q_cost_candidates[:q_end_idx])
        from_p_mincost = p_cost_candidates[p_min_idx]
        from_q_mincost = q_cost_candidates[q_min_idx]
        if from_p_mincost > from_q_mincost:
            mincost = from_q_mincost
            prevpath = q_paths[q_min_idx]
            minpath = q_path_candidates[q_min_idx]
        else:
            mincost = from_p_mincost
            prevpath = p_paths[p_min_idx]
            minpath = p_path_candidates[p_min_idx]
        p_costs_out[i] = mincost
        p_paths_out[i, :-3] = prevpath
        p_paths_out[i, -3:] = prevpath[-1] + minpath

    # compute right boundary
    t[0] = L1
    if p_is_last:  # No need steiner points on right boundary. Just check corner point.
        start_idx = len(q_pts) - 1
    elif q_is_initial:
        start_idx = 0
    else:  # LR corner already computed by the lower cell (MUST exclude for path track)
        start_idx = 1
    # Don't need to compute the last j (already done by P loop just above)
    for j in range(start_idx, len(q_pts) - 1):
        t[1] = delta_Q * j

        s[1] = 0
        if q_is_initial:  # No steiner points on bottom boundary; just check [0, 0]
            p_end_idx = 1
        else:
            p_end_idx = len(p_pts)
        for i in range(0, p_end_idx):
            s[0] = delta_P * i
            vert, cost = _st_owp(P1, u, Q1, v, b, s, t, line_point_cost, line_line_cost)
            p_cost_candidates[i] = p_costs[i] + cost
            p_path_candidates[i] = vert[1:] - vert[0]

        s[0] = 0
        if p_is_initial:  # No steiner points on left boundary; just check [0, 0]
            q_end_idx = 1
        else:
            q_end_idx = j + 1
        q_cost_candidates[0] = p_cost_candidates[0]  # cost from [0, 0] already known.
        q_path_candidates[0] = p_path_candidates[0]  # path from [0, 0] already known.
        for j_ in range(1, q_end_idx):  # cost from [0, 0] already known.
            s[1] = delta_Q * j_
            vert, cost = _st_owp(P1, u, Q1, v, b, s, t, line_point_cost, line_line_cost)
            q_cost_candidates[j_] = q_costs[j_] + cost
            q_path_candidates[j_] = vert[1:] - vert[0]

        p_min_idx = np.argmin(p_cost_candidates[:p_end_idx])
        q_min_idx = np.argmin(q_cost_candidates[:q_end_idx])
        from_p_mincost = p_cost_candidates[p_min_idx]
        from_q_mincost = q_cost_candidates[q_min_idx]
        if from_p_mincost > from_q_mincost:
            mincost = from_q_mincost
            prevpath = q_paths[q_min_idx]
            minpath = q_path_candidates[q_min_idx]
        else:
            mincost = from_p_mincost
            prevpath = p_paths[p_min_idx]
            minpath = p_path_candidates[p_min_idx]
        q_costs_out[j] = mincost
        q_paths_out[j, :-3] = prevpath
        q_paths_out[j, -3:] = prevpath[-1] + minpath

    q_costs_out[-1] = p_costs_out[-1]
    q_paths_out[-1] = p_paths_out[-1]

    return q_costs_out[:1], q_paths_out[:1], p_costs_out[:1], p_paths_out[:1]


@njit(cache=True)
def _st_owp(P1, u, Q1, v, b, s, t, line_point_cost, line_line_cost):
    """Optimal warping path between two points in a cell and its cost."""
    # Find steiner points in curve space
    P_s = P1 + u * s[0]
    P_t = P1 + u * t[0]
    Q_s = Q1 + v * s[1]
    Q_t = Q1 + v * t[1]

    if s[1] > s[0] + b:
        cs = np.array([s[1] - b, s[1]])
    else:
        cs = np.array([s[0], s[0] + b])
    if t[1] < t[0] + b:
        ct = np.array([t[1] - b, t[1]])
    else:
        ct = np.array([t[0], t[0] + b])

    if cs[0] < ct[0]:  # pass through lm
        P_cs = P1 + u * cs[0]
        P_ct = P1 + u * ct[0]
        Q_cs = Q1 + v * cs[1]
        Q_ct = Q1 + v * ct[1]

        if s[1] > s[0] + b:  # right
            s_to_cs = line_point_cost(P_s, P_cs, Q_s)
        else:  # up
            s_to_cs = line_point_cost(Q_s, Q_cs, P_s)

        cs_to_ct = line_line_cost(P_cs, P_ct, Q_cs, Q_ct)

        if t[1] > t[0] + b:  # up
            ct_to_t = line_point_cost(Q_ct, Q_t, P_t)
        else:  # right
            ct_to_t = line_point_cost(P_ct, P_t, Q_t)

        cost = s_to_cs + cs_to_ct + ct_to_t
        verts = np.stack((s, cs, ct, t))

    else:  # pass c'
        if s[1] > s[0] + b:  # right -> up
            cost = line_point_cost(P_s, P_t, Q_s) + line_point_cost(Q_s, Q_t, P_t)
            c_prime = np.array((t[0], s[1]))
        else:  # up -> right
            cost = line_point_cost(Q_s, Q_t, P_s) + line_point_cost(P_s, P_t, Q_t)
            c_prime = np.array((s[0], t[1]))
        # Force len=4 because homogeneous output type is easier to handle
        verts = np.stack((s, c_prime, c_prime, t))
    return verts, cost


def _sample_ifd_pts(P, Q, delta):
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
    return (
        P_edge_len,
        P_subedges_num,
        P_pts,
        Q_edge_len,
        Q_subedges_num,
        Q_pts,
    )


@njit(cache=True)
def _sample_pts(vert, delta):
    N, D = vert.shape
    vert_diff = vert[1:] - vert[:-1]
    edge_lens = np.empty(N - 1, dtype=np.float_)
    for i in range(N - 1):
        edge_lens[i] = np.linalg.norm(vert_diff[i])
    subedges_num = np.ceil(edge_lens / delta).astype(np.int_)

    pts = np.empty((np.sum(subedges_num) + 1, D), dtype=np.float_)
    count = 0
    for cell_idx in range(N - 1):
        P0 = vert[cell_idx]
        v = vert_diff[cell_idx]
        n = subedges_num[cell_idx]
        for i in range(n):
            pts[count + i] = P0 + (i / n) * v
        count += n
    pts[count] = vert[N - 1]
    return edge_lens, subedges_num, pts


@njit(cache=True)
def _cell_info(P_pts, L1, Q_pts, L2):
    P1 = P_pts[0]
    P2 = P_pts[-1]
    Q1 = Q_pts[0]
    Q2 = Q_pts[-1]

    P1P2 = P2 - P1
    Q1Q2 = Q2 - Q1

    if L1 < EPSILON:
        u = np.array([0, 0], np.float_)
    else:
        u = (P1P2) / L1
    if L2 < EPSILON:
        v = np.array([0, 0], np.float_)
    else:
        v = (Q1Q2) / L2

    # Find lm: y = x + b.
    # Can be acquired by finding points where distance is minimum.
    w = P1 - Q1
    u_dot_v = np.dot(u, v)
    if np.abs(1 - u_dot_v**2) > EPSILON:
        # Find points P(s) and Q(t) where P and Q intersects.
        # (s, t) is on y = x + b
        A = np.array([[1, -u_dot_v], [-u_dot_v, 1]], dtype=np.float_)
        B = np.array([-np.dot(u, w), np.dot(v, w)], dtype=np.float_)
        s, t = np.linalg.solve(A, B)
        b = t - s
    else:
        # P and Q are parallel; equations degenerate into s - (u.v)t = -u.w
        b = np.dot(u, w) / u_dot_v

    delta_P = L1 / (len(P_pts) - 1)
    delta_Q = L2 / (len(Q_pts) - 1)

    return P1, Q1, u, v, b, delta_P, delta_Q


@njit(cache=True)
def _refine_path(path):
    prev = path[0]
    count = 1
    for i in range(1, len(path)):
        current = path[i]
        if np.all(prev == current):
            continue
        else:
            path[count] = current
            prev = current
            count += 1
    return path[:count]


@njit(cache=True)
def _line_point_integrate(a, b, p):
    r"""Analytic integration from AP to BP.

    .. math::
        \int_0^1 \lVert (A - P) + (B - A) t \rVert \cdot \lVert (B - A) \rVert dt
    """
    # Goal: integrate sqrt(A*t**2 + B*t + C) * sqrt(A) dt over t [0, 1]
    # where A = dot(ab, ab), B = 2 * dot(pa, ab) and C = dot(pa, pa).
    # Can be simplified to A * integral sqrt(t**2 + B/A*t + C/A) dt.
    # Rewrite: A * integral sqrt(t**2 + B*t + C) dt over t [0, 1]
    # where B = 2 * dot(pa, ab) / A and C = dot(pa, pa) / A.
    ab = b - a
    A = np.dot(ab, ab)
    if A < EPSILON:
        # Degenerate: ab does not form line segement.
        return 0
    pa = a - p
    if np.abs(cross2d(ab, pa)) < EPSILON:
        # Degenerate: a, b, p all on a same line.
        t = np.dot(ab, pa) / A
        L1 = np.dot(pa, pa)
        bp = p - b
        L2 = np.dot(bp, bp)
        if t < 0:
            return (L2 - L1) / 2
        elif t > 1:
            return (L1 - L2) / 2
        else:
            return (L1 + L2) / 2
    B = 2 * np.dot(ab, pa) / A
    C = np.dot(pa, pa) / A
    integ = (
        4 * np.sqrt(1 + B + C)
        + 2 * B * (-np.sqrt(C) + np.sqrt(1 + B + C))
        - (B**2 - 4 * C)
        * np.log((2 + B + 2 * np.sqrt(1 + B + C)) / (B + 2 * np.sqrt(C)))
    ) / 8
    return A * integ


@njit(cache=True)
def _line_line_integrate(a, b, c, d):
    r"""Analytic integration from AC to BD.

    .. math::
        \int_0^1 \lVert (A - C) + (B - A + C - D)t \rVert \cdot
        \left( \lVert B - A \rVert + \lVert D - C \rVert \right) dt
    """
    # Goal: integrate sqrt(A*t**2 + B*t + C) * (sqrt(D) + sqrt(E)) dt over t [0, 1]
    # where A = dot(vu, vu), B = 2 * dot(vu, w), C = dot(w, w), D = dot(u, u),
    # and E = dot(v, v); where u = b - a, v = d - c and w = a - c.
    # Rewrite: (sqrt(A*D) + sqrt(A*E)) * integral sqrt(t**2 + B*t + C) dt over t [0, 1]
    # where B = 2 * dot(vu, w) / A and C = dot(w, w) / A
    u, v, w = b - a, d - c, a - c
    vu = u - v
    A = np.dot(vu, vu)
    C, D, E = np.dot(w, w), np.dot(u, u), np.dot(v, v)
    if A < EPSILON:
        # Degenerate: ab and cd has same direction and magnitude
        return np.sqrt(C * D) + np.sqrt(C * E)
    B, C = 2 * np.dot(vu, w) / A, C / A
    if np.abs(B) < EPSILON and C < EPSILON:
        # Degenerate: B and C are 0 (either w = 0 or A is too large)
        return (np.sqrt(A * D) + np.sqrt(A * E)) / 2
    numer = 2 + B + 2 * np.sqrt(1 + B + C)
    denom = B + 2 * np.sqrt(C)
    if np.abs(numer) < EPSILON or np.abs(denom) < EPSILON:
        # Degenerate: u-v and w are on the opposite direction
        # B**2 = 4*C, therefore integral sqrt((t + B/2)**2) dt over t [0, 1]
        if B > 0 or B < -2:
            integ = (np.abs(B / 2) + np.abs(1 + B / 2)) / 2
        else:
            integ = (np.abs(B / 2) ** 2) / 2 + (np.abs(1 + B / 2) ** 2) / 2
    else:
        integ = (
            4 * np.sqrt(1 + B + C)
            + 2 * B * (-np.sqrt(C) + np.sqrt(1 + B + C))
            - (B**2 - 4 * C) * np.log(numer / denom)
        ) / 8
    return (np.sqrt(A * D) + np.sqrt(A * E)) * integ
