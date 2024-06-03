"""Integral Fréchet distance."""

import numpy as np
from numba import njit
from numba.np.extensions import cross2d

__all__ = [
    "ifd",
    "ifd_owp",
]


EPSILON = np.finfo(np.float_).eps


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
    \times [0, 1]`. In this implementation, we choose the Euclidean distance
    as :math:`\delta` and the Manhattan norm as :math:`\lVert \cdot \rVert`.

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
        The integral Fréchet distance between P and Q.

    Notes
    -----
    This function implements the algorithm of Brankovic et al [#]_.

    References
    ----------
    .. [#] Brankovic, M., et al. "(k, l)-Medians Clustering of Trajectories Using
       Continuous Dynamic Time Warping." Proceedings of the 28th International
       Conference on Advances in Geographic Information Systems. 2020.

    See Also
    --------
    ifd_owp : Integral Fréchet distance with optimal warping path.

    Examples
    --------
    >>> ifd([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]], 0.1)
    2.0
    """
    P = np.asarray(P, dtype=np.float_)
    Q = np.asarray(Q, dtype=np.float_)

    P_subedges_num, P_pts = _sample_pts(P, delta)
    Q_subedges_num, Q_pts = _sample_pts(Q, delta)
    return _ifd(P_subedges_num, P_pts, Q_subedges_num, Q_pts)


@njit(cache=True)
def _ifd(P_subedges_num, P_pts, Q_subedges_num, Q_pts):
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
    p0 = P_costs[0]  # will be updated during i-loop when j == 0
    for i in range(NP - 1):
        p_pts = P_pts[P_vert_indices[i] : P_vert_indices[i + 1] + 1]

        q0 = Q_costs[0]  # will be updated during j-loop
        for j in range(NQ - 1):
            q_pts = Q_pts[Q_vert_indices[j] : Q_vert_indices[j + 1] + 1]

            if j == 0:
                p_costs = np.concatenate(
                    (
                        np.array((p0,)),
                        P_costs[P_vert_indices[i] + 1 : P_vert_indices[i + 1] + 1],
                    )
                )
            else:
                p_costs = P_costs[P_vert_indices[i] : P_vert_indices[i + 1] + 1]
            q_costs = np.concatenate(
                (
                    np.array((q0,)),
                    Q_costs[Q_vert_indices[j] + 1 : Q_vert_indices[j + 1] + 1],
                )
            )

            p1, q1 = _cell_owcs(
                p_pts,
                p_costs,
                P_costs[P_vert_indices[i] : P_vert_indices[i + 1] + 1],
                q_pts,
                q_costs,
                Q_costs[Q_vert_indices[j] : Q_vert_indices[j + 1] + 1],
                j == 0,
                i == 0,
            )

            # store for the next loops
            if j == 0:
                p0 = p1
            q0 = q1

    return Q_costs[-1]


@njit(cache=True)
def _cell_owcs(
    p_pts, p_costs, p_costs_out, q_pts, q_costs, q_costs_out, p_is_initial, q_is_initial
):
    """Apply _st_owc() to border points in a cell."""
    P1, Q1, L1, L2, u, v, b, delta_P, delta_Q = _cell_info(p_pts, q_pts)

    # Cost arrays not initialized yet.
    if p_is_initial:
        p_costs[-1] = p_costs[0] + _line_point_integrate(p_pts[0], p_pts[-1], q_pts[0])
    if q_is_initial:
        q_costs[-1] = q_costs[0] + _line_point_integrate(q_pts[0], q_pts[-1], p_pts[0])

    # Will be reused for each border point (t) to find best starting point (s).
    cost_candidates = np.empty(len(p_pts) + len(q_pts) - 1, dtype=np.float_)

    p_costs_out[0] = q_costs[-1]
    for i in range(1, len(p_pts)):  # Fill p_costs_out[i]
        t = np.array([delta_P * i, L2], dtype=np.float_)
        count = 0

        if q_is_initial:  # no steiner points on left boundary; just check [0, 0]
            s = np.array([0, 0], dtype=np.float_)
            cost = _st_owc(P1, u, Q1, v, b, s, t)
            cost_candidates[count] = q_costs[0] + cost
            count += 1
        else:
            for j in range(len(q_pts)):  # let left border points be (s). (to up)
                s = np.array([0, delta_Q * j], dtype=np.float_)
                cost = _st_owc(P1, u, Q1, v, b, s, t)
                cost_candidates[count] = q_costs[j] + cost
                count += 1

        if p_is_initial:  # no steiner points on bottom boundary
            pass  # s = [0, 0] already visited
        else:
            for i_ in range(i):  # let bottom border points be (s). (to right)
                s = np.array([delta_P * (i_ + 1), 0], dtype=np.float_)
                cost = _st_owc(P1, u, Q1, v, b, s, t)
                cost_candidates[count] = p_costs[i_] + cost
                count += 1

        p_costs_out[i] = np.min(cost_candidates[:count])

    q_costs_out[0] = p_costs[-1]
    # Don't need to compute the last j (already done by P loop just above)
    for j in range(1, len(q_pts) - 1):
        t = np.array([L1, delta_Q * j], dtype=np.float_)
        count = 0

        if p_is_initial:  # no steiner points on down boundary; just check [0, 0]
            s = np.array([0, 0], dtype=np.float_)
            cost = _st_owc(P1, u, Q1, v, b, s, t)
            cost_candidates[count] = p_costs[0] + cost
            count += 1
        else:
            for i in range(len(p_pts)):  # let down border points be (s). (to right)
                s = np.array([delta_P * i, 0], dtype=np.float_)
                cost = _st_owc(P1, u, Q1, v, b, s, t)
                cost_candidates[count] = p_costs[i] + cost
                count += 1

        if q_is_initial:  # no steiner points on bottom boundary
            pass  # s = [0, 0] already visited
        else:
            for j_ in range(j):  # let left border points be (s). (to up)
                s = np.array([0, delta_Q * (j_ + 1)], dtype=np.float_)
                cost = _st_owc(P1, u, Q1, v, b, s, t)
                cost_candidates[count] = q_costs[j_] + cost
                count += 1

        q_costs_out[j] = np.min(cost_candidates[:count])
    q_costs_out[-1] = p_costs_out[-1]

    return p_costs[-1], q_costs[-1]


@njit(cache=True)
def _st_owc(P1, u, Q1, v, b, s, t):
    """Optimal warping cost from s to t in a cell (without its path)."""
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
            s_to_cs = _line_point_integrate(P_s, P_cs, Q_s)
        else:  # up
            s_to_cs = _line_point_integrate(Q_s, Q_cs, P_s)

        cs_to_ct = _line_line_integrate(P_cs, P_ct, Q_cs, Q_ct)

        if t[1] > t[0] + b:  # up
            ct_to_t = _line_point_integrate(Q_ct, Q_t, P_t)
        else:  # right
            ct_to_t = _line_point_integrate(P_ct, P_t, Q_t)

        ret = s_to_cs + cs_to_ct + ct_to_t

    else:  # pass c'
        if s[1] > s[0] + b:  # right -> up
            ret = _line_point_integrate(P_s, P_t, Q_s) + _line_point_integrate(
                Q_s, Q_t, P_t
            )
        else:  # up -> right
            ret = _line_point_integrate(Q_s, Q_t, P_s) + _line_point_integrate(
                P_s, P_t, Q_t
            )
    return ret


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
        The integral Fréchet distance between P and Q.
    owp : ndarray
        Optimal warping path.
    """
    P = np.asarray(P, dtype=np.float_)
    Q = np.asarray(Q, dtype=np.float_)

    P_subedges_num, P_pts = _sample_pts(P, delta)
    P_costs = _edge_costs(P_pts, P_subedges_num, Q[0])

    Q_subedges_num, Q_pts = _sample_pts(Q, delta)
    Q_costs = _edge_costs(Q_pts, Q_subedges_num, P[0])

    return _ifd_owp(P_subedges_num, P_pts, P_costs, Q_subedges_num, Q_pts, Q_costs)


@njit(cache=True)
def _ifd_owp(P_subedges_num, P_pts, P_costs, Q_subedges_num, Q_pts, Q_costs):
    # Same as _ifd(), but stores paths so needs more memory.
    # NP = len(P_subedges_num) + 1
    # P_vert_indices = np.empty(NP, dtype=np.int_)
    # P_vert_indices[0] = 0
    # P_vert_indices[1:] = np.cumsum(P_subedges_num)

    # NQ = len(Q_subedges_num) + 1
    # Q_vert_indices = np.empty(NQ, dtype=np.int_)
    # Q_vert_indices[0] = 0
    # Q_vert_indices[1:] = np.cumsum(Q_subedges_num)

    # # Path passes (NP + NQ - 1) cells, and has 4 vertices in each cell.
    # # (our vertices are [s, cs, ct, t] or [s, c', c', t])
    # # There are (NP + NQ - 2) boundaries between cells where vertices overlap.
    # MAX_PATH_VERT_NUM = (NP + NQ - 1) * 4 - (NP + NQ - 2)
    # P_paths = np.empty(
    #     (len(P_pts), MAX_PATH_VERT_NUM, P_pts.shape[-1]), dtype=np.float_
    # )
    # Q_paths = np.empty(
    #     (len(Q_pts), MAX_PATH_VERT_NUM, Q_pts.shape[-1]), dtype=np.float_
    # )
    ...


@njit(cache=True)
def _cell_owps(
    P_pts,
    P_costs,
    P_costs_out,
    P_paths,
    P_paths_out,
    Q_pts,
    Q_costs,
    Q_costs_out,
    Q_paths,
    Q_paths_out,
):
    """Apply _st_owp() to border points in a cell."""
    P1, Q1, L1, L2, u, v, b, delta_P, delta_Q = _cell_info(P_pts, Q_pts)

    costs = np.empty(len(P_pts) + len(Q_pts) - 1, dtype=np.float_)
    paths = np.empty((len(P_pts) + len(Q_pts) - 1, 4, 2), dtype=np.float_)

    P_costs_out[0] = Q_costs[-1]
    for i in range(1, len(P_pts)):
        t = np.array([delta_P * i, L2], dtype=np.float_)
        for j in range(len(Q_pts)):
            s = np.array([0, delta_Q * j], dtype=np.float_)
            cost, path = _st_owp(P1, u, Q1, v, b, s, t)
            costs[j] = Q_costs[j] + cost
            paths[j] = path
        for i_ in range(i):
            s = np.array([delta_P * (i_ + 1), 0], dtype=np.float_)
            cost, path = _st_owp(P1, u, Q1, v, b, s, t)
            costs[len(Q_pts) + i_] = P_costs[i_] + cost
            paths[len(Q_pts) + i_] = path
        min_idx = np.argmin(costs[: len(Q_pts) + i_])
        P_costs_out[i] = costs[min_idx]
        # TODO: P_paths_out[i] = np.concatenate([prev_P_paths[min_idx], paths[min_idx]])

    Q_costs_out[0] = P_costs[-1]
    for j in range(1, len(Q_pts) - 1):
        t = np.array([L1, delta_Q * j], dtype=np.float_)
        for i in range(len(P_pts)):  # let down border points be (s). (to right)
            s = np.array([delta_P * i, 0], dtype=np.float_)
            cost, path = _st_owp(P1, u, Q1, v, b, s, t)
            costs[i] = P_costs[i] + cost
            paths[i] = path
        for j_ in range(j):  # let left border points be (s). (to up)
            s = np.array([0, delta_Q * (j_ + 1)], dtype=np.float_)
            cost, path = _st_owp(P1, u, Q1, v, b, s, t)
            costs[len(P_pts) + j_] = Q_costs[j_] + cost
            paths[len(P_pts) + j_] = path
        min_idx = np.argmin(costs[: len(P_pts) + j_])
        Q_costs_out[j] = costs[min_idx]
        # TODO: Q_paths_out[j] = np.concatenate([prev_Q_paths[min_idx], paths[min_idx]])
    Q_costs_out[-1] = P_costs_out[-1]
    Q_paths_out[-1] = P_paths_out[-1]


@njit(cache=True)
def _st_owp(P1, u, Q1, v, b, s, t):
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
    if t[1] > t[1] + b:
        ct = np.array([t[1] - b, t[1]])
    else:
        ct = np.array([t[0], t[0] + b])

    if cs[0] < ct[0]:  # pass through lm
        P_cs = P1 + u * cs[0]
        P_ct = P1 + u * ct[0]
        Q_cs = Q1 + v * cs[1]
        Q_ct = Q1 + v * ct[1]

        if s[1] > s[0] + b:  # right
            s_to_cs = _line_point_integrate(P_s, P_cs, Q_s)
        else:  # up
            s_to_cs = _line_point_integrate(Q_s, Q_cs, P_s)

        cs_to_ct = _line_line_integrate(P_cs, P_ct, Q_cs, Q_ct)

        if t[1] > t[0] + b:  # up
            ct_to_t = _line_point_integrate(Q_ct, Q_t, P_t)
        else:  # right
            ct_to_t = _line_point_integrate(P_ct, P_t, Q_t)

        cost = s_to_cs + cs_to_ct + ct_to_t
        verts = np.stack((s, cs, ct, t))

    else:  # pass c'
        if s[1] > s[0] + b:  # right -> up
            cost = _line_point_integrate(P_s, P_t, Q_s) + _line_point_integrate(
                Q_s, Q_t, P_t
            )
            c_prime = np.array((t[0], s[1]))
        else:  # up -> right
            cost = _line_point_integrate(Q_s, Q_t, P_s) + _line_point_integrate(
                P_s, P_t, Q_t
            )
            c_prime = np.array((s[0], t[1]))
        # Force len=4 because homogeneous output type is easier to handle
        verts = np.stack((s, c_prime, c_prime, t))
    return verts, cost


@njit(cache=True)
def _sample_pts(vert, delta):
    N, D = vert.shape
    vert_diff = np.empty((N - 1, D), dtype=np.float_)
    for i in range(N - 1):
        vert_diff[i] = vert[i + 1] - vert[i]
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
    return subedges_num, pts


@njit(cache=True)
def _edge_costs(pts, subedges_num, p):
    pts_costs = np.empty(len(pts), dtype=np.float_)
    count = 0
    pts_costs[0] = 0
    for n in subedges_num:
        a = pts[count]
        for i in range(n):
            b = pts[count + i + 1]
            integ = _line_point_integrate(a, b, p)
            pts_costs[count + i + 1] = pts_costs[count] + integ
        count += n
    return pts_costs


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
    ap = p - a
    if np.abs(cross2d(ab, ap)) < EPSILON:
        # Degenerate: a, b, p all on a same line.
        t = np.dot(ab, ap) / A
        L1 = np.dot(ap, ap)
        bp = p - b
        L2 = np.dot(bp, bp)
        if t < 0:
            return (L2 - L1) / 2
        elif t > 1:
            return (L1 - L2) / 2
        else:
            return (L1 + L2) / 2
    pa = -ap
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
    if B < EPSILON and C < EPSILON:
        # Degenerate: B and C are 0 (either w = 0 or A is too large)
        return (np.sqrt(A * D) + np.sqrt(A * E)) / 2
    denom = B + 2 * np.sqrt(C)
    if denom < EPSILON:
        # Degenerate: u-v and w are on the opposite direction
        if B > 0 or B < -2:
            integ = (-1 - B) / 2
        else:
            integ = (2 + 2 * B + B**2) / 4
    else:
        integ = (
            4 * np.sqrt(1 + B + C)
            + 2 * B * (-np.sqrt(C) + np.sqrt(1 + B + C))
            - (B**2 - 4 * C) * np.log((2 + B + 2 * np.sqrt(1 + B + C)) / denom)
        ) / 8
    return (np.sqrt(A * D) + np.sqrt(A * E)) * integ


@njit(cache=True)
def _cell_info(P_pts, Q_pts):
    P1 = P_pts[0]
    P2 = P_pts[-1]
    Q1 = Q_pts[0]
    Q2 = Q_pts[-1]

    P1P2 = P2 - P1
    Q1Q2 = Q2 - Q1
    L1 = np.linalg.norm(P1P2)
    L2 = np.linalg.norm(Q1Q2)

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
    if np.abs(cross2d(u, v)) > EPSILON:
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

    return P1, Q1, L1, L2, u, v, b, delta_P, delta_Q
