"""Integral Fréchet distance."""

import numpy as np
from numba import njit
from numba.np.extensions import cross2d

__all__ = [
    "ifd",
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

    Examples
    --------
    >>> ifd([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]], 0.1)
    1.9...
    """
    P = np.asarray(P, dtype=np.float_)
    Q = np.asarray(Q, dtype=np.float_)

    P_subedges_num, P_pts = _sample_pts(P, delta)
    P_costs = _edge_costs(P_pts, P_subedges_num, Q[0])

    Q_subedges_num, Q_pts = _sample_pts(Q, delta)
    Q_costs = _edge_costs(Q_pts, Q_subedges_num, P[0])

    return _ifd(P_subedges_num, P_pts, P_costs, Q_subedges_num, Q_pts, Q_costs)


@njit(cache=True)
def _ifd(P_subedges_num, P_pts, P_costs, Q_subedges_num, Q_pts, Q_costs):
    NP = len(P_subedges_num) + 1
    P_vert_indices = np.empty(NP, dtype=np.int_)
    P_vert_indices[0] = 0
    P_vert_indices[1:] = np.cumsum(P_subedges_num)

    NQ = len(Q_subedges_num) + 1
    Q_vert_indices = np.empty(NQ, dtype=np.int_)
    Q_vert_indices[0] = 0
    Q_vert_indices[1:] = np.cumsum(Q_subedges_num)

    # Instead of constructing the full steiner point arrays of (N, M), construct
    # (N,) arrays and keep updating values to reduce memory usage.
    # Careful: adjacent cells share corner points (current_cell[-1] == next_cell[0]).
    # Former cell MUST NOT update the shared points! Only the latter cell should.
    for i in range(NP - 1):  # TODO: parallelize this loop
        p_pts = P_pts[P_vert_indices[i] : P_vert_indices[i + 1] + 1]
        corner_costs = np.empty(NQ, dtype=np.float_)
        corner_costs[0] = P_costs[P_vert_indices[i + 1]]
        for j in range(NQ - 1):
            q_pts = Q_pts[Q_vert_indices[j] : Q_vert_indices[j + 1] + 1]

            p_costs = np.empty(P_subedges_num[i] + 1, dtype=np.float_)
            p_costs[:-1] = P_costs[P_vert_indices[i] : P_vert_indices[i + 1]]
            p_costs[-1] = corner_costs[j]

            q_costs = Q_costs[Q_vert_indices[j] : Q_vert_indices[j + 1] + 1]

            new_p_costs, new_q_costs = _cell_pts_costs(p_pts, p_costs, q_pts, q_costs)

            P_costs[P_vert_indices[i] : P_vert_indices[i + 1]] = new_p_costs[:-1]
            Q_costs[Q_vert_indices[j] : Q_vert_indices[j + 1]] = new_q_costs[:-1]
            corner_costs[j + 1] = new_q_costs[-1]
        # Update corner costs to Q point costs.
        # Q_costs now perfectly represents (i+1)-th column.
        Q_costs[Q_vert_indices] = corner_costs
        # TODO: crop i-th cell from P-related arrays (because i-th colum is cleared).
        # It may make parallalization difficult, though...

    return corner_costs[-1]


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
def _cell_pts_costs(P_pts, P_costs, Q_pts, Q_costs):
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

    # Find lm: y = x + b
    w = Q1 - P1
    if np.abs(cross2d(u, v)) > EPSILON:
        # b = -x + y where A.[x, y] = B
        u_dot_v = np.dot(u, v)
        A = np.array([[1, -u_dot_v], [u_dot_v, -1]], dtype=np.float_)
        B = np.array([-np.dot(u, w), -np.dot(v, w)], dtype=np.float_)
        x, y = np.linalg.solve(A, B)
        b = -x + y
    else:
        # P and Q are parallel.
        # Equations degenerate into x - y = -u.w, therefore b = u.w
        b = np.dot(u, w)

    delta_P = L1 / (len(P_pts) - 1)
    delta_Q = L2 / (len(Q_pts) - 1)

    new_P_costs = np.empty_like(P_costs)
    new_Q_costs = np.empty_like(Q_costs)

    new_P_costs[0] = Q_costs[-1]
    for i in range(1, len(new_P_costs)):
        t = np.array([delta_P * i, L2], dtype=np.float_)
        costs = np.empty(len(Q_pts) + i, dtype=np.float_)
        for j in range(len(Q_pts)):
            s = np.array([0, delta_Q * j], dtype=np.float_)
            costs[j] = Q_costs[j] + _cell_owp_integral(P1, u, Q1, v, b, s, t)
        for i_ in range(i):
            s = np.array([delta_P * (i_ + 1), 0], dtype=np.float_)
            costs[len(Q_pts) + i_] = P_costs[i_] + _cell_owp_integral(
                P1, u, Q1, v, b, s, t
            )
        new_P_costs[i] = np.min(costs)

    new_Q_costs[0] = P_costs[-1]
    for j in range(1, len(new_Q_costs) - 1):
        t = np.array([L1, delta_Q * j], dtype=np.float_)
        costs = np.empty(len(P_pts) + j, dtype=np.float_)
        for i in range(len(P_pts)):
            s = np.array([delta_P * i, 0], dtype=np.float_)
            costs[i] = P_costs[i] + _cell_owp_integral(P1, u, Q1, v, b, s, t)
        for j_ in range(j):
            s = np.array([0, delta_Q * (j_ + 1)], dtype=np.float_)
            costs[len(P_pts) + j_] = Q_costs[j_] + _cell_owp_integral(
                P1, u, Q1, v, b, s, t
            )
        new_Q_costs[j] = np.min(costs)
    new_Q_costs[-1] = new_P_costs[-1]

    return new_P_costs, new_Q_costs


@njit(cache=True)
def _cell_owp_integral(P1, u, Q1, v, b, s, t):
    """Integral along optimal warping path between two points in a cell."""
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
