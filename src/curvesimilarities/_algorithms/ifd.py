import numpy as np
from numba import njit

NAN = np.float64(np.nan)


@njit(cache=True)
def _ifd_acm_1d(P, Q, delta, dist_type):
    if len(P.shape) != 2:
        raise ValueError("P must be a 2-dimensional array.")
    if len(Q.shape) != 2:
        raise ValueError("Q must be a 2-dimensional array.")
    if P.shape[1] != Q.shape[1]:
        raise ValueError("P and Q must have the same number of columns.")

    P, Q = P.astype(np.float64), Q.astype(np.float64)
    P_subedges_num = _steiner_subedges(P, delta)
    Q_subedges_num = _steiner_subedges(Q, delta)

    p, q = len(P), len(Q)
    P_idxs = np.empty(p, dtype=np.int_)
    if p > 0:
        P_idxs[0] = 0
        P_idxs[1:] = np.cumsum(P_subedges_num)
    Q_idxs = np.empty(q, dtype=np.int_)
    if q > 0:
        Q_idxs[0] = 0
        Q_idxs[1:] = np.cumsum(Q_subedges_num)

    pp = 0 if p == 0 else np.sum(P_subedges_num) + 1
    qq = 0 if q == 0 else np.sum(Q_subedges_num) + 1
    B = np.empty(pp, dtype=np.float64)
    if pp > 0:
        B[0] = 0
        B[-1] = NAN
    L = np.empty(qq, dtype=np.float64)
    if qq > 0:
        L[0] = 0
        L[-1] = NAN

    # TODO: parallelize this i-loop.
    # Must ensure that cell (i - 1, j) is computed before (i, j).
    p0 = B[:1]  # will be updated during i-loop when j == 0
    for i in range(p - 1):
        p_pts = _steiner_pts(P[i : i + 2], P_subedges_num[i])
        q0 = L[:1]  # will be updated during j-loop
        for j in range(q - 1):
            q_pts = _steiner_pts(Q[j : j + 2], Q_subedges_num[j])

            if j == 0:
                p_costs = np.concatenate((p0, B[P_idxs[i] + 1 : P_idxs[i + 1] + 1]))
            else:
                p_costs = B[P_idxs[i] : P_idxs[i + 1] + 1].copy()
            q_costs = np.concatenate((q0, L[Q_idxs[j] + 1 : Q_idxs[j + 1] + 1]))

            p1, q1 = _update_cell(
                p_pts,
                q_pts,
                p_costs,
                B[P_idxs[i] : P_idxs[i + 1] + 1],
                q_costs,
                L[Q_idxs[j] : Q_idxs[j + 1] + 1],
                i == 0,
                j == 0,
                i == p - 2,
                j == q - 2,
                dist_type,
            )

            # store for the next loops
            if j == 0:
                p0 = p1
            q0 = q1

    return B, L


@njit(cache=True)
def _steiner_subedges(vert, delta):
    N, _ = vert.shape
    if N == 0:
        N = 1
    edge_lens = np.empty(N - 1, dtype=np.float64)
    for i in range(N - 1):
        edge_lens[i] = np.linalg.norm(vert[i + 1] - vert[i])
    subedges_num = np.ceil(edge_lens / delta).astype(np.int_)
    return subedges_num


@njit(cache=True)
def _steiner_pts(P1P2, n):
    _, D = P1P2.shape
    pts = np.empty((n + 1, D), dtype=np.float64)
    v = P1P2[1] - P1P2[0]
    for i in range(n):
        pts[i] = P1P2[0] + (i / n) * v
    pts[n] = P1P2[1]
    return pts


@njit(cache=True)
def _update_cell(
    p_pts,
    q_pts,
    p_costs,
    p_costs_out,
    q_costs,
    q_costs_out,
    p_is_initial,
    q_is_initial,
    p_is_last,
    q_is_last,
    dist_type,
):
    P1, Q1, L1, L2, u, v, b, delta_P, delta_Q = _cell_info(p_pts, q_pts)

    # Will be reused for each border point (t) to find best starting point (s).
    p_cost_candidates = np.empty(len(p_pts), dtype=np.float64)
    q_cost_candidates = np.empty(len(q_pts), dtype=np.float64)
    s = np.empty((2,), dtype=np.float64)
    t = np.empty((2,), dtype=np.float64)

    # compute upper boundary
    t[1] = L2
    if p_is_initial:  # No steiner points on left boundary; just check [0, 0]
        q_end_idx = 1
    else:
        q_end_idx = len(q_pts)
    if q_is_last:  # No need steiner points on upper boundary. Just check corner point.
        p_start_idx = len(p_pts) - 1
    else:
        p_start_idx = 0
    for i in range(p_start_idx, len(p_pts)):  # Fill p_costs_out[i]
        t[0] = delta_P * i

        s[0] = 0
        for j in range(0, q_end_idx):
            s[1] = delta_Q * j
            cost = _owc_in_cell(s, t, P1, Q1, u, v, b, dist_type)
            q_cost_candidates[j] = q_costs[j] + cost

        s[1] = 0
        if q_is_initial:  # No steiner points on bottom boundary; just check [0, 0]
            p_end_idx = 1
        else:
            p_end_idx = i + 1
        p_cost_candidates[0] = q_cost_candidates[0]  # cost from [0, 0] already known.
        for i_ in range(1, p_end_idx):  # let bottom border points be (s). (to right)
            s[0] = delta_P * i_
            cost = _owc_in_cell(s, t, P1, Q1, u, v, b, dist_type)
            p_cost_candidates[i_] = p_costs[i_] + cost

        p_costs_out[i] = min(
            np.min(p_cost_candidates[:p_end_idx]), np.min(q_cost_candidates[:q_end_idx])
        )

    # compute right boundary
    t[0] = L1
    if p_is_last:  # No need steiner points on right boundary. Just check corner point.
        q_start_idx = len(q_pts) - 1
    elif q_is_initial:
        q_start_idx = 0
    else:  # LR corner already computed by the lower cell
        q_start_idx = 1
    if q_is_initial:  # No steiner points on bottom boundary; just check [0, 0]
        p_end_idx = 1
    else:
        p_end_idx = len(p_pts)
    # Don't need to compute the last j (already done by P loop just above)
    for j in range(q_start_idx, len(q_pts) - 1):
        t[1] = delta_Q * j

        s[1] = 0
        for i in range(0, p_end_idx):
            s[0] = delta_P * i
            cost = _owc_in_cell(s, t, P1, Q1, u, v, b, dist_type)
            p_cost_candidates[i] = p_costs[i] + cost

        s[0] = 0
        if p_is_initial:  # No steiner points on left boundary; just check [0, 0]
            q_end_idx = 1
        else:
            q_end_idx = j + 1
        q_cost_candidates[0] = p_cost_candidates[0]  # cost from [0, 0] already known.
        for j_ in range(1, q_end_idx):  # cost from [0, 0] already known.
            s[1] = delta_Q * j_
            cost = _owc_in_cell(s, t, P1, Q1, u, v, b, dist_type)
            q_cost_candidates[j_] = q_costs[j_] + cost

        q_costs_out[j] = min(
            np.min(p_cost_candidates[:p_end_idx]), np.min(q_cost_candidates[:q_end_idx])
        )
    q_costs_out[-1] = p_costs_out[-1]

    # Lower-right corner and upper-left corner of cells.
    return q_costs_out[:1], p_costs_out[:1]


@njit(cache=True)
def _cell_info(P_pts, Q_pts):
    P1, P2 = P_pts[0], P_pts[-1]
    Q1, Q2 = Q_pts[0], Q_pts[-1]
    P1P2 = P2 - P1
    Q1Q2 = Q2 - Q1
    L1 = np.linalg.norm(P1P2)
    L2 = np.linalg.norm(Q1Q2)

    if L1 == 0:
        u = np.array([0, 0], np.float64)
    else:
        u = (P1P2) / L1
    if L2 == 0:
        v = np.array([0, 0], np.float64)
    else:
        v = (Q1Q2) / L2

    # Find lm: y = x + b
    w = P1 - Q1
    uv = np.dot(u, v)
    if uv == 1:
        # P and Q are parallel; equations degenerate into s - (u.v)t = -u.w
        b = np.dot(u, w)
    elif uv == -1:
        # P and Q are antiparallel.
        # Any value is OK, so just set b=0.
        b = np.float64(0)
    else:
        # P and Q intersects.
        # Find points P(s) and Q(t) where P and Q intersects.
        # (s, t) is on y = x + b
        A = np.array([[1, -uv], [-uv, 1]], dtype=np.float64)
        B = np.array([-np.dot(u, w), np.dot(v, w)], dtype=np.float64)
        s, t = np.linalg.solve(A, B)
        b = t - s

    delta_P = L1 / (len(P_pts) - 1)
    delta_Q = L2 / (len(Q_pts) - 1)
    return P1, Q1, L1, L2, u, v, b, delta_P, delta_Q


@njit(cache=True)
def _owc_in_cell(s, t, P1, Q1, u, v, b, dist_type):
    P_s = P1 + u * s[0]
    P_t = P1 + u * t[0]
    Q_s = Q1 + v * s[1]
    Q_t = Q1 + v * t[1]

    if s[1] > s[0] + b:
        cs_x, cs_y = s[1] - b, s[1]
    else:
        cs_x, cs_y = s[0], s[0] + b
    if t[1] < t[0] + b:
        ct_x, ct_y = t[1] - b, t[1]
    else:
        ct_x, ct_y = t[0], t[0] + b

    if cs_x < ct_x:  # pass through lm
        P_cs = P1 + u * cs_x
        P_ct = P1 + u * ct_x
        Q_cs = Q1 + v * cs_y
        Q_ct = Q1 + v * ct_y

        if s[1] > s[0] + b:  # right
            s_to_cs = _linepoint_cost(P_s, P_cs, Q_s, dist_type)
        else:  # up
            s_to_cs = _linepoint_cost(Q_s, Q_cs, P_s, dist_type)

        cs_to_ct = _lineline_cost(P_cs, P_ct, Q_cs, Q_ct, dist_type)

        if t[1] > t[0] + b:  # up
            ct_to_t = _linepoint_cost(Q_ct, Q_t, P_t, dist_type)
        else:  # right
            ct_to_t = _linepoint_cost(P_ct, P_t, Q_t, dist_type)

        cost = s_to_cs + cs_to_ct + ct_to_t

    else:  # pass c'
        if s[1] > s[0] + b:  # right -> up
            cost1 = _linepoint_cost(P_s, P_t, Q_s, dist_type)
            cost2 = _linepoint_cost(Q_s, Q_t, P_t, dist_type)
        else:  # up -> right
            cost1 = _linepoint_cost(Q_s, Q_t, P_s, dist_type)
            cost2 = _linepoint_cost(P_s, P_t, Q_t, dist_type)

        cost = cost1 + cost2
    return cost


@njit(cache=True)
def _linepoint_cost(a, b, p, dist_type):
    if dist_type == "euclidean":
        # TODO: implement numerical integration
        raise NotImplementedError
    elif dist_type == "squared_euclidean":
        ab = b - a
        pa = a - p
        A = np.dot(ab, ab)
        B = 2 * np.dot(ab, pa)
        C = np.dot(pa, pa)
        ret = (A / 3 + B / 2 + C) * np.sqrt(A)
    else:
        raise ValueError("Unknown type of distance.")
    return ret


@njit(cache=True)
def _lineline_cost(a, b, c, d, dist_type):
    if dist_type == "euclidean":
        # TODO: implement numerical integration
        raise NotImplementedError
    elif dist_type == "squared_euclidean":
        u = b - a
        v = d - c
        w = a - c
        vu = u - v
        A = np.dot(vu, vu)
        B = 2 * np.dot(vu, w)
        C = np.dot(w, w)
        D = np.dot(u, u)
        E = np.dot(v, v)
        ret = (A / 3 + B / 2 + C) * (np.sqrt(D) + np.sqrt(E))
    else:
        raise ValueError("Unknown type of distance.")
    return ret
