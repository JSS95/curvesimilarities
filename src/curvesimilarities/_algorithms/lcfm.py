"""Locally corret frechet matching."""

import numpy as np
from numba import njit

from .fd import _critical_b, _critical_c, _free_boundaries, _reachable_boundaries

EPSILON = np.finfo(np.float64).eps
NAN = np.float64(np.nan)
INF = np.float64(np.inf)


@njit(cache=True)
def _computeLCFM(P, Q, rel_tol, abs_tol, event_rel_tol, event_abs_tol):
    if len(P.shape) != 2:
        raise ValueError("P must be a 2-dimensional array.")
    if len(Q.shape) != 2:
        raise ValueError("Q must be a 2-dimensional array.")
    if P.shape[1] != Q.shape[1]:
        raise ValueError("P and Q must have the same number of columns.")

    p, q = len(P), len(Q)
    if p == 0 or q == 0:
        eps = NAN
        matching = np.empty((0, 2), dtype=np.float64)
    elif p == 1 and q == 1:
        eps = np.linalg.norm(P[0] - Q[0])
        matching = np.array([[0, 0]], dtype=np.float64)
    elif p == 1:
        eps = max(np.linalg.norm(P[0] - Q[0]), np.linalg.norm(P[0] - Q[1]))
        matching = np.array([[0, 0], [0, q - 1]], dtype=np.float64)
    elif q == 1:
        eps = max(np.linalg.norm(P[0] - Q[0]), np.linalg.norm(P[1] - Q[0]))
        matching = np.array([[0, 0], [p - 1, 0]], dtype=np.float64)
    elif p == 2 and q == 2:
        eps = max(np.linalg.norm(P[0] - Q[0]), np.linalg.norm(P[1] - Q[1]))
        matching = np.array([[0, 0], [1, 1]], dtype=np.float64)
    else:
        eps, e_r, e_r_horiz = _e_r(P, Q, rel_tol, abs_tol, event_rel_tol, event_abs_tol)

        i0, j0 = e_r[0]
        P1 = _idx2subcurve(P, 0, i0)
        Q1 = _idx2subcurve(Q, 0, j0)
        i1, j1 = e_r[-1]
        P2 = _idx2subcurve(P, i1, len(P) - 1)
        Q2 = _idx2subcurve(Q, j1, len(Q) - 1)

        _, mu_1 = _computeLCFM(P1, Q1, rel_tol, abs_tol, event_abs_tol, event_rel_tol)
        _, mu_2 = _computeLCFM(P2, Q2, rel_tol, abs_tol, event_abs_tol, event_rel_tol)
        # Scale mu s.t. indices are w.r.t full curves instead of subcurves
        if e_r_horiz:
            J, T = int(j0), j0 - int(j0)
            thres = len(Q1) - 2
            for idx in range(len(mu_1)):
                _, j = mu_1[idx]
                if j > thres:
                    t = j - J
                    mu_1[idx, 1] = J + t * T
            mu_2[:, 0] += i1
            for idx in range(len(mu_2)):
                _, j = mu_2[idx]
                if j < 1:
                    mu_2[idx, 1] = J + T + j * (1 - T)
                else:
                    mu_2[idx, 1] += J
        else:
            I, T = int(i0), i0 - int(i0)
            thres = len(P1) - 2
            for idx in range(len(mu_1)):
                i, _ = mu_1[idx]
                if i > thres:
                    t = i - I
                    mu_1[idx, 0] = I + t * T
            mu_2[:, 1] += j1
            for idx in range(len(mu_2)):
                i, _ = mu_2[idx]
                if i < 1:
                    mu_2[idx, 0] = I + T + i * (1 - T)
                else:
                    mu_2[idx, 0] += I

        matching = np.concatenate((mu_1[:-1], e_r, mu_2[1:]))
    return eps, matching


@njit(cache=True)
def _e_r(P, Q, rel_tol, abs_tol, event_rel_tol, event_abs_tol):
    p, q = len(P), len(Q)

    crit = np.empty(p * (q - 1) + (p - 1) * q, dtype=np.float64)
    count = 0
    for i in range(p - 1):
        for j in range(q):
            crit[count], _ = _critical_b(P[i], P[i + 1], Q[j])
            count += 1
    for i in range(p):
        for j in range(q - 1):
            crit[count], _ = _critical_b(Q[j], Q[j + 1], P[i])
            count += 1
    crit = np.sort(crit[:count])
    # Compute minimum feasible epsilons: 1. Binary search
    start, end = 0, len(crit)
    BF, LF, BR, LR = _feasible_boundaries(P, Q, crit[start])
    if BR[-1, -1, 1] == 1 or LR[-1, -1, 1] == 1:
        end = start
    while end - start > 1:
        mid = (start + end) // 2
        BF, LF, BR, LR = _feasible_boundaries(P, Q, crit[mid])
        if BR[-1, -1, 1] == 1 or LR[-1, -1, 1] == 1:
            end = mid
        else:
            start = mid
    # Compute minimum feasible epsilons: 2. Parametric search
    e1, e2 = crit[start], crit[end]
    while e2 - e1 > max(rel_tol * e2, abs_tol):
        mid = (e1 + e2) / 2
        if (mid - e1 < EPSILON) or (e2 - mid < EPSILON):
            break
        BF, LF, BR, LR = _feasible_boundaries(P, Q, mid)
        if BR[-1, -1, 1] == 1 or LR[-1, -1, 1] == 1:
            e2 = mid
        else:
            e1 = mid
    eps = e2
    if not (BR[-1, -1, 1] == 1 or LR[-1, -1, 1] == 1):  # compute one more last time
        BF, LF, BR, LR = _feasible_boundaries(P, Q, eps)

    BP, LP = _passable_boundaries(BR, LR, BR, LR)

    BE, LE, B_err, L_err = _realizing_set(
        P, Q, eps, BF, LF, BP, LP, event_rel_tol, event_abs_tol
    )

    # Select the most significant event.
    if len(LE) == 0:
        e_r_horiz = False
        B_minerr_idx = np.argmin(B_err)
    elif len(BE) == 0:
        e_r_horiz = True
        L_minerr_idx = np.argmin(L_err)
    else:
        B_minerr_idx = np.argmin(B_err)
        L_minerr_idx = np.argmin(L_err)
        if B_err[B_minerr_idx] < L_err[L_minerr_idx]:
            e_r_horiz = False
        else:
            e_r_horiz = True

    if e_r_horiz:
        i0, i1, jt = LE[L_minerr_idx]
        if i0 == i1:
            e_r = np.array([[i0, jt]], dtype=np.float64)
        else:
            e_r = np.array([[i0, jt], [i1, jt]], dtype=np.float64)
    else:
        it, j0, j1 = BE[B_minerr_idx]
        if j0 == j1:
            e_r = np.array([[it, j0]], dtype=np.float64)
        else:
            e_r = np.array([[it, j0], [it, j1]], dtype=np.float64)

    return eps, e_r, e_r_horiz


@njit(cache=True)
def _feasible_boundaries(P, Q, eps):
    BF, LF = _free_boundaries(P, Q, eps)
    BF[0, 0] = [0, 1]
    LF[0, 0] = [0, 1]
    BF[-1, -1] = [0, 1]
    LF[-1, -1] = [0, 1]
    BR, LR = _reachable_boundaries(BF, LF, np.empty_like(BF), np.empty_like(LF))
    return BF, LF, BR, LR


@njit(cache=True)
def _passable_boundaries(BR, LR, BP_out, LP_out):
    # Uppermost boundary
    if BR[-1, -1, 1] == 1:
        BP_out[-1, -1] = BR[-1, -1]
    else:
        BP_out[-1, -1] = [NAN, NAN]
    for i in range(1, BP_out.shape[0]):
        if BP_out[-i, -1, 0] == 0 and BR[-i - 1, -1, 1] == 1:
            BP_out[-i - 1, -1] = BR[-i - 1, -1]
        else:
            BP_out[-i - 1, -1] = [NAN, NAN]

    # Rightmost boundary
    if LR[-1, -1, 1] == 1:
        LP_out[-1, -1] = LR[-1, -1]
    else:
        LP_out[-1, -1] = [NAN, NAN]
    for j in range(1, LP_out.shape[1]):
        if LP_out[-1, -j, 0] == 0 and LR[-1, -j - 1, 1] == 1:
            LP_out[-1, -j - 1] = LR[-1, -j - 1]
        else:
            LP_out[-1, -j - 1] = [NAN, NAN]

    for i in range(BP_out.shape[0]):
        for j in range(LP_out.shape[1]):
            _, prevB_end = BP_out[-i - 1, -j - 1]
            B_start, B_end = BR[-i - 1, -j - 2]
            _, prevL_end = LP_out[-i - 1, -j - 1]
            L_start, L_end = LR[-i - 2, -j - 1]

            if not np.isnan(prevL_end):
                BP_out[-i - 1, -j - 2] = [B_start, B_end]
            elif prevB_end >= B_start:
                BP_out[-i - 1, -j - 2] = [B_start, min(prevB_end, B_end)]
            else:
                BP_out[-i - 1, -j - 2] = [NAN, NAN]

            if not np.isnan(prevB_end):
                LP_out[-i - 2, -j - 1] = [L_start, L_end]
            elif prevL_end >= L_start:
                LP_out[-i - 2, -j - 1] = [L_start, min(prevL_end, L_end)]
            else:
                LP_out[-i - 2, -j - 1] = [NAN, NAN]

    return BP_out, LP_out


@njit(cache=True)
def _realizing_set(P, Q, eps, BF, LF, BP, LP, rel_tol, abs_tol):
    L_E_minerr, L_E_closest_event = INF, np.empty((1, 3), dtype=np.float64)  # Fallback
    # each row of L_E = i0, i1, j + t
    L_E = np.empty((LP.shape[0] * LP.shape[1], 3), dtype=np.float64)
    L_E_err = np.empty(LP.shape[0] * LP.shape[1], dtype=np.float64)
    count = 0
    for j in range(LP.shape[1]):
        g_star = 0
        for i in range(1, LP.shape[0] - 1):  # skip event ending on rightmost boundary
            if not np.isnan(BP[i - 1, j, 0]) or LF[g_star, j, 0] <= LF[i, j, 0]:
                g_star = i
                d, t = _critical_b(Q[j], Q[j + 1], P[i])
            else:
                d, t = _critical_c(Q[j], Q[j + 1], P[g_star], P[i])
            if t == 1 and j < LP.shape[1] - 1 and LP[i, j + 1, 0] == 0:
                # Let upper cell deal with this event to prevent duplication.
                continue

            err = abs(eps - d)
            # Keep minimum critical value in case L_E is empty by incorrect tols.
            if err < L_E_minerr:
                L_E_minerr = err
                L_E_closest_event[0, :] = [g_star, i, j + t]
            if j < LP.shape[1] - 1 and LP[i, j, 0] == 1 and LP[i, j + 1, 0] == 0:
                is_singleton = False
            elif j > 0 and LP[i, j, 1] == 0 and LP[i, j - 1, 1] == 1:
                is_singleton = False
            elif LP[i, j, 0] == LP[i, j, 1]:
                is_singleton = True
            else:  # Singleton detection may have failed due to floating point error
                is_realizing = err <= max(rel_tol * max(abs(eps), abs(d)), abs_tol)
                is_singleton = is_realizing & ~np.isnan(LP[i, j, 0])
            if is_singleton:
                L_E[count] = [g_star, i, j + t]
                L_E_err[count] = err
                count += 1
                g_star = i
    L_E = L_E[:count]
    L_E_err = L_E_err[:count]

    B_E_minerr, B_E_closest_event = INF, np.empty((1, 3), dtype=np.float64)  # Fallback
    # each row of B_E = i + t, j0, j1
    B_E = np.empty((BP.shape[0] * BP.shape[1], 3), dtype=np.float64)
    B_E_err = np.empty(BP.shape[0] * BP.shape[1], dtype=np.float64)
    count = 0
    for i in range(BP.shape[0]):
        g_star = 0
        for j in range(1, BP.shape[1] - 1):  # skip event ending on uppermost boundary
            if not np.isnan(LP[i, j - 1, 0]) or BF[i, g_star, 0] <= BF[i, j, 0]:
                g_star = j
                d, t = _critical_b(P[i], P[i + 1], Q[j])
            else:
                d, t = _critical_c(P[i], P[i + 1], Q[g_star], Q[j])
            if t == 1 and i < BP.shape[0] - 1 and BP[i + 1, j, 0] == 0:
                # Let rhs cell deal with this event to prevent duplication.
                continue

            err = abs(eps - d)
            # Keep minimum critical value in case B_E is empty by incorrect tols.
            if err < B_E_minerr:
                B_E_minerr = err
                B_E_closest_event[0, :] = [g_star, i, j + t]
            if i < BP.shape[0] - 1 and BP[i, j, 0] == 1 and BP[i + 1, j, 0] == 0:
                is_singleton = False
            elif i > 0 and BP[i, j, 1] == 0 and BP[i - 1, j, 1] == 1:
                is_singleton = False
            elif BP[i, j, 0] == BP[i, j, 1]:
                is_singleton = True
            else:  # Singleton detection may have failed due to floating point error
                is_realizing = err <= max(rel_tol * max(abs(eps), abs(d)), abs_tol)
                is_singleton = is_realizing & ~np.isnan(BP[i, j, 0])
            if is_singleton:
                B_E[count] = [i + t, g_star, j]
                B_E_err[count] = err
                count += 1
                g_star = j
    B_E = B_E[:count]
    B_E_err = B_E_err[:count]

    if len(B_E) == 0 and len(L_E) == 0:
        # Realizing event not detected because of too harsh tolerances.
        # Since at least one realizing event must exist, we regard the event whose value
        # is the closest to epsilon as the only realizing event.
        if B_E_minerr < L_E_minerr:
            B_E = B_E_closest_event
            B_E_err = np.array((B_E_minerr,), dtype=np.float64)
        else:
            L_E = L_E_closest_event
            L_E_err = np.array((L_E_minerr,), dtype=np.float64)

    return B_E, L_E, B_E_err, L_E_err


@njit(cache=True)
def _idx2subcurve(curve, idx0, idx1):
    i0, t0 = int(np.ceil(idx0)) - 1, idx0 - int(idx0)
    if t0 == 0:
        start = curve[i0 + 1]
    else:
        start = curve[i0] + t0 * (curve[i0 + 1] - curve[i0])

    if idx0 == idx1:
        return start.reshape(1, -1)

    i1, t1 = int(np.ceil(idx1)) - 1, idx1 - int(idx1)
    if t1 == 0:
        end = curve[i1 + 1]
    else:
        end = curve[i1] + t1 * (curve[i1 + 1] - curve[i1])
    return np.concatenate(
        (start.reshape(1, -1), curve[int(idx0) + 1 : i1 + 1], end.reshape(1, -1))
    )
