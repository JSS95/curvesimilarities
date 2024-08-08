import numpy as np
from numba import njit

EPSILON = np.finfo(np.float64).eps
NAN = np.float64(np.nan)
INF = np.float64(np.inf)


@njit(cache=True)
def _fd(P, Q, rel_tol, abs_tol):
    """Algorithm 3 of Alt & Godau (1995)."""
    if len(P.shape) != 2:
        raise ValueError("P must be a 2-dimensional array.")
    if len(Q.shape) != 2:
        raise ValueError("Q must be a 2-dimensional array.")
    if P.shape[1] != Q.shape[1]:
        raise ValueError("P and Q must have the same number of columns.")

    p, q = len(P), len(Q)

    if not (p > 0 and q > 0):
        return NAN

    crit = _critical_values(P, Q)

    # binary search
    start, end = 0, len(crit) - 1
    B, L = _reachable_boundaries_1d(P, Q, crit[start])
    if B[-1, 1] == 1 or L[-1, 1] == 1:
        end = start
    while end - start > 1:
        mid = (start + end) // 2
        B, L = _reachable_boundaries_1d(P, Q, crit[mid])
        if B[-1, 1] == 1 or L[-1, 1] == 1:
            end = mid
        else:
            start = mid

    # parametric search
    e1, e2 = crit[start], crit[end]
    while e2 - e1 > max(rel_tol * e2, abs_tol):
        mid = (e1 + e2) / 2
        if (mid - e1 < EPSILON) or (e2 - mid < EPSILON):
            break
        B, L = _reachable_boundaries_1d(P, Q, mid)
        if B[-1, 1] == 1 or L[-1, 1] == 1:
            e2 = mid
        else:
            e1 = mid
    return e2


@njit(cache=True)
def _critical_values(P, Q):
    p, q = len(P), len(Q)
    MAX_A = max(np.linalg.norm(P[0] - Q[0]), np.linalg.norm(P[-1] - Q[-1]))
    crit_a = np.array((MAX_A,))

    crit_b = np.empty(p * (q - 1) + (p - 1) * q, dtype=np.float64)
    count = 0
    for i in range(p - 1):
        for j in range(q):
            dist, _ = _critical_b(P[i], P[i + 1], Q[j])
            if dist > MAX_A:
                crit_b[count] = dist
                count += 1
    for i in range(p):
        for j in range(q - 1):
            dist, _ = _critical_b(Q[j], Q[j + 1], P[i])
            if dist > MAX_A:
                crit_b[count] = dist
                count += 1
    crit_b = crit_b[:count]

    return np.sort(np.concatenate((crit_a, crit_b)))


@njit(cache=True)
def _critical_b(A, B, P):
    v = B - A
    w = P - A
    vv = np.dot(v, v)
    if vv == 0:
        t = np.float64(0)
        return np.linalg.norm(w), t
    t = np.dot(v, w) / vv
    if t < 0:
        dist = np.linalg.norm(w)
        t = np.float64(0)
    elif t > 1:
        dist = np.linalg.norm(P - B)
        t = np.float64(1)
    else:
        dist = np.linalg.norm(t * v - w)
    return dist, t


@njit(cache=True)
def _critical_c(A, B, P1, P2):
    M = (P1 + P2) / 2
    AB = B - A
    MA = A - M
    PP = P2 - P1

    a = np.dot(AB, PP)
    b = np.dot(MA, PP)
    if a == 0:
        if np.abs(np.dot(AB, MA)) == np.linalg.norm(AB) * np.linalg.norm(MA):
            # M is on AB
            ret = np.linalg.norm(A - P1)
            t = np.float64(0)
        else:
            ret = NAN
            t = NAN
    else:
        t = -b / a
        if t < 0 or t > 1:
            ret = NAN
        else:
            ret = np.linalg.norm(A + AB * t - P1)
    return ret, t


@njit(cache=True)
def _free_boundaries(P, Q, eps):
    p, q = len(P), len(Q)
    BF = np.empty((p - 1, q, 2), dtype=np.float64)
    LF = np.empty((p, q - 1, 2), dtype=np.float64)

    for i in range(p - 1):
        for j in range(q):
            BF[i, j] = _free_interval(P[i], P[i + 1], Q[j], eps)
    for i in range(p):
        for j in range(q - 1):
            LF[i, j] = _free_interval(Q[j], Q[j + 1], P[i], eps)

    return BF, LF


@njit(cache=True)
def _reachable_boundaries(BF, LF, BR_out, LR_out):
    # Downmost boundary
    if BF[0, 0, 0] == 0:
        BR_out[0, 0] = BF[0, 0]
    else:
        BR_out[0, 0] = [NAN, NAN]
    for i in range(1, BR_out.shape[0]):
        if BR_out[i - 1, 0, 1] == 1 and BF[i, 0, 0] == 0:
            BR_out[i, 0] = BF[i, 0]
        else:
            BR_out[i, 0] = [NAN, NAN]

    # Leftmost boundary
    if LF[0, 0, 0] == 0:
        LR_out[0, 0] = LF[0, 0]
    else:
        LR_out[0, 0] = [NAN, NAN]
    for j in range(1, LR_out.shape[1]):
        if LR_out[0, j - 1, 1] == 1 and LF[0, j, 0] == 0:
            LR_out[0, j] = LF[0, j]
        else:
            LR_out[0, j] = [NAN, NAN]

    # Propagate
    for i in range(BR_out.shape[0]):
        for j in range(LR_out.shape[1]):
            prevB_start, _ = BR_out[i, j]
            B_start, B_end = BF[i, j + 1]
            prevL_start, _ = LR_out[i, j]
            L_start, L_end = LF[i + 1, j]

            if not np.isnan(prevL_start):
                BR_out[i, j + 1] = [B_start, B_end]
            elif prevB_start <= B_end:
                BR_out[i, j + 1] = [max(prevB_start, B_start), B_end]
            else:
                BR_out[i, j + 1] = [NAN, NAN]

            if not np.isnan(prevB_start):
                LR_out[i + 1, j] = [L_start, L_end]
            elif prevL_start <= L_end:
                LR_out[i + 1, j] = [max(prevL_start, L_start), L_end]
            else:
                LR_out[i + 1, j] = [NAN, NAN]

    return BR_out, LR_out


@njit(cache=True)
def _reachable_boundaries_1d(P, Q, eps):
    # Keep 1d array instead of 2d, and do free + reachable in one-shot.
    # Memory efficient, but cannot do backtracking.
    p, q = len(P), len(Q)
    B = np.empty((1, 2), dtype=np.float64)
    L = np.empty((q - 1, 2), dtype=np.float64)

    # Construct leftmost Ls
    prevL0_end = 1
    for j in range(q - 1):
        if prevL0_end == 1:
            start, end = _free_interval(Q[j], Q[j + 1], P[0], eps)
            if start == 0:
                L[j] = [start, end]
            else:
                L[j] = [NAN, NAN]
        else:
            L[j] = [NAN, NAN]
        _, prevL0_end = L[j]

    prevB0_end = 1
    for i in range(p - 1):
        # construct lowermost B
        if prevB0_end == 1:
            start, end = _free_interval(P[i], P[i + 1], Q[0], eps)
            if start == 0:
                B[0] = [start, end]
            else:
                B[0] = [NAN, NAN]
        else:
            B[0] = [NAN, NAN]
        _, prevB0_end = B[0]
        for j in range(q - 1):
            prevL_start, _ = L[j]
            prevB_start, _ = B[0]
            L_start, L_end = _free_interval(Q[j], Q[j + 1], P[i + 1], eps)
            B_start, B_end = _free_interval(P[i], P[i + 1], Q[j + 1], eps)

            if not np.isnan(prevB_start):
                L[j] = [L_start, L_end]
            elif prevL_start <= L_end:
                L[j] = [max(prevL_start, L_start), L_end]
            else:
                L[j] = [NAN, NAN]

            if not np.isnan(prevL_start):
                B[0] = [B_start, B_end]
            elif prevB_start <= B_end:
                B[0] = [max(prevB_start, B_start), B_end]
            else:
                B[0] = [NAN, NAN]

    return B, L


@njit(cache=True)
def _free_interval(A, B, P, eps):
    # resulting interval is always in [0, 1] or is [nan, nan].
    start = end = NAN

    pa = A - P
    pb = B - P
    if np.dot(pa, pa) <= eps**2:
        start = np.float64(0)
    if np.dot(pb, pb) <= eps**2:
        end = np.float64(1)

    ab = B - A
    a = np.dot(ab, ab)
    if a == 0:
        return [start, end]

    b = 2 * np.dot(ab, pa)
    c = np.dot(pa, pa) - eps**2
    Det = b**2 - 4 * a * c
    if Det >= 0:
        if np.isnan(start):  # pa.pa > eps ** 2
            start = (-b - Det**0.5) / 2 / a
        if np.isnan(end):  # pb.pb > eps ** 2
            end = (-b + Det**0.5) / 2 / a
    if start > 1 or end < 0:
        start = end = NAN
    return [start, end]
