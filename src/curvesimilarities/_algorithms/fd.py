import numpy as np
from numba import njit

EPSILON = np.finfo(np.float64).eps
NAN = np.float64(np.nan)


@njit(cache=True)
def _fd(P, Q, rel_tol, abs_tol):
    """Algorithm 3 of Alt & Godau (1995)."""
    if len(P.shape) != 2:
        raise ValueError("P must be a 2-dimensional array.")
    if len(Q.shape) != 2:
        raise ValueError("Q must be a 2-dimensional array.")
    if P.shape[1] != Q.shape[1]:
        raise ValueError("P and Q must have the same number of columns.")

    P, Q = P.astype(np.float64), Q.astype(np.float64)
    p, q = len(P), len(Q)
    ANALYTIC = rel_tol == 0 and abs_tol == 0

    if not (p > 0 and q > 0):
        return NAN

    MAX_A = max(np.linalg.norm(P[0] - Q[0]), np.linalg.norm(P[-1] - Q[-1]))
    crit_a = np.array((MAX_A,))

    crit_b = np.empty(p * (q - 1) + (p - 1) * q, dtype=np.float64)
    count = 0
    for i in range(p - 1):
        for j in range(q):
            dist = _critical_b(P[i], P[i + 1], Q[j])
            if dist > MAX_A:
                crit_b[count] = dist
                count += 1
    for i in range(p):
        for j in range(q - 1):
            dist = _critical_b(Q[j], Q[j + 1], P[i])
            if dist > MAX_A:
                crit_b[count] = dist
                count += 1
    crit_b = crit_b[:count]

    if ANALYTIC:
        crit_c = np.empty(
            int(p * (p - 1) * (q - 1) / 2 + q * (q - 1) * (p - 1) / 2),
            dtype=np.float64,
        )
        count = 0
        for i in range(p):
            for j in range(i + 1, p):
                for k in range(q - 1):
                    dist = _critical_c(Q[k], Q[k + 1], P[i], P[j])
                    if dist > MAX_A:
                        crit_c[count] = dist
                        count += 1
        for i in range(q):
            for j in range(i + 1, q):
                for k in range(p - 1):
                    dist = _critical_c(P[k], P[k + 1], Q[i], Q[j])
                    if dist > MAX_A:
                        crit_c[count] = dist
                        count += 1
        crit_c = crit_c[:count]

        crit = np.sort(np.concatenate((crit_a, crit_b, crit_c)))
    else:
        crit = np.sort(np.concatenate((crit_a, crit_b)))

    # binary search
    start, end = 0, len(crit) - 1
    B, L = _reachable_boundaries(P, Q, crit[start])
    if B[-1, -1, 1] == 1 or L[-1, -1, 1] == 1:
        end = start
    while end - start > 1:
        mid = (start + end) // 2
        B, L = _reachable_boundaries(P, Q, crit[mid])
        if B[-1, -1, 1] == 1 or L[-1, -1, 1] == 1:
            end = mid
        else:
            start = mid

    if ANALYTIC:
        ret = crit[end]
    else:
        # parametric search
        e1, e2 = crit[start], crit[end]
        while e2 - e1 > max(rel_tol * e2, abs_tol):
            mid = (e1 + e2) / 2
            if (mid - e1 < EPSILON) or (e2 - mid < EPSILON):
                break
            B, L = _reachable_boundaries(P, Q, mid)
            if B[-1, -1, 1] == 1 or L[-1, -1, 1] == 1:
                e2 = mid
            else:
                e1 = mid
        ret = e2
    return ret


@njit(cache=True)
def _reachable_boundaries(P, Q, eps):
    B = np.empty((len(P) - 1, len(Q), 2), dtype=np.float64)
    start, end = _free_interval(P[0], P[1], Q[0], eps)
    if start == 0:
        B[0, 0] = [start, end]
    else:
        B[0, 0] = [NAN, NAN]
    for i in range(1, len(P) - 1):
        _, prev_end = B[i - 1, 0]
        if prev_end == 1:
            start, end = _free_interval(P[i], P[i + 1], Q[0], eps)
            if start == 0:
                B[i, 0] = [start, end]
                continue
        B[i, 0] = [NAN, NAN]

    L = np.empty((len(P), len(Q) - 1, 2), dtype=np.float64)
    start, end = _free_interval(Q[0], Q[1], P[0], eps)
    if start == 0:
        L[0, 0] = [start, end]
    else:
        L[0, 0] = [NAN, NAN]
    for j in range(1, len(Q) - 1):
        _, prev_end = L[0, j - 1]
        if prev_end == 1:
            start, end = _free_interval(Q[j], Q[j + 1], P[0], eps)
            if start == 0:
                L[0, j] = [start, end]
                continue
        L[0, j] = [NAN, NAN]

    for i in range(len(P) - 1):
        for j in range(len(Q) - 1):
            prevL_start, _ = L[i, j]
            prevB_start, _ = B[i, j]
            L_start, L_end = _free_interval(Q[j], Q[j + 1], P[i + 1], eps)
            B_start, B_end = _free_interval(P[i], P[i + 1], Q[j + 1], eps)

            if not np.isnan(prevB_start):
                L[i + 1, j] = [L_start, L_end]
            elif prevL_start <= L_end:
                L[i + 1, j] = [max(prevL_start, L_start), L_end]
            else:
                L[i + 1, j] = [NAN, NAN]

            if not np.isnan(prevL_start):
                B[i, j + 1] = [B_start, B_end]
            elif prevB_start <= B_end:
                B[i, j + 1] = [max(prevB_start, B_start), B_end]
            else:
                B[i, j + 1] = [NAN, NAN]

    return B, L


@njit(cache=True)
def _free_interval(A, B, P, eps):
    # resulting interval is always in [0, 1] or is [nan, nan].
    coeff1 = B - A
    coeff2 = A - P
    a = np.dot(coeff1, coeff1)
    c = np.dot(coeff2, coeff2) - eps**2
    if a == 0:  # degenerate case
        if c > 0:
            interval = [NAN, NAN]
        else:
            interval = [np.float64(0), np.float64(1)]
        return interval
    b = 2 * np.dot(coeff1, coeff2)
    Det = b**2 - 4 * a * c
    if Det < 0:
        interval = [NAN, NAN]
    else:
        start = max((-b - Det**0.5) / 2 / a, np.float64(0))
        end = min((-b + Det**0.5) / 2 / a, np.float64(1))
        if start > 1 or end < 0:
            start = end = NAN
        interval = [start, end]
    return interval


@njit(cache=True)
def _critical_b(A, B, P):
    v = B - A
    w = P - A
    vv = np.dot(v, v)
    if vv == 0:
        return np.linalg.norm(w)
    t = np.dot(v, w) / vv
    if t < 0:
        dist = np.linalg.norm(w)
    elif t > 1:
        dist = np.linalg.norm(P - B)
    else:
        dist = np.linalg.norm(t * v - w)
    return dist


@njit(cache=True)
def _critical_c(A, B, P1, P2):
    M = (P1 + P2) / 2
    AB = B - A
    MA = A - M
    PP = P2 - P1

    a = np.dot(AB, PP)
    b = np.dot(MA, PP)
    if a == 0:
        ret = NAN
    else:
        t = -b / a
        if t < 0 or t > 1:
            ret = NAN
        else:
            MT = AB * t + MA
            ret = np.linalg.norm(MT)
    return ret
