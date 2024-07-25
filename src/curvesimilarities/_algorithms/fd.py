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

    P, Q = P.astype(np.float64), Q.astype(np.float64)
    p, q = len(P), len(Q)
    # It is impossible to get analytic result because of floating point error,
    # which makes reachability fail with analytic solution...
    # ANALYTIC = rel_tol == 0 and abs_tol == 0
    ANALYTIC = False

    if not (p > 0 and q > 0):
        return NAN

    crit = _critical_values(P, Q, ANALYTIC)

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

    if ANALYTIC:
        ret = crit[end]
    else:
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
        ret = e2
    return ret


@njit(cache=True)
def _critical_values(P, Q, ANALYTIC):
    p, q = len(P), len(Q)
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
    return crit


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
        if np.abs(np.dot(AB, MA)) == np.linalg.norm(AB) * np.linalg.norm(MA):
            # M is on AB
            ret = np.float64(0)
        else:
            ret = NAN
    else:
        t = -b / a
        if t < 0 or t > 1:
            ret = NAN
        else:
            P1T = A + AB * t - P1
            ret = np.linalg.norm(P1T)
    return ret


@njit(cache=True)
def _reachable_boundaries(P, Q, eps):
    p, q = len(P), len(Q)

    B = np.empty((p - 1, q, 2), dtype=np.float64)
    if p > 1:
        start, end = _free_interval(P[0], P[1], Q[0], eps)
        if start == 0:
            B[0, 0] = [start, end]
        else:
            B[0, 0] = [NAN, NAN]
    for i in range(1, p - 1):
        _, prev_end = B[i - 1, 0]
        if prev_end == 1:
            start, end = _free_interval(P[i], P[i + 1], Q[0], eps)
            if start == 0:
                B[i, 0] = [start, end]
                continue
        B[i, 0] = [NAN, NAN]

    L = np.empty((p, q - 1, 2), dtype=np.float64)
    if q > 1:
        start, end = _free_interval(Q[0], Q[1], P[0], eps)
        if start == 0:
            L[0, 0] = [start, end]
        else:
            L[0, 0] = [NAN, NAN]
    for j in range(1, q - 1):
        _, prev_end = L[0, j - 1]
        if prev_end == 1:
            start, end = _free_interval(Q[j], Q[j + 1], P[0], eps)
            if start == 0:
                L[0, j] = [start, end]
                continue
        L[0, j] = [NAN, NAN]

    for i in range(1, p):
        for j in range(1, q):
            prevL_start, _ = L[i - 1, j - 1]
            prevB_start, _ = B[i - 1, j - 1]
            L_start, L_end = _free_interval(Q[j - 1], Q[j], P[i], eps)
            B_start, B_end = _free_interval(P[i - 1], P[i], Q[j], eps)

            if not np.isnan(prevB_start):
                L[i, j - 1] = [L_start, L_end]
            elif prevL_start <= L_end:
                L[i, j - 1] = [max(prevL_start, L_start), L_end]
            else:
                L[i, j - 1] = [NAN, NAN]

            if not np.isnan(prevL_start):
                B[i - 1, j] = [B_start, B_end]
            elif prevB_start <= B_end:
                B[i - 1, j] = [max(prevB_start, B_start), B_end]
            else:
                B[i - 1, j] = [NAN, NAN]

    return B, L


@njit(cache=True)
def _reachable_boundaries_1d(P, Q, eps):
    # Equivalent to _reachable_boundaries, but keep 1d array instead of 2d.
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
def _fd_params(P, Q, rel_tol, abs_tol):
    if len(P.shape) != 2:
        raise ValueError("P must be a 2-dimensional array.")
    if len(Q.shape) != 2:
        raise ValueError("Q must be a 2-dimensional array.")
    if P.shape[1] != Q.shape[1]:
        raise ValueError("P and Q must have the same number of columns.")

    P, Q = P.astype(np.float64), Q.astype(np.float64)
    p, q = len(P), len(Q)

    if not (p > 0 and q > 0):
        return NAN, NAN, NAN

    crit = _critical_values(P, Q, False)

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

    # Compute passability
    for _i in range(p - 1):
        i = p - _i - 2
        for _j in range(q - 1):
            j = q - _j - 2
            b_up, b_down = B[i, j + 1], B[i, j]
            l_right, l_left = L[i + 1, j], L[i, j]

            # update b_down
            if not np.isnan(l_right[0]):
                pass
            elif b_up[1] >= b_down[0]:
                b_down[1] = min(b_up[1], b_down[1])
            else:
                b_down[:] = [NAN, NAN]

            # update l_left
            if not np.isnan(b_up[0]):
                pass
            elif l_right[1] >= l_left[0]:
                l_left[1] = min(l_right[1], l_left[1])
            else:
                l_left[:] = [NAN, NAN]

    B_crits = np.empty_like(B)
    for i in range(p - 1):
        a, b = P[i], P[i + 1]
        for j in range(q):
            c = Q[j]
            s, t = B[i, j]
            if np.isnan(s):
                B_crits[i, j, 0] = -INF
            else:
                B_crits[i, j, 0] = np.linalg.norm(a + (b - a) * s - c)
            if np.isnan(t):
                B_crits[i, j, 1] = -INF
            else:
                B_crits[i, j, 1] = np.linalg.norm(a + (b - a) * t - c)
    Bmax_idx_flat = np.argmax(B_crits)
    # unravel index
    Bmax_idx = np.empty(3, dtype=np.int64)
    Bmax_idx[0] = Bmax_idx_flat // (B_crits.shape[1] * B_crits.shape[2])
    Bmax_idx_flat -= Bmax_idx[0] * (B_crits.shape[1] * B_crits.shape[2])
    Bmax_idx[1] = Bmax_idx_flat // B_crits.shape[2]
    Bmax_idx_flat -= Bmax_idx[1] * B_crits.shape[2]
    Bmax_idx[2] = Bmax_idx_flat
    Bmax = B_crits[Bmax_idx[0], Bmax_idx[1], Bmax_idx[2]]

    L_crits = np.empty_like(L)
    for i in range(p):
        c = P[i]
        for j in range(q - 1):
            a, b = Q[j], Q[j + 1]
            s, t = L[i, j]
            if np.isnan(s):
                L_crits[i, j, 0] = -INF
            else:
                L_crits[i, j, 0] = np.linalg.norm(a + (b - a) * s - c)
            if np.isnan(t):
                L_crits[i, j, 1] = -INF
            else:
                L_crits[i, j, 1] = np.linalg.norm(a + (b - a) * t - c)
    Lmax_idx_flat = np.argmax(L_crits)
    # unravel index
    Lmax_idx = np.empty(3, dtype=np.int64)
    Lmax_idx[0] = Lmax_idx_flat // (L_crits.shape[1] * L_crits.shape[2])
    Lmax_idx_flat -= Lmax_idx[0] * (L_crits.shape[1] * L_crits.shape[2])
    Lmax_idx[1] = Lmax_idx_flat // L_crits.shape[2]
    Lmax_idx_flat -= Lmax_idx[1] * L_crits.shape[2]
    Lmax_idx[2] = Lmax_idx_flat
    Lmax = L_crits[Lmax_idx[0], Lmax_idx[1], Lmax_idx[2]]

    P_cumsum = np.empty(len(P), dtype=np.float64)
    P_cumsum[0] = 0
    for i in range(len(P) - 1):
        P_cumsum[i + 1] = np.linalg.norm(P[i + 1] - P[i])
    Q_cumsum = np.empty(len(Q), dtype=np.float64)
    Q_cumsum[0] = 0
    for j in range(len(Q) - 1):
        Q_cumsum[j + 1] = np.linalg.norm(Q[j + 1] - Q[j])

    if Bmax < Lmax:
        fd = Bmax
        i, j, k = Bmax_idx
        param0 = P_cumsum[i] + (P_cumsum[i + 1] - P_cumsum[i]) * B[i, j, k]
        param1 = Q_cumsum[j]
    else:
        fd = Lmax
        i, j, k = Lmax_idx
        param0 = P_cumsum[i]
        param1 = Q_cumsum[j] + (Q_cumsum[j + 1] - Q_cumsum[j]) * L[i, j, k]

    return fd, param0, param1
