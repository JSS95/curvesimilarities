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

    if ANALYTIC:
        crit_c = np.empty(
            int(p * (p - 1) * (q - 1) / 2 + q * (q - 1) * (p - 1) / 2),
            dtype=np.float64,
        )
        count = 0
        for i in range(p):
            for j in range(i + 1, p):
                for k in range(q - 1):
                    dist, _ = _critical_c(Q[k], Q[k + 1], P[i], P[j])
                    if dist > MAX_A:
                        crit_c[count] = dist
                        count += 1
        for i in range(q):
            for j in range(i + 1, q):
                for k in range(p - 1):
                    dist, _ = _critical_c(P[k], P[k + 1], Q[i], Q[j])
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
        return NAN, np.empty((0, 2), dtype=np.float64)

    P_cumsum = np.empty(len(P), dtype=np.float64)
    P_cumsum[0] = 0
    for i in range(len(P) - 1):
        P_cumsum[i + 1] = P_cumsum[i] + np.linalg.norm(P[i + 1] - P[i])
    Q_cumsum = np.empty(len(Q), dtype=np.float64)
    Q_cumsum[0] = 0
    for j in range(len(Q) - 1):
        Q_cumsum[j + 1] = Q_cumsum[j] + np.linalg.norm(Q[j + 1] - Q[j])

    # critical values and their relevant indices
    crit_a = np.empty(2, dtype=np.float64)
    crit_a_params = np.empty((len(crit_a), 1, 2), dtype=np.float64)
    crit_a[0] = np.linalg.norm(P[0] - Q[0])
    crit_a_params[0, 0] = [P_cumsum[0], Q_cumsum[0]]
    crit_a[1] = np.linalg.norm(P[-1] - Q[-1])
    crit_a_params[1, 0] = [P_cumsum[-1], Q_cumsum[-1]]

    crit_b = np.empty((p - 1) * q + p * (q - 1), dtype=np.float64)
    crit_b_params = np.empty((len(crit_b), 1, 2), dtype=np.float64)
    idx = 0
    for i in range(p - 1):
        for j in range(q):
            dist, t = _critical_b(P[i], P[i + 1], Q[j])
            crit_b[idx] = dist
            crit_b_params[idx, 0, 0] = P_cumsum[i] + t * (P_cumsum[i + 1] - P_cumsum[i])
            crit_b_params[idx, 0, 1] = Q_cumsum[j]
            idx += 1
    for i in range(p):
        for j in range(q - 1):
            dist, t = _critical_b(Q[j], Q[j + 1], P[i])
            crit_b[idx] = dist
            crit_b_params[idx, 0, 0] = P_cumsum[i]
            crit_b_params[idx, 0, 1] = Q_cumsum[j] + t * (Q_cumsum[j + 1] - Q_cumsum[j])
            idx += 1

    crit_c = np.empty(
        int(p * (p - 1) * (q - 1) / 2 + q * (q - 1) * (p - 1) / 2), dtype=np.float64
    )
    crit_c_params = np.empty((len(crit_c), 2, 2), dtype=np.float64)
    idx = 0
    for i in range(p):
        for j in range(i + 1, p):
            for k in range(q - 1):
                dist, t = _critical_c(Q[k], Q[k + 1], P[i], P[j])
                if np.isnan(dist):
                    continue
                crit_c[idx] = dist
                crit_c_params[idx, 0, 0] = P_cumsum[i]
                crit_c_params[idx, 0, 1] = Q_cumsum[k] + t * (
                    Q_cumsum[k + 1] - Q_cumsum[k]
                )
                crit_c_params[idx, 1, 0] = P_cumsum[j]
                crit_c_params[idx, 1, 1] = Q_cumsum[k] + t * (
                    Q_cumsum[k + 1] - Q_cumsum[k]
                )
                idx += 1
    for i in range(q):
        for j in range(i + 1, q):
            for k in range(p - 1):
                dist, t = _critical_c(P[k], P[k + 1], Q[i], Q[j])
                if np.isnan(dist):
                    continue
                crit_c[idx] = dist
                crit_c_params[idx, 0, 0] = P_cumsum[k] + t * (
                    P_cumsum[k + 1] - P_cumsum[k]
                )
                crit_c_params[idx, 0, 1] = Q_cumsum[i]
                crit_c_params[idx, 1, 0] = P_cumsum[k] + t * (
                    P_cumsum[k + 1] - P_cumsum[k]
                )
                crit_c_params[idx, 1, 1] = Q_cumsum[j]
                idx += 1
    crit_c = crit_c[:idx]
    crit_c_params = crit_c_params[:idx]

    # binary search
    THRES = np.max(crit_a)
    crit = np.sort(np.concatenate((crit_a[crit_a >= THRES], crit_b[crit_b >= THRES])))
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
    fred_dist = e2

    # Find critical value closest to fd
    crit_a = np.abs(crit_a - fred_dist)
    a_minidx = np.argmin(crit_a)
    crit_b = np.abs(crit_b - fred_dist)
    b_minidx = np.argmin(crit_b)
    crit_c = np.abs(crit_c - fred_dist)
    c_minidx = np.argmin(crit_c)
    mins = np.array([crit_a[a_minidx], crit_b[b_minidx], crit_c[c_minidx]])
    val = np.min(mins)
    if val == crit_a[a_minidx]:
        param = crit_a_params[a_minidx]
    elif val == crit_b[b_minidx]:
        param = crit_b_params[b_minidx]
    else:
        param = crit_c_params[c_minidx]
    return fred_dist, param
