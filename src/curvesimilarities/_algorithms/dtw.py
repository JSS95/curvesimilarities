import numpy as np
from numba import njit


@njit(cache=True)
def _dtw_acm(P, Q, dist):
    if len(P.shape) != 2:
        raise ValueError("P must be a 2-dimensional array.")
    if len(Q.shape) != 2:
        raise ValueError("Q must be a 2-dimensional array.")
    if P.shape[1] != Q.shape[1]:
        raise ValueError("P and Q must have the same number of columns.")

    if dist == "euclidean":
        dist_func = _euclidean
    elif dist == "squared_euclidean":
        dist_func = _squared_euclidean
    else:
        raise ValueError(f"Unknown type of distance: {dist}")

    P, Q = P.astype(np.float64), Q.astype(np.float64)
    p, q = len(P), len(Q)
    ret = np.empty((p, q), dtype=np.float64)

    for i in range(p):
        if i == 0:
            ret[i, 0] = dist_func(P[0] - Q[0])
        else:
            ret[i, 0] = ret[i - 1, 0] + dist_func(P[i] - Q[0])
    for j in range(1, q):
        ret[0, j] = ret[0, j - 1] + dist_func(P[0] - Q[j])
    for i in range(1, p):
        for j in range(1, q):
            ret[i, j] = min(
                ret[i - 1, j], ret[i, j - 1], ret[i - 1, j - 1]
            ) + dist_func(P[i] - Q[j])
    return ret


@njit(cache=True)
def _dtw_owp(acm):
    p, q = acm.shape
    path = np.empty((p + q - 1, 2), dtype=np.int_)
    path_len = np.int_(0)

    i, j = p - 1, q - 1
    path[path_len] = [i, j]
    path_len += np.int_(1)

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            d = min(acm[i - 1, j], acm[i, j - 1], acm[i - 1, j - 1])
            if acm[i - 1, j] == d:
                i -= 1
            elif acm[i, j - 1] == d:
                j -= 1
            else:
                i -= 1
                j -= 1

        path[path_len] = [i, j]
        path_len += np.int_(1)
    return path[-(len(path) - path_len + 1) :: -1, :]


@njit(cache=True)
def _euclidean(x):
    return np.linalg.norm(x)


@njit(cache=True)
def _squared_euclidean(x):
    return np.linalg.norm(x) ** 2
