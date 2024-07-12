import numpy as np
from numba import njit


@njit(cache=True)
def _dfd_ca(P, Q):
    # Eiter, T., & Mannila, H. (1994)
    if len(P.shape) != 2:
        raise ValueError("P must be a 2-dimensional array.")
    if len(Q.shape) != 2:
        raise ValueError("Q must be a 2-dimensional array.")
    if P.shape[1] != Q.shape[1]:
        raise ValueError("P and Q must have the same number of columns.")

    P, Q = P.astype(np.float64), Q.astype(np.float64)
    p, q = len(P), len(Q)
    ret = np.empty((p, q), dtype=np.float64)

    for i in range(p):
        if i == 0:
            ret[i, 0] = np.linalg.norm(P[0] - Q[0])
        else:
            ret[i, 0] = max(ret[i - 1, 0], np.linalg.norm(P[i] - Q[0]))

    for j in range(1, q):
        ret[0, j] = max(ret[0, j - 1], np.linalg.norm(P[0] - Q[j]))

    for i in range(1, p):
        for j in range(1, q):
            ret[i, j] = max(
                min(ret[i - 1, j], ret[i, j - 1], ret[i - 1, j - 1]),
                np.linalg.norm(P[i] - Q[j]),
            )

    return ret


@njit(cache=True)
def _dfd_idxs(ca):
    p, q = ca.shape
    i, j = p - 1, q - 1

    while i > 0 or j > 0:
        current = ca[i, j]
        LEFT = np.inf if i == 0 else ca[i - 1, j]
        DOWN = np.inf if j == 0 else ca[i, j - 1]
        DIAG = np.inf if (i == 0 or j == 0) else ca[i - 1, j - 1]
        prev = min(LEFT, DOWN, DIAG)
        if current > prev:
            break
        elif current == LEFT:
            i -= 1
        elif current == DOWN:
            j -= 1
        else:
            i -= 1
            j -= 1
    return (i, j)
