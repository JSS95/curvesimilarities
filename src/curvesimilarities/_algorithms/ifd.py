import numpy as np
from numba import njit


@njit(cache=True)
def _steiner_pts(vert, delta):
    N, D = vert.shape
    edge_lens = np.empty(N - 1, dtype=np.float64)
    for i in range(N - 1):
        edge_lens[i] = np.linalg.norm(vert[i + 1] - vert[i])
    subedges_num = np.ceil(edge_lens / delta).astype(np.int_)

    pts = np.empty((np.sum(subedges_num) + 1, D), dtype=np.float64)
    count = 0
    for i in range(N - 1):
        P0 = vert[i]
        v = vert[i + 1] - vert[i]
        n = subedges_num[i]
        for j in range(n):
            pts[count + j] = P0 + (j / n) * v
        count += n
    pts[count] = vert[N - 1]
    return pts, subedges_num
