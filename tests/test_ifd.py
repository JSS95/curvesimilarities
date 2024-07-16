import numpy as np

from curvesimilarities import ifd, ifd_owp


def test_ifd():
    P, Q = [[0, 0], [1, 0]], [[0, 1], [1, 1]]
    assert ifd(np.asarray(P), np.asarray(Q), 1, "squared_euclidean") == 2
    assert ifd(np.asarray(Q), np.asarray(P), 1, "squared_euclidean") == 2

    P, Q = [[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]]
    assert ifd(np.asarray(P), np.asarray(Q), 0.5, "squared_euclidean") == 2
    assert ifd(np.asarray(Q), np.asarray(P), 0.5, "squared_euclidean") == 2

    P, Q = [[0, 0], [0.5, 0], [1, 0]], [[0, 1], [0.5, 1], [1, 1]]
    assert ifd(np.asarray(P), np.asarray(Q), 0.5, "squared_euclidean") == 2
    assert ifd(np.asarray(Q), np.asarray(P), 0.5, "squared_euclidean") == 2


def test_ifd_degenerate(P_vert, Q_vert):
    np.isnan(ifd(P_vert[:0], Q_vert[:0], 0.1, "squared_euclidean"))


def test_ifd_owp():
    P, Q = [[0, 0], [1, 0]], [[0, 1], [1, 1]]
    assert ifd_owp(np.asarray(P), np.asarray(Q), 1, "squared_euclidean")[0] == 2
    assert ifd_owp(np.asarray(Q), np.asarray(P), 1, "squared_euclidean")[0] == 2

    P, Q = [[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]]
    assert ifd_owp(np.asarray(P), np.asarray(Q), 0.5, "squared_euclidean")[0] == 2
    assert ifd_owp(np.asarray(Q), np.asarray(P), 0.5, "squared_euclidean")[0] == 2

    P, Q = [[0, 0], [0.5, 0], [1, 0]], [[0, 1], [0.5, 1], [1, 1]]
    assert ifd_owp(np.asarray(P), np.asarray(Q), 0.5, "squared_euclidean")[0] == 2
    assert ifd_owp(np.asarray(Q), np.asarray(P), 0.5, "squared_euclidean")[0] == 2


def test_ifd_owp_failedcases():
    P = [
        [403, 250],
        [403, 253],
        [402, 254],
    ]
    Q = [
        [355.75, 243.0],
        [355.89, 244.5],
        [355.75, 246.0],
    ]
    _, owp = ifd_owp(P, Q, 5.0, "squared_euclidean")
    assert owp[-1, 0] == np.sum(np.linalg.norm(np.diff(P, axis=0), axis=-1))
    assert owp[-1, 1] == np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=-1))


def test_ifd_owp_vertices_refined():
    P, Q, delta = [[0, 0], [0.5, 0], [1, 0]], [[0.5, 1], [1.5, 1]], 0.1
    _, path = ifd_owp(P, Q, delta, "squared_euclidean")
    assert not np.any(np.linalg.norm(np.diff(path, axis=0), axis=-1) == 0)

    P, Q, delta = [[0, 0], [1, 1]], [[0, 1], [1, 0]], 0.1
    _, path = ifd_owp(P, Q, delta, "squared_euclidean")
    assert not np.any(np.linalg.norm(np.diff(path, axis=0), axis=-1) == 0)

    P = [[0, 0], [2, 2], [4, 2], [4, 4], [2, 1], [5, 1], [7, 2]]
    Q = [[2, 0], [1, 3], [5, 3], [5, 2], [7, 3]]
    _, path = ifd_owp(P, Q, 0.1, "squared_euclidean")
    assert not np.any(np.linalg.norm(np.diff(path, axis=0), axis=-1) == 0)
