import numpy as np

from curvesimilarities import ifd, ifd_owp
from curvesimilarities.integfrechet import _cell_info, _line_point_square_integrate


def test_ifd_degenerate():

    P = np.asarray([[0, 0]], dtype=np.float64)
    Q = np.asarray([[0, 1], [1, 1]], dtype=np.float64)
    assert ifd(P, Q, 0.1) == _line_point_square_integrate(Q[0], Q[1], P[0])

    P = np.asarray([[0, 1], [1, 1]], dtype=np.float64)
    Q = np.asarray([[0, 0]], dtype=np.float64)
    assert ifd(P, Q, 0.1) == _line_point_square_integrate(P[0], P[1], Q[0])


def test_lm():
    P = np.array([[0.5, 0], [1, 0]], dtype=np.float64)
    L1 = np.linalg.norm(np.diff(P, axis=0), axis=-1)
    Q = np.array([[0, 1], [1, 1]], dtype=np.float64)
    L2 = np.linalg.norm(np.diff(Q, axis=0), axis=-1)
    assert _cell_info(P, L1, Q, L2)[4] == 0.5


def test_ifd():
    assert ifd([[0, 0], [1, 0]], [[0, 1], [1, 1]], 0.1) == 2.0
    assert ifd([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]], 0.1) == 2.0
    assert ifd([[0, 0], [1, 0]], [[0, 1], [0.5, 1], [1, 1]], 0.1) == 2.0
    assert ifd([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [0.5, 1], [1, 1]], 0.1) == 2.0


def test_ifd_dtype():
    assert (
        type(ifd([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [0.5, 1], [1, 1]], 0.1)) is float
    )


def test_ifd_owp():

    def check_value(P, Q, delta):
        assert ifd_owp(P, Q, delta)[0] == ifd(P, Q, delta)

    check_value([[0, 0], [1, 0]], [[0, 1], [1, 1]], 0.1)
    check_value([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]], 0.1)
    check_value([[0, 0], [1, 0]], [[0, 1], [0.5, 1], [1, 1]], 0.1)
    check_value([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [0.5, 1], [1, 1]], 0.1)


def test_ifd_owp_dtype():
    dist, path = ifd_owp([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [0.5, 1], [1, 1]], 0.1)
    assert type(dist) is float
    assert path.dtype == np.float64


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
    _, owp = ifd_owp(P, Q, 5.0)
    assert owp[-1, 0] == np.sum(np.linalg.norm(np.diff(P, axis=0), axis=-1))
    assert owp[-1, 1] == np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=-1))


def test_ifd_owp_vertices_refined():
    P, Q, delta = [[0, 0], [0.5, 0], [1, 0]], [[0.5, 1], [1.5, 1]], 0.1
    _, path = ifd_owp(P, Q, delta)
    assert not np.any(np.linalg.norm(np.diff(path, axis=0), axis=-1) == 0)

    P, Q, delta = [[0, 0], [1, 1]], [[0, 1], [1, 0]], 0.1
    _, path = ifd_owp(P, Q, delta)
    assert not np.any(np.linalg.norm(np.diff(path, axis=0), axis=-1) == 0)

    P = [[0, 0], [2, 2], [4, 2], [4, 4], [2, 1], [5, 1], [7, 2]]
    Q = [[2, 0], [1, 3], [5, 3], [5, 2], [7, 3]]
    _, path = ifd_owp(P, Q, 0.1)
    assert not np.any(np.linalg.norm(np.diff(path, axis=0), axis=-1) == 0)
