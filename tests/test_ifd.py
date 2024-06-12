import numpy as np

from curvesimilarities import ifd, ifd_owp
from curvesimilarities.integfrechet import (
    _cell_info,
    _line_line_integrate,
    _line_point_integrate,
)


def test_ifd_degenerate():

    P = np.asarray([[0, 0]], dtype=np.float_)
    Q = np.asarray([[0, 1], [1, 1]], dtype=np.float_)
    assert ifd(P, Q, 0.1) == _line_point_integrate(Q[0], Q[1], P[0])

    P = np.asarray([[0, 1], [1, 1]], dtype=np.float_)
    Q = np.asarray([[0, 0]], dtype=np.float_)
    assert ifd(P, Q, 0.1) == _line_point_integrate(P[0], P[1], Q[0])


def test_integration_degenerates():
    # test if integration can handle degenerate cases without error.
    A = np.array([1, 0], dtype=np.float_)
    B = np.array([2, 0], dtype=np.float_)

    _line_point_integrate(A, A, np.array([3, 3], dtype=np.float_))
    _line_point_integrate(A, B, (A + B) / 2)

    _line_line_integrate(A, B, A + 1, B + 1)
    _line_line_integrate(A, B, A, np.array([3, 3], dtype=np.float_))
    _line_line_integrate(A, B, B, A)


def test_lm():
    P = np.array([[0.5, 0], [1, 0]], dtype=np.float_)
    L1 = np.linalg.norm(np.diff(P, axis=0), axis=-1)
    Q = np.array([[0, 1], [1, 1]], dtype=np.float_)
    L2 = np.linalg.norm(np.diff(Q, axis=0), axis=-1)
    assert _cell_info(P, L1, Q, L2)[4] == 0.5


def test_ifd():
    assert ifd([[0, 0], [1, 0]], [[0, 1], [1, 1]], 0.1) == 2.0
    assert ifd([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]], 0.1) == 2.0
    assert ifd([[0, 0], [1, 0]], [[0, 1], [0.5, 1], [1, 1]], 0.1) == 2.0
    assert ifd([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [0.5, 1], [1, 1]], 0.1) == 2.0


def test_ifd_owp():

    def check_value(P, Q, delta):
        assert ifd_owp(P, Q, delta)[0] == ifd(P, Q, delta)

    check_value([[0, 0], [1, 0]], [[0, 1], [1, 1]], 0.1)
    check_value([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]], 0.1)
    check_value([[0, 0], [1, 0]], [[0, 1], [0.5, 1], [1, 1]], 0.1)
    check_value([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [0.5, 1], [1, 1]], 0.1)


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
