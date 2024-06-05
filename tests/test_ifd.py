import numpy as np

from curvesimilarities import ifd, ifd_owp
from curvesimilarities.integfrechet import (
    _cell_info,
    _line_line_integrate,
    _line_point_integrate,
)


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
    Q = np.array([[0, 1], [1, 1]], dtype=np.float_)
    assert _cell_info(P, Q)[6] == 0.5


def test_ifd():
    assert ifd([[0, 0], [1, 0]], [[0, 1], [1, 1]], 0.1) == 2.0
    assert ifd([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]], 0.1) == 2.0
    assert ifd([[0, 0], [1, 0]], [[0, 1], [0.5, 1], [1, 1]], 0.1) == 2.0
    assert ifd([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [0.5, 1], [1, 1]], 0.1) == 2.0


def test_ifd_owp():
    assert ifd_owp([[0, 0], [1, 0]], [[0, 1], [1, 1]], 0.1)[0] == 2.0
    assert ifd_owp([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]], 0.1)[0] == 2.0
    assert ifd_owp([[0, 0], [1, 0]], [[0, 1], [0.5, 1], [1, 1]], 0.1)[0] == 2.0
    assert (
        ifd_owp([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [0.5, 1], [1, 1]], 0.1)[0] == 2.0
    )
