import numpy as np

from curvesimilarities import qafd, qafd_owp
from curvesimilarities.averagefrechet import _line_point_square_integrate


def test_qafd_dtype():
    assert type(qafd([[0, 0], [1, 0]], [[0, 1], [1, 1]], 0.1)) is float


def test_qafd_owp_dtype():
    dist, path = qafd_owp([[0, 0], [1, 0]], [[0, 1], [1, 1]], 0.1)
    assert type(dist) is float
    assert path.dtype == np.float64


def test_qafd_degenerate():

    def check(P, Q):
        P = np.asarray(P, dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)
        if len(P) == 1:
            point, curve = P, Q
        else:
            point, curve = Q, P
        curve_len = np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=-1))
        integ = _line_point_square_integrate(curve[0], curve[1], point[0])
        assert qafd(P, Q, 0.1) == np.sqrt(integ / curve_len)

    check([[0, 0]], [[0, 1], [2, 1]])
    check([[0, 1], [2, 1]], [[0, 0]])
