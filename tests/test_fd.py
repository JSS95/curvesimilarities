import numpy as np
from scipy.spatial.distance import cdist

from curvesimilarities.frechet import _decision_problem, dfd, dfd_idxs, fd


def test_fd_analytic():
    P, Q = [[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]]
    dist = fd(P, Q, rel_tol=0, abs_tol=0)
    assert dist == 1.0


def test_fd_dtype():
    assert type(fd([[0, 0], [1, 0]], [[0, 1], [1, 1]])) is float


def test_fd_degenerate():

    def check(P, Q):
        assert fd(P, Q) == np.max(cdist(P, Q))

    check([[0, 0]], [[0, 1]])
    check([[0, 0], [1, 0]], [[0, 1]])
    check([[0, 0]], [[0, 1], [1, 1]])


def test_decision_problem():
    P = np.array([[0, 0], [0.5, 0], [1, 0]], dtype=np.float64)
    Q = np.array([[0, 1], [1, 1]], dtype=np.float64)
    assert _decision_problem(P, Q, 1.0)


def test_dfd_idxs(P_pts, Q_pts):
    dist = cdist(P_pts, Q_pts)
    d, i, j = dfd_idxs(P_pts, Q_pts)
    assert d == dist[i, j]
    assert d == dfd(P_pts, Q_pts)
