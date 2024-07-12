import numpy as np
from scipy.spatial.distance import cdist

from curvesimilarities.frechet import dfd, dfd_idxs, fd


def test_fd_analytic():
    P, Q = [[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]]
    dist = fd(np.asarray(P), np.asarray(Q), rel_tol=0, abs_tol=0)
    assert dist == 1.0


def test_fd_degenerate():

    assert np.isnan(fd(np.empty((0, 2)), np.empty((0, 2))))

    def check(P, Q):
        assert fd(np.asarray(P), np.asarray(Q)) == np.max(cdist(P, Q))

    check([[0, 0]], [[0, 1]])
    check([[0, 0], [1, 0]], [[0, 1]])
    check([[0, 0]], [[0, 1], [1, 1]])


def test_fd_duplicate(P_vert, Q_vert):
    P_dup = np.repeat(P_vert, 2, axis=0)
    Q_dup = np.repeat(Q_vert, 2, axis=0)
    assert fd(P_dup, Q_dup) == fd(P_vert, Q_vert)


def test_dfd_degenerate():

    assert np.isnan(dfd(np.empty((0, 2)), np.empty((0, 2))))

    def check(P, Q):
        P, Q = np.asarray(P), np.asarray(Q)
        assert dfd(P, Q) == np.max(cdist(P, Q))

    check([[0, 0]], [[0, 1]])
    check([[0, 0], [1, 0]], [[0, 1]])
    check([[0, 0]], [[0, 1], [1, 1]])


def test_dfd_duplicate(P_pts, Q_pts):
    P_dup = np.repeat(P_pts, 2, axis=0)
    Q_dup = np.repeat(Q_pts, 2, axis=0)
    assert dfd(P_dup, Q_dup) == dfd(P_pts, Q_pts)


def test_dfd_idxs(P_pts, Q_pts):
    dist = cdist(P_pts, Q_pts)
    d, i, j = dfd_idxs(P_pts, Q_pts)
    assert d == dist[i, j]
    assert d == dfd(P_pts, Q_pts)
