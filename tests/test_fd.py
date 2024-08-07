import numpy as np
from scipy.spatial.distance import cdist

from curvesimilarities.frechet import decision_problem, dfd, dfd_idxs, fd, fd_matching


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


def test_fd_decision_problem(P_vert, Q_vert):
    dist = fd(P_vert, Q_vert)
    assert not decision_problem(P_vert, Q_vert, dist / 2)
    assert decision_problem(P_vert, Q_vert, dist)


def test_fd_matching(P_vert, Q_vert):
    assert fd(P_vert[:1], Q_vert[:1]) == fd_matching(P_vert[:1], Q_vert[:1])[0]
    assert fd(P_vert[:1], Q_vert[:2]) == fd_matching(P_vert[:1], Q_vert[:2])[0]
    assert fd(P_vert[:2], Q_vert[:1]) == fd_matching(P_vert[:2], Q_vert[:1])[0]
    assert fd(P_vert[:1], Q_vert[:3]) == fd_matching(P_vert[:1], Q_vert[:3])[0]
    assert fd(P_vert[:3], Q_vert[:1]) == fd_matching(P_vert[:3], Q_vert[:1])[0]
    assert fd(P_vert[:2], Q_vert[:2]) == fd_matching(P_vert[:2], Q_vert[:2])[0]
    assert fd(P_vert, Q_vert) == fd_matching(P_vert, Q_vert)[0]


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
