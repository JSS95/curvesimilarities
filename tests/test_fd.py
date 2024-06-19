import numpy as np
from scipy.spatial.distance import cdist

from curvesimilarities.frechet import _decision_problem, dfd, fd


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


def test_dfd_dtype():
    assert type(dfd([[0, 0], [1, 0]], [[0, 1], [1, 1]])) is float
