import numpy as np

from curvesimilarities.frechet import _decision_problem


def test_decision_problem():
    P = np.array([[0, 0], [0.5, 0], [1, 0]], dtype=np.float_)
    Q = np.array([[0, 1], [1, 1]], dtype=np.float_)
    assert _decision_problem(P, Q, 1.0)
