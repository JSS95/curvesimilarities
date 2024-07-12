import numpy as np
from scipy.spatial.distance import cdist

from curvesimilarities import dtw


def test_dtw_degenerate():

    assert np.isnan(dtw(np.empty((0, 2)), np.empty((0, 2))))

    def check(P, Q):
        P, Q = np.asarray(P), np.asarray(Q)
        assert dtw(P, Q) == np.sum(cdist(P, Q))

    check([[0, 0]], [[0, 1]])
    check([[0, 0], [1, 0]], [[0, 1]])
    check([[0, 0]], [[0, 1], [1, 1]])
