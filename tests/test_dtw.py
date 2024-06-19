import numpy as np
from scipy.spatial.distance import cdist

from curvesimilarities import dtw, sdtw


def test_dtw_dtype():
    assert type(dtw([[0, 0], [1, 0]], [[0, 1], [1, 1]])) is float


def test_dtw_degenerate():

    def check(P, Q):
        assert dtw(P, Q) == np.sum(cdist(P, Q))

    check([[0, 0]], [[0, 1]])
    check([[0, 0], [1, 0]], [[0, 1]])
    check([[0, 0]], [[0, 1], [1, 1]])


def test_sdtw_dtype():
    assert type(sdtw([[0, 0], [1, 0]], [[0, 1], [1, 1]])) is float
