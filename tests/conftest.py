import numpy as np
import pytest

from curvesimilarities.util import sample_polyline

P_VERT = [[0, 0], [2, 2], [4, 2], [4, 4], [2, 1], [5, 1], [7, 2]]
Q_VERT = [[2, 0], [1, 3], [5, 3], [5, 2], [7, 3]]


@pytest.fixture
def P_pts():
    param = np.linspace(0, np.sum(np.linalg.norm(np.diff(P_VERT, axis=0), axis=-1)), 10)
    return sample_polyline(P_VERT, param)


@pytest.fixture
def Q_pts():
    param = np.linspace(0, np.sum(np.linalg.norm(np.diff(Q_VERT, axis=0), axis=-1)), 10)
    return sample_polyline(Q_VERT, param)
