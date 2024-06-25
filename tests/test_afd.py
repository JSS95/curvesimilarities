import numpy as np

from curvesimilarities import qafd, qafd_owp


def test_qafd_dtype():
    assert type(qafd([[0, 0], [1, 0]], [[0, 1], [1, 1]], 0.1)) is float


def test_qafd_owp_dtype():
    dist, path = qafd_owp([[0, 0], [1, 0]], [[0, 1], [1, 1]], 0.1)
    assert type(dist) is float
    assert path.dtype == np.float64
