from curvesimilarities import afd, afd_owp, ifd_owp


def test_afd():
    assert afd([[0, 0], [1, 0]], [[0, 1], [1, 1]], 0.1) == 1.0
    assert afd([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]], 0.1) == 1.0
    assert afd([[0, 0], [1, 0]], [[0, 1], [0.5, 1], [1, 1]], 0.1) == 1.0
    assert afd([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [0.5, 1], [1, 1]], 0.1) == 1.0


def test_afd_owp():
    assert afd_owp([[0, 0], [1, 0]], [[0, 1], [1, 1]], 0.1)[0] == 1.0
    assert afd_owp([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]], 0.1)[0] == 1.0
    assert afd_owp([[0, 0], [1, 0]], [[0, 1], [0.5, 1], [1, 1]], 0.1)[0] == 1.0
    assert (
        afd_owp([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [0.5, 1], [1, 1]], 0.1)[0] == 1.0
    )
