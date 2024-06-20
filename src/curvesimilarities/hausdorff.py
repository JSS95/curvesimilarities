"""Hausdorff distance."""

__all__ = [
    "hd",
]


def hd(P, Q):
    """(Continuous) Hausdorff distance between two open polygonal curves.

    See Also
    --------
    scipy.spatial.distance.directed_hausdorff :
        Directed Hausdorff distance between discrete sets of points.
    """
