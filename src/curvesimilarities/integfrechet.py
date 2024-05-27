"""Integral Fréchet distance."""

import numpy as np
from numba import njit

__all__ = [
    "ifd",
]


EPSILON = np.finfo(np.float_).eps


def ifd(P, Q):
    r"""Integral Fréchet distance between two open polygonal curves.

    Let :math:`f, g: [0, 1] \to \Omega` be curves defined in a metric space
    :math:`\Omega`. Let :math:`\alpha, \beta: [0, 1] \to [0, 1]` be continuous
    non-decreasing surjections, and define :math:`\pi: [0, 1] \to [0, 1] \times
    [0, 1]` such that :math:`\pi(t) = \left(\alpha(t), \beta(t)\right)`.
    The integral Fréchet distance between :math:`f` and :math:`g` is defined as

    .. math::

        \inf_{\pi} \int_0^1
        \delta\left(\pi(t)\right) \cdot
        \lVert \pi'(t) \rVert
        \mathrm{d}t,

    where :math:`\delta\left(\pi(t)\right)` is a distance between
    :math:`f\left(\alpha(t)\right)` and :math:`g\left(\beta(t)\right)` in
    :math:`\Omega` and :math:`\lVert \cdot \rVert` is a norm in :math:`[0, 1]
    \times [0, 1]`. In this implementation, we choose the Euclidean distance
    as :math:`\delta` and the Manhattan norm as :math:`\lVert \cdot \rVert`.

    Parameters
    ----------
    P : array_like
        A :math:`p` by :math:`n` array of :math:`p` vertices in an
        :math:`n`-dimensional space.
    Q : array_like
        A :math:`q` by :math:`n` array of :math:`q` vertices in an
        :math:`n`-dimensional space.

    Returns
    -------
    dist : double
        The integral Fréchet distance between P and Q.

    Notes
    -----
    This function implements the algorithm of Brankovic et al [#]_.

    References
    ----------
    .. [#] Brankovic, M., et al. "(k, l)-Medians Clustering of Trajectories Using
       Continuous Dynamic Time Warping." Proceedings of the 28th International
       Conference on Advances in Geographic Information Systems. 2020.
    """


@njit(cache=True)
def _line_point_integrate(a, b, p):
    r"""Analytic integration from AP to BP.

    .. math::
        \int_0^1 \lVert (A - P) + (B - A) t \rVert \cdot \lVert (B - A) \rVert dt
    """
    # Goal: integrate sqrt(A*t**2 + B*t + C) * sqrt(A) dt over t [0, 1]
    # where A = dot(b - a, b - a), B = 2 * dot(a - p, b - a) and C = dot(a - p, a - p).
    # Can be simplified to A * integral sqrt(t**2 + B/A*t + C/A) dt.
    # Rewrite: A * integral sqrt(t**2 + B*t + C) dt over t [0, 1]
    # where B = 2 * dot(a - p, b - a) / A and C = dot(a - p, a - p) / A.
    A = np.dot(b - a, b - a)
    if A < EPSILON:
        # Degenerate: ab does not form line segement.
        return 0
    B = 2 * np.dot(b - a, a - p) / A
    C = np.dot(a - p, a - p) / A
    integ = (
        4 * np.sqrt(1 + B + C)
        + 2 * B * (-np.sqrt(C) + np.sqrt(1 + B + C))
        - (B**2 - 4 * C)
        * np.log((2 + B + 2 * np.sqrt(1 + B + C)) / (B + 2 * np.sqrt(C)))
    ) / 8
    return A * integ


@njit(cache=True)
def _line_line_integrate(a, b, c, d):
    r"""Analytic integration from AC to BD.

    .. math::
        \int_0^1 \lVert (A - C) + (B - A + C - D)t \rVert \cdot
        \left( \lVert B - A \rVert + \lVert D - C \rVert \right) dt
    """
    # Goal: integrate sqrt(A*t**2 + B*t + C) * (sqrt(D) + sqrt(E)) dt over t [0, 1]
    # where A = dot(u - v, u - v), B = 2 * dot(u - v, w), C = dot(w, w), D = dot(u, u),
    # and E = dot(v, v); where u = b - a, v = d - c and w = a - c.
    # Rewrite: (sqrt(A*D) + sqrt(A*E)) * integral sqrt(t**2 + B*t + C) dt over t [0, 1]
    # where B = 2 * dot(u - v, w) / A and C = dot(w, w) / A
    u, v, w = b - a, d - c, a - c
    A = np.dot(u - v, u - v)
    C, D, E = np.dot(w, w), np.dot(u, u), np.dot(v, v)
    if A < EPSILON:
        return np.sqrt(C * D) + np.sqrt(C * E)
    B, C = 2 * np.dot(u - v, w) / A, C / A
    integ = (
        4 * np.sqrt(1 + B + C)
        + 2 * B * (-np.sqrt(C) + np.sqrt(1 + B + C))
        - (B**2 - 4 * C)
        * np.log((2 + B + 2 * np.sqrt(1 + B + C)) / (B + 2 * np.sqrt(C)))
    ) / 8
    return (np.sqrt(A * D) + np.sqrt(A * E)) * integ
