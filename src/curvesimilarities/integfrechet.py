"""Integral Fréchet distance."""

import numpy as np
from numba import njit
from numba.np.extensions import cross2d

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

    Examples
    --------
    >>> ifd([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]])
    """
    ...


@njit(cache=True)
def _cell_owp_integral(P1, P2, Q1, Q2, s, t):
    """Integral along optimal warping path between two points in a cell."""
    P1P2 = P2 - P1
    Q1Q2 = Q2 - Q1
    L1 = np.linalg.norm(P1P2)
    L2 = np.linalg.norm(Q1Q2)

    if L1 < EPSILON:
        u = np.array([0, 0], np.float_)
    else:
        u = (P1P2) / L1
    if L2 < EPSILON:
        v = np.array([0, 0], np.float_)
    else:
        v = (Q1Q2) / L2

    # Find lm: y = x + b
    w = Q1 - P1
    if np.abs(cross2d(P1P2, Q1Q2)) > EPSILON:
        # b = -x + y where A.[x, y] = B
        u_dot_v = np.dot(u, v)
        A = np.array([[1, -u_dot_v], [u_dot_v, -1]], dtype=np.float_)
        B = np.array([-np.dot(u, w), -np.dot(v, w)], dtype=np.float_)
        b = np.dot(np.array([-1, 1]), np.linalg.solve(A, B))
    else:
        # P and Q are parallel.
        # Equations degenerate into x - y = -u.w, therefore b = u.w
        b = np.dot(u, w)

    # Find steiner points in curve space
    P_s = P1 + u * s[0]
    P_t = P1 + u * t[0]
    Q_s = Q1 + v * s[1]
    Q_t = Q1 + v * t[1]

    if s[1] > s[0] + b:
        cs = np.array([s[1] - b, s[1]])
    else:
        cs = np.array([s[0], s[0] + b])
    if t[1] > t[1] + b:
        ct = np.array([t[1] - b, t[1]])
    else:
        ct = np.array([t[0], t[0] + b])

    if cs[0] < ct[0]:  # pass through lm
        P_cs = P1 + u * cs[0]
        P_ct = P1 + u * ct[0]
        Q_cs = Q1 + v * cs[1]
        Q_ct = Q1 + v * ct[1]

        if s[1] > s[0] + b:  # right
            s_to_cs = _line_point_integrate(P_s, P_cs, Q_s)
        else:  # up
            s_to_cs = _line_point_integrate(Q_s, Q_cs, P_s)

        cs_to_ct = _line_line_integrate(P_cs, P_ct, Q_cs, Q_ct)

        if t[1] > t[0] + b:  # up
            ct_to_t = _line_point_integrate(Q_ct, Q_t, P_t)
        else:  # right
            ct_to_t = _line_point_integrate(P_ct, P_t, Q_t)

        ret = s_to_cs + cs_to_ct + ct_to_t

    else:  # pass c'
        if s[1] > s[0] + b:  # right -> up
            ret = (
                _line_point_integrate(P_s, P_t, Q_s)
                + _line_point_integrate(Q_s, Q_t, P_t)
            )
        else:  # up -> right
            ret = (
                _line_point_integrate(Q_s, Q_t, P_s)
                + _line_point_integrate(P_s, P_t, Q_t)
            )
    return ret


@njit(cache=True)
def _line_point_integrate(a, b, p):
    r"""Analytic integration from AP to BP.

    .. math::
        \int_0^1 \lVert (A - P) + (B - A) t \rVert \cdot \lVert (B - A) \rVert dt
    """
    # Goal: integrate sqrt(A*t**2 + B*t + C) * sqrt(A) dt over t [0, 1]
    # where A = dot(ab, ab), B = 2 * dot(pa, ab) and C = dot(pa, pa).
    # Can be simplified to A * integral sqrt(t**2 + B/A*t + C/A) dt.
    # Rewrite: A * integral sqrt(t**2 + B*t + C) dt over t [0, 1]
    # where B = 2 * dot(pa, ab) / A and C = dot(pa, pa) / A.
    ab = b - a
    A = np.dot(ab, ab)
    if A < EPSILON:
        # Degenerate: ab does not form line segement.
        return 0
    ap = p - a
    if np.abs(cross2d(ab, ap)) < EPSILON:
        # Degenerate: a, b, p all on a same line.
        t = np.dot(ab, ap) / A
        L1 = np.dot(ap, ap)
        bp = p - b
        L2 = np.dot(bp, bp)
        if t < 0:
            return (L2 - L1) / 2
        elif t > 1:
            return (L1 - L2) / 2
        else:
            return (L1 + L2) / 2
    pa = -ap
    B = 2 * np.dot(ab, pa) / A
    C = np.dot(pa, pa) / A
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
    # where A = dot(vu, vu), B = 2 * dot(vu, w), C = dot(w, w), D = dot(u, u),
    # and E = dot(v, v); where u = b - a, v = d - c and w = a - c.
    # Rewrite: (sqrt(A*D) + sqrt(A*E)) * integral sqrt(t**2 + B*t + C) dt over t [0, 1]
    # where B = 2 * dot(vu, w) / A and C = dot(w, w) / A
    u, v, w = b - a, d - c, a - c
    vu = u - v
    A = np.dot(vu, vu)
    C, D, E = np.dot(w, w), np.dot(u, u), np.dot(v, v)
    if A < EPSILON:
        # Degenerate: ab and cd has same direction and magnitude
        return np.sqrt(C * D) + np.sqrt(C * E)
    B, C = 2 * np.dot(vu, w) / A, C / A
    if B < EPSILON and C < EPSILON:
        # Degenerate: B and C are 0 (either w = 0 or A is too large)
        return (np.sqrt(A * D) + np.sqrt(A * E)) / 2
    denom = B + 2 * np.sqrt(C)
    if denom < EPSILON:
        # Degenerate: u-v and w are on the opposite direction
        if B > 0 or B < -2:
            integ = (-1 - B) / 2
        else:
            integ = (2 + 2 * B + B**2) / 4
    else:
        integ = (
            4 * np.sqrt(1 + B + C)
            + 2 * B * (-np.sqrt(C) + np.sqrt(1 + B + C))
            - (B**2 - 4 * C) * np.log((2 + B + 2 * np.sqrt(1 + B + C)) / denom)
        ) / 8
    return (np.sqrt(A * D) + np.sqrt(A * E)) * integ
