"""Average Fréchet distance and its variants."""

import numpy as np

from .integfrechet import ifd, ifd_owp

__all__ = [
    "qafd",
    "qafd_owp",
]


def qafd(P, Q, delta, dist="euclidean"):
    r"""Quadratic average Fréchet distance between two open polygonal curves.

    The quadratic average Fréchet distance is defined as

    .. math::

        \inf_{\pi}
        \sqrt{
            \frac{
                \int_0^1
                dist\left(\pi(t)\right)^2 \cdot
                \lVert \pi'(t) \rVert_1
                \mathrm{d}t
            }{
                \int_0^1
                \lVert \pi'(t) \rVert_1
                \mathrm{d}t
            }
        },

    where :math:`dist` is the Euclidean distance and :math:`\pi` is the continuous
    nondecreasing path in the parameter space. Because we use the Manhattan norm
    :math:`\lVert \cdot \rVert_1`, the formula can be reduced to

    .. math::

        \frac{1}{\sqrt{\lvert f \rvert + \lvert g \rvert}}
        \inf_{\pi}
        \sqrt{
        \int_0^1
        dist\left(\pi(t)\right)^2 \cdot
        \lVert \pi'(t) \rVert_1
        \mathrm{d}t
        },

    where :math:`\lvert \cdot \rvert` denotes the length of a polygonal curve.

    Parameters
    ----------
    P : array_like
        A :math:`p` by :math:`n` array of :math:`p` vertices in an
        :math:`n`-dimensional space.
    Q : array_like
        A :math:`q` by :math:`n` array of :math:`q` vertices in an
        :math:`n`-dimensional space.
    delta : double
        Maximum length of edges between Steiner points.
    dist : {'euclidean'}
        Type of :math:`dist`. Refer to the Notes section for more information.

    Returns
    -------
    double
        The quadratic average Fréchet distance between *P* and *Q*, NaN if any
        vertice is empty or both vertices consist of a single point.

    Raises
    ------
    ValueError
        If *P* and *Q* are not 2-dimensional arrays with same number of columns.

    See Also
    --------
    afd : Average Fréchet distance.
    qafd_owp : Quadratic average Fréchet distance with optimal warping path.

    Examples
    --------
    >>> qafd([[0, 0], [0.5, 0], [1, 0]], [[0, 1], [1, 1]], 0.1)
    1.0
    """
    if dist == "euclidean":
        square_ifd = ifd(P, Q, delta, dist="squared_euclidean")
    else:
        raise ValueError(f"Unknown type of distance: {dist}")
    P_len = np.sum(np.linalg.norm(np.diff(P, axis=0), axis=-1))
    Q_len = np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=-1))
    return float(np.sqrt(square_ifd / (P_len + Q_len)))


def qafd_owp(P, Q, delta, dist="euclidean"):
    """Quadratic average Fréchet distance and its optimal warping path.

    Parameters
    ----------
    P : array_like
        A :math:`p` by :math:`n` array of :math:`p` vertices in an
        :math:`n`-dimensional space.
    Q : array_like
        A :math:`q` by :math:`n` array of :math:`q` vertices in an
        :math:`n`-dimensional space.
    delta : double
        Maximum length of edges between Steiner points.
    dist : {'euclidean'}
        Type of :math:`dist`. Refer to :func:`qafd`.

    Returns
    -------
    dist : double
        The quadratic average Fréchet distance between *P* and *Q*, NaN if any
        vertice is empty or both vertices consist of a single point.
    owp : ndarray
        Optimal warping path, empty if any vertice is empty or both vertices
        consist of a single point.

    Raises
    ------
    ValueError
        If *P* and *Q* are not 2-dimensional arrays with same number of columns.

    Examples
    --------
    >>> dist, path = qafd_owp([[0, 0], [0.5, 0], [1, 0]], [[0.5, 1], [1.5, 1]], 0.1)
    >>> import matplotlib.pyplot as plt #doctest: +SKIP
    >>> plt.plot(*path.T)  #doctest: +SKIP
    """
    if dist == "euclidean":
        dist, owp = ifd_owp(P, Q, delta, dist="squared_euclidean")
    else:
        raise ValueError(f"Unknown type of distance: {dist}")
    return float(np.sqrt(dist / np.sum(owp[-1]))), owp
