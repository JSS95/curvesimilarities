.. template taken from Scipy

.. module:: curvesimilarities

*******************************
CurveSimilarities documentation
*******************************

.. plot:: plot-header.py
    :include-source: False

A collection of curve similarity measures.

There are tons of similar packages published on PyPI, but this one aims
to be the most NumPy-friendly and the easiest to use.

API reference
=============

To reproduce the examples, run the following code first:

.. literalinclude:: plot_pre_code
    :language: python

.. warning::
    The similarity functions do not sanitize the vertices of input polygonal curves.
    Make sure your curves do not have consecutive duplicate vertices.

Fréchet distance
----------------

.. automodule:: curvesimilarities.frechet
    :members:

Dynamic time warping distance
-----------------------------

.. automodule:: curvesimilarities.dtw
    :members:

Integral Fréchet distance
-------------------------

.. automodule:: curvesimilarities.integfrechet
    :members:

Utilities
---------

.. automodule:: curvesimilarities.util
    :members:

Glossary
========

.. glossary::

    arc-length parametrization
        Parametrization of curve based on its arc-length.
        This is the default parametrization used in CurveSimilarities.

        Let :math:`P` be a point on a curve :math:`\gamma`. The parameter of :math:`P`
        equals to the arc length of a subcurve of :math:`\gamma` which begins at the
        starting point of :math:`\gamma` and ends at :math:`P`.

        Also known as "natural parametrization".

    vertex parametrization
        Parametrization of polyline based on vertex indices and edge lengths.
        Useful for dealing with vertices.

        Let :math:`P_i` be an :math:`i`-th vertex of a polyline.
        A point which a parameter :math:`t` represents is defined as
        :math:`P_{\lfloor t \rfloor} + (t - \lfloor t \rfloor) (P_{\lfloor t \rfloor + 1} - P_{\lfloor t \rfloor})`

    matching
        A matching between two curves (either continuous or discrete) :math:`P` and
        :math:`Q` is an ordered set of tuple of points
        :math:`\{(p, q) | p \in P \land q \in Q\}`.

        A matching in curve space is uniquely represented by a path in parameter space.

    optimal warping path
        A matching determined by dynamic time warping (DTW) or integral Fréchet distance
        (IFD).

        The term has been widely used for DTW, and we also use it for IFD since it is
        commonly referred to as "continuous dynamic time warping".
