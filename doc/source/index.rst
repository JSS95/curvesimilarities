.. template taken from Scipy

.. module:: curvesimilarities

*******************************
CurveSimilarities documentation
*******************************

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

Average Fréchet distance
------------------------

.. automodule:: curvesimilarities.averagefrechet
    :members:
