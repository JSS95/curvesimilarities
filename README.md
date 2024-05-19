# CurveSimilarities

[![License](https://img.shields.io/github/license/JSS95/curvesimilarities)](https://github.com/JSS95/curvesimilarities/blob/master/LICENSE)
[![CI](https://github.com/JSS95/curvesimilarities/actions/workflows/ci.yml/badge.svg)](https://github.com/JSS95/curvesimilarities/actions/workflows/ci.yml)
[![CD](https://github.com/JSS95/curvesimilarities/actions/workflows/cd.yml/badge.svg)](https://github.com/JSS95/curvesimilarities/actions/workflows/cd.yml)
[![Docs](https://readthedocs.org/projects/curvesimilarities/badge/?version=latest)](https://curvesimilarities.readthedocs.io/en/latest/?badge=latest)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/curvesimilarities.svg)](https://pypi.python.org/pypi/curvesimilarities/)
[![PyPI Version](https://img.shields.io/pypi/v/curvesimilarities.svg)](https://pypi.python.org/pypi/curvesimilarities/)

Curve similarity measures, implemented using Numba.

## Usage

```python
>>> from curvesimilarities import fd  # (Continuous) Fréchet distance
>>> fd([[0, 0], [1, 3], [2, 0]], [[0, 1], [2, 1]])
2.0
```

## Installation

CurveSimilarities can be installed using `pip`.

```
$ pip install curvesimilarities
```

## Documentation

CurveSimilarities is documented with [Sphinx](https://pypi.org/project/Sphinx/).
The manual can be found on Read the Docs:

> https://curvesimilarities.readthedocs.io/

If you want to build the document yourself, get the source code and install with `[doc]` dependency.
Then, go to `doc` directory and build the document:

```
$ pip install .[doc]
$ cd doc
$ make html
```

Document will be generated in `build/html` directory. Open `index.html` to see the central page.
