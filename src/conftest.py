"""Configure doctesting by pytest."""

import numpy as np
import pytest

import curvesimilarities
import curvesimilarities.util


@pytest.fixture(autouse=True)
def doctest_pre_code(doctest_namespace):
    """Import modules for doctesting.

    This fixture is equivalent to::

        import numpy as np
        from curvesimilarities import *
        from curvesimilarities.util import *
    """
    doctest_namespace["np"] = np
    for var in curvesimilarities.__all__:
        doctest_namespace[var] = getattr(curvesimilarities, var)
    for var in curvesimilarities.util.__all__:
        doctest_namespace[var] = getattr(curvesimilarities.util, var)
