"""Utility functions."""

import functools

import numpy as np


def sanitize_vertices(owp):
    """Decorator to sanitize the vertices."""

    def decorator(func):

        @functools.wraps(func)
        def wrapper(P, Q, *args, **kwargs):
            P = np.asarray(P, dtype=np.float_)
            Q = np.asarray(Q, dtype=np.float_)

            if len(P.shape) != 2:
                raise ValueError("P must be a 2-dimensional array.")
            if len(Q.shape) != 2:
                raise ValueError("Q must be a 2-dimensional array.")
            if P.shape[1] != Q.shape[1]:
                raise ValueError("P and Q must have the same number of columns.")

            if P.size == 0 or Q.size == 0:
                if owp:
                    return np.float_(np.nan), np.empty((0, 2), dtype=np.int_)
                else:
                    return np.float_(np.nan)
            return func(P, Q, *args, **kwargs)

        return wrapper

    return decorator
