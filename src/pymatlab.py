"""
    Implementation of native matlab methods in python
    Examples: (some differences between matlab and python)
        size((1, n)) -> np.size(n)
"""
import numpy as np


def size(array, axis=0):
    if axis is None:
        return array.shape
    if array.ndim > 1:
        return np.size(array, axis=axis)
    return np.size(array)


def length(array):
    if array.ndim > 1:
        n, p = array.shape
        return n if n >= p else p
    return np.size(array)


def isempty(array):
    return np.size(array) == 0 or np.all(array == 0)


def isfinite(array):
    return np.isfinite(array, np.ones(array.shape))
