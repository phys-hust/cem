import pytest
import numpy as np
import random

from cem.backend import backend, NumpyBackend

try:
    from cem.backend import CuPyBackend
    import cupy as cp
    skip_cupy_test = False
except ImportError:
    skip_cupy_test = True


def test_numpy_backend():
    X = random.randint(0, 10) * 10
    Y = random.randint(0, 10) * 10

    zeros = backend.zeros((X, Y))
    ones = backend.ones((X, Y))

    assert isinstance(backend, NumpyBackend)
    assert isinstance(zeros, np.ndarray)
    assert isinstance(ones, np.ndarray)
    assert backend.int == np.int64
    assert backend.float == np.float64
    assert zeros.shape == (X, Y)
    assert ones.shape == (X, Y)
    assert backend.sin(ones).any() == np.sin(ones).any()
    assert backend.cos(ones).any() == np.cos(ones).any()


@pytest.mark.skipif(skip_cupy_test, reason='CuPy is not installed.')
def test_cupy_backend():
    backend.set_backend('cupy')
    X = random.randint(0, 10) * 10
    Y = random.randint(0, 10) * 10

    zeros = backend.zeros((X, Y))
    ones = backend.ones((X, Y))

    assert isinstance(backend, CuPyBackend)
    assert isinstance(zeros, cp.ndarray)
    assert isinstance(ones, cp.ndarray)
    assert backend.int == cp.int64
    assert backend.float == cp.float64
    assert zeros.shape == (X, Y)
    assert ones.shape == (X, Y)
    assert backend.sin(ones).all() == cp.sin(ones).all()
    assert backend.cos(ones).all() == cp.cos(ones).all()


@pytest.mark.skipif(skip_cupy_test, reason='CuPy is not installed.')
def test_set_backend():
    backend.set_backend('numpy')
    assert isinstance(backend, NumpyBackend)
    backend.set_backend('cupy')
    assert isinstance(backend, CuPyBackend)
