import pytest
import numpy as np
from cem.planewave import C, Space


def test_speed_light():
    assert C == pytest.approx(299792458)


def test_space():
    space = Space(size=(1, 10), step_size=(0.1, 0.01), wvlen=5)
    assert space._Nx == pytest.approx(1 / 0.1)
    assert space._Ny == pytest.approx(10 / 0.01)
    assert space._k == pytest.approx(2 * np.pi / 5)
    assert space._omega == pytest.approx(C * space._k)


def test_prop():
    wvlen = 5
    space = Space(size=(1, 10), step_size=(0.1, 1), wvlen=wvlen)
    assert space._E0 == pytest.approx(1)
    assert space._phi == pytest.approx(0)
    T0 = wvlen / C
    space.propagate(T0 / 4)
    assert space._Ex[:, 0] == pytest.approx(np.zeros(space._Nx))
    space.propagate(T0 / 2)
    assert space._Ex[:, 0] == pytest.approx(-np.ones(space._Nx))
    space.propagate(T0)
    assert space._Ex[:, 0] == pytest.approx(np.ones(space._Nx))
