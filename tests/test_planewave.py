import pytest
import numpy as np
from random import random
from cem.planewave import C, Space


def test_speed_light():
    assert C == pytest.approx(299792458)


def test_space():
    X = random() * 10
    Y = random() * 10
    dX = random() / 10
    dY = random() / 10
    wvlen = random()

    space = Space(size=(X, Y), step_size=(dX, dY), wvlen=wvlen)

    assert space._Nx == X // dX
    assert space._Ny == Y // dY
    assert space._k == pytest.approx(2 * np.pi / wvlen)
    assert space._omega == pytest.approx(C * space._k)

    assert space._E0 == pytest.approx(1)
    assert space._phi == pytest.approx(0)

    T0 = wvlen / C
    space.propagate(T0 / 4)
    assert space._Ex[:, 0] == pytest.approx(np.zeros(space._Nx))
    space.propagate(T0 / 2)
    assert space._Ex[:, 0] == pytest.approx(-np.ones(space._Nx))
    space.propagate(T0)
    assert space._Ex[:, 0] == pytest.approx(np.ones(space._Nx))
    space.propagate(T0 * 2)
    assert space._Ex[:, 0] == pytest.approx(np.ones(space._Nx))
