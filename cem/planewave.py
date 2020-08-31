from numbers import Number
from typing import Tuple

import numpy as np

# SI units are used here
# Constant: speed of light in vacuum
C = 299792458.0


class Space:
    def __init__(
            self,
            size: Tuple[Number, Number],
            step_size: Tuple[Number, Number],
            wvlen: Number,
    ):
        self._step_size = step_size

        # Magnitude of electric field
        self._E0 = 1.0
        # Initial phase
        self._phi = 0.0

        # Wavenumber
        self._k = 2 * np.pi / wvlen
        # Angular frequency
        self._omega = C * self._k

        self._Nx, self._Ny = self._compute_num_grids(size, step_size)

        self._Ex = np.zeros((self._Nx, self._Ny))

    @staticmethod
    def _compute_num_grids(size, step_size):
        assert (len(size) == 2)
        assert (len(step_size) == 2)
        return (int(x[0] / x[1]) for x in zip(size, step_size))

    @property
    def get_step_size(self):
        return self._step_size

    @property
    def get_Ex(self):
        return self._Ex

    def propagate(self, t):
        y = np.arange(self._Ny) * self._step_size[1]
        self._Ex[...] = self._E0 * np.cos(self._k * y - self._omega * t +
                                          self._phi)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as ani

    # Set a wavelength of 1m
    wvlen = 1
    # Set a space of 1m x 10m with a step size of 1cm
    space = Space((1, 10), (0.01, 0.01), wvlen)
    # Visualize for 2ns with a step size of 100ps
    time = np.arange(0, 2e-9, 1e-10)

    step_size_x, step_size_y = space.get_step_size

    def draw(t):
        space.propagate(t)
        image = plt.imshow(space.get_Ex, cmap='seismic')
        plt.title('$\lambda$ = ' + str(wvlen) + 'm, t = ' +
                  str(round(t * 1e9, 1)) + 'ns')
        plt.ylabel('x (' + str(step_size_x) + 'm)')
        plt.xlabel('y (' + str(step_size_y) + 'm)')
        return image,

    fig = plt.figure()
    animation = ani.FuncAnimation(fig, draw, frames=time, blit=True)
    animation.save('animation.gif', writer=ani.PillowWriter(fps=5))
