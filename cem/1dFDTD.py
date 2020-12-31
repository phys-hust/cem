#######################################################
# Author: Qixin Hu
# Email:  hqx11@hust.edu.cn
# Version: 1.0
#######################################################
#   1d-FDTD with TF/SF and ABC
#######################################################

from numbers import Number
from typing import Tuple

from backend import backend as bd

import matplotlib.pyplot as plt
import matplotlib.collections as collections
import math
import os

# UNITES
meters = 1
centimeters = 1e-2 * meters
millimeters = 1e-3 * meters
micrometer = 1e-6 * meters
nanometer = 1e-9 * meters

seconds = 1
hertz = 1 / seconds
kilohertz = 1e3 * hertz
megahertz = 1e6 * hertz
gigahertz = 1e9 * hertz

# CONSTANTS
c0 = 299792458 * meters / seconds
e0 = 8.8541878176e-12 * 1 / meters
u0 = 1.2566370614e-6 * 1 / meters


class GaussSource:
    """
    Basic Gauss Source.
    The program automatic choose tau and t0
    """

    def __init__(self, fmax: Number = 5.0 * gigahertz, nz_src: Number = 2):
        # the max frequency of Gauss source
        assert (fmax != 0)
        self.fmax = fmax
        self.tau = 0.5 / fmax
        self.t0 = 5 * self.tau
        # The source's grid location (default 2)
        self.nz_src = nz_src

    @property
    def get_fmax(self):
        return self.fmax

    @property
    def get_t0(self):
        return self.t0

    @property
    def get_src(self):
        return self.nz_src

    def compute_source(self, t):
        return bd.exp(- ((t - self.t0) / self.tau) ** 2)


class SlabDevice:
    """
    Basic permittivity slab device.
    """

    def __init__(self, dslab: Number = 0, erslab: Number = 1):
        # the size here is real length.
        self.dslab = dslab
        self.erslab = erslab

    @property
    def get_size(self):
        return self.dslab

    @property
    def get_er(self):
        return self.erslab


class FDTD1dGrid:
    """
    1dFDTD update equation with TF/SF and ABC,
    The only thing you need to do is setup source parameters(fmax) and slab device parameters(size and ER).
    you can add simple device by call
    """

    def __init__(self, source: GaussSource, device: SlabDevice, visualize=False):
        self.source = source
        self.device = device
        self.nz_src = self.source.get_src

        # GRID SUPER PARAMETER
        # How many points we simulate through one-wave length (Time steps)
        self.N_lam = 10
        # Grid resolution to resolve the smallest device features
        self.N_dim = 1
        # Device两边的缓冲区域
        self.N_bufz = (100, 100)

        # Initial optimizer grad parameters(dz, Nz, dt, STEPS).
        self.dz, self.Nz, self.dt, self.STEPS = 0.0, 0, 0.0, 0
        self.initial_grid()

        # Applied device on grid
        self.ER = bd.ones(self.Nz)
        self.UR = bd.ones(self.Nz)
        self.applied_device()

        # FDTD fields and update coefficients
        self.mEy = (c0 * self.dt) / self.ER
        self.mHx = (c0 * self.dt) / self.UR
        self.Ey = bd.zeros(self.Nz)
        self.Hx = bd.zeros(self.Nz)

        # ABC parameters (we choose a special time_step dt to implement 1d ABC)
        self.H3, self.H2, self.H1 = 0, 0, 0
        self.E3, self.E2, self.E1 = 0, 0, 0

        # TF/SF parameters
        self.delay = self.dz / (2 * c0) + self.dt / 2
        self.normH = - bd.sqrt(self.ER[self.nz_src] / self.UR[self.nz_src])

        # used for show the results
        self.fig, self.ax = plt.subplots()
        self.za = bd.arange(self.Nz) * self.dz

        # If visualize results:
        self.VISUALIZE = visualize

    def run(self):

        for i in range(self.STEPS):
            self.update_H(i)
            self.update_E(i)

            # 每20个iter播放一帧
            if ((i + 1) % 20) == 0 and self.VISUALIZE:
                self.visualization_frame(i)

    def update_H(self, i):
        # update H from E with ABC
        self.Hx[:-1] = self.Hx[:-1] + self.mHx[:-1] * (self.Ey[1:] - self.Ey[:-1]) / self.dz
        self.Hx[-1] = self.Hx[-1] + self.mHx[-1] * (self.E3 - self.Ey[-1]) / self.dz
        self.H3 = self.H2
        self.H2 = self.H1
        self.H1 = self.Hx[0]

        # add TF/SF source
        self.Hx[self.nz_src - 1] -= self.mHx[self.nz_src - 1] * self.source.compute_source(i * self.dt) / self.dz

    def update_E(self, i):
        # update E from H with ABC
        self.Ey[0] = self.Ey[0] + self.mEy[0] * (self.Hx[0] - self.H3) / self.dz
        self.Ey[1:] = self.Ey[1:] + self.mEy[1:] * (self.Hx[1:] - self.Hx[:-1]) / self.dz
        self.E3 = self.E2
        self.E2 = self.E1
        self.E1 = self.Ey[-1]

        # add TF/SF source
        self.Ey[self.nz_src] -= self.mEy[self.nz_src] * self.normH * self.source.compute_source(
            i * self.dt + self.delay) / self.dz

    def visualization_frame(self, i):
        self.ax.cla()
        self.ax.plot(self.za, self.Ey, 'r', label='Ey')
        self.ax.plot(self.za, self.Hx, 'b', label='Hx')

        # plot device
        collection = collections.BrokenBarHCollection.span_where(
            self.za, ymin=-1.5, ymax=1.5, where=self.ER > 1.0
        )
        self.ax.add_collection(collection)

        plt.title(r"Gauss source frequency:5GHz, t={:.3}ns".format(i * self.dt * 1e9))
        plt.xlabel("z(m)")
        plt.ylabel("normalized EM field.")
        plt.ylim(-1.5, 1.5)
        plt.legend()
        # plt.savefig(os.path.join('./examples', str("1dFDTD"), str(i)+'.png'))
        plt.pause(0.01)

    def initial_grid(self):
        """
        Initial the grid parameters,
        cal dz, Nz, dt, STEPS and apply device on grid.
        """
        # Find the ER_max and n_max
        ermax = max([1.0, self.device.get_er])  # max([erair, erslab])
        nmax = bd.sqrt(ermax)

        # compute wvlen
        wvlen = c0 / self.source.get_fmax
        # compute dz and NZ
        if self.device.get_size == 0:
            self.dz = wvlen / nmax / self.N_lam
        else:
            self.dz = min([wvlen / nmax / self.N_lam, self.device.get_size / self.N_dim])
        nz = math.ceil(self.device.get_size / self.dz)
        self.dz = self.device.get_size / nz
        self.Nz = int(round(nz)) + sum(self.N_bufz) + 3

        # compute dt and STEPS
        nbc = 1.0  # Boundary n
        # In order to perform TF/SF and ABC,
        # we choose this dt (2 time steps move one dz grid)
        self.dt = nbc * self.dz / (2 * c0)

        # Compute total time steps
        tprop = nmax * self.Nz * self.dz / c0
        T = 2 * self.source.get_t0 + 1.5 * tprop
        self.STEPS = math.ceil(T / self.dt)

    def applied_device(self):
        """
        Add device on the grid.
        """
        nz1 = 2 + self.N_bufz[0] + 1
        nz2 = int(nz1 + round(self.device.get_size / self.dz) - 1)
        self.ER[nz1:nz2] = self.device.get_er


if __name__ == "__main__":
    # Example:
    # Set a GaussSource with fmax = 5.0GHz
    # and a SlabDevice with 5cm and ER = 12
    source = GaussSource(5.0 * gigahertz)

    device = SlabDevice(5 * centimeters, erslab=12.0)

    fdtd1d = FDTD1dGrid(source, device, True)

    fdtd1d.run()
