#######################################################
# Author: Qixin Hu
# Email:  hqx11@hust.edu.cn
# Version: 1.0
#######################################################
# The backends contain Numpy and CuPy(optional).
# Default one is Numpy, you can choose:
#   1. numpy
#   2. cupy (GPU if it's available)
# we use int64 and float64 as our data type.
#######################################################
# Usage: In the fdtd main program, call:
#
#     from .backend import backend as bd
#
# Then replace np/cp with bd, e.g:
# 1. a = np.array([1,2,3]) ---> a = bd.array([1,2,3])
# 2. np.sin(a)             ---> bd.sin(a)
# 3. np.exp(a)             ---> bd.exp(a)
#
# You can change different backend by call:
# bd.set_backend('cupy') or bd.set_backend('numpy')
#######################################################

# Basic(default) numpy backend
import numpy

# Optional cupy backend
try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# You can add other backend here (e.g: Pytorch, TensorFlow, Mars...)


class Backend:
    """ Basic backend class """

    def __repr__(self):
        return self.__class__.__name__


    def set_backend(self, name: str):
        """
        Set backend to FDTD simulation program.
        Two backends: 'cupy' and 'numpy'
        """

        # check if cupy is installed.
        if name == 'cupy' and not CUPY_AVAILABLE:
            raise RuntimeError(
                """
                CuPy backend is not available. Have you installed cupy?
                Please check https://docs.cupy.dev/en/stable/install.html 
                to see how to install cupy on your device.
                """
            )

        if name == 'numpy':
            self.__class__ = NumpyBackend
        elif name == 'cupy':
            self.__class__ = CuPyBackend
        else:
            raise RuntimeError(r'unknown backend "{}"'.format(name))
        #

# numpy backend
class NumpyBackend(Backend):
    """
    Numpy backend class,
    here we use np.ndarray as our bd.ndarray
    """

    # Types
    int = numpy.int64
    float = numpy.float64

    # Math calculation
    # you can add more if needed
    exp = staticmethod(numpy.exp)
    sin = staticmethod(numpy.sin)
    cos = staticmethod(numpy.cos)
    sqrt = staticmethod(numpy.sqrt)

    # Create array
    # you can add more if needed
    array = staticmethod(numpy.array)
    ones = staticmethod(numpy.ones)
    zeros = staticmethod(numpy.zeros)
    arange = staticmethod(numpy.arange)

    @staticmethod
    def is_array(arr):
        """ check if arr is a numpy ndarray."""
        return isinstance(arr, numpy.ndarray)

    def numpy(self, arr):
        """ if the arr is numpy.ndarray, return it. Or raise TypeError"""
        if self.is_array(arr):
            return arr
        else:
            raise TypeError("The input should be a numpy.ndarray.")


if CUPY_AVAILABLE:
    class CuPyBackend(Backend):
        """
        CuPy backend class, GPU accelerate.
        """

        # Types
        int = cupy.int64
        float = cupy.float64

        # Math calculation
        # you can add more if needed
        exp = staticmethod(cupy.exp)
        sin = staticmethod(cupy.sin)
        cos = staticmethod(cupy.cos)
        sqrt = staticmethod(cupy.sqrt)

        # Create array
        array = staticmethod(cupy.array)
        ones = staticmethod(cupy.ones)
        zeros = staticmethod(cupy.zeros)
        arange = staticmethod(cupy.arange)

        @staticmethod
        def is_array(arr):
            """ check if arr is a cupy ndarray."""
            return isinstance(arr, cupy.ndarray)

        def numpy(self, arr):
            """
            Convert cupy.ndarray into numpy.ndarray.
            The data will transfer between CPU and the GPU,
            which is costly in terms of performance. So I
            recommend call this function after all computation.
            """
            if self.is_array(arr):
                return cupy.asnumpy(arr)
            else:
                raise TypeError("The input should be a cupy.ndarray.")

# Default Backend: numpy
# All the data types and operations is based on the backend.
# User should choose backend at the very beginning.
# By calling bd.set_backend('cupy') to set cupy as backend.
backend = NumpyBackend()

# Test Unit
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    bd = backend
    print(bd)
    # a = bd.ones((10, 10)
    # plt.imshow()
    # plt.show()
    bd.set_backend('abc')
    print(bd)