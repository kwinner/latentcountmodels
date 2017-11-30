"""
gdual.pyx

cython test of wrapping gdual_exp

"""

import cython
from lsgdual.logsign import LS_DTYPE

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# This defines the struct datatype for logsign numbers. It needs to match
# the datatype of the passed in numpy array as well as the definition of
# the ls struct in libgdual.h
cdef struct LogSign:
    np.float64_t mag
    np.int8_t sgn

# declare the interface to the C code
cdef extern void gdual_exp (LogSign *u, LogSign *v, size_t n)

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_gdual_exp(np.ndarray[LogSign, ndim=1, mode="c"] u not None):
    """
    gdual_exp (u)

    Computes exp of gdual u

    param: array -- a 1-d numpy array

    """
    cdef size_t n = u.shape[0]
    cdef np.ndarray[LogSign, ndim=1, mode="c"] v = np.zeros(n, dtype=LS_DTYPE)

    gdual_exp(<LogSign *> u.data, <LogSign*> v.data, n)

    return v
