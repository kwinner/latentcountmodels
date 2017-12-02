"""
cygdual.pyx

cython wrapper for libgdual

"""

import cython

from logsign import LS_DTYPE

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np


cdef extern from "gdual.h":
    struct ls:
        double mag
        int sign

    void gdual_exp (ls *u, ls *v, size_t n)

@cython.boundscheck(False)
@cython.wraparound(False)
def cygdual_exp(np.ndarray[ls, ndim=1, mode="c"] u not None):
    """
    gdual_exp (u)

    Computes exp of gdual u

    param: array -- a 1-d numpy array

    """
    cdef size_t n = u.shape[0]
    cdef np.ndarray[ls, ndim=1, mode="c"] v = np.zeros(n, dtype=LS_DTYPE)

    gdual_exp(<ls *> u.data, <ls*> v.data, n)

    return v
