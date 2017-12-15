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

    void gdual_exp( ls* res, ls* u, size_t n)
    void gdual_log( ls* res, ls* u, size_t n)
    void gdual_inv( ls* res, ls* u, size_t n)

    void gdual_mul( ls* res, ls* u, ls* w, size_t n )
    void gdual_add( ls* res, ls* u, ls* w, size_t n )
    void gdual_div( ls* res, ls* u, ls* w, size_t n )

    void gdual_pow( ls* res, ls* u, double r, size_t n)
    void gdual_scalar_mul( ls* res, ls* u, double c, size_t n)
    void gdual_u_plus_cw( ls* res, ls* u, ls* w, double c, size_t n)



def exp(np.ndarray[ls, ndim=1, mode="c"] u not None):
    return unary_op(u, 'exp')

def log(np.ndarray[ls, ndim=1, mode="c"] u not None):
    return unary_op(u, 'log')

def inv(np.ndarray[ls, ndim=1, mode="c"] u not None):
    return unary_op(u, 'inv')

def mul(np.ndarray[ls, ndim=1, mode="c"] u not None,
        np.ndarray[ls, ndim=1, mode="c"] w not None):
    return binary_op(u, w, 'mul')

def add(np.ndarray[ls, ndim=1, mode="c"] u not None,
        np.ndarray[ls, ndim=1, mode="c"] w not None):
    return binary_op(u, w, 'add')

def div(np.ndarray[ls, ndim=1, mode="c"] u not None,
        np.ndarray[ls, ndim=1, mode="c"] w not None):
    return binary_op(u, w, 'div')


ctypedef void (*UNARY_OP)(ls*, ls*, size_t)

def unary_op(np.ndarray[ls, ndim=1, mode="c"] u not None, op not None):

    cdef size_t n = u.shape[0]
    cdef np.ndarray[ls, ndim=1, mode="c"] v = np.zeros(n, dtype=LS_DTYPE)

    cdef UNARY_OP fun

    if op == 'exp':
        fun = &gdual_exp
    elif op == 'log':
        fun = &gdual_log
    elif op == 'inv':
        fun = &gdual_inv
    else:
        raise('unrecognized unary operation on gduals: ' + op)

    fun(<ls *> v.data, <ls*> u.data, n)

    return v

ctypedef void (*BINARY_OP)(ls*, ls*, ls*, size_t)

def binary_op(np.ndarray[ls, ndim=1, mode="c"] u not None,
              np.ndarray[ls, ndim=1, mode="c"] w not None,
              op not None):

    assert(u.shape[0] == w.shape[0])

    cdef size_t n = u.shape[0]
    cdef np.ndarray[ls, ndim=1, mode="c"] v = np.zeros(n, dtype=LS_DTYPE)

    cdef BINARY_OP fun
    if op == 'mul':
        fun = &gdual_mul
    elif op == 'add':
        fun = &gdual_add
    elif op == 'div':
        fun = &gdual_div
    else:
        raise('unrecognized operation : ' + op)

    fun(<ls *> v.data, <ls *> u.data, <ls*> w.data, n)

    return v


def pow(np.ndarray[ls, ndim=1, mode="c"] u not None, r):

    cdef size_t n = u.shape[0]
    cdef np.ndarray[ls, ndim=1, mode="c"] v = np.zeros(n, dtype=LS_DTYPE)

    gdual_pow(<ls *> v.data, <ls *> u.data, <double> r, <size_t> n)

    return v
