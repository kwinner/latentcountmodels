"""
cygdual.pyx

cython wrapper for libgdual

"""

import cython

import logsign
from logsign import DTYPE as LS_DTYPE

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
    void gdual_neg( ls* res, ls* u, size_t n)

    void gdual_mul( ls* res, size_t n, ls* u, size_t u_len, ls* w, size_t w_len )

    void gdual_mul_same( ls* res, ls* u, ls* w, size_t n )
    void gdual_compose_same( ls* res, ls* u, ls* w, size_t n )
    void gdual_mul_fft( ls* res, ls* u, ls* w, size_t n )
    void gdual_add( ls* res, ls* u, ls* w, size_t n )
    void gdual_div( ls* res, ls* u, ls* w, size_t n )

    void gdual_pow( ls* res, ls* u, double r, size_t n)
    void gdual_scalar_mul( ls* res, ls* u, double c, size_t n)
    void gdual_u_plus_cw( ls* res, ls* u, ls* w, double c, size_t n)


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
    elif op == 'neg':
        fun = &gdual_neg
    else:
        raise(ValueError('unrecognized unary operation on gduals: ' + op))

    fun(<ls *> v.data, <ls*> u.data, n)

    return v

def exp(np.ndarray[ls, ndim=1, mode="c"] u not None):
    return unary_op(u, 'exp')

def log(np.ndarray[ls, ndim=1, mode="c"] u not None):
    return unary_op(u, 'log')

def inv(np.ndarray[ls, ndim=1, mode="c"] u not None):
    return unary_op(u, 'inv')

def neg(np.ndarray[ls, ndim=1, mode="c"] u not None):
    return unary_op(u, 'neg')

ctypedef void (*BINARY_OP)     (ls*, size_t, ls*, size_t, ls*, size_t) # different sized operands
ctypedef void (*BINARY_OP_SAME)(ls*, ls*, ls*, size_t)                 # same size operands

def binary_op(np.ndarray[ls, ndim=1, mode="c"] u not None,
              np.ndarray[ls, ndim=1, mode="c"] w not None,
              op not None,
              truncation_order = None):

    cdef size_t u_len = len(u)
    cdef size_t w_len = len(w)
    cdef size_t n               # output length

    if truncation_order is not None:
        n = truncation_order
    else:
        n = max(len(u), len(w))

    # create output array
    cdef np.ndarray[ls, ndim=1, mode="c"] v = np.zeros(n, dtype=LS_DTYPE)
    
    cdef BINARY_OP fun
    if op == 'mul':
        fun = &gdual_mul
    else:
        raise(ValueError('unrecognized operation: ' + op))

    fun(<ls *> v.data, n, <ls *> u.data, u_len, <ls*> w.data, w_len)

    return v
        
def binary_op_same(np.ndarray[ls, ndim=1, mode="c"] u not None,
                   np.ndarray[ls, ndim=1, mode="c"] w not None,
                   op not None):

    assert(u.shape[0] == w.shape[0])

    cdef size_t n = u.shape[0]
    cdef np.ndarray[ls, ndim=1, mode="c"] v = np.zeros(n, dtype=LS_DTYPE)

    cdef BINARY_OP_SAME fun
    if op == 'mul':
        fun = &gdual_mul_same
    elif op == 'mul_fft':
        fun = &gdual_mul_fft
    elif op == 'add':
        fun = &gdual_add
    elif op == 'div':
        fun = &gdual_div
    elif op == 'compose':
        fun = &gdual_compose_same
    else:
        raise('unrecognized operation : ' + op)

    fun(<ls *> v.data, <ls *> u.data, <ls*> w.data, n)

    return v


def mul(np.ndarray[ls, ndim=1, mode="c"] u not None,
        np.ndarray[ls, ndim=1, mode="c"] w not None,
        truncation_order = None):
    return binary_op(u, w, 'mul', truncation_order=truncation_order)

def mul_same(np.ndarray[ls, ndim=1, mode="c"] u not None,
             np.ndarray[ls, ndim=1, mode="c"] w not None):
    return binary_op_same(u, w, 'mul')

def mul_fft(np.ndarray[ls, ndim=1, mode="c"] u not None,
        np.ndarray[ls, ndim=1, mode="c"] w not None):
    return binary_op_same(u, w, 'mul_fft')

def add(np.ndarray[ls, ndim=1, mode="c"] u not None,
        np.ndarray[ls, ndim=1, mode="c"] w not None):
    return binary_op_same(u, w, 'add')

def div(np.ndarray[ls, ndim=1, mode="c"] u not None,
        np.ndarray[ls, ndim=1, mode="c"] w not None):
    return binary_op_same(u, w, 'div')

def rdiv(np.ndarray[ls, ndim=1, mode="c"] u not None,
         np.ndarray[ls, ndim=1, mode="c"] w not None):
    return binary_op_same(w, u, 'div')


def rsub(np.ndarray[ls, ndim=1, mode="c"] u not None,
         np.ndarray[ls, ndim=1, mode="c"] w not None):
    return sub(w, u)

def sub(np.ndarray[ls, ndim=1, mode="c"] u not None,
        np.ndarray[ls, ndim=1, mode="c"] w not None):

    assert(u.shape[0] == w.shape[0])

    cdef size_t n = u.shape[0]
    cdef np.ndarray[ls, ndim=1, mode="c"] v = np.zeros(n, dtype=LS_DTYPE)

    gdual_u_plus_cw(<ls *> v.data, <ls *> u.data, <ls*> w.data, -1.0, n)

    return v

def compose(np.ndarray[ls, ndim=1, mode="c"] u not None,
            np.ndarray[ls, ndim=1, mode="c"] w not None):
    return binary_op_same(u, w, 'compose')

def pow(np.ndarray[ls, ndim=1, mode="c"] u not None, r):

    cdef size_t n = u.shape[0]
    cdef np.ndarray[ls, ndim=1, mode="c"] v = np.zeros(n, dtype=LS_DTYPE)

    gdual_pow(<ls *> v.data, <ls *> u.data, <double> r, <size_t> n)

    return v
