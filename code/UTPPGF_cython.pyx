# distutils: language = c++
# distutils: libraries = stdc++
# distutils: library_dirs = /usr/local/lib
# distutils: extra_compile_args = -O3 -w -std=c++0x
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

from cython cimport boundscheck, cdivision, nonecheck, wraparound

import numpy as np
cimport numpy as np

from algopy import UTPM

cpdef np.ndarray utp_compose_cython(np.ndarray G, np.ndarray F):
    cdef:
        double     g_scalar = G[0]        # value of G^{(0)}
        double     f_scalar = F[0]        # value of F^{(0)}
        int        d        = G.shape[0]  # length of G, F, out
        int        i                      # Horner's method index
        np.ndarray out      = np.zeros(d) # return value: G o F

    # temporarily zero out first element of G and F
    # this lets us do the convolution "in place" as it were
    # the entries are restored before the function exits
    G[0] = 0
    F[0] = 0

    # Horner's method truncated to d
    out[0] = G[d - 1]
    for i in range(d - 2, -1, -1):
        out = np.convolve(out, F)[:d]
        out[0] += G[i]

    out[0] = g_scalar

    # restore the first entries of G and F
    G[0] = g_scalar
    F[0] = f_scalar

    return out

# new utps typically have two nonzero coefficients
# composing them can be done in linear time using only the second nonzero coefficient
cpdef np.ndarray utp_compose_affine_cython(np.ndarray G, np.ndarray F):
    cdef:
        int        d        = G.shape[0]  # length of G, F, out
        np.ndarray out      = np.zeros(d) # return value: G o F

    # no need for Horner's method, utp composition uses only the 2nd and higher coefficients, of which F has only 1 nonzero
    out = G * np.power(F[1], range(0, d))

    return out

cpdef np.ndarray utp_mul_cython(np.ndarray F, np.ndarray G):
    cdef:
        int d          = max(F.shape[0], G.shape[0])

    return np.convolve(F, G)[:d]