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
import scipy as sp

from algopy import UTPM


cpdef int falling_factorial_cython(int k, int i):
    return sp.special.poch(i - k + 1, k)


cpdef np.ndarray falling_factorial_vec_cython(int k, np.ndarray i):
    return sp.special.poch(i - k + 1, k)


cpdef np.ndarray utpvec_compose_cython(np.ndarray G, np.ndarray F):
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
cpdef np.ndarray utpvec_compose_affine_cython(np.ndarray G, np.ndarray F):
    cdef:
        int        d        = G.shape[0]  # length of G, F, out
        np.ndarray out      = np.zeros(d) # return value: G o F

    if F.shape[0] <= 1:
        return G

    # no need for Horner's method, utp composition uses only the 2nd and higher coefficients, of which F has only 1 nonzero
    out = G * np.power(F[1], range(0, d))

    return out


cpdef np.ndarray utpvec_mul_cython(np.ndarray F, np.ndarray G):
    cdef:
        int d          = max(F.shape[0], G.shape[0])

    return np.convolve(F, G)[:d]


cpdef np.ndarray new_utpvec_cython(double x, int d):
    cdef:
        np.ndarray out = np.zeros(d, dtype=np.double)

    out[0] = x
    if d > 1:
        out[1] = 1

    return out


cpdef np.ndarray utpvec_deriv_cython(np.ndarray x, int k):
    cdef:
        int d = len(x)
        np.ndarray fact = falling_factorial_vec_cython(k, np.arange(d))
        np.ndarray out = np.array(x * fact, copy=True)

    # shift by k places
    out = out[k:]

    return out


cpdef double utpvecpgf_mean_cython(np.ndarray F):
    return F[1]


cpdef double utpvecpgf_var_cython(np.ndarray F):
    return (2 * F[2]) - np.power(F[1], 2) + F[1]


cpdef np.ndarray utpvec_exp_cython(np.ndarray F):
    cdef:
        int d = F.shape[0]
        np.ndarray out = np.empty_like(F)
        np.ndarray Ftilde = F[1:].copy()

    out[0] = np.exp(F[0])
    # for i in range(1, d):
    #     Ftilde[i - 1] *= i
    Ftilde *= range(1, d) # equivalent to previous two lines
    for i in range(1, d):
        out[i] = np.sum(out[:i][::-1]*Ftilde[:i], axis=0) / i

    return out


cpdef np.ndarray utpvec_log_cython(np.ndarray F):
    cdef:
        int d = F.shape[0]
        np.ndarray out = np.empty_like(F)

    out[0] = np.log(F[0])

    for i in range(1, d):
        out[i] = (F[i]*i - np.sum(F[1:i][::-1]*out[1:i],axis=0))
        out[i] /= F[0]
    for i in range(1, d):
        out[i] /= i

    return out


cpdef np.ndarray utpvec_pow_cython(np.ndarray F, double k):
    return utpvec_exp_cython(k * utpvec_log_cython(F))