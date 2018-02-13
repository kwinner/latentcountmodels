# distutils: language = c++
# distutils: libraries = stdc++
# distutils: library_dirs = /usr/local/lib
# distutils: extra_compile_args = -O3 -w -std=c++0x
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

from cython cimport boundscheck, cdivision, nonecheck, wraparound

from UTPPGF_cython import *

import numpy as np
cimport numpy as np

cpdef np.ndarray[np.double_t, ndim=1] poisson_utppgf_cython(np.ndarray[np.double_t, ndim=1] s,
                                                            np.ndarray[np.double_t, ndim=1] theta):
    cdef:
        np.double_t                     lmbda = theta[0]
        np.ndarray[np.double_t, ndim=1] out   = s.copy()
    out[0] -= 1
    return utpvec_exp_cython(lmbda * out)


cpdef np.ndarray[np.double_t, ndim=1] bernoulli_utppgf_cython(np.ndarray[np.double_t, ndim=1] s,
                                                              np.ndarray[np.double_t, ndim=1] theta):
    cdef:
        np.double_t                     p = theta[0]
        np.ndarray[np.double_t, ndim=1] out   = s.copy()
    out *= p
    out[0] += (1 - p)
    return out


cpdef np.ndarray[np.double_t, ndim=1] binomial_utppgf_cython(np.ndarray[np.double_t, ndim=1] s,
                                                             np.ndarray[np.double_t, ndim=1] theta):
    cdef:
        np.double_t                     n   = theta[0]
        np.double_t                     p   = theta[1]
        np.ndarray[np.double_t, ndim=1] out = s.copy()
    out *= p
    out[0] += (1 - p)
    return utpvec_pow_cython(out, n)


cpdef np.ndarray[np.double_t, ndim=1] negbin_utppgf_cython(np.ndarray[np.double_t, ndim=1] s,
                                                           np.ndarray[np.double_t, ndim=1] theta):
    cdef:
        np.double_t                     r   = theta[0]
        np.double_t                     p   = theta[1]
        np.ndarray[np.double_t, ndim=1] out = s.copy()

    out *= (p-1)
    out[0] += 1
    out = utpvec_reciprocal_cython(out)
    out = p * out
    out = utpvec_pow_cython(out, r)

    return out


cpdef np.ndarray[np.double_t, ndim=1] logarithmic_utppgf_cython(np.ndarray[np.double_t, ndim=1] s,
                                                                np.ndarray[np.double_t, ndim=1] theta):
    cdef:
        np.double_t                     p   = theta[0]
        np.ndarray[np.double_t, ndim=1] out = s.copy()
    out *= -p
    out[0] += 1
    out = utpvec_log_cython(out)
    return (out / np.log(1 - p))


# PGF for geometric with support 0, 1, 2, ...
cpdef np.ndarray[np.double_t, ndim=1] geometric_utppgf_cython(np.ndarray[np.double_t, ndim=1] s,
                                                              np.ndarray[np.double_t, ndim=1] theta):
    cdef:
        np.double_t                     p   = theta[0]
        np.ndarray[np.double_t, ndim=1] out = s.copy()
    out *= (p - 1)
    out[0] += 1
    return (p * utpvec_reciprocal_cython(out))


# PGF for geometric with support 1, 2, ...
# note: likely to be /significantly/ slower than the previous definition
cpdef np.ndarray[np.double_t, ndim=1] geometric2_utppgf_cython(np.ndarray[np.double_t, ndim=1] s,
                                                               np.ndarray[np.double_t, ndim=1] theta):
    cdef:
        np.double_t                     p   = theta[0]
        np.ndarray[np.double_t, ndim=1] out = s.copy()
    out *= (p - 1)
    out[0] += 1
    out = utpvec_reciprocal_cython(out)
    return utpvec_mul_cython(p * s, out)