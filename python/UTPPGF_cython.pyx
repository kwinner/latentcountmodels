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

from libc.math cimport lgamma, exp

cdef double poch(double x, double n):
    return exp(lgamma(x + n) - lgamma(x))

cdef double falling_factorial_c(double k, double i):
    return poch(i - k + 1, k)

cpdef np.ndarray[np.double_t, ndim=1] falling_factorial_vec_cython(double k,
                                                                   np.ndarray[np.double_t, ndim=1] i):
    cdef:
        int                             index
        int                             d   = i.shape[0]
        np.ndarray[np.double_t, ndim=1] out = np.empty_like(i, dtype=np.double)
    for index in range(d):
        out[index] = poch(i[index] - k + 1, k)

    return out
    #return scipy.special.poch(i - k + 1, k)

cpdef np.ndarray[np.double_t, ndim=1] utpvec_compose_cython(np.ndarray[np.double_t, ndim=1] G,
                                                          np.ndarray[np.double_t, ndim=1] F):
    cdef:
        np.double_t                     g_scalar = G[0]                         # value of G^{(0)}
        np.double_t                     f_scalar = F[0]                         # value of F^{(0)}
        int                             d        = G.shape[0]                   # length of G, F, out
        int                             i                                       # Horner's method index
        np.ndarray[np.double_t, ndim=1] out      = np.zeros(d, dtype=np.double) # return value: G o F

    # temporarily zero out first element of G and F
    # this lets us do the convolution "in place" as it were
    # the entries are restored before the function exits
    G[0] = 0
    F[0] = 0

    # Horner's method truncated to d
    out[0] = G[d - 1]
    for i in range(d - 2, -1, -1):
        out = np.convolve(out, F)[:d]
        #out = scipy.signal.fftconvolve(out, F)[:d]
        #out = np.fft.ifft(np.multiply(np.fft.fft(out, 2*d - 1), np.fft.fft(F, 2*d - 1)))[:d]
        out[0] += G[i]

    out[0] = g_scalar

    # restore the first entries of G and F
    G[0] = g_scalar
    F[0] = f_scalar

    return out


# new utps typically have two nonzero coefficients
# composing them can be done in linear time using only the second nonzero coefficient
cpdef np.ndarray[np.double_t, ndim=1] utpvec_compose_affine_cython(np.ndarray[np.double_t, ndim=1] G,
                                                                 np.ndarray[np.double_t, ndim=1] F):
    cdef:
        int                           d        = G.shape[0]                   # length of G, F, out
        np.ndarray[np.double_t, ndim=1] out      = np.zeros(d, dtype=np.double) # return value: G o F

    if F.shape[0] <= 1:
        return G

    # no need for Horner's method, utp composition uses only the 2nd and higher coefficients, of which F has only 1 nonzero
    out = G * np.power(F[1], range(0, d))

    return out


cpdef np.ndarray[np.double_t, ndim=1] utpvec_mul_cython(np.ndarray[np.double_t, ndim=1] F,
                                                      np.ndarray[np.double_t, ndim=1] G):
    cdef:
        int d          = max(F.shape[0], G.shape[0])

    return np.convolve(F, G)[:d]
    #return scipy.signal.fftconvolve(F, G)[:d]
    #return np.fft.ifft(np.multiply(np.fft.fft(F, 2*d - 1), np.fft.fft(G, 2*d - 1)))[:d].astype(np.double)


cpdef np.ndarray[np.double_t, ndim=1] new_utpvec_cython(np.double_t x, int d):
    cdef:
        np.ndarray[np.double_t, ndim=1] out = np.zeros(d, dtype=np.double)

    out[0] = x
    if d > 1:
        out[1] = 1

    return out


cpdef np.ndarray[np.double_t, ndim=1] utpvec_deriv_cython(np.ndarray[np.double_t, ndim=1] x,
                                                        int k):
    cdef:
        int d = len(x)
        np.ndarray[np.double_t, ndim=1] fact = falling_factorial_vec_cython(k, np.arange(d, dtype=np.double))
        np.ndarray[np.double_t, ndim=1] out  = np.array(x * fact, copy=True)

    # shift by k places
    out = out[k:]

    return out


cpdef np.double_t utpvecpgf_mean_cython(np.ndarray[np.double_t, ndim=1] F):
    return F[1]


cpdef np.double_t utpvecpgf_var_cython(np.ndarray[np.double_t, ndim=1] F):
    return (2 * F[2]) - np.power(F[1], 2) + F[1]


cpdef np.ndarray[np.double_t, ndim=1] utpvec_exp_cython(np.ndarray[np.double_t, ndim=1] F):
    cdef:
        int d = F.shape[0]
        np.ndarray[np.double_t, ndim=1] out    = np.empty_like(F, dtype=np.double)
        np.ndarray[np.double_t, ndim=1] Ftilde = F[1:].copy()

    #print "exp"
    #print "in:"
    #print F

    out[0] = np.exp(F[0])
    # for i in range(1, d):
    #     Ftilde[i - 1] *= i
    Ftilde *= range(1, d) # equivalent to previous two lines
    for i in range(1, d):
        out[i] = np.sum(out[:i][::-1]*Ftilde[:i], axis=0) / i

    #print "out:"
    #print out
    #print ""

    return out


cpdef np.ndarray[np.double_t, ndim=1] utpvec_log_cython(np.ndarray[np.double_t, ndim=1] F):
    cdef:
        int d = F.shape[0]
        np.ndarray[np.double_t, ndim=1] out = np.empty_like(F, dtype=np.double)

    #print "log"
    #print "in:"
    #print F

    out[0] = np.log(F[0])

    for i in range(1, d):
        out[i] = (F[i]*i - np.sum(F[1:i][::-1]*out[1:i],axis=0))
        out[i] /= F[0]
    for i in range(1, d):
        out[i] /= i

    #print "out:"
    #print out
    #print ""

    return out


cpdef np.ndarray[np.double_t, ndim=1] utpvec_pow_cython(np.ndarray[np.double_t, ndim=1] F,
                                                      np.double_t k):
    return utpvec_exp_cython(k * utpvec_log_cython(F))


cpdef np.ndarray[np.double_t, ndim=1] utpvec_reciprocal_cython(np.ndarray[np.double_t, ndim=1] F):
    cdef:
        int d = F.shape[0]
        np.ndarray[np.double_t, ndim=1] out = np.zeros_like(F, dtype=np.double)

    out[0] = 1. / F[0]
    for i in range(1, d):
        out[i] = 1. / F[0] * (-np.sum(out[:i] * F[i:0:-1], axis = 0))

    return out