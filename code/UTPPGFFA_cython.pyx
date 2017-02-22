# distutils: language = c++
# distutils: libraries = stdc++
# distutils: library_dirs = /usr/local/lib
# distutils: extra_compile_args = -O3 -w -std=c++0x
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

from cython cimport boundscheck, cdivision, nonecheck, wraparound

from UTPPGF_cython import *
from distributions_cython import *

import numpy as np
cimport numpy as np

#import scipy as sp
from libc.math cimport lgamma

cpdef tuple utppgffa_cython(int[::1] y,
                            str arrival_pgf_cython_name,
                            np.ndarray[np.double_t, ndim=2] theta_arrival,
                            str branch_pgf_cython_name,
                            np.ndarray[np.double_t, ndim=2] theta_branch,
                            np.ndarray[np.double_t, ndim=1] theta_observ,
                            int d=1):
    cdef:
        int K = len(y)
        #list Alpha = [None] * K
        double[::1] logZ = np.zeros(K)
        np.ndarray[np.double_t, ndim=1] alpha

    alpha = lift_A(new_utpvec_cython(1., 1),
           K - 1,
           d,
           y,
           arrival_pgf_cython_name,
           theta_arrival,
           branch_pgf_cython_name,
           theta_branch,
           theta_observ,
           #Alpha,
           logZ)

    return alpha, logZ


cpdef np.ndarray[np.double_t, ndim=1] lift_A(np.ndarray[np.double_t, ndim=1] s,
                                             int k,
                                             int d_k,
                                             int[::1] y,
                                             str arrival_pgf_cython_name,
                                             np.ndarray[np.double_t, ndim=2] theta_arrival,
                                             str branch_pgf_cython_name,
                                             np.ndarray[np.double_t, ndim=2] theta_branch,
                                             np.ndarray[np.double_t, ndim=1] theta_observ,
                                             #list Alpha,
                                             double[::1] logZ):
    # define the return value alpha, a utppgf
    cdef:
        np.ndarray[np.double_t, ndim=1] alpha

    # base case for k = -1, a constant utppgf = [1]
    if k < 0:
        alpha = np.zeros(d_k)
        alpha[0] = 1.

        #Alpha[k] = alpha
        return alpha

    # define all the utppgfs we'll need, plus the normalizing constant Z
    cdef:
        np.ndarray[np.double_t, ndim=1] u
        np.ndarray[np.double_t, ndim=1] F
        np.ndarray[np.double_t, ndim=1] G
        np.ndarray[np.double_t, ndim=1] s_prev
        np.ndarray[np.double_t, ndim=1] u_du
        np.ndarray[np.double_t, ndim=1] beta
        np.ndarray[np.double_t, ndim=1] s_ds
        double Z

    # scalar mul
    u = s * (1 - theta_observ[k])

    # lifted branch GF @ u
    u_du = new_utpvec_cython(u, d_k + y[k])
    if   branch_pgf_cython_name == 'poisson':
        F = poisson_utppgf_cython(u_du, theta_branch[k-1,:])
    elif branch_pgf_cython_name == 'bernoulli':
        F = bernoulli_utppgf_cython(u_du, theta_branch[k-1,:])
    elif branch_pgf_cython_name == 'binomial':
        F = binomial_utppgf_cython(u_du, theta_branch[k-1,:])
    elif branch_pgf_cython_name == 'negbin':
        F = negbin_utppgf_cython(u_du, theta_branch[k-1,:])
    elif branch_pgf_cython_name == 'logarithmic':
        F = logarithmic_utppgf_cython(u_du, theta_branch[k-1,:])
    elif branch_pgf_cython_name == 'geometric':
        F = geometric_utppgf_cython(u_du, theta_branch[k-1,:])
    elif branch_pgf_cython_name == 'geometric2':
        F = geometric2_utppgf_cython(u_du, theta_branch[k-1,:])

    s_prev = new_utpvec_cython(F[0], 1)
    # recurse
    beta = utpvec_compose_cython(lift_A(s_prev,
                                        k - 1,
                                        d_k + y[k],
                                        y,
                                        arrival_pgf_cython_name,
                                        theta_arrival,
                                        branch_pgf_cython_name,
                                        theta_branch,
                                        theta_observ,
                                        #Alpha,
                                        logZ),
                                 F)

    # lifted arrival GF @ u
    if   arrival_pgf_cython_name == 'poisson':
        G = poisson_utppgf_cython(u_du, theta_arrival[k,:])
    elif arrival_pgf_cython_name == 'bernoulli':
        G = bernoulli_utppgf_cython(u_du, theta_arrival[k,:])
    elif arrival_pgf_cython_name == 'binomial':
        G = binomial_utppgf_cython(u_du, theta_arrival[k,:])
    elif arrival_pgf_cython_name == 'negbin':
        G = negbin_utppgf_cython(u_du, theta_arrival[k,:])
    elif arrival_pgf_cython_name == 'logarithmic':
        G = logarithmic_utppgf_cython(u_du, theta_arrival[k,:])
    elif arrival_pgf_cython_name == 'geometric':
        G = geometric_utppgf_cython(u_du, theta_arrival[k,:])
    elif arrival_pgf_cython_name == 'geometric2':
        G = geometric2_utppgf_cython(u_du, theta_arrival[k,:])

    # utp mul
    beta = utpvec_mul_cython(beta, G)

    # observe
    s_ds = new_utpvec_cython(s, d_k)
    # deriv
    alpha = utpvec_deriv_cython(beta, y[k])
    # correct dual
    alpha = utpvec_compose_affine_cython(alpha, (s_ds * (1 - theta_observ[k])))
    # mul by (s*rho) ^ y_k
    alpha = utpvec_mul_cython(alpha, utpvec_pow_cython(s_ds * theta_observ[k], y[k]))

    # divide by y_k! (incorporate into Z immediately)
    #logZ[k] = -sp.special.gammaln(y[k] + 1)
    logZ[k] = -lgamma(y[k] + 1)

    # normalize the alpha messages
    if np.any(alpha):
        Z = np.max(np.abs(alpha))
        logZ[k] += np.log(Z)
        alpha /= Z

    #Alpha[k] = alpha

    return alpha