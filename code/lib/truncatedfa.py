import numpy as np
from scipy import stats, signal

import warnings, sys
#warnings.filterwarnings('error')

"""
NOTE: If p(n_k|n_k-1) is very unlikely, convolution will return negative
probabilities, which will be truncated to zero
"""

def forward(y, arrival_params, branching_params, rho, arrival_pmf,
            branching_pmf, n_max=100, print_warnings=False):
    """
    Input:
    - arrival_dist    : probability distribution object of new arrivals
                        (e.g. stats.poisson, stats.nbinom)
    - arrival_params  : matrix (K x n_params) of parameters of new arrivals
                        (e.g. [[lambda_1], ..., [lambda_K]] for Poisson,
                        [[r_1, p_1], ..., [r_K, p_K]] for NB)
    - branching_fn    : function that takes n_max as the first argument and
                        branching_params as the remaining arguments, and returns a
                        distribution matrix of the branching process
    - branching_params: matrix (K x n_params) of arguments to branching_fn
                        (e.g. [[gamma_1], ..., [gamma_K]] for Poisson,
                        [[delta_1], ..., [delta_K]] for binomial)
    - rho             : list (K) of detection probabilities
    - y               : list (K) of evidence
    - n_max           : maximum abundance at each k (inclusive)

    Output:
    - alpha : matrix (n_max x K) of forward messages
    - z     : list (K) of normalizing constants
    """
    global warn
    warn = print_warnings

    n_max += 1 # so the maximum support is inclusive of n_max
    K = len(y)
    alpha = np.zeros((n_max, K))
    z = np.zeros(K)

    # k = 0
    pi = arrival_vector(arrival_pmf, arrival_params[0], n_max) # initial state distn
    evidence_k = evidence_vector(rho[0], y[0], n_max)
    alpha_k, z_k = normalize(evidence_k * pi)
    alpha[:, 0] = alpha_k
    z[0] = z_k

    for k in xrange(1, K):
        trans_k = trans_matrix(arrival_pmf, arrival_params[k], branching_pmf, branching_params[k - 1], n_max)
        evidence_k = evidence_vector(rho[k], y[k], n_max)
        alpha_k, z_k = normalize(evidence_k * trans_k.T.dot(alpha_k))
        alpha[:, k] = alpha_k
        z[k] = z_k

    if not np.any(np.isnan(z)) and np.all(z > 0):
        ll = np.sum(np.log(z))
    else:
        if warn: print 'Warning: taking log(0) = -inf due to 0 likelihood'
        ll = float('-inf')

    return ll, alpha, z

def normalize(v):
    z = np.sum(v)
    if z > 0:
        alpha = v / z
    else:
        alpha = [0] * len(v)
    return alpha, z

def trans_matrix(arrival_pmf, arrival_params_k, branching_pmf,
                 branching_params_k, n_max):
    """
    Output: n_max x n_max matrix of transition probabilities
    """
    arrival = arrival_vector(arrival_pmf, arrival_params_k, n_max)
    branching = branching_pmf(n_max, *branching_params_k)
    
    trans_k = signal.fftconvolve(arrival.reshape(1, -1), branching)[:, :n_max]
    neg_probs = trans_k < 0
    if np.any(neg_probs):
        if warn: print 'Warning: truncating negative transition probabilities to zero'
        trans_k[np.where(neg_probs)] = 0

    # True distn of Poisson arrival + Poisson branching, for comparison
    #n_k = np.arange(n_max).reshape((-1, 1))
    #trans_k = stats.poisson.pmf(np.arange(n_max), n_k * branching_params_k[0] + arrival_params_k[0])
    
    return trans_k

def arrival_vector(pmf, params, n_max):
    return pmf(np.arange(n_max), *params)

def evidence_vector(rho_k, y_k, n_max):
    return stats.binom.pmf(y_k, np.arange(n_max), rho_k)
