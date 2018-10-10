import time
import numpy as np
from scipy import stats, signal
from scipy.special import logsumexp
import warnings

"""
NOTE: If p(n_k|n_k-1) is very unlikely, convolution will return negative
probabilities, which will be truncated to zero
"""

def truncated_forward(arrival_dist, arrival_params, branching_fn,
                      branching_params, rho, y, n_max=None, silent=False, conv_method='auto'):
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
    if n_max is None: n_max = np.max(y) * 5

    n_max += 1 # so the maximum support is inclusive of n_max
    K = len(y)
    alpha = np.zeros((n_max, K))
    z = np.zeros(K)

    # k = 0
    pi = arrival_vector(arrival_dist, arrival_params[0], n_max) # initial state distn
    evidence_k = evidence_vector(rho[0], y[0], n_max)
    alpha_k = evidence_k + pi
    alpha[:, 0] = alpha_k

    for k in range(1, K):
        #trans_k = trans_matrix(arrival_dist, arrival_params[k], delta[k - 1], n_max)
        trans_k = trans_matrix(arrival_dist, arrival_params[k], branching_fn, branching_params[k-1], n_max, silent, conv_method=conv_method)
        if np.any(np.isnan(trans_k)):
            trans_k = trans_matrix(arrival_dist, arrival_params[k], branching_fn, branching_params[k - 1], n_max,
                                   silent, conv_method=conv_method)
        evidence_k = evidence_vector(rho[k], y[k], n_max)
        alpha[:, k] = logsumexp(alpha[:, k-1, None] + trans_k, axis=0) + evidence_k

    logz = logsumexp(alpha[:,-1])
    return alpha, logz

def normalize(v):
    z = np.sum(v)
    alpha = v / z
    return alpha, z


def trans_matrix(arrival_dist, arrival_params_k, branching_fn,
                 branching_params_k, n_max, silent, conv_method='auto'):
    """
    Output: n_max x n_max matrix of transition probabilities
    """
    arrival = arrival_vector(arrival_dist, arrival_params_k, n_max)
    branching = branching_fn(n_max, *branching_params_k)

    shift_arrival = arrival.max()
    shift_branching = branching.max(axis=1).reshape(-1,1)
    
    arrival   -= shift_arrival
    branching -= shift_branching
    
    # Convolve each row of branching matrix w/ arrival distribution
    trans_k = signal.convolve(np.exp(branching),
                              np.exp(arrival.reshape(1, -1)),
                              method=conv_method)[:n_max,:n_max]

    neg_probs = trans_k < 0
    if not silent and np.any(neg_probs):
        print('Warning: truncating negative transition probabilities to zero')
    trans_k[np.where(neg_probs)] = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trans_k_log = np.log(trans_k) + shift_arrival + shift_branching

    # True distn of Poisson arrival + Poisson branching, for comparison
    #n_k = np.arange(n_max).reshape((-1, 1))
    #trans_k = stats.poisson.pmf(np.arange(n_max), n_k * branching_params_k[0] + arrival_params_k[0])
    
    return trans_k_log

def arrival_vector(dist, params, n_max):
    return dist.logpmf(np.arange(n_max), *params)

def evidence_vector(rho_k, y_k, n_max):
    return stats.binom.logpmf(y_k, np.arange(n_max), rho_k)
    #return np.zeros(n_max)

def likelihood(z, log=True):
    """
    Output: log likelihood if log is set to True, likelihood otherwise
    """
    ll = np.sum(np.log(z))
    return ll if log else np.exp(ll)

def poisson_branching(n_max, gamma_k):
    from_count = np.arange(n_max).reshape((-1,1))  # column vector
    to_count   = np.arange(n_max).reshape((1,-1))  # row vector    
    return stats.poisson.logpmf(to_count, from_count * gamma_k)

def binomial_branching(n_max, delta_k):
    n_k = np.arange(n_max).reshape((-1, 1))
    return stats.binom.logpmf(np.arange(n_max), n_k, delta_k)

def nbinom_branching(n_max, p_k):
    n_k = np.arange(n_max).reshape((-1, 1))
    return stats.nbinom.logpmf(np.arange(n_max), n_k, p_k)


if __name__ == "__main__":
    # Poisson arrival, binomial branching

    # y = np.array([6,8,10,6,8,10,6,8,10])
    # lmbda = np.array([16, 20, 24, 16, 20, 24, 16, 20, 24]).reshape((-1, 1))
    # delta = np.array([0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4]).reshape((-1, 1))
    # rho = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])

    y     = np.array([2, 5, 3])
    lmbda = np.array([ 10 ,  0.  , 0.  ]).reshape((-1,1))
    delta = 50.0*np.array([ 1.0 ,  1.0 , 1.0 ]).reshape((-1,1))
    rho   = np.array([ 0.25,  0.25, 0.25])
    n_max = 250

    alpha, logz = truncated_forward(stats.poisson, lmbda, poisson_branching,
                                    delta, rho, y, n_max, silent=False, conv_method='fft')

    print(logz)
    
    #lik = likelihood(z, False)
    #print(lik, 2.30542690568e-29)

    # # Poisson arrival, Poisson branching
    # _, z = truncated_forward(stats.poisson, lmbda, poisson_branching,
    #                          delta, rho, y, n_max)
    # lik = likelihood(z, False)
    # print(lik, 1.78037453027e-27)
    """
    # NB arrival, binomial branching
    r = [16, 20, 24, 16, 20, 24, 16, 20, 24]
    p = [0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.8]
    arrival_params = np.array([param for param in zip(r, p)])

    _, z = truncated_forward(stats.nbinom, arrival_params, binomial_branching,
                             delta, rho, y, n_max)
    ll = likelihood(z)
    assert abs(ll - (-64.5405)) < 1e-5, 'Error too big'
    print ll

    # Poisson arrival, poisson branching
    y = np.array([  5,  11,  16,  30,  44,  73, 104, 165, 230])
    n_max = 300
    _, z = truncated_forward(stats.poisson, lmbda, poisson_branching,
                             delta, rho, y, n_max)
    print likelihood(z)

    # Runtime test
    reps = 100
    t_start = time.clock()
    for i in xrange(reps):
        truncated_forward(stats.nbinom, arrival_params, rho, delta, y, n_max)
    total_time = time.clock() - t_start
    print total_time / reps
    """
