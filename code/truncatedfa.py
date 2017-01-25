import time
import numpy as np
from scipy import stats, signal

"""
NOTE: If p(n_k|n_k-1) is very unlikely, convolution will return negative
probabilities, which will be truncated to zero
"""

def truncated_forward(arrival_dist, arrival_params, branching_fn,
                      branching_params, rho, y, n_max=40):
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
    n_max += 1 # so the maximum support is inclusive of n_max
    K = len(y)
    alpha = np.zeros((n_max, K))
    z = np.zeros(K)

    # k = 0
    pi = arrival_vector(arrival_dist, arrival_params[0], n_max) # initial state distn
    evidence_k = evidence_vector(rho[0], y[0], n_max)
    alpha_k, z_k = normalize(evidence_k * pi)
    alpha[:, 0] = alpha_k
    z[0] = z_k

    for k in xrange(1, K):
        #trans_k = trans_matrix(arrival_dist, arrival_params[k], delta[k - 1], n_max)
        trans_k = trans_matrix(arrival_dist, arrival_params[k], branching_fn, branching_params[k - 1], n_max)
        evidence_k = evidence_vector(rho[k], y[k], n_max)
        alpha_k, z_k = normalize(evidence_k * trans_k.T.dot(alpha_k))
        alpha[:, k] = alpha_k
        z[k] = z_k

    return alpha, z

def normalize(v):
    z = np.sum(v)
    alpha = v / z
    return alpha, z

def trans_matrix(arrival_dist, arrival_params_k, branching_fn,
                 branching_params_k, n_max):
    """
    Output: n_max x n_max matrix of transition probabilities
    """
    arrival = arrival_vector(arrival_dist, arrival_params_k, n_max)
    branching = branching_fn(n_max, *branching_params_k)
    
    trans_k = signal.fftconvolve(arrival.reshape(1, -1), branching)[:, :n_max]
    neg_probs = trans_k < 0
    if np.any(neg_probs):
        print 'Warning: truncating negative transition probabilities to zero'
        trans_k[np.where(neg_probs)] = 0

    # True distn of Poisson arrival + Poisson branching, for comparison
    #n_k = np.arange(n_max).reshape((-1, 1))
    #trans_k = stats.poisson.pmf(np.arange(n_max), n_k * branching_params_k[0] + arrival_params_k[0])
    
    return trans_k

def arrival_vector(dist, params, n_max):
    return dist.pmf(np.arange(n_max), *params)

def evidence_vector(rho_k, y_k, n_max):
    return stats.binom.pmf(y_k, np.arange(n_max), rho_k)

def likelihood(z, log=True):
    """
    Output: log likelihood if log is set to True, likelihood otherwise
    """
    ll = np.sum(np.log(z))
    return ll if log else np.exp(ll)

def poisson_branching(n_max, gamma_k):
    n_k = np.arange(n_max).reshape((-1, 1))
    return stats.poisson.pmf(np.arange(n_max), n_k * gamma_k)

def binomial_branching(n_max, delta_k):
    n_k = np.arange(n_max).reshape((-1, 1))
    return stats.binom.pmf(np.arange(n_max), n_k, delta_k)

if __name__ == "__main__":
    # Poisson arrival, binomial branching
    y = np.array([6,8,10,6,8,10,6,8,10])
    lmbda = np.array([16, 20, 24, 16, 20, 24, 16, 20, 24]).reshape((-1, 1))
    delta = np.array([0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4]).reshape((-1, 1))
    rho = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    n_max = 10

    _, z = truncated_forward(stats.poisson, lmbda, binomial_branching,
                             delta, rho, y, n_max)
    ll = likelihood(z)
    assert abs(ll - (-85.4080)) < 1e-5, 'Error too big'
    print ll

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

    """
    # Runtime test
    reps = 100
    t_start = time.clock()
    for i in xrange(reps):
        truncated_forward(stats.nbinom, arrival_params, rho, delta, y, n_max)
    total_time = time.clock() - t_start
    print total_time / reps
    """