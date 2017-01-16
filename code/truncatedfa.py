import time
import numpy as np
from scipy import stats, signal

def truncated_forward(arrival_dist, arrival_params, rho, delta, y, n_max=40):
    """
    Input:
    - arrival_dist   : probability distribution object of new arrivals
                       (e.g. stats.poisson, stats.nbinom)
    - arrival_params : matrix (K x n_params) of parameters of new arrivals
                       (e.g. [[lambda_1], ..., [lambda_K]] for Poisson,
                       [[r_1, p_1], ..., [r_K, p_K]] for NB)
    - rho            : list (K) of detection probabilities
    - delta          : list (K-1) of survival probabilities
    - y              : list (K) of evidence
    - n_max          : maximum abundance at each k (inclusive)

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
        trans_k = trans_matrix(arrival_dist, arrival_params[k], delta[k - 1], n_max)
        evidence_k = evidence_vector(rho[k], y[k], n_max)
        alpha_k, z_k = normalize(evidence_k * trans_k.T.dot(alpha_k))
        alpha[:, k] = alpha_k
        z[k] = z_k

    return alpha, z

def normalize(v):
    z = np.sum(v)
    alpha = v / z
    return alpha, z

def trans_matrix(arrival_dist, arrival_params_k, delta_k, n_max):
    """
    Output: n_max x n_max matrix of transition probabilities
    """
    arrival = arrival_vector(arrival_dist, arrival_params_k, n_max)
    survival = survival_matrix(delta_k, n_max)
    return signal.fftconvolve(arrival.reshape(1, -1), survival)[:, :n_max]

def arrival_vector(dist, params, n_max):
    return dist.pmf(np.arange(n_max), *params)

def survival_matrix(delta_k, n_max):
    n_k = np.arange(n_max).reshape((-1, 1))
    return stats.binom.pmf(np.arange(n_max), n_k, delta_k)

def evidence_vector(rho_k, y_k, n_max):
    return stats.binom.pmf(y_k, np.arange(n_max), rho_k)

def likelihood(z, log=True):
    """
    Output: log likelihood if log is set to True, likelihood otherwise
    """
    ll = np.sum(np.log(z))
    return ll if log else np.exp(ll)

# Poisson arrival
y = np.array([6,8,10,6,8,10,6,8,10])
lmbda = np.array([16, 20, 24, 16, 20, 24, 16, 20, 24]).reshape((-1, 1))
delta = np.array([0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4])
rho = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
n_max = 10

alpha, z = truncated_forward(stats.poisson, lmbda, rho, delta, y, n_max)
#print alpha
print likelihood(z)

# NB arrival
r = [16, 20, 24, 16, 20, 24, 16, 20, 24]
p = [0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.8]
arrival_params = np.array([param for param in zip(r, p)])

alpha, z = truncated_forward(stats.nbinom, arrival_params, rho, delta, y, n_max)
#print alpha
print likelihood(z)

# Runtime test
reps = 100
t_start = time.clock()
for i in xrange(reps):
    truncated_forward(stats.nbinom, arrival_params, rho, delta, y, n_max)
total_time = time.clock() - t_start
print total_time / reps
