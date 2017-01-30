import time, warnings, sys
import numpy as np
from scipy.special import gammaln
from scipy.misc import logsumexp
from UTPPGF_util import *

np.seterr(divide='ignore')
warnings.filterwarnings('error')

"""
NOTE: Breaks if y > ~1500
"""

def pgf_forward(lmbda, rho, delta, y):
    """
    lmbda : new arrival rates, len(lmbda) = K
    rho   : detection probabilities, len(rho) = K
    delta : survival probabilities, len(delta) = K - 1
    y     : observations, len(y) = K
    """
    K = len(y)
    a = 0
    b = 0
    f = np.array([1])

    for k in xrange(K):
        a, b = arrivals(a, b, lmbda[k])
        a, b, f = evidence(a, b, f, y[k], rho[k])
        b, f, _ = normalize(b, f)
        if k < K - 1:
            a, b, f = survivors(a, b, f, delta[k])

    return a, b, f

def arrivals(a, b, lmbda):
    a_prime = a + lmbda
    b_prime = b - lmbda
    return a_prime, b_prime

def evidence(a, b, f, y, rho):
    a_prime = a * (1 - rho)
    b_prime = b

    max_deriv = min(y + 1, len(f))

    # Compute sum_{l=0}^y of f^(l)(s) * a^(y-l) / (l!(y-l)!)
    log_g = np.full((max_deriv, len(f)), -np.inf)
    df = f
    
    for l in xrange(max_deriv):
        # Normalize
        b_prime, df, max_df = normalize(b_prime, df)
        log_g[:l+1, :] = log_g[:l+1, :] - max_df

        # a^(y-l) / (l!(y-l)!)
        log_c = (y-l) * np.log(a) - gammaln(l+1) - gammaln(y-l+1)
        
        # f^(l)(s) * log_c
        log_g[l, :len(df)] = np.log(df) + log_c

        # f^(l+1)(s)
        if l < y: df = poly_der(df)

    # sum_{l=0}^y of log_g
    log_g = logsumexp(log_g, axis=0)

    # Normalize
    b_prime, log_g, _ = normalize(b_prime, log_g, log=True)

    # g(s(1 - rho))
    h = np.arange(1, len(f)+1) * np.log(1-rho)
    log_g = log_g + h

    g = np.exp(log_g)                         # scale back to non-log space
    g = np.append(np.zeros(y), g) # g * s**y
    b_prime = b_prime + y * np.log(rho)       # g * rho**y

    return a_prime, b_prime, g

def survivors(a, b, f, delta):
    a_prime = a * delta
    b_prime = b + a*(1 - delta)
    g = compose_poly_horner_special(f, [1 - delta, delta]) # f(delta * s + 1 - delta)

    return a_prime, b_prime, g

def normalize(b, f, log=False):
    c = np.max(f)
    if log and c > -np.inf:
        b_prime = b + c
        g = f - c
        log_c = c
    elif not log and c > 0:
        b_prime = b + np.log(c)
        g = f / c
        log_c = np.log(c)
    else:
        b_prime, g = b, f
        log_c = 0

    return b_prime, g, log_c

def likelihood(a, b, f, log=True):
    ll = np.log(np.sum(f)) + a + b
    return ll if log else np.exp(ll)

if __name__ == "__main__":
    
    # y = np.array([ 785, 1712, 1683, 1524, 1303, 1489, 1454, 1890])
    # lmbda = np.array([1000, 1500, 1320, 680, 880, 900, 1100, 1280])
    # delta = np.array([0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6])
    # rho = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    # a, b, f = pgf_forward(lmbda, rho, delta, y)
    # print likelihood(a, b, f, False), 1.96541052172e-17

    y = np.array([6,8,10,6,8,10,6,8,10])
    lmbda = np.array([16, 20, 24, 16, 20, 24, 16, 20, 24])
    delta = np.array([0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4])
    rho = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    a, b, f = pgf_forward(lmbda, rho, delta, y)
    print likelihood(a, b, f, False), 2.30542691e-29
    """"
    # N-mixture
    y = np.array([112, 128, 129, 124, 118, 123, 121, 125, 126])
    K = len(y)
    lmbda = [250] + [0] * (K - 1)
    delta = [1] * (K - 1)
    rho = [0.5] * K
    a, b, f = pgf_forward(lmbda, rho, delta, y)
    print likelihood(a, b, f, False), 1.11963529571e-13
    
    # Runtime test
    reps = 10
    t_start = time.clock()
    for i in xrange(reps):
        pgf_forward(lmbda, rho, delta, y)
    total_time = time.clock() - t_start
    print total_time / reps
    """
