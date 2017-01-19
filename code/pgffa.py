import time
import numpy as np
from scipy.misc import factorial

"""
NOTE: Breaks if y > 100
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
    f = np.poly1d(1)

    for k in xrange(K):
        a, b = arrivals(a, b, lmbda[k])
        a, f = evidence(a, f, y[k], rho[k])
        b, f = normalize(b, f)
        if k < K - 1:
            a, b, f = survivors(a, b, f, delta[k])

    return a, b, f

def arrivals(a, b, lmbda):
    a_prime = a + lmbda
    b_prime = b - lmbda
    return a_prime, b_prime

def evidence(a, f, y, rho):
    a_prime = a * (1 - rho)
    
    # Compute sum_{l=0}^y of f^(l)(s) / (l!(y-l)!) * a^(y-l)
    g = np.poly1d(0)
    df = f
    for l in xrange(y + 1):
        #log_c = (y-l) * np.log(a) - np.sum(np.log(np.arange(1, l+1))) - np.sum(np.log(np.arange(1, y-l+1)))
        log_c = (y-l) * np.log(a) - np.log(factorial(l)) - np.log(factorial(y-l))
        g = g + df * np.exp(log_c)
        df = np.polyder(df)

    g = np.polyval(g, np.poly1d([1-rho, 0])) # g(s(1 - rho))
    g = g * rho**y * np.poly1d([1] + [0]*y)

    return a_prime, g

def survivors(a, b, f, delta):
    a_prime = a * delta
    b_prime = b + a*(1 - delta)
    g = np.polyval(f, np.poly1d([delta, 1 - delta])) # f(delta * s + 1 - delta)

    return a_prime, b_prime, g

def normalize(b, f):
    c = np.max(f)
    return b + np.log(c), f/c

def likelihood(a, b, f):
    return np.polyval(f, 1) * np.exp(a + b)

y = np.array([6,8,10,6,8,10,6,8,10])
lmbda = np.array([16, 20, 24, 16, 20, 24, 16, 20, 24])
delta = np.array([0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4])
rho = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
print likelihood(*pgf_forward(lmbda, rho, delta, y)), 2.30542691e-29

# N-mixture
y = np.array([112, 128, 129, 124, 118, 123, 121, 125, 126])
K = len(y)
lmbda = [250] + [0] * (K - 1)
delta = [1] * (K - 1)
rho = [0.5] * K
print likelihood(*pgf_forward(lmbda, rho, delta, y))

"""
# Runtime test
reps = 100
t_start = time.clock()
for i in xrange(reps):
    pgf_forward(lmbda, rho, delta, y)
total_time = time.clock() - t_start
print total_time / reps
"""
