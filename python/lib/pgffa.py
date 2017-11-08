import time, warnings, sys
import numpy as np
from scipy.special import gammaln, gamma

warnings.filterwarnings('error')

"""
NOTE: Breaks if y > 100
"""

def forward(y, lmbda, delta, rho):
    """
    lmbda : new arrival rates, len(lmbda) = K
    delta : survival probabilities, len(delta) = K - 1
    rho   : detection probabilities, len(rho) = K
    y     : observations, len(y) = K
    """

    # Flatten into 1d array if lambda or delta is a 2d array
    if np.array(lmbda).ndim == 2: lmbda = lmbda.reshape(-1)
    if np.array(delta).ndim == 2: delta = delta.reshape(-1)

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

    lik = np.polyval(f, 1) * np.exp(a + b)
    if lik > 0:
        ll = np.log(lik)
    else:
        print 'Warning: taking log(0) = -inf due to 0 likelihood'
        ll = float('-inf')

    return ll, a, b, f

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
        try:
            if a > 0:
                log_c = (y-l) * np.log(a) - gammaln(l+1) - gammaln(y-l+1)
                c = np.exp(log_c)
            else:
                c = a**(y-l) / (gamma(l+1) * gamma(y-l+1))
        except RuntimeWarning as w:
            print 'Computing c:', w
            print 'y, l, a =', y, l, a
            sys.exit(0)

        try:
            g = g + df * c
        except RuntimeWarning as w:
            print 'Running sum of g:', w
            print c
            sys.exit(0)

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
    try:
        return (b + np.log(c), f/c) if c > 0 else (b, f)
    except RuntimeWarning as w:
        print 'Error in normalize:', w
        sys.exit(0)
