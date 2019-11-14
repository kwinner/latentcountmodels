import scipy
import numpy as np
import gdual as gd
from gdual import GDual, LSGDual, diff, diff_grad, exp, log
from scipy.special import gammaln

def poisson_pgf(s, theta):
    lmbda = theta[0]
    return np.exp(lmbda * (s - 1))

def bernoulli_pgf(s, theta):
    p = theta[0]
    return (1 - p) + (p * s)

def binomial_pgf(s, theta):
    n, p = theta[:]
    return (((1 - p) + (p * s)) ** n)

def negbin_pgf(s, theta):
    r, p = theta[:]

    return ((p / (1 - ((1 - p) * s))) ** r)

def logarithmic_pgf(s, theta):
    p = theta[0]
    return gd.log(1 - (p * s)) / np.log(1 - p)

# PGF for geometric with support 0, 1, 2, ...
def geometric_pgf(s, theta):
    p = theta[0]
    return p / (1 - ((1 - p) * s))

# PGF for geometric with support 1, 2, ...
def geometric2_pgf(s, theta):
    p = theta[0]
    return (p * s) / (1 - ((1 - p) * s))

def mixed_offspring_pgf(s, theta):
    lmbda, p = theta[:]

    return poisson_pgf(s, [lmbda]) * bernoulli_pgf(s, [p])

def forward(y,
            immigration_pgf,
            theta_immigration,
            offspring_pgf,
            theta_offspring,
            rho,
            GDualType=gd.LSGDual,
            d = 0):

    K = len(y) # K = length of chain/number of observations

    Alpha = [None] * K # Alpha = list of gdual objects for each alpha message
    
    def A(s, k):
        if k < 0:
            return 1.0

        Gamma_k = lambda u_k: Gamma( u_k, k )
        
        const = s**y[k]
        const *= GDualType.const(y[k]*np.log(rho[k]) - gammaln(y[k] + 1), as_log=True)
        alpha = const * diff(Gamma_k, s*(1 - rho[k]), y[k], GDualType=GDualType )
        alpha.trunc_neg_coefs()
        Alpha[k] = alpha
        return alpha
        
    def Gamma(u, k):
        F = lambda u:   offspring_pgf(u, theta_offspring[k-1])
        G = lambda u: immigration_pgf(u, theta_immigration[k])
        return A(F(u), k-1) * G(u)

    A_final = lambda s: A(s, K-1)
    
    if d == 0:
        alpha = A_final( 1.0 )
    else:
        alpha = A_final( GDualType(1.0, d) )
    
    logZ = alpha.get(0, as_log=True)

    # temporary code for plotting marginals vs apgffwd
    # A_final = lambda s: Gamma(s, K - 1)
    # print('ERROR: BAD CODE DETECTED')
    # g = A_final(GDualType(0.0, 1000))
    # g /= GDualType.const(logZ, as_log=True)
    # return g

    def marginals(k):
        a  = A_final( GDualType(0.0, k) )
        a /= GDualType.const(logZ, as_log=True)
        return a
        
    return logZ, alpha, marginals

if __name__ == "__main__":
    import time

    # y     = 10*np.array([2, 5])
    # lmbda = 5000*np.array([ 20 ,  10.]).reshape(-1,1)
    # delta = np.array([ 0.3]).reshape(-1,1)
    # rho   = np.array([ 0.25,  0.25])

    y = 300*np.array([1,2,3,1,3])
    lmbda = 300*np.array([2.5, 6, 6, 6, 6]).reshape(-1,1)
    delta = np.array([0.5, 0.5, 0.5, 0.5]).reshape(-1,1)
    rho = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # y = np.array([2])
    # lmbda = np.array([5.]).reshape(-1, 1)
    # delta = np.array([0.3]).reshape(-1, 1)
    # rho = np.array([0.25])

    rt = 0
    for i in range(0,10):
        start = time.process_time()
        logZ, alpha, marginals = forward(y,
                                         poisson_pgf,
                                         lmbda,
                                         poisson_pgf,
                                         delta,
                                         rho,
                                         GDualType=gd.LSGDual,
                                         d = 0)
        rt = rt + time.process_time() - start
    
    print(logZ)
    print(alpha)
    print(rt)
