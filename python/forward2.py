import scipy
import numpy as np
import gdual as gd
from gdual import GDual, LSGDual, diff
from scipy.special import gammaln

def poisson_pgf(s, lmbda):
    return gd.exp(lmbda * (s - 1))

def bernoulli_pgf(s, p):
    return (1 - p) + (p * s)

def binomial_pgf(s, theta):
    n, p = theta[:]
    return gd.pow((1 - theta[1]) + (p * s), n)

def negbin_pgf(s, theta):
    r, p = theta[:]

    return gd.pow(p / (1 - ((1 - p) * s)), r)

def logarithmic_pgf(s, p):
    return gd.log(1 - (p * s)) / np.log(1 - p)

# PGF for geometric with support 0, 1, 2, ...
def geometric_pgf(s, theta):
    p = theta[0]
    return p / (1 - ((1 - p) * s))

# PGF for geometric with support 1, 2, ...
def geometric2_pgf(s, theta):
    p = theta[0]
    return (p * s) / (1 - ((1 - p) * s))
    
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
        F = lambda u:   offspring_pgf(u, theta_offspring[k])
        G = lambda u: immigration_pgf(u, theta_immigration[k])
        return A(F(u), k-1) * G(u)

    A_final = lambda s: A(s, K-1)
    
    if d == 0:
        alpha = A_final( 1.0 )
    else:
        alpha = A_final( GDualType(1.0, d) )
    
    logZ = alpha.get(0, as_log=True)

    def marginals(k):
        a  = A_final( GDualType(0.0, k) )
        a /= GDualType.const(logZ, as_log=True)
        return a
        
    return logZ, alpha, marginals

import cygdual

if __name__ == "__main__":

    y     = 20*np.array([2, 5, 3])
    lmbda = np.array([ 10 ,  0.  , 0.  ])
    delta = np.array([ 1.0 ,  1.0 , 1.0 ])
    rho   = np.array([ 0.25,  0.25, 0.25])

    logZ, alpha, marginals = forward(y,
                                     poisson_pgf,
                                     lmbda,
                                     bernoulli_pgf,
                                     delta,
                                     rho,
                                     GDualType=gd.GDual,
                                     d = 0)
    
    print(logZ)
