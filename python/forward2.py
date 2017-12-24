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
        print "A: ", s, k

        if k < 0:
            return 1.0
        
        Gamma_k = lambda u_k: Gamma( u_k, k )        
        const = (s * rho[k])**y[k] / GDualType.const(gammaln(y[k] + 1), as_log=True)
        alpha = const * diff(Gamma_k, s*(1 - rho[k]), y[k] )
        Alpha[k] = alpha        
        return alpha
        
    def Gamma(u, k):
        print "Gamma: ", u, k
        F = lambda u:   offspring_pgf(u, theta_offspring[k])
        G = lambda u: immigration_pgf(u, theta_immigration[k])
        return A(F(u), k-1) * G(u)

    A_final = lambda s: A(s, K-1)
    
    diff(A_final, 0.0, d)

    return Alpha

if __name__ == "__main__":

    y     = np.array([2, 5, 3])
    lmbda = np.array([ 10 ,  10.  , 10.  ])
    delta = np.array([ 1.0 ,  1.0 , 1.0 ])
    rho   = np.array([ 0.25,  0.25, 0.25])
    
    Alpha = forward(y,
                    poisson_pgf,
                    lmbda,
                    bernoulli_pgf,
                    delta,
                    rho,
                    GDualType=gd.LSGDual,
                    d = 20)

    lik = Alpha[-1].as_real()[0]

    print lik

    
