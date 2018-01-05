import scipy
import numpy as np
import gdual as gd
from gdual import log, exp
from gdual import GDual, LSGDual, diff
from scipy.special import gammaln

def poisson_pgf(s, lmbda):
    return exp(lmbda * (s - 1))

def poisson_pgf_grad(s, lmbda):
    y  = exp(lmbda * (s - 1))
    dy = y * lmbda
    return y, dy
    
def bernoulli_pgf(s, p):
    return (1 - p) + (p * s)

def bernoulli_pgf_grad(s, p):
    y  = (1 - p) + (p * s)
    dy = p
    return y, dy

def forward_grad(y,
                 immigration_pgf_grad,
                 theta_immigration,
                 offspring_pgf_grad,
                 theta_offspring,
                 rho,
                 GDualType=gd.LSGDual,
                 d = 0):

    K = len(y) # K = length of chain/number of observations

    def A_grad(s, k):
        """Return A(s, k) and d/ds A(s, k)"""
        if k < 0:
            return 1.0, 0.0

        ## TODO: derivatives wrt params
        ## rho[k]
        
        # Forward prop
        a = s * (1 - rho[k])
        
        Gamma_k_grad = lambda u: Gamma_grad( u, k )

        ## TODO: the diff function only works for scalar outputs
        b, db_da = diff(Gamma_k_grad, a, y[k], GDualType=GDualType)

        const = GDualType.const(y[k]*np.log(rho[k]) - gammaln(y[k] + 1),
                                as_log=True)
        s_to_y = s**y[k]
        c = const * s_to_y
        f = b*c

        # Back prop
        df = 1.0
        dc = df * b
        db = df * c
        da = db * db_da
        ds = da*(1-rho[k]) + dc * const * y[k] * (s_to_y / s)
        
        return f, ds
    
    def Gamma_grad(u, k):
        """Return Gamma(u, k) and d/du Gamma(u, k)"""
        
        ## TODO: derivatives wrt params
        ##   theta_offspring[k]
        ##   theta_immigration[k]

        F_grad = lambda u:   offspring_pgf_grad(u, theta_offspring[k])
        G_grad = lambda u: immigration_pgf_grad(u, theta_immigration[k])

        # Forward prop
        a, da_du = F_grad(u)
        b, db_da = A_grad(a, k-1)
        c, dc_du = G_grad(u)

        f =  b * c  # output value

        # Back prop
        df = 1.0
        dc = df * b
        db = df * c
        da = db * db_da
        du = da * da_du + dc * dc_du
        
        return f, du

    A_final = lambda s: A_grad(s, K-1)
    
    if d == 0:
        alpha, dalpha_ds = A_final( 1.0 )
    else:
        alpha, dalpha_ds = A_final( GDualType(1.0, d) )
        
    logZ = log( alpha )
    dlogZ_dalpha = 1/alpha
    dlogZ_ds = dlogZ_dalpha * dalpha_ds
    
    return alpha, logZ, dlogZ_ds

if __name__ == "__main__":

    y     = np.array([2, 5, 3])
    lmbda = np.array([ 10 ,  0.  , 0.  ])
    delta = np.array([ 1.0 ,  1.0 , 1.0 ])
    rho   = np.array([ 0.25,  0.25, 0.25])
    
    alpha, logZ, dlogZ_ds = forward_grad(y,
                                         poisson_pgf_grad,
                                         lmbda,
                                         bernoulli_pgf_grad,
                                         delta,
                                         rho,
                                         GDualType=gd.LSGDual,
                                         d = 2)
    
    
    print logZ, dlogZ_ds
