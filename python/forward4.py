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
    dlmbda = y * (s - 1)
    return y, dy, dlmbda
    
def bernoulli_pgf(s, p):
    return (1 - p) + (p * s)

def bernoulli_pgf_grad(s, p):
    y  = (1 - p) + (p * s)
    dy = p
    dp = -1 + s
    return y, dy, dp

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

        # Forward prop
        a = s * (1 - rho[k])
        
        Gamma_k_grad = lambda u: Gamma_grad( u, k )

        b, db_da, *db_dtheta = diff(Gamma_k_grad, a, y[k], GDualType=GDualType)
        
        const = GDualType.const(y[k]*np.log(rho[k]) - gammaln(y[k] + 1), as_log=True)
        c = const * s**y[k]
        f = b*c

        # Back prop
        df = 1.0
        dc = df * b
        db = df * c
        da = db * db_da
        ds = da*(1-rho[k]) + dc * const * y[k] * (s**(y[k]-1))

        drho = da * -s + \
               dc * (s**y[k]) * y[k] * GDualType.const( (y[k]-1)*np.log(rho[k]) - gammaln(y[k]+1), as_log=True)

        dtheta = [db * x for x in db_dtheta] + [drho]

        return [f, ds] + dtheta
    
    def Gamma_grad(u, k):
        """Return Gamma(u, k) and d/du Gamma(u, k)"""
        
        F_grad = lambda u:   offspring_pgf_grad(u, theta_offspring[k])
        G_grad = lambda u: immigration_pgf_grad(u, theta_immigration[k])

        # Forward prop
        a, da_du, *da_dtheta_offspring_k = F_grad(u)
        b, db_da, *db_dtheta_lt_k = A_grad(a, k-1)
        c, dc_du, *dc_dtheta_immigration_k = G_grad(u)

        f =  b * c  # output value

        # Back prop
        df = 1.0
        dc = df * b
        db = df * c
        da = db * db_da
        du = da * da_du + dc * dc_du

        dtheta = [db * x for x in db_dtheta_lt_k] + \
                 [da * x for x in da_dtheta_offspring_k] + \
                 [dc * x for x in dc_dtheta_immigration_k]
        
        return [f, du] + dtheta

    A_final = lambda s: A_grad(s, K-1)
    
    if d == 0:
        alpha, dalpha_ds, *dtheta = A_final( 1.0 )
    else:
        alpha, dalpha_ds, *dtheta = A_final( GDualType(1.0, d) )

    logZ = log( alpha )
    
    dlogZ_dalpha = 1/alpha
    dlogZ_ds = dlogZ_dalpha * dalpha_ds
    dlogZ_dtheta = [dlogZ_dalpha * x for x in dtheta]
    
    return logZ, dlogZ_ds, dlogZ_dtheta
    

if __name__ == "__main__":
    
    y     = 10*np.array([2, 5, 3])
    delta = np.array([ 1.0 ,  1.0 , 1.0 ])
    lmbda = np.array([ 10 ,  10.  , 10.  ])
    rho   = np.array([ 0.25,  0.25, 0.25])

    def pack_params(delta, lmbda, rho):
        '''Pack all parameters into single vector'''
        theta = np.zeros((3*len(y),))
        theta[0::3] = delta
        theta[1::3] = lmbda
        theta[2::3] = rho
        return theta
    
    def unpack_params(theta):
        '''Unpack parameters into delta, lmbda, rho'''
        delta = theta[0::3]
        lmbda = theta[1::3]
        rho   = theta[2::3]
        return delta, lmbda, rho

    theta0 = pack_params(delta, lmbda, rho)

    def unpack_single(x):
        '''Copy single parameter into theta0 and return entire vector'''
        theta = theta0.copy()
        theta[1] = x[0]
        return unpack_params(theta)
            
    def nll_grad(theta):

        delta, lmbda, rho = unpack_single(theta)
        
        (logZ, ds, dtheta) = forward_grad(y,
                                          poisson_pgf_grad,
                                          lmbda,
                                          poisson_pgf_grad,
                                          delta,
                                          rho,
                                          GDualType=gd.LSGDual,
                                          d = 0)

        nll = -logZ.get(0)
        grad = np.array([-x.get(0) for x in dtheta])
        return nll, grad[1:2]
        
    def nll(theta):
        nll, _ = nll_grad(theta)
        return nll
        
    def grad(theta):
        _, grad = nll_grad(theta)
        return grad


    optimize = True
    
    if not optimize:
        (logZ, ds, dtheta) = forward_grad(y,
                                          poisson_pgf_grad,
                                          lmbda,
                                          poisson_pgf_grad,
                                          delta,
                                          rho,
                                          GDualType=gd.LSGDual,
                                          d = 0)
    else:
        
        x0 = np.array([lmbda[0]])
        
        import time
        from scipy.optimize import check_grad, minimize
        
        #print("Gradient check");
        #print(check_grad(nll, grad, x0))
        
        start_time = time.time()
        print("Starting optimization....")
        theta = minimize(nll_grad, x0, jac=True)
        print("done in %f seconds\n" % (time.time() - start_time))
