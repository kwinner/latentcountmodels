import scipy
import numpy as np
import gdual as gd
from gdual import log, exp
from gdual import GDual, LSGDual, diff_grad
from scipy.special import gammaln

class Parameter():
    def __init__(self, val, need_grad=False, grad=None):
        self.val = val
        self.need_grad = need_grad
        self.grad = grad

def poisson_pgf_grad(s, lmbda):
    # Forward
    y  = exp(lmbda.val * (s - 1))

    # Backward
    def backprop_dy_ds(dy):
        ds = dy * (y * lmbda.val)
        if lmbda.need_grad:
            lmbda.grad = dy * (y * (s - 1))
        return ds
    
    return y, backprop_dy_ds

def bernoulli_pgf_grad(s, p):

    # Forward
    y  = (1 - p.val) + (p.val * s)

    # Backward
    def backprop_dy_ds(dy):
        ds = dy * p.val
        if p.need_grad:
            p.grad = dy * (-1 + s)
        return ds
        
    return y, backprop_dy_ds
            
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
            def backprop_dalpha_ds(dalpha):
                return 0.0
            return 1.0, backprop_dalpha_ds
        
        # Forward pass
        a = s * (1 - rho[k].val)

        Gamma_k_grad = lambda u: Gamma_grad( u, k )

        Gamma_k_params = theta_immigration[:k+1] + \
                         theta_offspring[:k+1] + \
                         rho[:k]
        
        b, backprop_db_da = diff_grad(Gamma_k_grad, a,
                                      Gamma_k_params, y[k],
                                      GDualType=GDualType)
        
        const = GDualType.const(y[k]*np.log(rho[k].val) - \
                                gammaln(y[k] + 1),
                                as_log=True)
        
        c = const * s**y[k]
        alpha = b*c

        # Backward
        def backprop_dalpha_ds(dalpha):
            dc = dalpha * b
            db = dalpha * c
            da = backprop_db_da(db)
            ds = da * (1-rho[k].val) + \
                 dc * const * y[k] * (s**(y[k]-1))

            if rho[k].need_grad:
                rho[k].grad = da * -s + \
                              dc * (s**y[k]) * y[k] * \
                              GDualType.const( (y[k]-1)*np.log(rho[k].val) \
                                               - gammaln(y[k]+1),
                                               as_log=True)
            return ds

        return alpha, backprop_dalpha_ds
    
    def Gamma_grad(u, k):
        """Return Gamma(u, k) and d/du Gamma(u, k)"""
        
        F_grad = lambda u:   offspring_pgf_grad(u, theta_offspring[k])
        G_grad = lambda u: immigration_pgf_grad(u, theta_immigration[k])

        # Forward prop
        a, backprop_da_du = F_grad(u)
        b, backprop_db_da = A_grad(a, k-1)
        c, backprop_dc_du = G_grad(u)

        gamma =  b * c  # output value
        
        # Back prop
        def backprop_dgamma_du(dgamma):
            dc = dgamma * b
            db = dgamma * c
            da = backprop_db_da(db)
            du = backprop_da_du(da) + backprop_dc_du(dc)
            return du

        return gamma, backprop_dgamma_du

    if d == 0:
        alpha, backprop = A_grad( 1.0, K-1 )
    else:
        alpha, backprop = A_grad( GDualType(1.0, d), K-1 )
        
    logZ = alpha.get(0, as_log=True)

    # Backprop with 1/alpha = dlogZ/dalpha
    #  side effect: computes gradient of parameters
    backprop(1/alpha)
    
    return logZ


def unpack(theta, wrap=True):
    '''Convert numpy array into parameter vectors'''
    if wrap:
        theta = [Parameter(t, need_grad=True) for t in theta]

    k = int(len(theta) / 3)
    delta = theta[:k]
    lmbda = theta[k:2*k]
    rho   = theta[2*k:]
    return delta, lmbda, rho

def pack(delta, lmbda, rho):
    return np.concatenate((delta, lmbda, rho))

def recover_grad(params):
    return np.array([t.grad.get(0) for t in params])

if __name__ == "__main__":
    
    y     = np.array([2, 5, 3])
    delta = np.array([ 1.0 ,  1.0 , 1.0 ])
    lmbda = np.array([ 10 ,  0.  , 0.  ])
    rho   = np.array([ 0.25,  0.25, 0.25])

    def nll_grad(theta, y):
        
        delta, lmbda, rho = unpack(theta)
        
        logZ = forward_grad(y,
                            poisson_pgf_grad,
                            lmbda,
                            bernoulli_pgf_grad,
                            delta,
                            rho,
                            GDualType=gd.LSGDual,
                            d = 0)

        nll = -logZ
        grad = -recover_grad(delta + lmbda + rho)
        return nll, grad
        
    def nll(theta, y):
        nll, _ = nll_grad(theta, y)
        return nll
        
    def grad(theta, y):
        _, grad = nll_grad(theta, y)
        return grad


    #test = "simple"
    test = "grad_check"
    #test = "optimize"

    if test == "simple":

        theta0 = pack(delta, lmbda, rho)
        (nll, grad) = nll_grad(theta0, y)
        print("nll: ", nll)
        print("grad: ", grad)
        
    elif test == "grad_check":
        
        from scipy.optimize import check_grad

        for i in range(60):            
            f      = lambda theta: nll(theta, i*y)
            f_grad = lambda theta: grad(theta, i*y)
            theta0 = pack(delta, lmbda, rho)
            val = check_grad(f, f_grad, theta0)
            print(i, "gradient check: ", val)

    elif test == "optimize":

        theta0 = pack(delta, lmbda, rho)

        from scipy.optimize import minimize
        print("Starting optimization....")
        theta = minimize(nll_grad, theta0, jac=True)
        print("done")

    else:
        raise(ValueError('unknown test'))
