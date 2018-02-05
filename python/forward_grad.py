import scipy
import numpy as np
import gdual as gd
from gdual import log, exp
from gdual import GDual, LSGDual, diff_grad
from scipy.special import gammaln

from collections import Iterable

# Simple version of recursive map. All Iterables converted to lists
def recursive_map(f, theta):
    if isinstance(theta, Iterable):
        return [recursive_map(f, t) for t in theta]
    else:
        return f(theta)

# This recursively iterates through a pair of lists
def recursive_map_pair(f, theta, need_grad):
    if isinstance(theta, Iterable):
        return [recursive_map_pair(f, t, ng) for t, ng in zip(theta, need_grad)]
    else:
        return f(theta, need_grad)

# Flatten a 2d list into 1d
def flatten(l):
    return [i for row in l for i in row]

def get_grad(theta):
    f = lambda t: t.grad.get(0) if t.need_grad else None
    theta_grad = recursive_map(f, theta)
    return theta_grad

class Parameter():
    def __init__(self, val, need_grad=False, grad=None):
        self.val = val
        self.need_grad = need_grad
        self.grad = grad

    def __str__(self):
        return "Parameter: " + self.val.__str__()

    def __repr__(self):
        return "Parameter: " + self.val.__repr__()

    @classmethod
    def wrap(cls, theta, need_grad=False):
        if need_grad is True or need_grad is False:
            f = lambda val: cls(val, need_grad=need_grad)
            return recursive_map(f, theta)
        elif isinstance(need_grad, Iterable):
            f = lambda theta_val, need_grad_val: cls(theta_val, need_grad=need_grad_val)
            return recursive_map_pair(f, theta, need_grad)
        else:
            raise(ValueError('need_grad must be Boolean or a list'))
    
def poisson_pgf_grad(s, theta):
    # Forward
    lmbda = theta[0]
    y  = exp(lmbda.val * (s - 1))

    # Backward
    def backprop_dy_ds(dy):
        ds = dy * (y * lmbda.val)
        if lmbda.need_grad:
            lmbda.grad = dy * (y * (s - 1))
        return ds
    
    return y, backprop_dy_ds

def bernoulli_pgf_grad(s, theta):
    p = theta[0]
    
    # Forward
    y  = (1 - p.val) + (p.val * s)

    # Backward
    def backprop_dy_ds(dy):
        ds = dy * p.val
        if p.need_grad:
            p.grad = dy * (-1 + s)
        return ds
        
    return y, backprop_dy_ds

def geometric_pgf_grad(s, theta):
    p = theta[0]

    # Forward
    y = p.val / (1 - ((1 - p.val) * s))

    # Backward
    def backprop_dy_ds(dy):
        tmp = ((p.val - 1) * s + 1)**2
        ds = dy * -((p.val - 1) * p.val) / tmp # df/ds = df/dy * dy/ds
        if p.need_grad:
            p.grad = dy * (1 - s) / tmp # df/dp = df/dy * dy/dp
        return ds

    return y, backprop_dy_ds

def negbin_pgf_grad(s, theta):
    r, p = theta

    # Forward
    a = 1 - s*(1 - p.val)
    b = p.val / a
    y = b**r.val         # y = (p / (1 - s(1-p)))**r

    # Backward
    def backprop_dy_ds(dy):

        db = dy * r.val * (y / b)
        da = db / (a*a) * -p.val
        ds = da * (p.val - 1)
        
        if p.need_grad:
            p.grad = db/a + da*s

        if r.need_grad:
            r.grad = dy * y * log(b)
        
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

        Gamma_k_params = flatten(theta_immigration[:k+1]) + \
                         flatten(theta_offspring[:k])     + \
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
        
        F_grad = lambda u:   offspring_pgf_grad(u, theta_offspring[k-1])
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
        alpha, alpha_backprop = A_grad( 1.0, K-1 )
    else:
        alpha, alpha_backprop = A_grad( GDualType(1.0, d), K-1 )
        
    logZ = alpha.get(0, as_log=True)

    # Backprop with 1/alpha = dlogZ/dalpha
    #  side effect: computes gradient of parameters
    alpha_backprop(1/alpha)

    theta_immigration_grad = get_grad(theta_immigration)
    theta_offspring_grad = get_grad(theta_offspring)
    rho_grad = get_grad(rho)
        
    return logZ, theta_immigration_grad, theta_offspring_grad, rho_grad

if __name__ == "__main__":
    
    y     = 300*np.array([2, 5, 3])
    lmbda = np.array([ 10 ,  0.  , 0.  ]).reshape(-1,1)
    delta = np.array([ 1.0 ,  1.0]).reshape(-1,1)
    rho   = np.array([ 0.25,  0.25, 0.25])

    ## Test 1: simple usage
    
    lmbda = Parameter.wrap(lmbda, need_grad=True)
    delta = Parameter.wrap(delta, need_grad=True)
    rho   = Parameter.wrap(rho  , need_grad=False)

    logZ, lmbda_grad, delta_grad, rho_grad = forward_grad(y,
                                                          poisson_pgf_grad,
                                                          lmbda,
                                                          bernoulli_pgf_grad,
                                                          delta,
                                                          rho,
                                                          GDualType=gd.LSGDual,
                                                          d = 0)
    
    print("logZ: ", logZ)
    print("lmbda_grad: ", lmbda_grad)
    print("delta_grad: ", delta_grad)
    print("rho_grad: ", rho_grad)
    

    ## Test 1: more advanced usage. mimic how this function
    ## will be called from MLE

    T = np.arange(3)
    theta = np.array([1.0])
    
    hyperparam2param = lambda x, T: np.concatenate((x, np.zeros(len(T) - 1))).reshape((-1, 1))
    need_grad = lambda T: [[True]] + [[False]] * (len(T)-1)
    backprop = lambda dtheta: dtheta[0][0]

    lmbda = hyperparam2param(theta, T)
    grad_mask = need_grad(T)

    print("lmbda, grad_mask: ", lmbda, grad_mask)

    lmbda = Parameter.wrap(lmbda, need_grad=grad_mask)
    
    
    logZ, lmbda_grad, delta_grad, rho_grad = forward_grad(y,
                                                          poisson_pgf_grad,
                                                          lmbda,
                                                          bernoulli_pgf_grad,
                                                          delta,
                                                          rho,
                                                          GDualType=gd.LSGDual,
                                                          d = 0)

    hyperparam_grad = backprop(lmbda_grad)
    
    print("logZ: ", logZ)
    print("lmbda_grad: ", lmbda_grad)
    print("delta_grad: ", delta_grad)
    print("rho_grad: ", rho_grad)

    print("hyperparam_grad: ", hyperparam_grad)
