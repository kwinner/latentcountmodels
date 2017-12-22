import numpy as np
import gdual as gd
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

def logarithmic_pgf(s, theta):
    p = theta[0]
    return gd.log(1 - (p * s)) / gd.log(1 - p)

# PGF for geometric with support 0, 1, 2, ...
def geometric_pgf(s, theta):
    p = theta[0]
    return p / (1 - ((1 - p) * s))

# PGF for geometric with support 1, 2, ...
def geometric2_pgf(s, theta):
    p = theta[0]
    return (p * s) / (1 - ((1 - p) * s))

def forward(y,
            arrival_pgf,
            theta_arrival,
            branch_pgf,
            theta_branch,
            theta_observ,
            GDual=gd.LSGDual,
            d = 1):

    K = len(y) # K = length of chain/number of observations

    Alpha = [None] * K # Alpha = list of gdual objects for each alpha message
    
    def lift_A(s, k, q_k):
        # recursively compute alpha messages
        # k = observation indices, used to index into theta objects
        # q_k = length of gduals for index k (varies due to observation)

        # base case, alpha = 1
        if k < 0:
            alpha = GDual(1.0, q_k)
            return alpha

        # unroll to recurse to the next layer of lift_A
        u_du = GDual( s * (1 - theta_observ[k]), q_k + y[k])

        F = branch_pgf(u_du, theta_branch[k - 1])

        s_prev = F.as_real()[0]
        
        # recurse
        beta = lift_A(s_prev,
                      k - 1,
                      q_k + y[k])
        
        beta = beta.compose(F)

        # construct the arrival pgf, then mix with beta
        beta *= arrival_pgf(u_du, theta_arrival[k])

        s_ds = GDual(s, q_k)

        # observe
        alpha = beta.deriv(y[k])
        alpha = alpha.compose_affine(s_ds * (1 - theta_observ[k]))
        
        # UTP for (s * rho)^{y_k} (the conditioning correction)
        alpha *= pow(s_ds * theta_observ[k], y[k])

        # divide by y[k]! (in log space)
        alpha /= GDual.const(gammaln(y[k] + 1), q_k, as_log=True)
        
        Alpha[k] = alpha
        return alpha

    lift_A(1.0, K - 1, d)

    return Alpha

if __name__ == "__main__":

    y     = np.array([2, 5, 3])
    lmbda = np.array([   10. ,  0.  , 0.  ])
    delta = np.array([ 1.0 ,  1.0 , 1.0 ])
    rho   = np.array([ 0.25,  0.25, 0.25])
    
    Alpha = forward(y,
                    poisson_pgf,
                    lmbda,
                    bernoulli_pgf,
                    delta,
                    rho,
                    GDual=gd.LSGDual,
                    d = 1)

    lik = Alpha[-1].as_real()[0]

    print lik

    
