import numpy as np
from scipy.special import gammaln
import gdual as gd

import forward as fwd

# get parameters of a poisson by matching moments with some dist'n p
def MM_Poiss(mean_p):
    lambda_q = mean_p
    theta_q = [lambda_q]

    return theta_q

# get parameters of a nb by matching moments with some dist'n p
def MM_NB(mean_p, var_p):
    assert var_p > mean_p, 'Error in MM_NB: cannot approximate a distn with mean >= var'
    r_q = (mean_p ** 2) / (var_p - mean_p)
    p_q = 1 - (mean_p / var_p)
    theta_q = [r_q, p_q]

    return theta_q

# get parameters of a binomial by matching moments with some dist'n p
def MM_Binom(mean_p, var_p):
    assert mean_p > var_p, 'Error in MM_Binom: cannot approximate a distn with mean <= var'

    n_q = mean_p / (1 - (var_p / mean_p))

    ###
    # note: for a binomial, n_q must be integral. Further examination is needed to determine the best/correct way,
    # but for now, I'm just rounding it to the nearest integer, then setting $p$ correspondingly
    ###

    n_q = np.round(n_q)
    p_q = mean_p / n_q
    theta_q = [n_q, p_q]

    return theta_q

def APGF(F_p, logZ = None, GDualType = gd.LSGDual, return_type = ['name', 'param', 'lambda']):
    # compute the normalization constant for F if it wasn't provided
    if logZ is None:
        logZ = F_p(GDualType.const(1)).get(0, as_log=True)

    # renormalize the distribution before computing moments (this is now done later)
    # F_star = lambda s: F_p(s) / Z

    # construct M_p, the MGF of p, from F_p
    M_p = lambda t: F_p(np.exp(t))

    # use the MGF to compute the first k moments of p
    k = 2
    t0_gd = GDualType(0, q=k)

    # extract the moments (note the renormalization by logZ)
    # note: this assumes the mean and variance of F_p are >= 0, which they should be for a count distribution
    # but we go ahead and force that for stability reasons
    log_moments_p = M_p(t0_gd)
    log_moments_p.trunc_neg_coefs()
    log_moments_p = log_moments_p.get(range(1,k+1), as_log=True) + gammaln(np.arange(2, k + 2)) - logZ

    log_mean_p = log_moments_p[0]

    # variance is a little harder to do in log space, because none of the logsumexp/logaddexp can do subtraction
    # this is just manually doing logSUBexp between m1 and m0^2
    log_m1 = log_moments_p[1]
    var_Z = np.maximum(2 * log_mean_p, log_m1)
    log_var_p = np.log(np.exp(log_m1 - var_Z) - np.exp( 2 * log_mean_p - var_Z)) + var_Z

    mean_p = np.exp(log_mean_p)
    var_p  = np.exp(log_var_p)

    assert np.isfinite(mean_p) and np.isfinite(var_p)

    # assert not np.isposinf(log_mean_p) and not np.isnan(log_mean_p)
    # assert not np.isposinf(log_var_p) and not np.isnan(log_var_p)

    # if mean and var are "close" (to numerical stability) treat them as equal
    if np.abs(mean_p - var_p) < 1e-6:
        distn = 'poiss'
        theta = MM_Poiss(mean_p)
        lmbda = lambda s, theta = theta, logZ = logZ: fwd.poisson_pgf(s, theta) / np.exp(logZ)
    elif mean_p < var_p:
        distn = 'nb'
        theta = MM_NB(mean_p, var_p)
        lmbda = lambda s, theta = theta, logZ = logZ: fwd.negbin_pgf(s, theta) / np.exp(logZ)
    elif mean_p > var_p:
        distn = 'binom'
        theta = MM_Binom(mean_p, var_p)
        lmbda = lambda s, theta = theta, logZ = logZ: fwd.binomial_pgf(s, theta) / np.exp(logZ)
    else:
        raise Exception('Unable to approximate PGF.')

    return_vals = {'name': distn, 'param': theta, 'lambda': lmbda}
    return([return_vals[key] for key in return_type])

def APGF_Forward_symb(y,
                      immigration_pgf,
                      theta_immigration,
                      offspring_pgf,
                      theta_offspring,
                      rho,
                      GDualType=gd.LSGDual,
                      d=0):
    def Gamma_k(u_k, k):
        s_kminus1 = offspring_pgf(u_k, theta_offspring[k-1])
        return Alpha_k(s_kminus1, k - 1) * immigration_pgf(u_k, theta_immigration[k])

    def Alpha_k(s_k, k):
        # base case, Alpha_0 = 1.0
        if k < 0:
            return GDualType.const(1.0, q = s_k.order())

        u_k = s_k * (1 - rho[k])
        Gamma = lambda u, k=k: Gamma_k(u, k)

        # select a parametric distn to approximate Gamma
        gamma_logZ = Gamma(GDualType.const(1)).get(0, as_log=True)
        apgf_res = APGF(Gamma, gamma_logZ, GDualType=GDualType, return_type=['name', 'param'])
        apgf_distn = apgf_res[0]
        apgf_theta = apgf_res[1]

        # compute the parts of A_k that don't depend on Gamma
        val = s_k ** y[k]
        val *= GDualType.const(y[k] * np.log(rho[k]) - gammaln(y[k] + 1), as_log=True)

        # add the parts that depend on Gamma
        if apgf_distn == 'poiss':
            lmbda = apgf_theta[0]
            # const parts
            val *= GDualType.const(y[k] * np.log(lmbda) - lmbda, as_log = True)
            # gdual parts
            val *= np.exp(s_k * lmbda * (1 - rho[k]))
        elif apgf_distn == 'nb':
            r = apgf_theta[0]
            p = apgf_theta[1]
            # const parts
            val *= GDualType.const(r * np.log(p) + y[k] * np.log(1 - p) + gammaln(r + y[k]) - gammaln(r), as_log = True)
            # gdual parts
            val *= (1 - (1 - rho[k]) * (1 - p) * s_k) ** (-r - y[k])
        elif apgf_distn == 'binom':
            n = apgf_theta[0]
            p = apgf_theta[1]
            # const parts
            val *= GDualType.const(y[k] * np.log(p) + gammaln(n + 1) - gammaln(n - y + 1), as_log = True)
            # gdual parts
            val *= (1 - p + p * (1 - rho[k]) * s_k) ** (n - y[k])

        val /= GDualType.const(gamma_logZ, as_log = True)

        return val

    K = len(y)
    A_K = Alpha_k(GDualType.const(1.0), K - 1)

    return A_K

if __name__ == '__main__':
    # currently np.exp(any LSGDual) throws this warning, but it can be ignored until we figure out why it happens
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in exp")

    # F = lambda s: fwd.binomial_pgf(s, [10000, 0.2])
    #
    # print(APGF(F))
    #
    # F = lambda s: fwd.poisson_pgf(s, [10000])
    #
    # print(APGF(F))
    #
    # F = lambda s: fwd.negbin_pgf(s, [10000, 0.2])
    #
    # print(APGF(F))
    #
    # F = lambda s: fwd.geometric2_pgf(s, [0.3])
    #
    # print(APGF(F))
    #
    # F = lambda s: 1/156 * fwd.poisson_pgf(s, [10000])
    #
    # print(APGF(F))

    y = np.array([1, 2, 3, 1, 3])
    lmbda = np.array([2.5, 6, 6, 6, 6]).reshape(-1, 1)
    delta = np.array([0.5, 0.5, 0.5, 0.5]).reshape(-1, 1)
    rho = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    A = APGF_Forward_symb(y,
                                     fwd.poisson_pgf,
                                     lmbda,
                                     fwd.poisson_pgf,
                                     delta,
                                     rho,
                                     GDualType=gd.LSGDual,
                                     d=0)

    print(np.log(A))

    logZ, alpha, marginals = fwd.forward(y,
                                     fwd.poisson_pgf,
                                     lmbda,
                                     fwd.poisson_pgf,
                                     delta,
                                     rho,
                                     GDualType=gd.LSGDual,
                                     d=0)

    print(logZ)