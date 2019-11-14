import numpy as np
from scipy.special import gammaln

import gdual as gd
import forward as fwd

# construct a poisson PGF by matching moments with some dist'n p
# if provided, renormalize the PGF w/ Z
def APGF_Poiss(mean_p, Z=1.0, report_distn=False):
    lambda_q = mean_p
    theta_q = [lambda_q]

    if report_distn:
        return lambda s, Z=Z, theta=theta_q: Z * fwd.poisson_pgf(s, theta), theta_q
    else:
        return lambda s, Z=Z, theta=theta_q: Z * fwd.poisson_pgf(s, theta)


# construct a NB PGF by matching moments with some dist'n p
# if provided, renormalize the PGF w/ Z
def APGF_NB(mean_p, var_p, Z=1.0, report_distn=False):
    assert var_p > mean_p, 'Error in APGF_NB: cannot approximate a distn with mean >= var'
    r_q = (mean_p ** 2) / (var_p - mean_p)
    # p_q = 1 - (mean_p / var_p)
    p_q = mean_p / var_p
    theta_q = [r_q, p_q]

    if report_distn:
        return lambda s, Z=Z, theta=theta_q: Z * fwd.negbin_pgf(s, theta), theta_q
    else:
        return lambda s, Z=Z, theta=theta_q: Z * fwd.negbin_pgf(s, theta)


# construct a binom PGF by matching moments with some dist'n p
# if provided, renormalize the PGF w/ Z
def APGF_Binom(mean_p, var_p, Z=1.0, report_distn=False):
    assert mean_p > var_p, 'Error in APGF_Binom: cannot approximate a distn with mean <= var'

    n_q = mean_p / (1 - (var_p / mean_p))

    ###
    # note: for a binomial, n_q must be integral. Further examination is needed to determine the best/correct way,
    # but for now, I'm just rounding it to the nearest integer, then setting $p$ correspondingly
    ###

    n_q = np.round(n_q)
    p_q = mean_p / n_q
    theta_q = [n_q, p_q]
    if report_distn:
        return lambda s, Z=Z, theta=theta_q: Z * fwd.binomial_pgf(s, theta), theta_q
    else:
        return lambda s, Z=Z, theta=theta_q: Z * fwd.binomial_pgf(s, theta)


def APGF(F_p, report_distn = False, GDualType = gd.LSGDual):
    # compute the normalization constant for F
    Z = F_p(1)

    # if isinstance(Z, gd.GDual) or isinstance(Z, gd.LSGDual):
    #     Z = Z.get(0, as_log = False)

    # renormalize the distribution before computing moments
    F_star = lambda s: F_p(s) / Z

    # construct M_p, the MGF of p, from F_star
    M_p = lambda t: F_star(np.exp(t))

    # use the MGF to compute the first k moments of p
    k = 2
    t0_gd = GDualType(0, q=k)
    moments_p = M_p(t0_gd)
    moments_p = np.exp(moments_p.unwrap_coefs(moments_p.coefs, as_log=True)[1:] + gammaln(np.arange(2, k + 2)))

    mean_p = moments_p[0]
    var_p = moments_p[1] - (moments_p[0] ** 2)

    # if mean_p == var_p:
    # if mean_p/var_p - 1.0 < 1e-10: # weaker test to account for some stability errors
    # if mean_p >= var_p - 1e-4: # new test, can only use NB if p is overdispersed!
    #     return APGF_Poiss(mean_p, Z)
    # else:
    #     return APGF_NB(mean_p, var_p, Z)

    if np.abs(mean_p - var_p) < 1e-4:
    # if True:
        distn = 'poiss'
        if report_distn:
            Fhat, theta = APGF_Poiss(mean_p, Z, report_distn)
        else:
            Fhat = APGF_Poiss(mean_p, Z, report_distn)
    elif mean_p < var_p:
        distn = 'nb'
        if report_distn:
            Fhat, theta = APGF_NB(mean_p, var_p, Z, report_distn)
        else:
            Fhat = APGF_NB(mean_p, var_p, Z, report_distn)
    elif mean_p > var_p:
        distn = 'binom'
        if report_distn:
            Fhat, theta = APGF_Binom(mean_p, var_p, Z, report_distn)
        else:
            Fhat = APGF_Binom(mean_p, var_p, Z, report_distn)
    else:
        distn = 'poiss'
        if report_distn:
            Fhat, theta = APGF_Poiss(mean_p, Z, report_distn)
        else:
            Fhat = APGF_Poiss(mean_p, Z, report_distn)

    if report_distn:
        return Fhat, distn, theta
    else:
        return Fhat


# GDualType = gd.LSGDual

def APGF_Forward(y,
                 immigration_pgf,
                 theta_immigration,
                 offspring_pgf,
                 theta_offspring,
                 rho,
                 GDualType=gd.LSGDual,
                 d=0):
    def Gamma_k(u_k, Alpha_PGF, k):
        # F(.) = offspring_pgf(u_k, theta_offspring[k-1])
        # G(.) = immigration_pgf(u_k, theta_immigration[k])
        #         print(offspring_pgf(u_k, theta_offspring[k-1]))
        res = Alpha_PGF(offspring_pgf(u_k, theta_offspring[k - 1])) * immigration_pgf(u_k, theta_immigration[k])
        return res

    def Alpha_k(s_k, Gamma_PGF, k):
        # s_k needs to be a dual number of order > y[k] in order to compute derivatives correctly below
        # if s_k passed in was a scalar or too low order, extend it
        # if np.isscalar(s_k):
        #     s_k = GDualType.const(s_k, q = y[k] + 1)
        # elif isinstance(s_k, gd.GDualBase):
        #     if s_k.order() <= y[k]:
        #         s_k = GDualType(s_k, q = y[k] + 1)

        if np.isscalar(s_k):
            const = GDualType.const(y[k] * (np.log(s_k) + np.log(rho[k])) - gammaln(y[k] + 1), as_log = True)
        elif isinstance(s_k, gd.GDualBase):
            const = s_k ** y[k]
            const *= GDualType.const(y[k] * np.log(rho[k]) - gammaln(y[k] + 1), as_log = True)
        res = const * gd.diff(Gamma_PGF, s_k * (1 - rho[k]), y[k], GDualType=GDualType)

        res.trunc_neg_coefs()

        return res

    K = len(y)

    Gamma_PGFs = [None] * K
    Gamma_APGF_names = [None] * K
    Alpha_PGFs = [None] * (K + 1)
    # Z = [None] * K
    Alpha_PGFs[0] = lambda s_k: s_k.__class__.const(1.0, q = s_k.order()) if isinstance(s_k, gd.GDualBase) else 1.0

    for i in range(K):
        Gamma = lambda u_k, k=i: Gamma_k(u_k, Alpha_PGFs[k], k)
        Gamma_PGFs[i], Gamma_APGF_names[i], _ = APGF(Gamma, report_distn=True, GDualType=GDualType)
        # res = APGF(Gamma)
        # Gamma_PGFs[i] = res[0]
        # Z[i] = res[1]

        Alpha_PGFs[i + 1] = lambda s_k, k=i: Alpha_k(s_k, Gamma_PGFs[k], k)

    if d == 0:
        alpha = Alpha_PGFs[-1](1.0)
        logZ = np.log(alpha)
    else:
        alpha = Alpha_PGFs[-1](GDualType(1.0, d))
        logZ = alpha.get(0, as_log=True)

    return logZ, alpha, Alpha_PGFs, Gamma_APGF_names

if __name__ == "__main__":
    # np.seterr(divide='ignore')

    y = np.array([1, 2, 3, 1, 3])
    lmbda = np.array([2.5, 6, 6, 6, 6]).reshape(-1, 1)
    delta = np.array([0.5, 0.5, 0.5, 0.5]).reshape(-1, 1)
    rho = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # res = APGF_Forward(y,
    #                       fwd.poisson_pgf,
    #                       lmbda,
    #                       fwd.poisson_pgf,
    #                       delta,
    #                       rho,
    #                       GDualType=gd.LSGDual)
    #
    # print(res)

    # y = 0 * np.array([40, 20, 40])
    # lmbda = np.array([25, 25, 25]).reshape(-1, 1)
    # delta = np.array([0.2, 0.9]).reshape(-1, 1)
    # rho = np.array([0.15, 0.15, 0.15])

    res = APGF_Forward(y,
                       fwd.poisson_pgf,
                       lmbda,
                       fwd.bernoulli_pgf,
                       delta,
                       rho)

    # print(res[0][-1](1))
    print(res)
    print(res[3])

    res2 = fwd.forward(y,
                       fwd.poisson_pgf,
                       lmbda,
                       fwd.bernoulli_pgf,
                       delta,
                       rho)
    print(res2)

    # print(np.log(res[-1](1)))