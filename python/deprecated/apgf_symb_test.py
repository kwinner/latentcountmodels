import numpy as np
from scipy.special import gammaln
from scipy.misc import factorial
import gdual as gd

import forward as fwd

from apgf_forward_log import APGF as APGF_baseline
from apgf_forward_log import APGF_Forward

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

def APGF(F_p, Z = None, GDualType = gd.LSGDual, return_type = ['name', 'param', 'lambda']):
    # compute the normalization constant for F if it wasn't provided
    if Z is None:
        Z = F_p(GDualType.const(1)).get(0, as_log=False)

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
    moments_p = M_p(t0_gd)
    # log_moments_p.trunc_neg_coefs()
    moments_p = moments_p.get(range(1,k+1), as_log=False) * factorial(np.arange(1, k + 1)) / Z

    mean_p = moments_p[0]
    var_p  = moments_p[1] - (mean_p ** 2)

    # mean_p = np.exp(log_mean_p)
    # var_p  = np.exp(log_var_p)

    assert np.isfinite(mean_p) and np.isfinite(var_p)

    # assert not np.isposinf(log_mean_p) and not np.isnan(log_mean_p)
    # assert not np.isposinf(log_var_p) and not np.isnan(log_var_p)

    # if mean and var are "close" (to numerical stability) treat them as equal
    # if np.abs(mean_p - var_p) < 1e-6:
    if True:
        distn = 'poiss'
        theta = MM_Poiss(mean_p)
        lmbda = lambda s, theta = theta, Z = Z: Z * fwd.poisson_pgf(s, theta)
    elif mean_p < var_p:
        distn = 'nb'
        theta = MM_NB(mean_p, var_p)
        lmbda = lambda s, theta = theta, Z = Z: Z * fwd.negbin_pgf(s, theta)
    elif mean_p > var_p:
        distn = 'binom'
        theta = MM_Binom(mean_p, var_p)
        lmbda = lambda s, theta = theta, Z = Z: Z * fwd.binomial_pgf(s, theta)
    else:
        raise Exception('Unable to approximate PGF.')

    return_vals = {'name': distn, 'param': theta, 'lambda': lmbda}
    return([return_vals[key] for key in return_type])

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in exp")


GDualType = gd.LSGDual

F = fwd.bernoulli_pgf
G = fwd.poisson_pgf
y = np.array([1, 2, 3], dtype=np.int32)
lmbda = np.array([2.5, 6, 6]).reshape(-1, 1)
delta = np.array([0.5, 0.5]).reshape(-1, 1)
rho = np.array([0.2, 0.2, 0.2])



G_1 = lambda u_1: G(u_1, lmbda[0])
Z_1 = G_1(1)
Ghat_1 = APGF(G_1, Z_1, return_type=['name', 'param'])
Gtest_1 = APGF_baseline(G_1, report_distn=True)

# print(Ghat_1)
# print(Gtest_1)

mu = Ghat_1[1][0]
A_1 = lambda s_1: ((mu * rho[0]) ** y[0]) * Z_1 / factorial(y[0]) / np.exp(mu) * (s_1 ** y[0]) * np.exp(mu * (1 - rho[0]) * s_1)

Gtest_1 = Gtest_1[0]
s_1 = GDualType.const(1.1)

def Atest_1(s_1):
    if np.isscalar(s_1):
        const = GDualType.const(y[0] * (np.log(s_1) + np.log(rho[0])) - gammaln(y[0] + 1), as_log=True)
    elif isinstance(s_1, gd.GDualBase):
        const = s_1 ** y[0]
        const *= GDualType.const(y[0] * np.log(rho[0]) - gammaln(y[0] + 1), as_log=True)
    res = const * gd.diff(Gtest_1, s_1 * (1 - rho[0]), y[0], GDualType=GDualType)
    return res

# print(A_1(s_1))
# print(Atest_1(s_1))



G_2 = lambda u_2: A_1(F(u_2, delta[0])) * G(u_2, lmbda[1])
Gtest_2 = lambda u_2: Atest_1(F(u_2, delta[0])) * G(u_2, lmbda[1])

# print(G_2(GDualType(0.3, q = 4)))
# print(Gtest_2(GDualType(0.3, q = 4)))

Z_2 = G_2(1)
Ghat_2 = APGF(G_2, Z_2, return_type=['name', 'param'])
Ghattest_2 = APGF_baseline(G_2, report_distn=True)

# print(Ghat_2)
# print(Ghattest_2)

mu = Ghat_2[1][0]
A_2 = lambda s_2: ((mu * rho[1]) ** y[1]) * Z_2 / factorial(y[1]) / np.exp(mu) * (s_2 ** y[1]) * np.exp(mu * (1 - rho[1]) * s_2)

Ghattest_2 = Ghattest_2[0]
s_2 = GDualType.const(1.3)

def Atest_2(s_2):
    if np.isscalar(s_2):
        const = GDualType.const(y[1] * (np.log(s_2) + np.log(rho[1])) - gammaln(y[1] + 1), as_log=True)
    elif isinstance(s_2, gd.GDualBase):
        const = s_2 ** y[1]
        const *= GDualType.const(y[1] * np.log(rho[1]) - gammaln(y[1] + 1), as_log=True)
    res = const * gd.diff(Ghattest_2, s_2 * (1 - rho[1]), y[1], GDualType=GDualType)
    return res

# print(A_2(s_2))
# print(Atest_2(s_2))


G_3 = lambda u_3: A_2(F(u_3, delta[1])) * G(u_3, lmbda[2])
Gtest_3 = lambda u_3: Atest_2(F(u_3, delta[1])) * G(u_3, lmbda[2])

# print(G_3(GDualType(0.3, q = 4)))
# print(Gtest_3(GDualType(0.3, q = 4)))

Z_3 = G_3(1)
Ghat_3 = APGF(G_3, Z_3, return_type=['name', 'param'])
Ghattest_3 = APGF_baseline(G_3, report_distn=True)

# print(Ghat_3)
# print(Ghattest_3)

mu = Ghat_3[1][0]
A_3 = lambda s_3: ((mu * rho[2]) ** y[2]) * Z_3 / factorial(y[2]) / np.exp(mu) * (s_3 ** y[2]) * np.exp(mu * (1 - rho[2]) * s_3)

Ghattest_3 = Ghattest_3[0]
s_3 = GDualType.const(1.0)

def Atest_3(s_3):
    if np.isscalar(s_3):
        const = GDualType.const(y[2] * (np.log(s_3) + np.log(rho[2])) - gammaln(y[2] + 1), as_log=True)
    elif isinstance(s_3, gd.GDualBase):
        const = s_3 ** y[2]
        const *= GDualType.const(y[2] * np.log(rho[2]) - gammaln(y[2] + 1), as_log=True)
    res = const * gd.diff(Ghattest_3, s_3 * (1 - rho[2]), y[2], GDualType=GDualType)
    return res

print(A_3(s_3))
print(Atest_3(s_3))

res = APGF_Forward(y, G, lmbda, F, delta, rho)

print(np.exp(res[0]))

from apgf_forward_symb3 import APGF_Forward_symb

res = APGF_Forward_symb(y, G, lmbda, F, delta, rho)

print(res)

# F = lambda s: fwd.binomial_pgf(s, [10000, 0.2])
#
# print(APGF(F))
# print(APGF_baseline(F, report_distn=True))
#
# F = lambda s: fwd.poisson_pgf(s, [10000])
#
# print(APGF(F))
# print(APGF_baseline(F, report_distn=True))
#
# F = lambda s: fwd.negbin_pgf(s, [10000, 0.2])
#
# print(APGF(F))
# print(APGF_baseline(F, report_distn=True))
#
# F = lambda s: fwd.geometric2_pgf(s, [0.3])
#
# print(APGF(F))
# print(APGF_baseline(F, report_distn=True))

# F = lambda s: 1/156 * fwd.poisson_pgf(s, [10000])
#
# print(APGF(F))
# print(APGF_baseline(F, report_distn=True))