import numpy as np
from apgf_forward_symb_log import MM_Binom, MM_NB
import forward as fwd
import matplotlib.pyplot as plt
import gdual as gd
from scipy.special import gamma, gammaln

def gbin_pgf(s, mu, sigma2):
    assert mu != sigma2

    D = sigma2/mu

    return((D + (1 - D) * s) ** (mu / (1 - D)))

def gbin_deriv(s, y, mu, sigma2):
    assert mu != sigma2

    D = sigma2/mu
    D2 = mu / (1 - D)

    return(((1 - D) ** y) *
           gamma(D2 + 1) /
           gamma(D2 - y + 1) *
           (D + (1 - D) * s) ** (D2 - y))

def gbin_logderiv(s, y, mu, sigma2):
    assert mu != sigma2

    D = sigma2/mu
    D2 = mu / (1 - D)

    return(y * np.log(1 - D) +
           gammaln(D2 + 1) -
           gammaln(D2 - y + 1) +
           (D2 - y) * np.log(D + (1 - D) * s))

def test(F_p):
    M_p = lambda t: F_p(np.exp(t))

    # use the MGF to compute the first k moments of p
    k = 2
    t0_gd = gd.LSGDual(0, q=k)

    # extract the moments (note the renormalization by logZ)
    # note: this assumes the mean and variance of F_p are >= 0, which they should be for a count distribution
    # but we go ahead and force that for stability reasons
    moments_p = M_p(t0_gd)
    # log_moments_p.trunc_neg_coefs()
    moments_p = np.exp(moments_p.get(range(1, k + 1), as_log=True) + gammaln(np.arange(2, k + 2)) - logZ)

    mean_p = moments_p[0]
    var_p = moments_p[1] - (mean_p ** 2)

s_vals = np.linspace(0, 2, 51)
s_deriv_val = 1.0
q = 50

mu = 5
sigma2 = 2.0001

gbin_val = np.array([gbin_pgf(s, mu, sigma2) for s in s_vals])

s_gd = gd.LSGDual(s_deriv_val, q = q)
# gbin_derivs = gbin_pgf(s_gd, mu, sigma2).get(range(0, q), as_log = True)
gbin_derivs = gbin_pgf(s_gd, mu, sigma2)
# gbin_closedderivs = [gbin_logderiv(s_deriv_val, y, mu, sigma2) for y in range(0, q)]
gbin_closedderivs = [gbin_deriv(s_deriv_val, y, mu, sigma2) for y in range(0, q)]

if(sigma2 > mu):
    theta = MM_NB(mu, sigma2)

    reg_val = np.array([fwd.negbin_pgf(s, theta) for s in s_vals])
    legend_name = 'NB'

    reg_derivs = fwd.negbin_pgf(s_gd, theta).get(range(0, q), as_log = True)

if(sigma2 < mu):
    theta = MM_Binom(mu, sigma2)

    reg_val = np.array([fwd.binomial_pgf(s, theta) for s in s_vals])
    legend_name = 'Binom'

    reg_derivs = fwd.binomial_pgf(s_gd, theta).get(range(0, q), as_log = True)

plt.plot(s_vals, reg_val, '-', s_vals, gbin_val, '--')
plt.legend(([legend_name, 'GBin']))
plt.xlabel('s')
plt.ylabel('F(s)')
plt.title('mu = {}, sigma^2 = {}'.format(mu, sigma2))
plt.show()

plt.plot(s_vals, np.abs(reg_val-gbin_val) / reg_val)
# plt.legend(([legend_name, 'GBin']))
plt.xlabel('s')
plt.ylabel('|A(s) - B(s)| / A(s)')
plt.title('mu = {}, sigma^2 = {}'.format(mu, sigma2))
plt.show()

plt.plot(range(0,q), reg_derivs, '-', range(0,q), gbin_derivs, '--')
plt.legend(([legend_name, 'GBin']))
plt.xlabel('y')
plt.ylabel('log d^y/ds^y F(s)')
plt.title('mu = {}, sigma^2 = {}, @s = {}'.format(mu, sigma2, s_deriv_val))
plt.show()

plt.plot(range(0,q), np.abs(reg_derivs-gbin_derivs) / reg_derivs)
# plt.legend(([legend_name, 'GBin']))
plt.xlabel('y')
plt.ylabel('relative error of derivatives')
plt.title('mu = {}, sigma^2 = {}, @s = {}'.format(mu, sigma2, s_deriv_val))
plt.show()