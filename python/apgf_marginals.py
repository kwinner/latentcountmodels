import forward as fwd
import apgf_forward_symb_log as apgffwd
import pgffa

import numpy as np
from scipy.stats import binom, nbinom

import matplotlib.pyplot as plt

# y = np.array([5,1,5])
# y = np.array([50,1,50,1,50,1,50])
y = np.array([1,82,104,111,391,499,665])#,360,192,167,142,126,69,29,30,17,39,17,8,6]) # case 2
K = len(y)
x = range(250,450)

# lmbda = np.array(y * 1.0).reshape(-1,1)
lmbda = np.array(y * 0.3).reshape(-1, 1)
delta = np.array([0.2] * K).reshape(-1,1)
rho = np.array([0.25] * K)

imm_pgf = fwd.poisson_pgf
# off_pgf = fwd.bernoulli_pgf
off_pgf = fwd.poisson_pgf

res = apgffwd.APGF_Forward_symb(y, imm_pgf, lmbda, off_pgf, delta, rho)
if res[0][0] == 'binom':
    res_y = binom.pmf(x, res[0][1][0], res[0][1][1])# * np.exp(res[1])
elif res[0][0] == 'nb':
    res_y = nbinom.pmf(x, res[0][1][0], res[0][1][1])

res2 = fwd.forward(y, imm_pgf, lmbda, off_pgf, delta, rho).get(x, as_log=False)

# s = np.zeros(K+1)
# u = np.zeros(K)
#
# s[K] = 1.0
# for i in range(K)[::-1]:
#     u[i] = s[i+1] * (1 - rho[i])
#     s[i] = off_pgf(u[i], theta=delta[i])
#
# Alpha_0 = lambda s_0: s_0.__class__.const(1.0, q = s_0.order()) if isinstance(s_0, gd.GDualBase) else 1.0
#
# Gamma_1 = lambda u_1: Alpha_0(off_pgf(u_1, delta[0])) * imm_pgf(u_1, lmbda[0])

lmbda = lmbda.reshape(-1)
delta = delta.reshape(-1)
res3 = pgffa.pgf_forward(lmbda, rho, delta, y)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)
plt.plot(x, res2, 'b--', x, res_y, 'r:', lw=2)
plt.legend(("exact", "approx"))
# plt.title(r"PMF of $\Gamma$ message")
plt.show()

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)
plt.plot(x, res2, 'b--', x, res_y, 'r:', lw=2)
plt.legend(("exact", "approx"))
plt.yscale('log')
# plt.title(r"PMF of $\Gamma$ message")
plt.show()

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=18)
# plt.plot(x, res2, 'bx ', x, res_y, 'r+ ', lw=2)
# plt.legend(("gdual-fwd", "APGF-fwd"))
# plt.title(r"PMF of $\Gamma$ message")
# plt.show()

print('Done.')