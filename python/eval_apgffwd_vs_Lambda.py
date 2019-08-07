import gdual as gd
# from apgf_forward_log import APGF_Forward
from apgf_forward_symb_log import APGF_Forward_symb
import forward as fwd

import time
import numpy as np

import matplotlib.pyplot as plt

# subtitle = ', alternating ys, Poiss/Poiss'
# subtitle = ''

filename_prefix = 'evalvsLambda'

LAMBDAS = np.array([1, 5, 10, 20, 50, 100])
K = len(LAMBDAS)

n_reps = 20

# y = np.array([3, 2, 4, 1, 5, 2, 3])
# lmbda = np.array([5, 5, 5, 5, 5, 5, 5]).reshape(-1, 1)
# lmbda = np.array([5, 5, 5, 5, 5, 5, 5])
# delta = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4]).reshape(-1, 1)
# rho = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

# pattern = 'standard'
pattern = 'alternating'

y = np.array([5, 1, 5, 1, 5, 1, 5])
# lmbda = np.array([5, 5, 5, 5, 5, 5, 5]).reshape(-1, 1)
lmbda = np.array([5, 5, 5, 5, 5, 5, 5])
delta = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4]).reshape(-1, 1)
rho = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

# imm_pgf = fwd.poisson_pgf
imm_pgf = fwd.negbin_pgf
# off_pgf = fwd.bernoulli_pgf
off_pgf = fwd.poisson_pgf

distn_name = 'NB+P'

rt_fwd             = np.zeros(K)
# rt_apgffwd         = np.zeros(K)
rt_apgffwdsymb     = np.zeros(K)
nll_fwd            = np.zeros(K)
# nll_apgffwd        = np.zeros(K)
nll_apgffwdsymb    = np.zeros(K)


for k in range(K):
    x = LAMBDAS[k]

    y_k = x * y
    # y_k = y
    # lmbda_k = x * lmbda
    lmbda_k = np.stack(((lmbda * x), [0.5] * len(lmbda)), 1)

    for n in range(n_reps):
        start = time.process_time()
        res = fwd.forward(y_k, imm_pgf, lmbda_k, off_pgf, delta, rho, GDualType=gd.LSGDual, d = 0)[0]
        end = time.process_time() - start

        nll_fwd[k] = nll_fwd[k] + (res / n_reps)
        rt_fwd[k] = rt_fwd[k] + (end / n_reps)

        # start = time.process_time()
        # res = APGF_Forward(y_k, imm_pgf, lmbda_k, off_pgf, delta, rho, GDualType=gd.LSGDual, d = 0)[0].get(0)
        # end = time.process_time() - start

        # nll_apgffwd[k] = nll_apgffwd[k] + (res / n_reps)
        # rt_apgffwd[k] = rt_apgffwd[k] + (end / n_reps)

        start = time.process_time()
        res = APGF_Forward_symb(y_k, imm_pgf, lmbda_k, off_pgf, delta, rho, GDualType=gd.LSGDual).get(0, as_log=True)
        end = time.process_time() - start

        nll_apgffwdsymb[k] = nll_apgffwdsymb[k] + (res / n_reps)
        rt_apgffwdsymb[k] = rt_apgffwdsymb[k] + (end / n_reps)

        # nll_mean_rel_error[k] = nll_mean_rel_error[k] + (np.abs(nll_fwd - nll_apgffwd) / np.abs(nll_fwd) / n_reps)

Y = np.sum(y)
Ys = Y * LAMBDAS
Y_max = np.max(y)
Y_maxs = Y_max * LAMBDAS

# plt.plot(LAMBDAS, nll_fwd, LAMBDAS, nll_apgffwd, LAMBDAS, nll_apgffwdsymb)
# plt.legend((['fwd', 'apgffwd', 'apgffwd_symb']))
plt.plot(LAMBDAS, nll_fwd, LAMBDAS, nll_apgffwdsymb)
plt.legend((['fwd', 'apgffwd_symb']))
plt.xlabel('Lambda')
plt.ylabel('nll')
plt.title('NLL vs pop scale, pattern = {},{}'.format(pattern, distn_name))
plt.show()
plt.savefig('{}_{}_{}.nll.png'.format(filename_prefix, pattern, distn_name))

plt.plot(LAMBDAS, np.abs(nll_fwd - nll_apgffwdsymb) / np.abs(nll_fwd))
# plt.legend((['fwd', 'apgffwd', 'apgffwd_symb']))
plt.xlabel('Lambda')
plt.ylabel('nll relative error')
plt.title('NLL Relative Error, pattern = {},{}'.format(pattern, distn_name))
plt.show()
plt.savefig('{}_{}_{}.relerr.png'.format(filename_prefix, pattern, distn_name))

# plt.plot(Ys, rt_fwd, Ys, rt_apgffwd)
# plt.plot(LAMBDAS, rt_fwd, LAMBDAS, rt_apgffwd, LAMBDAS, rt_apgffwdsymb)
# plt.legend((['fwd', 'apgffwd', 'apgffwd_symb']))
plt.plot(LAMBDAS, rt_fwd, LAMBDAS, rt_apgffwdsymb)
plt.legend((['fwd', 'apgffwd_symb']))
# plt.xlabel('Y')
plt.xlabel('Lambda')
plt.ylabel('rt')
plt.title('RT vs pop scale, pattern = {},{}'.format(pattern, distn_name))
plt.show()
plt.savefig('{}_{}_{}.rt.png'.format(filename_prefix, pattern, distn_name))

# plt.plot(Y_maxs, rt_fwd, Y_maxs, rt_apgffwd)
# plt.legend((['fwd', 'apgffwd']))
# plt.xlabel('max(y)')
# plt.ylabel('rt')
# plt.title('RT vs max count')
# plt.show()

print('Done.')