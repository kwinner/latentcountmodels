import gdual as gd
# from apgf_forward_log import APGF_Forward
from apgf_forward_symb_log import APGF_Forward_symb as APGF_Forward
import forward as fwd

import time
import numpy as np

import matplotlib.pyplot as plt

# Ks = np.array([2, 6, 10, 15, 20,])
Ks = np.array([2, 6, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200])
# Ks = np.array([2, 6, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100])
n_K = len(Ks)

filename_prefix = 'evalvsK'

# ['early', 'mid', 'late', 'uniform']
pattern = 'early'

n_reps = 20

YMAX_SCALE   = 300
# YMAX_SCALE = 10
LAMBDA_SCALE = 0.8
DELTA        = 0.2
RHO          = 0.75

# y = np.array([3, 2, 4])
# lmbda = np.array([5, 5, 5]).reshape(-1, 1)
# delta = np.array([0.4, 0.4]).reshape(-1, 1)
# rho = np.array([0.25, 0.25, 0.25])

imm_pgf = fwd.poisson_pgf
off_pgf = fwd.poisson_pgf
distn_name = 'P+P'

rt_gdfwd    = np.zeros(n_K)
rt_apgffwd  = np.zeros(n_K)
nll_gdfwd   = np.zeros(n_K)
nll_apgffwd = np.zeros(n_K)

for i_k in range(n_K):
    K = Ks[i_k]

    if pattern == 'uniform':
        y     = np.array([YMAX_SCALE] * K,       dtype=np.int32)
        lmbda = (y * LAMBDA_SCALE).reshape(-1, 1)
        # lmbda = np.stack(((y * LAMBDA_SCALE), [0.5] * K), 1)
        delta = np.array([DELTA] * (K - 1)).reshape(-1, 1)
        rho   = np.array([RHO]   * K)
    elif pattern == 'mid':
        y     = np.array([1] * K, dtype=np.int32)
        y[int(np.ceil(K / 2))] = YMAX_SCALE
        lmbda = (y * LAMBDA_SCALE).reshape(-1, 1)
        # lmbda = np.stack(((y * LAMBDA_SCALE), [0.5] * K), 1)
        delta = np.array([DELTA] * (K - 1)).reshape(-1, 1)
        rho   = np.array([RHO]   * K)
    elif pattern == 'early':
        y = np.array([1] * K, dtype=np.int32)
        y[0] = YMAX_SCALE
        lmbda = (y * LAMBDA_SCALE).reshape(-1, 1)
        # lmbda = np.stack(((y * LAMBDA_SCALE), [0.5] * K), 1)
        delta = np.array([DELTA] * (K - 1)).reshape(-1, 1)
        rho = np.array([RHO] * K)
    elif pattern == 'late':
        y = np.array([1] * K, dtype=np.int32)
        y[-1] = YMAX_SCALE
        lmbda = (y * LAMBDA_SCALE).reshape(-1, 1)
        # lmbda = np.stack(((y * LAMBDA_SCALE), [0.5] * K), 1)
        delta = np.array([DELTA] * (K - 1)).reshape(-1, 1)
        rho = np.array([RHO] * K)

    for n in range(n_reps):
        start = time.process_time()
        nll = fwd.forward(y, imm_pgf, lmbda, off_pgf, delta, rho, GDualType=gd.LSGDual, d = 0)[0]
        end = time.process_time() - start

        nll_gdfwd[i_k] = nll_gdfwd[i_k] + (nll / n_reps)
        rt_gdfwd[i_k]  = rt_gdfwd[i_k]  + (end / n_reps)

        start = time.process_time()
        nll = APGF_Forward(y, imm_pgf, lmbda, off_pgf, delta, rho, GDualType=gd.LSGDual).get(0, as_log=True)
        end = time.process_time() - start

        nll_apgffwd[i_k] = nll_apgffwd[i_k] + (nll / n_reps)
        rt_apgffwd[i_k]  = rt_apgffwd[i_k]  + (end / n_reps)

        # nll_mean_rel_error[k] = nll_mean_rel_error[k] + (np.abs(nll_fwd - nll_apgffwd) / np.abs(nll_fwd) / n_reps)

# Y = np.sum(y)
# Ys = Y * LAMBDAS
# Y_max = np.max(y)
# Y_maxs = Y_max * LAMBDAS

plt.plot(Ks, np.abs(nll_gdfwd - nll_apgffwd) / np.abs(nll_gdfwd))
# plt.legend((['fwd', 'apgffwd', 'apgffwd_symb']))
plt.xlabel('K')
plt.ylabel('nll relative error')
plt.title('NLL Relative Error, Lambda = {}, pattern = {},{}'.format(YMAX_SCALE, pattern, distn_name))
plt.show()
plt.savefig('{}_L{}_{}_{}.relerr.png'.format(filename_prefix, YMAX_SCALE, pattern, distn_name))

plt.plot(Ks, nll_gdfwd, Ks, nll_apgffwd)
plt.xlabel('K')
plt.ylabel('nll')
plt.title('nll vs K, Lambda = {}, pattern = {},{}'.format(YMAX_SCALE, pattern, distn_name))
plt.legend((['gdfwd', 'apgffwd']))
plt.show()
plt.savefig('{}_L{}_{}_{}.nll.png'.format(filename_prefix, YMAX_SCALE, pattern, distn_name))

plt.plot(Ks, rt_gdfwd, Ks, rt_apgffwd)
plt.xlabel('K')
plt.ylabel('rt (s)')
plt.title('rt vs K, Lambda = {}, pattern = {},{}'.format(YMAX_SCALE, pattern, distn_name))
plt.legend((['gdfwd', 'apgffwd']))
plt.show()
plt.savefig('{}_L{}_{}_{}.rt.png'.format(filename_prefix, YMAX_SCALE, pattern, distn_name))

# # plt.plot(Ys, rt_fwd, Ys, rt_apgffwd)
# plt.plot(LAMBDAS, rt_fwd, LAMBDAS, rt_apgffwd)
# plt.legend((['fwd', 'apgffwd']))
# # plt.xlabel('Y')
# plt.xlabel('Lambda')
# plt.ylabel('rt')
# plt.title('RT vs total pop size')
# plt.show()

# plt.plot(Y_maxs, rt_fwd, Y_maxs, rt_apgffwd)
# plt.legend((['fwd', 'apgffwd']))
# plt.xlabel('max(y)')
# plt.ylabel('rt')
# plt.title('RT vs max count')
# plt.show()

print('Done.')