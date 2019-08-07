import gdual as gd
# from apgf_forward_log import APGF_Forward
from apgf_forward_symb_log import APGF_Forward_symb
import forward as fwd

import time
import numpy as np

import matplotlib.pyplot as plt
import pickle

# subtitle = ', alternating ys, Poiss/Poiss'
# subtitle = ''

filename_prefix = 'evalvsDelta2'

DELTAS = np.linspace(0.05, 0.95, 19)
n_DELTAS = len(DELTAS)

n_reps = 1

# y = np.array([1,3,4,8,23,21,56,138,132,207,244,199,182,134,115,89,55,27,34,22,18,10,5,8,7,1]) # case 1
y = np.array([1,82,104,111,391,499,665,360,192,167,142,126,69,29,30,17,39,17,8,6]) # case 2
# y = np.array([1,2,3])
K = len(y)
# lmbda = np.array([5, 5, 5, 5, 5, 5, 5]).reshape(-1, 1)
lmbda = np.array(y * 0.3).reshape(-1, 1)
rho = np.array([0.25] * K)

imm_pgf_name = 'poisson'
if imm_pgf_name == 'poisson':
    imm_pgf = fwd.poisson_pgf
elif imm_pgf_name == 'negbin':
    imm_pgf = fwd.negbin_pgf

off_pgf_name = 'poisson'
if off_pgf_name == 'bernoulli':
    off_pgf = fwd.bernoulli_pgf
elif off_pgf_name == 'poisson':
    off_pgf = fwd.poisson_pgf

distn_name = 'P+P'

rt_fwd             = np.zeros(n_DELTAS)
# rt_apgffwd         = np.zeros(K)
rt_apgffwdsymb     = np.zeros(n_DELTAS)
nll_fwd            = np.zeros(n_DELTAS)
# nll_apgffwd        = np.zeros(K)
nll_apgffwdsymb    = np.zeros(n_DELTAS)


for k in range(n_DELTAS):
    print(k)
    delta = np.array([DELTAS[k]] * K).reshape(-1, 1)

    for n in range(n_reps):
        start = time.process_time()
        res = fwd.forward(y, imm_pgf, lmbda, off_pgf, delta, rho, GDualType=gd.LSGDual, d = 0)[0]
        end = time.process_time() - start

        nll_fwd[k] = nll_fwd[k] + (res / n_reps)
        rt_fwd[k] = rt_fwd[k] + (end / n_reps)

        # start = time.process_time()
        # res = APGF_Forward(y_k, imm_pgf, lmbda_k, off_pgf, delta, rho, GDualType=gd.LSGDual, d = 0)[0].get(0)
        # end = time.process_time() - start

        # nll_apgffwd[k] = nll_apgffwd[k] + (res / n_reps)
        # rt_apgffwd[k] = rt_apgffwd[k] + (end / n_reps)

        start = time.process_time()
        res = APGF_Forward_symb(y, imm_pgf, lmbda, off_pgf, delta, rho, GDualType=gd.LSGDual).get(0, as_log=True)
        end = time.process_time() - start

        nll_apgffwdsymb[k] = nll_apgffwdsymb[k] + (res / n_reps)
        rt_apgffwdsymb[k] = rt_apgffwdsymb[k] + (end / n_reps)

        # nll_mean_rel_error[k] = nll_mean_rel_error[k] + (np.abs(nll_fwd - nll_apgffwd) / np.abs(nll_fwd) / n_reps)

result = {}
result['experiment'] = 'eval_apgffwd_vs_Delta'
result['n_reps'] = n_reps
result['x'] = DELTAS
result['rt'] = [rt_fwd, rt_apgffwdsymb]
result['nll'] = [nll_fwd, nll_apgffwdsymb]
result['imm_pgf'] = imm_pgf_name
result['off_pgf'] = off_pgf_name
result['y'] = y
result['lmbda'] = lmbda
result['rho'] = rho

# pickle.dump()

# Y = np.sum(y)
# Ys = Y * LAMBDAS
# Y_max = np.max(y)
# Y_maxs = Y_max * LAMBDAS

# plt.plot(LAMBDAS, nll_fwd, LAMBDAS, nll_apgffwd, LAMBDAS, nll_apgffwdsymb)
# plt.legend((['fwd', 'apgffwd', 'apgffwd_symb']))
plt.plot(DELTAS, nll_fwd, DELTAS, nll_apgffwdsymb)
plt.legend((['fwd', 'apgffwd_symb']))
plt.xlabel('Delta')
plt.ylabel('nll')
plt.title('NLL vs pop scale,{}'.format(distn_name))
plt.show()
# plt.savefig('{}_{}.nll.png'.format(filename_prefix, distn_name))

# plt.plot(DELTAS, np.abs(nll_fwd - nll_apgffwdsymb) / np.abs(nll_fwd))
# # plt.legend((['fwd', 'apgffwd', 'apgffwd_symb']))
# plt.xlabel('Delta')
# plt.ylabel('nll relative error')
# plt.title('NLL Relative Error,{}'.format(distn_name))
# plt.show()
# plt.savefig('{}_{}.relerr.png'.format(filename_prefix, distn_name))

# # plt.plot(Ys, rt_fwd, Ys, rt_apgffwd)
# # plt.plot(LAMBDAS, rt_fwd, LAMBDAS, rt_apgffwd, LAMBDAS, rt_apgffwdsymb)
# # plt.legend((['fwd', 'apgffwd', 'apgffwd_symb']))
# plt.plot(DELTAS, rt_fwd, DELTAS, rt_apgffwdsymb)
# plt.legend((['fwd', 'apgffwd_symb']))
# # plt.xlabel('Y')
# plt.xlabel('Delta')
# plt.ylabel('rt')
# plt.title('RT vs pop scale,{}'.format(distn_name))
# plt.show()
# plt.savefig('{}_{}.rt.png'.format(filename_prefix, distn_name))

# plt.plot(Y_maxs, rt_fwd, Y_maxs, rt_apgffwd)
# plt.legend((['fwd', 'apgffwd']))
# plt.xlabel('max(y)')
# plt.ylabel('rt')
# plt.title('RT vs max count')
# plt.show()

print('Done.')