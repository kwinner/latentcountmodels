import csv, os
import numpy as np

from mle_distributions import *
from mle import *

# One-hundredth of H1N1 cases in the US during 2009 outbreak
# First wave
#y = np.array([ 1, 14, 25, 18, 28, 32, 40, 43, 46, 33, 27, 28, 22, 20, 20, 15])
# First half of the first wave
y = np.array([[ 1, 14, 25, 18, 28, 32, 40, 43, 46]])

# All H1N1 cases in the US, first half of the first wave
#y = np.array([14, 1354, 2484, 1741, 2796, 3144, 3967, 4203, 4587])

# All H1N1 cases in New England, first half of the first wave
#y = np.array([[1, 82, 104, 111, 391, 499]])

K = len(y[0])
print(K)

def flatten(l):
    return [i for row in l for i in row]

var_poisson_branch = {
    'learn_mask': [True] * (K - 1),
    'pmf': poisson_branching,
    'pgf': poisson_pgf,
    'pgf_grad': poisson_pgf_grad,
    'need_grad': lambda T: [[True]] * (len(T)-1),
    'sample': lambda n, gamma: stats.poisson.rvs(n * gamma),
    'hyperparam2param': lambda x, T: x.reshape((-1, 1)),
    'backprop': lambda dtheta: flatten(dtheta),
    'init': lambda y: [1.5] * (K - 1),
    'bounds': lambda y: [(0.1, None)] * (K - 1)
}

var_nbinom_branch = {
    'learn_mask': [True] * (K - 1),
    'pmf': nbinom_branching,
    'pgf': 'geometric',
    'sample': stats.nbinom.rvs,
    'hyperparam2param': lambda x, T: 1/(x.reshape((-1, 1)) + 1),
    'init': lambda y: [1] * (K - 1),
    'bounds': lambda y: [(1e-6, None)] * (K - 1)
}

grr_nbinom_branch = {
    'learn_mask': [True] * (K + 1),
    'pmf': nbinom_branching,
    'pgf': 'geometric',
    'sample': stats.nbinom.rvs,
    'hyperparam2param': lambda x, T: 1/(x[2:].reshape((-1, 1)) + 1),
    'init': lambda y: [1/np.var(y)] + [0.1] + [1] * (K - 1),
    'bounds': lambda y: [(None, None)] + [(1e-6, None)] * K # [sigma, R_0, R_t's]
}

lmbda = 30
fixed_poisson_arrival = {
    'learn_mask': False,
    'pmf': stats.poisson.pmf,
    'pgf': poisson_pgf,
    'sample': stats.poisson.rvs,
    'hyperparam2param': lambda x, T: np.concatenate(([lmbda], np.zeros(len(T) - 1))).reshape((-1, 1)),
    'init': lambda y: [],
    'bounds': lambda y: []
}

# Distributions
arrival = nmixture_poisson_arrival
#arrival = fixed_poisson_arrival

#branch = grr_nbinom_branch
#branch = var_nbinom_branch
branch = var_poisson_branch
#branch = constant_nbinom_branch
#branch = constant_poisson_branch

observ = constant_binom_observ
#observ = full_binom_observ

T = np.arange(K) # vector of observation times

theta_hat, ci_left, ci_right = run_mle(T, arrival, branch, observ,
                                       y=y.astype(np.int32), grad=True)
print(theta_hat, ci_left, ci_right)
"""
# Clean output format
r_hat = theta_hat[3:-1]
r_ci_left = ci_left[3:-1]
r_ci_right = ci_right[3:-1]

f = open('../data/case_study/pois_pois_esc_7wk_feb23.csv', 'w')
writer = csv.writer(f)
writer.writerow(['k', 'r_hat', 'ci_left', 'ci_right'])
for k, (r_k, ci1_k, ci2_k) in enumerate(zip(r_hat, r_ci_left, r_ci_right)):
	writer.writerow([k + 1, r_k, ci1_k, ci2_k])
f.close()


# Vary fixed arrival rate
path = '../data/case_study_pois_fix_lambda/'
if not os.path.exists(path):
    os.makedirs(path)

for lmbda in range(10, 101, 10):
    print lmbda
    fixed_poisson_arrival = {
        'learn_mask': False,
        'pmf': stats.poisson.pmf,
        'pgf': 'poisson',
        'sample': stats.poisson.rvs,
        'hyperparam2param': lambda x, T: np.concatenate(([lmbda], np.zeros(len(T) - 1))).reshape((-1, 1)),
        'init': lambda y: [],
        'bounds': lambda y: []
    }
    arrival = fixed_poisson_arrival
    out = None #path + str(lmbda) +'.csv'
    log = None #path + 'warnings.log'

    run_mle(T, arrival, branch, observ, fa, out=None, log=log,
            y=y.astype(np.int32), n=1, max_iters=5)
"""
