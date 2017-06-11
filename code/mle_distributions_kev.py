from scipy import stats, special
from stabilityworkspace.generatingfunctions import *
import random


"""
Notes
-----
hyperparam2param
Input:  hyperparameters that are learned by MLE
Output: parameters that are used by FA of size (K, n_params)
"""

### Arrival ###

# Constant param across time
constant_poisson_arrival = {
    'learn_mask': True,
    'pgf': poisson_ngdual,
    'sample': stats.poisson.rvs,
    'hyperparam2param': lambda x, T: np.tile(x, (len(T), 1)),
    'init': lambda y: random.normalvariate(np.median(y), 1.0),
    'bounds': lambda y: (1e-6, 1000)
}

constant_nbinom_arrival = {
    'learn_mask': [True, True],
    'pgf': negbin_ngdual,
    'sample': stats.nbinom.rvs,
    'hyperparam2param': lambda x, T: np.tile(x, (len(T), 1)),
    'init': lambda y: [random.normalvariate(np.mean(y), 1.0), random.uniform(0.25,0.75)],
    'bounds': lambda y: [(1e-6, 1000), (1e-6, 1 - 1e-6)]
}

# N-mixture (parameter for the first time step, no subsequent new arrivals)
nmixture_poisson_arrival = {
    'learn_mask': True,
    'pgf': poisson_ngdual,
    'sample': stats.poisson.rvs,
    'hyperparam2param': lambda x, T: np.concatenate((x, np.zeros(len(T) - 1))).reshape((-1, 1)),
    'init': lambda y: y[0],
    'bounds': lambda y: (1e-6, None)
}

### Branching ###

# Constant parameter across time
constant_poisson_branch = {
    'learn_mask': True,
    'pgf': poisson_ngdual,
    'sample': lambda n, lmbda: stats.poisson.rvs(n * lmbda),
    'hyperparam2param': lambda x, T: np.tile(x, (len(T)-1, 1)),
    'init': lambda y: random.uniform(1e-6, 2.),
    'bounds': lambda y: (1e-6, None)
}

constant_nbinom_branch = {
    'learn_mask': True,
    'pgf': geometric_ngdual,
    'sample': stats.nbinom.rvs,
    'hyperparam2param': lambda x, T: 1/(np.tile(x, (len(T)-1, 1)) + 1),
    'init': lambda y: random.uniform(1e-6, 2.),
    'bounds': lambda y: (1e-6, None)
}

constant_binom_branch = {
    'learn_mask' : True,
    'pgf': bernoulli_ngdual,
    'sample': stats.binom.rvs,
    'hyperparam2param': lambda x, T: np.tile(x, (len(T)-1, 1)),
    # 'init': lambda y: 1e-6,
    'init': lambda y: random.uniform(1e-6, 1 - 1e-6),
    'bounds': lambda y: (1e-6, 1 - 1e-6)
    #'bounds': lambda y: (-np.inf, np.inf)
}

constant_logistic_binom_branch = {
    'learn_mask' : True,
    'pgf': bernoulli_ngdual,
    'sample': stats.binom.rvs,
    'hyperparam2param': lambda x, T: np.tile(1 / (1+np.exp(-x)), (len(T)-1, 1)),
    # 'init': lambda y: 1e-6,
    'init': lambda y: random.uniform(-2, 2),
    'bounds': lambda y: (None, None)
}

"""
# Time-varying parameters across time
var_poisson_branch = {
    'learn_mask': [True] * (K - 1),
    'pgf': 'poisson',
    'sample': lambda n, gamma: stats.poisson.rvs(n * gamma),
    'hyperparam2param': lambda x, T: x.reshape((-1, 1)),
    'init': lambda y: [1] * (K - 1),
    'bounds': lambda y: [(1e-6, None)] * (K - 1)
}

var_nbinom_branch = {
    'learn_mask': [True] * (K - 1),
    'pgf': 'geometric',
    'sample': stats.nbinom.rvs,
    'hyperparam2param': lambda x, T: 1/(x.reshape((-1, 1)) + 1),
    'init': lambda y: [1] * (K - 1),
    'bounds': lambda y: [(1e-6, None)] * (K - 1)
}
"""
### Observation ###

# Binomial observation with constant param across time
constant_binom_observ = {
    'learn_mask': True,
    'pgf': None,
    'sample': stats.binom.rvs,
    'hyperparam2param': lambda x, T: np.tile(x, len(T)),
    # 'init': lambda y: 0.5,
    #'init': lambda y: 0.6,
    'init': lambda y: random.uniform(1e-3, 1 - 1e-3),
    # 'bounds': lambda y: (0.1, 1-1e-6)
    'bounds': lambda y: (1e-3, 1 - 1e-3)
    #'bounds': lambda y: (0.6, 0.6)
}

fix_binom_observ = {
    'learn_mask': False,
    'pgf': None,
    'sample': stats.binom.rvs,
    'hyperparam2param': lambda x, T: np.tile(0.6, len(T)),
    'init': lambda y: None,
    'bounds': lambda y: None
}

# Fully observed binomial (detection prob is 1)
full_binom_observ = {
    'learn_mask': False,
    'pgf': None,
    'sample': stats.binom.rvs,
    'hyperparam2param': lambda x, T: np.tile(1, len(T)),
    'init': lambda y: None,
    'bounds': lambda y: None
}
