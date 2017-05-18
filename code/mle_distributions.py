from scipy import stats
from stabilityworkspace.generatingfunctions import *
from truncatedfa import poisson_branching, binomial_branching, nbinom_branching

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
    'init': lambda y: np.median(y),
    'bounds': lambda y: (1e-6, 1000)
}

constant_nbinom_arrival = {
    'learn_mask': [True, True],
    'pgf': negbin_ngdual,
    'sample': stats.nbinom.rvs,
    'hyperparam2param': lambda x, T: np.tile(x, (len(T), 1)),
    'init': lambda y: [np.mean(y), 0.5],
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
    'init': lambda y: 1,
    'bounds': lambda y: (1e-6, None)
}

constant_nbinom_branch = {
    'learn_mask': True,
    'pgf': geometric_ngdual,
    'sample': stats.nbinom.rvs,
    'hyperparam2param': lambda x, T: 1/(np.tile(x, (len(T)-1, 1)) + 1),
    'init': lambda y: 1,
    'bounds': lambda y: (1e-6, None)
}

constant_binom_branch = {
    'learn_mask' : True,
    'pgf': bernoulli_ngdual,
    'sample': stats.binom.rvs,
    'hyperparam2param': lambda x, T: np.tile(x, (len(T)-1, 1)),
    'init': lambda y: 1e-6,
    'bounds': lambda y: (1e-6, 1 - 1e-6)
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
    'init': lambda y: 0.5,
    'bounds': lambda y: (0.1, 1-1e-6)
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
