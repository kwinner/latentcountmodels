from scipy import stats
from forward import *
from forward_grad import *
from truncatedfa import poisson_branching, binomial_branching, nbinom_branching

"""
Notes
-----
hyperparam2param
Input:  hyperparameters that are learned by MLE
Output: parameters that are used by FA of size (K, n_params)

n_params: if 0 do not learn
"""

### Arrival ###

# Constant across time
constant_poisson_arrival = {
    'n_params': lambda T: 1,
    'pgf': poisson_pgf,
    'pgf_grad': poisson_pgf_grad,
    'sample': stats.poisson.rvs,
    'hyperparam2param': lambda x, T: np.tile(x, (len(T), 1)),
    'need_grad': lambda T: [[True]] * len(T),
    'backprop': lambda dtheta, x: np.sum(dtheta, axis=0), # should return 1d numpy array
    'init': lambda y, T: np.median(y),
    'bounds': lambda y, T: (1e-6, 1000)
}

constant_nbinom_arrival = {
    'n_params': lambda T: 2,
    'pgf': negbin_pgf,
    'pgf_grad': negbin_pgf_grad,
    'sample': stats.nbinom.rvs,
    'hyperparam2param': lambda x, T: np.tile(x, (len(T), 1)),
    'need_grad': lambda T: [[True, True]] * len(T),
    'backprop': lambda dtheta, x: np.sum(dtheta, axis=0),
    'init': lambda y, T: [np.median(y), 0.5],
    'bounds': lambda y, T: [(1e-6, 1000), (1e-6, 1 - 1e-6)]
}

# Fix (do not learn)
fix_poisson_arrival = {
    'n_params': lambda T: 0,
    'pgf': poisson_pgf,
    'pgf_grad': poisson_pgf_grad,
    'sample': stats.poisson.rvs,
    'hyperparam2param': lambda x, T: np.tile(x, (len(T), 1)),
    'need_grad': lambda T: [[False]] * len(T),
    'backprop': lambda dtheta, x: [],
    'init': lambda y, T: [],
    'bounds': lambda y, T: []
}

fix_nbinom_arrival = {
    'n_params': lambda T: 0,
    'pgf': negbin_pgf,
    'pgf_grad': negbin_pgf_grad,
    'sample': stats.nbinom.rvs,
    'hyperparam2param': lambda x, T: np.tile(x, (len(T), 1)),
    'need_grad': lambda T: [[False, False]] * len(T),
    'backprop': lambda dtheta, x: [],
    'init': lambda y, T: [],
    'bounds': lambda y, T: []
}

# N-mixture (parameter for the first time step, no subsequent new arrivals)
nmixture_poisson_arrival = {
    'n_params': lambda T: 1,
    'pgf': poisson_pgf,
    'pgf_grad': poisson_pgf_grad,
    'sample': stats.poisson.rvs,
    'hyperparam2param': lambda x, T: np.concatenate((x, np.zeros(len(T) - 1))).reshape((-1, 1)),
    'need_grad': lambda T: [[True]] + [[False]] * (len(T)-1),
    'backprop':  lambda dtheta, x: dtheta[0],
    'init': lambda y, T: np.median(y[:, 0]),
    'bounds': lambda y, T: (1e-6, None)
}

### Branching ###

# Constant across time
constant_poisson_branch = {
    'n_params': lambda T: 1,
    'pgf': poisson_pgf,
    'pgf_grad': poisson_pgf_grad,
    'sample': lambda n, lmbda: stats.poisson.rvs(n * lmbda),
    'hyperparam2param': lambda x, T: np.tile(x, (len(T)-1, 1)),
    'need_grad': lambda T: [[True]] * len(T),
    'backprop': lambda dtheta, x: np.sum(dtheta, axis=0),
    'init': lambda y, T: 1.0,
    'bounds': lambda y, T: (1e-6, None)
}

constant_nbinom_branch = {
    'n_params': lambda T: 1,
    'pgf': geometric_pgf,
    'pgf_grad': geometric_pgf_grad,
    'sample': stats.nbinom.rvs,
    'hyperparam2param': lambda x, T: 1/(np.tile(x, (len(T)-1, 1)) + 1),
    'need_grad': lambda T: [[True]] * len(T),
    'backprop': lambda dtheta, x: np.sum(dtheta, axis=0) * -1.0 / (np.array(x)+1)**2,
    'init': lambda y, T: 1.0,
    'bounds': lambda y, T: (1e-6, None)
}

constant_binom_branch = {
    'n_params': lambda T: 1,
    'pgf': bernoulli_pgf,
    'pgf_grad': bernoulli_pgf_grad,
    'sample': stats.binom.rvs,
    'hyperparam2param': lambda x, T: np.tile(x, (len(T)-1, 1)),
    'need_grad': lambda T: [[True]] * len(T),
    'backprop': lambda dtheta, x: np.sum(dtheta, axis=0),
    'init': lambda y, T: 0.5,
    'bounds': lambda y, T: (1e-6, 1 - 1e-6)
}

# Varying across time
var_poisson_branch = {
    'n_params': lambda T: (len(T) - 1),
    'pgf': poisson_pgf,
    'pgf_grad': poisson_pgf_grad,
    'sample': lambda n, gamma: stats.poisson.rvs(n * gamma),
    'hyperparam2param': lambda x, T: np.reshape(x, (-1, 1)),
    'need_grad': lambda T: [[True]] * len(T),
    'backprop': lambda dtheta, x: np.reshape(dtheta, -1), # df/dx = df/dtheta * dtheta/dx
    'init': lambda y, T: [1.0] * (len(T) - 1),
    'bounds': lambda y, T: [(1e-6, None)] * (len(T) - 1)
}

var_nbinom_branch = {
    'n_params': lambda T: (len(T) - 1),
    'pgf': geometric_pgf,
    'pgf_grad': geometric_pgf_grad,
    'sample': stats.nbinom.rvs,
    'hyperparam2param': lambda x, T: 1/(np.reshape(x, (-1, 1)) + 1),
    'need_grad': lambda T: [[True]] * len(T),
    'backprop': lambda dtheta, x: np.reshape(dtheta, -1) * - 1.0 / (np.array(x) + 1)**2,
    'init': lambda y, T: [1.0] * (len(T) - 1),
    'bounds': lambda y, T: [(1e-6, None)] * (len(T) - 1)
}

var_binom_branch = {
    'n_params': lambda T: (len(T) - 1),
    'pgf': bernoulli_pgf,
    'pgf_grad': bernoulli_pgf_grad,
    'sample': stats.binom.rvs,
    'hyperparam2param': lambda x, T: np.reshape(x, (-1, 1)),
    'need_grad': lambda T: [[True]] * len(T),
    'backprop': lambda dtheta, x: np.reshape(dtheta, -1), # df/dx = df/dtheta * dtheta/dx
    'init': lambda y, T: [0.5] * (len(T) - 1),
    'bounds': lambda y, T: [(1e-6, 1 - 1e-6)] * (len(T) - 1)
}

### Observation ###

# Binomial observation with constant param across time
constant_binom_observ = {
    'n_params': lambda T: 1,
    'pgf': None,
    'sample': stats.binom.rvs,
    'hyperparam2param': lambda x, T: np.tile(x, len(T)),
    'need_grad': lambda T: [True] * len(T),
    'backprop': lambda dtheta, x: np.sum(dtheta, keepdims=True),
    'init': lambda y, T: 0.5,
    'bounds': lambda y, T: (0.1, 1-1e-6)
}

# Fix binomial observation param, do not learn
fix_binom_observ = {
    'n_params': lambda T: 0,
    'pgf': None,
    'sample': stats.binom.rvs,
    'hyperparam2param': lambda x, T: np.tile(x, len(T)),
    'need_grad': lambda T: [False] * len(T),
    'backprop': lambda dtheta, x: [],
    'init': lambda y, T: [],
    'bounds': lambda y, T: []
}

"""
# Fully observed binomial (detection prob is 1)
full_binom_observ = {
    'learn_mask': False,
    'pgf': None,
    'sample': stats.binom.rvs,
    'hyperparam2param': lambda x, T: np.tile(1, len(T)),
    'init': lambda y: None,
    'bounds': lambda y: None
}

# Fix binomial observation param, do not learn
fix_binom_observ = {
    'learn_mask': False,
    'pgf': None,
    'sample': stats.binom.rvs,
    'hyperparam2param': lambda x, T: np.tile(0.6, len(T)),
    'init': lambda y: 0.6,
    'bounds': lambda y: (None, None)
}
"""
