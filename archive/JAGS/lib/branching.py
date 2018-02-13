import numpy as np
from scipy import stats

class BranchingDistribution(object):
    def __init__(self):
        self.sample = self.dist.rvs

class BinomialBranching(BranchingDistribution):
    def __init__(self):
        self.dist = stats.binom
        self.param_names = ['delta']       # p of binomial
        self.hyperparam_names = ['lambda'] # rate of exponential

    def pmf(self, n_max, delta_k):
        n_k = np.arange(n_max).reshape((-1, 1))
        return self.dist.pmf(np.arange(n_max), n_k, delta_k)

    def init_hyperparams(self, y, T):
        return [1] #[np.std(T)]

    def hyperparam_bounds(self, y, T):
        return [(1e-6, np.max(T)-np.min(T))]

    def expand_params(self, params, T):
        delta = self.hyperparams_to_params(params, T)
        return delta.reshape((-1, 1))

    def hyperparams_to_params(self, hyperparams, T):
        x = np.array(T[1:]) - T[:-1]
        delta = stats.expon.cdf(x, scale=1/hyperparams[0])
        return delta

    def generate_next(self, n, delta):
        return self.dist.rvs(n, delta)

class PoissonBranching(BranchingDistribution):
    def __init__(self):
        self.dist = stats.poisson 
        self.param_names = ['lambda']
        self.hyperparam_names = ['lambda']

    def pmf(self, n_max, gamma_k):
        n_k = np.arange(n_max).reshape((-1, 1))
        return self.dist.pmf(np.arange(n_max), n_k * gamma_k)

    def init_hyperparams(self, y, T):
        return [1]

    def hyperparam_bounds(self, y, T):
        return [(1e-6, np.max(T)-np.min(T))]

    def expand_params(self, params, T):
        K = len(T)
        return np.repeat(params, K - 1).reshape((-1, 1))

    def hyperparams_to_params(self, params, T):
        return np.repeat(params, len(T))

    def generate_next(self, n, gamma):
        return self.dist.rvs(n * gamma)

binom = BinomialBranching()
poisson = PoissonBranching()
