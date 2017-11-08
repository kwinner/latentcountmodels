import numpy as np
from scipy import stats

class ArrivalDistribution(object):
    def __init__(self):
        self.pmf = self.dist.pmf
        self.sample = self.dist.rvs

    def init_hyperparams(self, y, T):
        mu = np.mean(T)
        sigma = np.std(T)
        c = np.sum(y)

        return [mu, sigma, c]

    def hyperparam_bounds(self, y, T):
        mu = (np.min(T)-np.var(T), np.max(T)+np.var(T))
        sigma = (1e-6, np.var(T))
        c = (1, 100*np.sum(y))
        return [mu, sigma, c]

    def _mean_arrivals(self, hyperparams, T):
        mu, sigma, c = hyperparams
        return stats.norm.pdf(T, mu, sigma) * c

class PoissonArrival(ArrivalDistribution):
    def __init__(self):
        self.dist = stats.poisson
        self.hyperparam_names = ['mu', 'sigma', 'c']
        self.param_names = ['lambda']
        super(PoissonArrival, self).__init__()

    def init_params(self, y):
        return [np.mean(y)]

    def param_bounds(self, y, T=None):
        return [(0, None)]

    def hyperparams_to_params(self, hyperparams, T):
        return self._mean_arrivals(hyperparams, T)

    def expand_params(self, params, T):
        if np.isscalar(T):
            # n-mixture: [[lambda], [0], ...]
            return np.vstack((params, np.zeros((T-1, 1))))
        else:
            # zonneveld: [[lambda_1], ..., [lambda_K]]
            lmbda = self.hyperparams_to_params(params, T)
            return lmbda.reshape((-1, 1))

    def generate_data(self, hyperparams, T):
        lmbda = self.hyperparams_to_params(hyperparams, T)
        return self.dist.rvs(lmbda)

class NBinomArrival(ArrivalDistribution):
    def __init__(self):
        self.dist = stats.nbinom
        self.hyperparam_names = ['mu', 'sigma', 'c', 'r']
        self.param_names = ['r', 'p']
        super(NBinomArrival, self).__init__()

    def init_params(self, y):
        r, p = np.var(y), 0.5
        return [r, p]

    def init_hyperparams(self, y, T):
        mu, sigma, c = super(NBinomArrival, self).init_hyperparams(y, T)
        r = np.mean(y)**2 / (np.var(y)-np.mean(y))
        return [mu, sigma, c, r]

    def param_bounds(self, y):
        return [(1e-6, None), (1e-6, 1)]

    def hyperparam_bounds(self, y, T):
        mu, sigma, c = super(NBinomArrival, self).hyperparam_bounds(y, T)
        r = (1e-6, None)
        return [mu, sigma, c, r]

    def hyperparams_to_params(self, hyperparams, T):
        lmbda = self._mean_arrivals(hyperparams[:-1], T)
        r = hyperparams[-1]
        p = r / (r + lmbda)
        return r, p

    def expand_params(self, params, T):
        if np.isscalar(T):
            # n-mixture: [[r, p], [1, 1], ...]
            return np.vstack((params, np.ones((T-1, 2))))
        else:
            # zonneveld: [[r_1, p_1], ..., [r_K, p_K]]
            r, p = self.hyperparams_to_params(params, T)
            return np.array([(r, p_k) for p_k in p])

    def generate_data(self, hyperparams, T):
        r, p = self.hyperparams_to_params(hyperparams, T)
        return self.dist.rvs(r, p)

class GeomArrival(ArrivalDistribution):
    def __init__(self):
        self.dist = stats.geom
        self.hyperparam_names = ['mu', 'sigma', 'c']
        self.param_names = ['p']
        super(GeomArrival, self).__init__()

    def init_params(self, y, T=None):
        return [0.5]

    def param_bounds(self, y, T=None):
        return [(0, 1)]

    def hyperparams_to_params(self, hyperparams, T):
        lmbda = self._mean_arrivals(hyperparams, T)
        p = 1 / (lmbda + 1)
        loc = -1 # shift location by -1 to get [0, inf) support
        return p, loc

    def expand_params(self, params, T):
        if np.isscalar(T):
            # n-mixture: [[p, -1], [1, -1], ...]
            p = np.vstack((params, np.ones((T-1, 1))))
            loc = -np.ones((T, 1)) # shift location by -1 to get [0, inf) support
            return np.hstack((p, loc))
        else:
            # zonneveld: [[p_1], ..., [p_K]]
            p, loc = self.hyperparams_to_params(params, T)
            return np.array([(p_k, loc) for p_k in p])

    def generate_data(self, hyperparams, T):
        p, loc = self.hyperparams_to_params(hyperparams, T)
        return self.dist.rvs(p, loc)
 
poisson = PoissonArrival()
geom = GeomArrival()
nbinom = NBinomArrival()
