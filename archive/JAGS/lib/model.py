import sys
import numpy as np
import pgffa, truncatedfa
import arrival, branching
from scipy import stats, optimize

class Model(object):
    def __init__(self, arrival_dist, branching_dist):
        self.arrival_dist = arrival_dist
        self.branching_dist = branching_dist

        # Warn location shift by -1 for geom arrival
        if arrival_dist is arrival.geom:
            print 'Warning: shifting geometric arrivals by loc=-1'

class NMixture(Model):
    """
    NOTE:
    - Only supports Poisson, NB, and geometric arrival distributions
    - For geometric arrival, shift location by -1 to get [0, inf) support
    """

    def __init__(self, arrival_dist, branching_dist=branching.binom):
        if branching_dist is not branching.binom:
            print 'Error: n-mixture model only supports binomial branching'
            sys.exit(0)

        super(NMixture, self).__init__(arrival_dist, branching_dist)

    def mle(self, y, T=None, theta0=None, bounds=None, fa=pgffa, n_max=-1):
        # For PGFFA, make sure Poisson arrival and binomial branching
        if fa is pgffa:
            assert self.arrival_dist is arrival.poisson and \
                   self.branching_dist is branching.binom, \
                   'Error: PGFFA only supports Poisson arrival and binomial branching'

        if T is not None:
            print 'Warning: T will be ignored'

        # Set defaults
        if theta0 is None: theta0 = self._default_theta0(y)
        if bounds is None: bounds = self._default_bounds(y)
        if fa is truncatedfa and n_max == -1: n_max = np.max(y) * 5

        # Call the optimizer
        fa_args = tuple() if fa is pgffa else (self.arrival_dist.pmf, self.branching_dist.pmf, n_max)
        objective_args = (y, fa, fa_args)
        theta, fmin, info = optimize.fmin_l_bfgs_b(self.objective, theta0,
            args=objective_args, approx_grad=True, bounds=bounds)
        params = self.theta2dict(theta)

        return params, fmin, info

    def objective(self, theta, y, fa, fa_args):
        # Make sure there is no nan in theta
        if np.any(np.isnan(theta)): return float('inf')
    
        # Get params from theta
        K = len(y)
        arrival_params, branching_params, rho = self._get_params_from_theta(theta, K)
        
        return -fa.forward(y, arrival_params, branching_params, rho, *fa_args)[0]

    def _default_theta0(self, y):
        arrival_params = self.arrival_dist.init_params(y)
        rho = 0.5
        return np.hstack((arrival_params, rho))

    def _default_bounds(self, y):
        arrival_bounds = self.arrival_dist.param_bounds(y)
        rho_bounds = [(0, 1)]
        return arrival_bounds + rho_bounds

    def _get_params_from_theta(self, theta, K):
        arrival_params = theta[:-1]
        rho = theta[-1]

        arrival_params = self.arrival_dist.expand_params(arrival_params, K)
        branching_params = np.array([1] * (K - 1)).reshape((-1, 1))
        rho = [rho] * K

        return arrival_params, branching_params, rho

    def theta2dict(self, theta):
        arrival_names = self.arrival_dist.param_names
        names = arrival_names + ['rho']
        params = {name: value for name, value in zip(names, theta)}
        return params

    def generate_data(self, arrival_params, rho, K):
        n = self.arrival_dist.sample(*arrival_params)
        y = stats.binom.rvs(n, rho, size=K)
        return y

class Zonneveld(Model):
    """
    NOTE:
    - Only supports Poisson, NB, and geometric arrival distributions
    - Only supports binomial and Poisson branching distributions
    - The parameter of Poisson branching is currently constant across time
    """

    def objective(self, theta, y, T, fa, fa_args):
        # Make sure there is no nan in theta
        if np.any(np.isnan(theta)): return float('inf')
    
        # Get params from theta
        arrival_params, branching_params, rho = self._get_params_from_theta(theta, T)
        
        return -fa.forward(y, arrival_params, branching_params, rho, *fa_args)[0]

    def mle(self, y, T=None, theta0=None, bounds=None, fa=pgffa, n_max=-1):
        # For PGFFA, make sure Poisson arrival and binomial branching
        if fa is pgffa:
            assert self.arrival_dist is arrival.poisson and \
                   self.branching_dist is branching.binom, 'Error: \
                   PGFFA only supports Poisson arrival and binomial branching'

        # Set defaults
        if T is None: T = np.arange(len(y))
        if theta0 is None: theta0 = self._default_theta0(y, T)
        if bounds is None: bounds = self._default_bounds(y, T)
        if fa is truncatedfa and n_max == -1: n_max = np.max(y) * 5

        # Call the optimizer
        fa_args = tuple() if fa is pgffa else (self.arrival_dist.pmf, self.branching_dist.pmf, n_max)
        objective_args = (y, T, fa, fa_args)
        theta, fmin, info = optimize.fmin_l_bfgs_b(self.objective, theta0,
            args=objective_args, approx_grad=True, bounds=bounds)        
        params = self.theta2dict(theta)

        return params, fmin, info

    def _default_theta0(self, y, T):
        arrival_params = self.arrival_dist.init_hyperparams(y, T)
        branching_params = self.branching_dist.init_hyperparams(y, T)
        rho = 0.5
        return np.hstack((arrival_params, branching_params, rho))

    def _default_bounds(self, y, T):
        arrival_bounds = self.arrival_dist.hyperparam_bounds(y, T)
        branching_bounds = self.branching_dist.hyperparam_bounds(y, T)
        rho_bounds = [(0, 1)]
        return arrival_bounds + branching_bounds + rho_bounds
    
    def _get_params_from_theta(self, theta, T):
        K = len(T)
        n_arrival_params = len(self.arrival_dist.hyperparam_names)
        arrival_hyperparams = theta[:n_arrival_params]
        branching_hyperparams = theta[n_arrival_params:-1]
        rho = theta[-1]

        arrival_params = self.arrival_dist.expand_params(arrival_hyperparams, T)
        branching_params = self.branching_dist.expand_params(branching_hyperparams, T)
        rho = [rho] * K
        return arrival_params, branching_params, rho

    def theta2dict(self, theta):
        arrival_names = self.arrival_dist.hyperparam_names
        branching_names = self.branching_dist.hyperparam_names
        names = arrival_names + branching_names + ['rho']
        params = {name: value for name, value in zip(names, theta)}
        return params

    def generate_data(self, arrival_hyperparams, branching_hyperparams, rho, T):
        K = len(T)
        m = self.arrival_dist.generate_data(arrival_hyperparams, T)
        n, y, z = np.zeros(K, dtype=int), np.zeros(K, dtype=int), np.zeros(K-1, dtype=int)
        delta = self.branching_dist.hyperparams_to_params(branching_hyperparams, T)
        for k in range(K):
            n[k] = m[k] + z[k-1] if k > 0 else m[k]
            if k < K-1: z[k] = self.branching_dist.generate_next(n[k], delta[k])
        y = stats.binom.rvs(n, rho)  

        return y
