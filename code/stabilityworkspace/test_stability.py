import numpy as np
from scipy import stats

import generatingfunctions
import gdualforward

# true params
Lambda_gen  = 100 * np.array([0.0257, 0.1163, 0.2104, 0.1504, 0.0428]).reshape(-1,1)
Delta_gen   = 2 * np.array([0.2636, 0.2636, 0.2636, 0.2636]).reshape(-1,1)
Rho_gen     = 0.5 * np.ones(5)

K = Lambda_gen.shape[0]

sample_counts = False

# configure distributions
arrival   = 'poisson'
offspring = 'bernoulli'
if arrival == 'poisson':
    arrival_distn     = stats.poisson
    arrival_pgf       = generatingfunctions.poisson_pgf
    arrival_liftedpgf = generatingfunctions.poisson_gdual
elif arrival == 'negbin':
    arrival_distn     = stats.nbinom
    arrival_pgf       = generatingfunctions.negbin_pgf
    arrival_liftedpgf = generatingfunctions.negbin_gdual
elif arrival == 'logser':
    arrival_distn     = stats.logser
    arrival_pgf       = generatingfunctions.logarithmic_pgf
    arrival_liftedpgf = generatingfunctions.logarithmic_gdual
elif arrival == 'geom':
    arrival_distn     = stats.geom
    arrival_pgf       = generatingfunctions.geometric_pgf
    arrival_liftedpgf = generatingfunctions.geometric_gdual

if offspring == 'bernoulli':
    offspring_distn     = stats.bernoulli
    offspring_pgf       = generatingfunctions.bernoulli_pgf
    offspring_liftedpgf = generatingfunctions.bernoulli_gdual
elif offspring == 'poisson':
    offspring_distn     = stats.poisson
    offspring_pgf       = generatingfunctions.poisson_pgf
    offspring_liftedpgf = generatingfunctions.poisson_gdual

if sample_counts == True:
    # sample data
    N = np.empty(K, dtype=np.int64)
    y = np.empty(K, dtype=np.int64)
    for k in xrange(K):
        # sample immigrants
        N[k] = arrival_distn.rvs(Lambda_gen[k, :])
        if k > 0:
            # sample offspring
            for i in xrange(N[k - 1]):
                N[k] += offspring_distn.rvs(Delta_gen[k - 1, :])
        # sample observation
        y[k] = stats.binom.rvs(N[k], Rho_gen[k])
else:
    # N = [22, 104, 214, 240, 162]
    # y = [11, 55,  115, 112, 72]
    y = [11,  89, 221, 231, 144]
    y = [ 2,  9,  12,  14,  9]

print y

# compute LL using test methods
Lambda_eval = Lambda_gen
Delta_eval  = Delta_gen
Rho_eval    = Rho_gen
Alpha, logZ = gdualforward.gdualforward2(y,
                                        arrival_liftedpgf,
                                        Lambda_eval,
                                        offspring_liftedpgf,
                                        Delta_eval,
                                        Rho_eval,
                                        d=1)
ll = np.log(Alpha[-1][0]) + np.sum(logZ)

print "LL from normalized algorithm:", ll

Alpha, logZ = gdualforward.gdualforward(y,
                                        arrival_liftedpgf,
                                        Lambda_eval,
                                        offspring_liftedpgf,
                                        Delta_eval,
                                        Rho_eval,
                                        d=1)
ll = np.log(Alpha[-1][0]) + np.sum(logZ)

print "LL from previous algorithm:  ", ll