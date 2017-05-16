import numpy as np
from scipy import stats

import generatingfunctions
import gdualforward
import ngdualforward
import truncatedfa

# true params
Lambda_gen  = 10000 * np.array([0.0257, 0.1163, 0.2104, 0.1504, 0.0428]).reshape(-1,1)
Delta_gen   = 2 * 0.2636 * np.ones(5).reshape(-1,1)
Rho_gen     = 0.5 * np.ones(5)
# Lambda_gen  = 1500 * np.array([0.0257, 0.05, 0.1163, 0.15, 0.2104, 0.17, 0.1504, 0.07, 0.0428]).reshape(-1,1)
# Delta_gen   = 2 * np.array([0.2636, 0.2636, 0.2636, 0.2636, 0.2636, 0.2636, 0.2636, 0.2636]).reshape(-1,1)
# Rho_gen     = 0.5 * np.ones(9)

K = Lambda_gen.shape[0]

sample_counts = True

# configure distributions
arrival   = 'poisson'
offspring = 'bernoulli'
if arrival == 'poisson':
    arrival_distn     = stats.poisson
    arrival_pgf       = generatingfunctions.poisson_pgf
    arrival_liftedpgf = generatingfunctions.poisson_gdual
    arrival_normliftedpgf = generatingfunctions.poisson_ngdual
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
    offspring_normliftedpgf = generatingfunctions.bernoulli_ngdual
elif offspring == 'poisson':
    offspring_distn     = stats.poisson
    offspring_pgf       = generatingfunctions.poisson_pgf
    offspring_liftedpgf = generatingfunctions.poisson_gdual
    offspring_normliftedpgf = generatingfunctions.poisson_ngdual

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
    y = [ 41, 138, 325, 377, 234]
    # y = [ 18,  50, 125, 203, 282, 271, 280, 213, 145]
    # y = [ 2,  9,  12,  14,  9]
    y = [45, 213, 454, 515, 352]

print y

# compute LL using test methods
Lambda_eval = Lambda_gen
Delta_eval  = Delta_gen
Rho_eval    = Rho_gen

Alpha = ngdualforward.ngdualforward(y,
                                    arrival_normliftedpgf,
                                    Lambda_eval,
                                    offspring_normliftedpgf,
                                    Delta_eval,
                                    Rho_eval,
                                    d=1)

ll = Alpha[-1][0] + np.log(Alpha[-1][1][0])
print "LL from ngdual algorithm:  ", ll

# Alpha, logZ = gdualforward.gdualforward(y,
#                                         arrival_liftedpgf,
#                                         Lambda_eval,
#                                         offspring_liftedpgf,
#                                         Delta_eval,
#                                         Rho_eval,
#                                         d=1)
# ll = np.log(Alpha[-1][0]) + np.sum(logZ)
#
# print "LL from gdual algorithm:  ", ll

# Alpha, logZ = gdualforward.gdualforward_original(y,
#                                         arrival_liftedpgf,
#                                         Lambda_eval,
#                                         offspring_liftedpgf,
#                                         Delta_eval,
#                                         Rho_eval,
#                                         d=1)
# ll = np.log(Alpha[-1][0]) + np.sum(logZ)
#
# print "LL from original gdual algorithm:  ", ll
#
# Alpha = gdualforward.gdualforward_unnormalized(y,
#                                         arrival_liftedpgf,
#                                         Lambda_eval,
#                                         offspring_liftedpgf,
#                                         Delta_eval,
#                                         Rho_eval,
#                                         d=1)
# ll = np.log(Alpha[-1][0])
#
# print "LL from unnormalized algorithm: ", ll

Alpha_tfa, z_tfa = truncatedfa.truncated_forward(arrival_distn,
                                                 Lambda_eval,
                                                 truncatedfa.binomial_branching,
                                                 Delta_eval,
                                                 Rho_eval,
                                                 y,
                                                 n_max=1000)

ll      = truncatedfa.likelihood(z_tfa, log=True)
print "LL from truncated algorithm:  ", ll