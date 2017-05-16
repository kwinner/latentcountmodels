import numpy as np
from scipy import stats
import time

import generatingfunctions
import ngdualforward
import truncatedfa

def runtime_vs_lambda():
    # test parameters
    nrep_overall = 5       # number of times to repeat the entire experiment
    nrep_each    = 10     # number of times to repeat each evaluation

    nsteps_lambda = 50     # number of different values of Lambda to iterate over
    window_lambda = 0.5    # width of a window to test around the true value of Lambda (as a percent of Lambda_true)

    epsilon_eq    = 1e-6   # how close must the truncated algorithm be to ngdual in order to stop
    epsilon_conv  = 1e-10  # if ll changes by < epsilon_conv, then stop
    N_LIMIT       = 1000   # absolute upper bound on n_max
    verbose       = 4

    # true params
    K = 5
    Lambda_true = 21.0
    Delta_true  = 0.5272
    Rho_true    = 0.75

    Lambda_gen  = Lambda_true * np.ones(K).reshape(-1,1)
    Delta_gen   = Delta_true  * np.ones(K).reshape(-1,1)
    Rho_gen     = Rho_true    * np.ones(K)

    # whether to randomly sample counts or use pregenerated ones (set below)
    sample_counts = True
    if sample_counts == False:
        y = [2, 9, 12, 14, 9]

    # configure distributions
    arrival   = 'poisson'
    offspring = 'bernoulli'
    if arrival == 'poisson':
        arrival_distn         = stats.poisson
        arrival_pgf           = generatingfunctions.poisson_pgf
        arrival_liftedpgf     = generatingfunctions.poisson_gdual
        arrival_normliftedpgf = generatingfunctions.poisson_ngdual
    elif arrival == 'negbin':
        arrival_distn         = stats.nbinom
        arrival_pgf           = generatingfunctions.negbin_pgf
        arrival_liftedpgf     = generatingfunctions.negbin_gdual
    elif arrival == 'logser':
        arrival_distn         = stats.logser
        arrival_pgf           = generatingfunctions.logarithmic_pgf
        arrival_liftedpgf     = generatingfunctions.logarithmic_gdual
    elif arrival == 'geom':
        arrival_distn         = stats.geom
        arrival_pgf           = generatingfunctions.geometric_pgf
        arrival_liftedpgf     = generatingfunctions.geometric_gdual

    if offspring == 'bernoulli':
        offspring_distn         = stats.bernoulli
        offspring_pgf           = generatingfunctions.bernoulli_pgf
        offspring_liftedpgf     = generatingfunctions.bernoulli_gdual
        offspring_normliftedpgf = generatingfunctions.bernoulli_ngdual
        branching_proc          = truncatedfa.binomial_branching
    elif offspring == 'poisson':
        offspring_distn         = stats.poisson
        offspring_pgf           = generatingfunctions.poisson_pgf
        offspring_liftedpgf     = generatingfunctions.poisson_gdual
        offspring_normliftedpgf = generatingfunctions.poisson_ngdual
        branching_proc          = truncatedfa.poisson_branching


    Lambda_min = Lambda_true - (Lambda_true * 0.5 * window_lambda)
    Lambda_max = Lambda_true + (Lambda_true * 0.5 * window_lambda)
    Lambda_test = np.linspace(Lambda_min, Lambda_max, num=nsteps_lambda)

    runtime_ngdual      = np.zeros([nsteps_lambda, nrep_each, nrep_overall])
    runtime_trunc_best  = np.zeros([nsteps_lambda, nrep_each, nrep_overall])
    runtime_trunc_total = np.zeros([nsteps_lambda, nrep_each, nrep_overall])
    n_max_trunc         = np.zeros([nsteps_lambda, nrep_each, nrep_overall])
    for i_overall in xrange(0, nrep_overall):
        # sample counts
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

        if verbose >= 1: print "Experiment", i_overall + 1, "of", nrep_overall, ", y =", y

        for i_rep in xrange(0, nrep_each):
            if verbose >= 2: print " Repetition", i_rep + 1, "of", nrep_each

            for i_lambda in xrange(0, Lambda_test.shape[0]):
                if verbose >= 3: print "  Lambda: ", Lambda_test[i_lambda]
                Lambda_eval = Lambda_test[i_lambda] * np.ones(K).reshape(-1,1)
                # Lambda_eval = Lambda_gen
                Delta_eval  = Delta_gen
                Rho_eval    = Rho_gen

                t_start = time.clock()
                Alpha_ngdual = ngdualforward.ngdualforward(y,
                                                           arrival_normliftedpgf,
                                                           Lambda_eval,
                                                           offspring_normliftedpgf,
                                                           Delta_eval,
                                                           Rho_eval,
                                                           d=1)

                ll_ngdual = Alpha_ngdual[-1][0] + np.log(Alpha_ngdual[-1][1][0])
                runtime_ngdual[i_lambda, i_rep, i_overall] = time.clock() - t_start
                if verbose >= 5: print "   LL from ngdual algorithm:", ll_ngdual

                ll_trunc = float('inf')
                ll_delta = float('inf')  # used to track trunc convergence
                if i_rep == 0:
                    n_max_trunc[i_lambda, i_rep, i_overall] = max(y) - 1
                else:
                    n_max_trunc[i_lambda, i_rep, i_overall] = max(max(y) - 1, n_max_trunc[i_lambda, 0, i_overall] - 5)
                t_start = time.clock()
                while abs(ll_trunc - ll_ngdual) >= epsilon_eq and \
                      ll_delta >= epsilon_conv and \
                      n_max_trunc[i_lambda, i_rep, i_overall] <= N_LIMIT:
                    n_max_trunc[i_lambda, i_rep, i_overall] += 1
                    t_loop = time.clock()
                    Alpha_trunc, z = truncatedfa.truncated_forward(arrival_distn,
                                                                   Lambda_eval,
                                                                   branching_proc,
                                                                   Delta_eval,
                                                                   Rho_eval,
                                                                   y,
                                                                   n_max=n_max_trunc[i_lambda, i_rep, i_overall])
                    ll_iter = truncatedfa.likelihood(z, log=True)
                    ll_delta = abs(ll_trunc - ll_iter)
                    ll_trunc = ll_iter
                    runtime_trunc_best[i_lambda, i_rep, i_overall] = time.clock() - t_loop
                runtime_trunc_total[i_lambda, i_rep, i_overall] = time.clock() - t_start

                if verbose >= 5: print "   LL from trunc algorithm:", ll_trunc, "n_max =", n_max_trunc[i_lambda, i_rep, i_overall]
                if verbose >= 4: print "   ngdual:", runtime_ngdual[i_lambda, i_rep, i_overall], ", trunc:", runtime_trunc_best[i_lambda, i_rep, i_overall]

    return runtime_ngdual, runtime_trunc_best, runtime_trunc_total, n_max_trunc


def runtime_vs_rho():
    # test parameters
    nrep_overall = 5       # number of times to repeat the entire experiment
    nrep_each    = 10      # number of times to repeat each evaluation

    nsteps_rho = 10        # number of different values of Rho to iterate over
    window_rho = 0.75       # width of a window to test around the true value of Lambda (as a percent of Lambda_true)

    epsilon_eq    = 1e-6   # how close must the truncated algorithm be to ngdual in order to stop
    epsilon_conv  = 1e-10  # if ll changes by < epsilon_conv, then stop
    N_LIMIT       = 1000   # absolute upper bound on n_max
    verbose       = 4

    # true params
    K = 6
    Lambda_true = 75.0
    Delta_true  = 0.75
    Rho_true    = 0.50

    Lambda_gen  = Lambda_true * np.ones(K).reshape(-1,1)
    Delta_gen   = Delta_true  * np.ones(K).reshape(-1,1)
    Rho_gen     = Rho_true    * np.ones(K)

    # whether to randomly sample counts or use pregenerated ones (set below)
    sample_counts = True
    if sample_counts == False:
        y = [2, 9, 12, 14, 9]

    # configure distributions
    arrival   = 'poisson'
    offspring = 'bernoulli'
    if arrival == 'poisson':
        arrival_distn         = stats.poisson
        arrival_pgf           = generatingfunctions.poisson_pgf
        arrival_liftedpgf     = generatingfunctions.poisson_gdual
        arrival_normliftedpgf = generatingfunctions.poisson_ngdual
    elif arrival == 'negbin':
        arrival_distn         = stats.nbinom
        arrival_pgf           = generatingfunctions.negbin_pgf
        arrival_liftedpgf     = generatingfunctions.negbin_gdual
    elif arrival == 'logser':
        arrival_distn         = stats.logser
        arrival_pgf           = generatingfunctions.logarithmic_pgf
        arrival_liftedpgf     = generatingfunctions.logarithmic_gdual
    elif arrival == 'geom':
        arrival_distn         = stats.geom
        arrival_pgf           = generatingfunctions.geometric_pgf
        arrival_liftedpgf     = generatingfunctions.geometric_gdual

    if offspring == 'bernoulli':
        offspring_distn         = stats.bernoulli
        offspring_pgf           = generatingfunctions.bernoulli_pgf
        offspring_liftedpgf     = generatingfunctions.bernoulli_gdual
        offspring_normliftedpgf = generatingfunctions.bernoulli_ngdual
        branching_proc          = truncatedfa.binomial_branching
    elif offspring == 'poisson':
        offspring_distn         = stats.poisson
        offspring_pgf           = generatingfunctions.poisson_pgf
        offspring_liftedpgf     = generatingfunctions.poisson_gdual
        offspring_normliftedpgf = generatingfunctions.poisson_ngdual
        branching_proc          = truncatedfa.poisson_branching


    Rho_min = Rho_true - (0.5 * window_rho)
    Rho_max = Rho_true + (0.5 * window_rho)
    Rho_test = np.linspace(Rho_min, Rho_max, num=nsteps_rho)

    runtime_ngdual      = np.zeros([nsteps_rho, nrep_each, nrep_overall])
    runtime_trunc_best  = np.zeros([nsteps_rho, nrep_each, nrep_overall])
    runtime_trunc_total = np.zeros([nsteps_rho, nrep_each, nrep_overall])
    n_max_trunc         = np.zeros([nsteps_rho, nrep_each, nrep_overall])
    for i_overall in xrange(0, nrep_overall):
        # sample counts
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

        if verbose >= 1: print "Experiment", i_overall + 1, "of", nrep_overall, ", y =", y

        for i_rep in xrange(0, nrep_each):
            if verbose >= 2: print " Repetition", i_rep + 1, "of", nrep_each

            for i_rho in xrange(0, Rho_test.shape[0]):
                if verbose >= 3: print "  Rho: ", Rho_test[i_rho]
                # Lambda_eval = Lambda_test[i_lambda] * np.ones(K).reshape(-1,1)
                Lambda_eval = Lambda_gen
                Delta_eval  = Delta_gen
                Rho_eval    = Rho_test[i_rho] * np.ones(K).reshape(-1,1)
                # Rho_eval    = Rho_gen

                t_start = time.clock()
                Alpha_ngdual = ngdualforward.ngdualforward(y,
                                                           arrival_normliftedpgf,
                                                           Lambda_eval,
                                                           offspring_normliftedpgf,
                                                           Delta_eval,
                                                           Rho_eval,
                                                           d=1)

                ll_ngdual = Alpha_ngdual[-1][0] + np.log(Alpha_ngdual[-1][1][0])
                runtime_ngdual[i_rho, i_rep, i_overall] = time.clock() - t_start
                if verbose >= 5: print "   LL from ngdual algorithm:", ll_ngdual

                ll_trunc = float('inf')
                ll_delta = float('inf')  # used to track trunc convergence
                if i_rep == 0:
                    n_max_trunc[i_rho, i_rep, i_overall] = max(y) - 1
                else:
                    n_max_trunc[i_rho, i_rep, i_overall] = max(max(y) - 1, n_max_trunc[i_rho, 0, i_overall] - 5)
                t_start = time.clock()
                while abs(ll_trunc - ll_ngdual) >= epsilon_eq and \
                      ll_delta >= epsilon_conv and \
                      n_max_trunc[i_rho, i_rep, i_overall] <= N_LIMIT:
                    n_max_trunc[i_rho, i_rep, i_overall] += 1
                    t_loop = time.clock()
                    Alpha_trunc, z = truncatedfa.truncated_forward(arrival_distn,
                                                                   Lambda_eval,
                                                                   branching_proc,
                                                                   Delta_eval,
                                                                   Rho_eval,
                                                                   y,
                                                                   n_max=n_max_trunc[i_rho, i_rep, i_overall])
                    ll_iter = truncatedfa.likelihood(z, log=True)
                    ll_delta = abs(ll_trunc - ll_iter)
                    ll_trunc = ll_iter
                    runtime_trunc_best[i_rho, i_rep, i_overall] = time.clock() - t_loop
                runtime_trunc_total[i_rho, i_rep, i_overall] = time.clock() - t_start

                if verbose >= 5: print "   LL from trunc algorithm:", ll_trunc, "n_max =", n_max_trunc[i_rho, i_rep, i_overall]
                if verbose >= 4: print "   ngdual:", runtime_ngdual[i_rho, i_rep, i_overall], ", trunc:", runtime_trunc_best[i_rho, i_rep, i_overall]

    return runtime_ngdual, runtime_trunc_best, runtime_trunc_total, n_max_trunc


# runtime_ngdual, runtime_trunc_best, runtime_trunc_total, n_max_trunc = runtime_vs_lambda()
runtime_ngdual, runtime_trunc_best, runtime_trunc_total, n_max_trunc = runtime_vs_rho()