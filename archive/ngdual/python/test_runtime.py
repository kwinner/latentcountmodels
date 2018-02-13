import numpy as np
from scipy import stats
import time
import matplotlib.pyplot as plt
import pickle

import generatingfunctions
import ngdualforward
import truncatedfa
import pgffa_kev as pgffa

def runtime_vs_lambda_eval():
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
    runtime_pgffa       = np.zeros([nsteps_lambda, nrep_each, nrep_overall])
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


def runtime_vs_rho_eval():
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
    K = 5
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
    runtime_pgffa       = np.zeros([nsteps_rho, nrep_each, nrep_overall])
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


def runtime_vs_rho_gen():
    # test parameters
    # nrep_overall = 50       # number of times to repeat the entire experiment
    nrep_overall = 10
    nrep_each    = 1      # number of times to repeat each evaluation

    n_Rho_test   = 19
    Rho_test_min = 0.05
    Rho_test_max = 0.95
    Rho_test = np.linspace(Rho_test_min, Rho_test_max, n_Rho_test)

    epsilon_eq    = 1e-6   # how close must the truncated algorithm be to ngdual in order to stop
    epsilon_conv  = 1e-10  # if ll changes by < epsilon_conv, then stop
    N_LIMIT       = 1000   # absolute upper bound on n_max
    verbose       = 5

    # other true params
    K = 5
    # Lambda_true = 75.0
    Delta_true  = 0.2636
    # Delta_true = 0.4

    # Lambda_gen  = Lambda_true * np.ones(K).reshape(-1,1)
    Delta_gen   = Delta_true  * np.ones(K).reshape(-1,1)

    # override some parameters explicitly
    # Lambda_gen = np.array([5.13, 23.26, 42.08, 30.09, 8.56]).reshape(-1,1)
    Lambda_gen = 5. * np.array([5.13, 23.26, 42.08, 30.09, 8.56]).reshape(-1, 1)
    # Lambda_gen = np.array([0.0398, 10.26, 74.93, 25.13, 4.14]).reshape(-1, 1)
    # Lambda_gen = np.array([80.0, 0., 0., 0., 0.]).reshape(-1, 1)
    assert Lambda_gen.shape[0] == K

    # whether to randomly sample counts or use pregenerated ones (set below)
    sample_counts = True
    if sample_counts == False:
        y = [0, 9, 12, 14, 9]

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

    runtime_ngdual      = np.zeros([n_Rho_test, nrep_each, nrep_overall])
    runtime_pgffa       = np.zeros([n_Rho_test, nrep_each, nrep_overall])
    runtime_trunc_best  = np.zeros([n_Rho_test, nrep_each, nrep_overall])
    runtime_trunc_total = np.zeros([n_Rho_test, nrep_each, nrep_overall])
    n_max_trunc         = np.zeros([n_Rho_test, nrep_each, nrep_overall], dtype=np.int64)
    for i_overall in xrange(0, nrep_overall):
        if verbose >= 1: print "Experiment", i_overall + 1, "of", nrep_overall

        for i_rho in xrange(0, n_Rho_test):
            Rho_gen = Rho_test[i_rho] * np.ones(K)

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

            if verbose >= 2: print " Rho =", Rho_test[i_rho], " y =", y

            for i_rep in xrange(0, nrep_each):
                if verbose >= 3: print "  Repetition", i_rep + 1, "of", nrep_each

                # Lambda_eval = Lambda_test[i_lambda] * np.ones(K).reshape(-1,1)
                Lambda_eval = Lambda_gen
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
                runtime_ngdual[i_rho, i_rep, i_overall] = time.clock() - t_start
                if verbose >= 5: print "   LL from ngdual algorithm:", ll_ngdual

                t_start = time.clock()
                a, b, f = pgffa.pgf_forward(Lambda_eval.reshape(-1),
                                            Rho_eval.reshape(-1),
                                            Delta_eval.reshape(-1),
                                            np.array(y, dtype=np.int32))

                ll_pgffa = pgffa.likelihood(a, b, f)
                runtime_pgffa[i_rho, i_rep, i_overall] = time.clock() - t_start
                if verbose >= 5: print "   LL from pgffa:", ll_pgffa

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

    return runtime_ngdual, runtime_pgffa, runtime_trunc_best, runtime_trunc_total, n_max_trunc, Rho_test


def runtime_vs_Lambda_gen():
    # test parameters
    # nrep_overall = 50       # number of times to repeat the entire experiment
    nrep_overall = 10
    nrep_each    = 1      # number of times to repeat each evaluation

    # n_mag_test    = 10
    # mag_test_min  = 0.5
    # mag_test_max  = 2000
    # mag_test = np.linspace(mag_test_min, mag_test_max, n_mag_test)
    mag_test = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    n_mag_test = len(mag_test)

    epsilon_eq    = 1e-6   # how close must the truncated algorithm be to ngdual in order to stop
    epsilon_conv  = 1e-10  # if ll changes by < epsilon_conv, then stop
    N_LIMIT       = 1000   # absolute upper bound on n_max
    verbose       = 5

    # other true params
    K = 5
    Delta_true  = 0.2636
    # Delta_true = 0.4
    Rho_true = 0.5

    # Lambda_gen  = Lambda_true * np.ones(K).reshape(-1,1)
    Delta_gen   = Delta_true  * np.ones(K).reshape(-1,1)
    Rho_gen     = Rho_true    * np.ones(K).reshape(-1,1)

    # override some parameters explicitly
    Lambda_gen = np.array([5.13, 23.26, 42.08, 30.09, 8.56]).reshape(-1,1)
    # Lambda_gen = np.array([0.0398, 10.26, 74.93, 25.13, 4.14]).reshape(-1, 1)
    # Lambda_gen = np.array([80.0, 0., 0., 0., 0.]).reshape(-1, 1)
    assert Lambda_gen.shape[0] == K

    # whether to randomly sample counts or use pregenerated ones (set below)
    sample_counts = True
    if sample_counts == False:
        y = [0, 9, 12, 14, 9]

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

    runtime_ngdual      = np.zeros([n_mag_test, nrep_each, nrep_overall])
    runtime_pgffa       = np.zeros([n_mag_test, nrep_each, nrep_overall])
    runtime_trunc_best  = np.zeros([n_mag_test, nrep_each, nrep_overall])
    runtime_trunc_total = np.zeros([n_mag_test, nrep_each, nrep_overall])
    n_max_trunc         = np.zeros([n_mag_test, nrep_each, nrep_overall], dtype=np.int64)
    for i_overall in xrange(0, nrep_overall):
        if verbose >= 1: print "Experiment", i_overall + 1, "of", nrep_overall

        for i_mag in xrange(0, n_mag_test):
            Mag_gen = mag_test[i_mag]
            Lambda_gen_iter = Mag_gen * Lambda_gen

            # sample counts
            if sample_counts == True:
                # sample data
                N = np.empty(K, dtype=np.int64)
                y = np.empty(K, dtype=np.int64)
                for k in xrange(K):
                    # sample immigrants
                    N[k] = arrival_distn.rvs(Lambda_gen_iter[k, :])
                    if k > 0:
                        # sample offspring
                        for i in xrange(N[k - 1]):
                            N[k] += offspring_distn.rvs(Delta_gen[k - 1, :])
                    # sample observation
                    y[k] = stats.binom.rvs(N[k], Rho_gen[k])

            if verbose >= 2: print " Mag =", mag_test[i_mag], " y =", y

            for i_rep in xrange(0, nrep_each):
                if verbose >= 3: print "  Repetition", i_rep + 1, "of", nrep_each

                # Lambda_eval = Lambda_test[i_lambda] * np.ones(K).reshape(-1,1)
                Lambda_eval = Lambda_gen_iter
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
                runtime_ngdual[i_mag, i_rep, i_overall] = time.clock() - t_start
                if verbose >= 5: print "   LL from ngdual algorithm:", ll_ngdual

                t_start = time.clock()
                a, b, f = pgffa.pgf_forward(Lambda_eval.reshape(-1),
                                            Rho_eval.reshape(-1),
                                            Delta_eval.reshape(-1),
                                            np.array(y, dtype=np.int32))

                ll_pgffa = pgffa.likelihood(a, b, f)
                runtime_pgffa[i_mag, i_rep, i_overall] = time.clock() - t_start
                if verbose >= 5: print "   LL from pgffa:", ll_pgffa

                ll_trunc = float('inf')
                ll_delta = float('inf')  # used to track trunc convergence
                if i_rep == 0:
                    n_max_trunc[i_mag, i_rep, i_overall] = max(y) - 1
                else:
                    n_max_trunc[i_mag, i_rep, i_overall] = max(max(y) - 1, n_max_trunc[i_mag, 0, i_overall] - 5)
                t_start = time.clock()
                while abs(ll_trunc - ll_ngdual) >= epsilon_eq and \
                      ll_delta >= epsilon_conv and \
                      n_max_trunc[i_mag, i_rep, i_overall] <= N_LIMIT:
                    n_max_trunc[i_mag, i_rep, i_overall] += 1
                    t_loop = time.clock()
                    Alpha_trunc, z = truncatedfa.truncated_forward(arrival_distn,
                                                                   Lambda_eval,
                                                                   branching_proc,
                                                                   Delta_eval,
                                                                   Rho_eval,
                                                                   y,
                                                                   n_max=n_max_trunc[i_mag, i_rep, i_overall])
                    ll_iter = truncatedfa.likelihood(z, log=True)
                    ll_delta = abs(ll_trunc - ll_iter)
                    ll_trunc = ll_iter
                    runtime_trunc_best[i_mag, i_rep, i_overall] = time.clock() - t_loop
                runtime_trunc_total[i_mag, i_rep, i_overall] = time.clock() - t_start

                if verbose >= 5: print "   LL from trunc algorithm:", ll_trunc, "n_max =", n_max_trunc[i_mag, i_rep, i_overall]

                if verbose >= 4: print "   ngdual:", runtime_ngdual[i_mag, i_rep, i_overall], ", trunc:", runtime_trunc_best[i_mag, i_rep, i_overall]

    return runtime_ngdual, runtime_pgffa, runtime_trunc_best, runtime_trunc_total, n_max_trunc, mag_test


def runtime_plot(runtime_ngdual, runtime_pgffa, runtime_trunc, x_vals):
    plotalpha = 1.0
    linewidth = 5.0
    elinewidth = 2.0
    capsize = 3.0

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    rt_mean_ngdual = np.mean(runtime_ngdual, axis=(1,2))
    rt_var_ngdual  = np.var(runtime_ngdual,  axis=(1,2))
    rt_mean_pgffa  = np.mean(runtime_pgffa, axis=(1,2))
    rt_var_pgffa   = np.var(runtime_pgffa,   axis=(1,2))
    rt_mean_trunc  = np.mean(runtime_trunc, axis=(1,2))
    rt_var_trunc   = np.var(runtime_trunc,   axis=(1,2))

    handle_trunc = plt.errorbar(np.ravel(x_vals),
                                np.ravel(rt_mean_trunc),
                                yerr=1.96 * np.sqrt(np.ravel(rt_var_trunc)) / np.sqrt(np.size(runtime_trunc)),
                                color="#EEB220", label=r"\texttt{Trunc}", alpha=plotalpha,
                                linewidth=linewidth, dashes=(3, 2),
                                elinewidth=elinewidth, capsize=capsize, capthick=elinewidth)  # ,linestyle=':')
    handle_pgffa = plt.errorbar(np.ravel(x_vals),
                                np.ravel(rt_mean_pgffa),
                                yerr=1.96 * np.sqrt(np.ravel(rt_var_pgffa)) / np.sqrt(np.size(runtime_pgffa)),
                                color="#DA5319", label=r"\texttt{PGF-FORWARD}", alpha=plotalpha,
                                linewidth=linewidth, linestyle='--',
                                elinewidth=elinewidth, capsize=capsize, capthick=elinewidth)
    handle_ngdual = plt.errorbar(np.ravel(x_vals),
                                 np.ravel(rt_mean_ngdual),
                                 yerr=1.96 * np.sqrt(np.ravel(rt_var_ngdual)) / np.sqrt(np.size(runtime_trunc)),
                                 color="#0072BE", label=r"\texttt{GDUAL-FORWARD}", alpha=plotalpha,
                                 linewidth=linewidth, linestyle='-',
                                 elinewidth=elinewidth, capsize=capsize, capthick=elinewidth)

    plt.legend(loc=2, fontsize=15)
    # plt.legend(loc=6, fontsize=15)
    plt.ylabel('Runtime (s)', fontsize=18)
    plt.xlabel(r'$\rho$', fontsize=20)
    # plt.title(r'Runtime vs $\rho$ for $\Lambda = %s$' % str(lambda_target), fontsize=18)
    plt.title(r'Runtime vs $\rho$ in PHMM', fontsize=18)
    # plt.title(r'Runtime vs $\rho$ in PHMM: $\Lambda = %s, \sigma = %s$' % (str(lambda_target), 2), fontsize=18)
    # plt.title(r'Runtime vs $\rho$ in Branching NMix: $\Lambda = %s, \delta = %s$' % (str(lambda_target), 0.4), fontsize=18)
    plt.xlim(np.min(x_vals), np.max(x_vals))
    # plt.ylim(0, max((np.max(rt_mean_trunc), np.max(rt_mean_pgffa), np.max(rt_mean_ngdual))))

    plt.show(block=True)


# runtime_ngdual, runtime_trunc_best, runtime_trunc_total, n_max_trunc = runtime_vs_lambda()
# runtime_ngdual, runtime_trunc_best, runtime_trunc_total, n_max_trunc = runtime_vs_rho_eval()
# runtime_ngdual, runtime_pgffa, runtime_trunc_best, runtime_trunc_total, n_max_trunc, x = runtime_vs_rho_gen()
runtime_ngdual, runtime_pgffa, runtime_trunc_best, runtime_trunc_total, n_max_trunc, x = runtime_vs_Lambda_gen()

# record = pickle.load(open("/Users/kwinner/Work/Data/Results/shannon_icml_2017/data2c.pickle"))
# runtime_ngdual = record['runtime_ngdual']
# runtime_pgffa = record['runtime_pgffa']
# runtime_trunc_best = record['runtime_trunc_best']
# Rho_test = record['rho_space']

runtime_plot(runtime_ngdual, runtime_pgffa, runtime_trunc_best, x)