import os
import sys
from glob import glob
import pwd
import time
import cProfile
import pickle
import itertools

import numpy as np
from scipy import stats, integrate
import matplotlib.pyplot as plt

import gdual as gd
import forward as gdfwd
import truncatedfa as trfwd

NMAX_RECORD_DTYPE = np.dtype([('N_max', '<i4'), ('LL', '<f8')])

def stability_experiment(
        Lambda_gen  = 10 * np.array([0.0257, 0.1163, 0.2104, 0.1504, 0.0428]).reshape(-1, 1),
        Delta_gen   = np.array([0.2636, 0.2636, 0.2636, 0.2636]).reshape(-1, 1),
        Rho_gen     = 0.5 * np.ones(5),
        epsilon     = 1e-5, # allowable error in truncated fa
        n_reps      = 10,   # number of times to repeat the experiment
        N_init      = 0,    # initial value of N_max to use in each experiment. if N_init < max(y), it will be set to max(y)
        N_LIMIT     = 25000, # hard cap on the max value for the truncated algorithm
        silent      = False,
        arrival     = 'poisson',
        branch      = 'bernoulli',
        Lambda_eval = None,
        Delta_eval  = None,
        Rho_eval    = None
        ):
    if(Lambda_eval is None): Lambda_eval = Lambda_gen
    if(Delta_eval  is None): Delta_eval  = Delta_gen
    if(Rho_eval    is None): Rho_eval    = Rho_gen

    K = len(Lambda_gen)

    # record of sampled data
    N = np.zeros((n_reps, K), dtype=np.int32)
    y = np.zeros((n_reps, K), dtype=np.int32)

    # record of computed final LL
    LL_trunc_dir = np.zeros(n_reps)
    LL_trunc_fft = np.zeros(n_reps)
    LL_gdual     = np.zeros(n_reps)
    LL_lsgdual   = np.zeros(n_reps)

    # record of runtime
    RT_trunc_dir = np.zeros(n_reps)
    RT_trunc_fft = np.zeros(n_reps)
    RT_gdual     = np.zeros(n_reps)
    RT_lsgdual   = np.zeros(n_reps)

    # record of LL vs N_max (of truncated alg) for each rep
    N_max_trunc_dir = [np.array([], dtype = NMAX_RECORD_DTYPE) for i in range(n_reps)]
    N_max_trunc_fft = [np.array([], dtype = NMAX_RECORD_DTYPE) for i in range(n_reps)]

    # set the arrival pmf (for data generation, truncated fa) and pgf (for pgffa)
    if arrival == 'poisson':
        arrival_pmf = stats.poisson
        arrival_pgf = gdfwd.poisson_pgf
    elif arrival == 'negbin':
        arrival_pmf = stats.nbinom
        arrival_pgf = gdfwd.negbin_pgf

    # same as above for branching distributions
    # note: truncfa uses a truncated pmf for the collective branching distribution (i.e. bernoulli -> truncated binomial)
    #       denoted here the trpmf, which is an abuse of notation
    if branch == 'poisson':
        branch_pmf   = stats.poisson
        branch_trpmf = trfwd.poisson_branching
        branch_pgf   =  gdfwd.poisson_pgf
    elif branch == 'bernoulli' or branch == 'binomial':
        branch_pmf   = stats.bernoulli
        branch_trpmf = trfwd.binomial_branching
        branch_pgf   = gdfwd.bernoulli_pgf

    for rep in range(n_reps):
        if not silent:
            if rep != 0:
                print("")
            print("#####")
            print("Iteration %d of %d" % (rep+1, n_reps))

        # sample data
        for i in range(0, K):
            if i == 0:
                N[rep, i] = arrival_pmf.rvs(Lambda_gen[i])
            else:
                if branch == 'bernoulli':
                    N[rep, i] = arrival_pmf.rvs(Lambda_gen[i]) + stats.binom.rvs(N[rep, i - 1], Delta_gen[i - 1])
                elif branch == 'poisson':
                    N[rep, i] = arrival_pmf.rvs(Lambda_gen[i]) + stats.poisson.rvs(N[rep, i - 1] * Delta_gen[i - 1])
            y[rep, i] = stats.binom.rvs(N[rep, i], Rho_gen[i])

        if N_init < np.max(y[rep, :]):
            N_init = np.max(y[rep, :])

        if not silent:
            print("y: %s" % str(y[rep, :]))

        start_time = time.process_time()
        LL_lsgdual[rep], _, _ = gdfwd.forward(y[rep, :],
                                              arrival_pgf,
                                              Lambda_eval,
                                              branch_pgf,
                                              Delta_eval,
                                              Rho_eval,
                                              GDualType=gd.LSGDual,
                                              d=0)
        RT_lsgdual[rep] = time.process_time() - start_time
        if not silent:
            print("LL (lsgd): %f, time = %f" % (LL_lsgdual[rep], RT_lsgdual[rep]))

        start_time = time.process_time()
        LL_gdual[rep],   _, _ = gdfwd.forward(y[rep, :],
                                              arrival_pgf,
                                              Lambda_eval,
                                              branch_pgf,
                                              Delta_eval,
                                              Rho_eval,
                                              GDualType=gd.GDual, # Note: different from above (GDual vs LSGDual)
                                              d=0)
        RT_gdual[rep] = time.process_time() - start_time
        if not silent:
            print("LL (gd): %f, diff = %f, time = %f" % (LL_gdual[rep], np.abs(LL_gdual[rep] - LL_lsgdual[rep]), RT_gdual[rep]))

        # stability check on the flavors of the truncated fwd algorithm requires tuning an appropriate threshold parameter N_max
        # in both cases below, we do iterative doubling on N_max beginning at N_init until the relative difference between successive
        # iterations is less than epsilon, a parameter
        # the record of N_max values and corresponding LL is logged for debugging purposes
        # extended to stop the iterative doubling if the relative difference between trfwd and either pgffa implementation converges also
        N_max = np.int32(N_init)
        delta = np.inf # delta (unqualified) is the smallest delta between either previous method or the previous iteration
        while N_max <= N_LIMIT and delta > epsilon:
            # run the trial
            start_time = time.process_time()
            _, logz = trfwd.truncated_forward(arrival_pmf,
                                                  Lambda_eval,
                                                  branch_trpmf,
                                                  Delta_eval,
                                                  Rho_eval,
                                                  y[rep, :],
                                                  N_max,
                                                  silent=False,
                                                  conv_method='direct')
            RT_trunc_dir[rep] = time.process_time() - start_time
            # compute LL
            LL = np.sum(logz)

            # compute delta, the relative difference in LL between this iteration and the previous iteration
            if N_max_trunc_dir[rep].size == 0:
                delta_self = np.inf
            else:
                delta_self = np.abs((LL - N_max_trunc_dir[rep][-1]['LL']) / max(LL, N_max_trunc_dir[rep][-1]['LL']))
            delta_lsgd = np.abs((LL - LL_lsgdual[rep]) / max(LL, LL_lsgdual[rep]))
            delta_gd   = np.abs((LL - LL_gdual[rep])   / max(LL, LL_gdual[rep]))

            delta = np.nanmin([delta_self, delta_lsgd, delta_gd])

            # log the (N_max, LL) tuple
            N_max_trunc_dir[rep] = np.append(N_max_trunc_dir[rep], np.array((N_max, LL), dtype = NMAX_RECORD_DTYPE))

            if not silent:
                print("Trfwd.dir trial, N_Max = %d, LL = %f, delta = %f" % (N_max, LL, delta))

            # double for next iteration
            if N_max < N_LIMIT:
                N_max = N_max * 2

                # if this bring N_max over the limit, try one last time at the limiting value, then quit (will hit the break below)
                if N_max > N_LIMIT:
                    N_max = N_LIMIT
            else:
                break

        LL_trunc_dir[rep] = LL
        if not silent:
            print("LL (trfwd.dir): %f, diff = %f, time = %f" % (LL_trunc_dir[rep], np.abs(LL_trunc_dir[rep] - LL_lsgdual[rep]), RT_trunc_dir[rep]))

        # repeat the iterative doubling stability check for trfwd w/ fft
        N_max = np.int32(N_init)
        delta = np.inf # delta (unqualified) is the smallest delta between either previous method or the previous iteration
        while N_max <= N_LIMIT and delta > epsilon:
            # run the trial
            start_time = time.process_time()
            _, logz = trfwd.truncated_forward(arrival_pmf,
                                                  Lambda_eval,
                                                  branch_trpmf,
                                                  Delta_eval,
                                                  Rho_eval,
                                                  y[rep, :],
                                                  N_max,
                                                  silent=False,
                                                  conv_method='fft')
            RT_trunc_fft[rep] = time.process_time() - start_time
            # compute LL
            LL = np.sum(logz)

            # compute delta, the relative difference in LL between this iteration and the previous iteration
            if N_max_trunc_fft[rep].size == 0:
                delta_self = np.inf
            else:
                delta_self = np.abs((LL - N_max_trunc_fft[rep][-1]['LL']) / max(LL, N_max_trunc_fft[rep][-1]['LL']))
            delta_lsgd = np.abs((LL - LL_lsgdual[rep]) / max(LL, LL_lsgdual[rep]))
            delta_gd   = np.abs((LL - LL_gdual[rep])   / max(LL, LL_gdual[rep]))

            delta = np.nanmin([delta_self, delta_lsgd, delta_gd])

            # log the (N_max, LL) tuple
            N_max_trunc_fft[rep] = np.append(N_max_trunc_fft[rep], np.array((N_max, LL), dtype = NMAX_RECORD_DTYPE))

            if not silent:
                print("Trfwd.fft trial, N_Max = %d, LL = %f, delta = %f" % (N_max, LL, delta))

            # double for next iteration
            if N_max < N_LIMIT:
                N_max = N_max * 2

                # if this bring N_max over the limit, try one last time at the limiting value, then quit (will hit the break below)
                if N_max > N_LIMIT:
                    N_max = N_LIMIT
            else:
                break

        LL_trunc_fft[rep] = LL
        if not silent:
            print("LL (trfwd.fft): %f, diff = %f, time = %f" % (LL_trunc_fft[rep], np.abs(LL_trunc_fft[rep] - LL_lsgdual[rep]), RT_trunc_fft[rep]))

        if not silent:
            print("#######")

    return LL_lsgdual, LL_gdual, LL_trunc_dir, LL_trunc_fft, y, N_max_trunc_dir, N_max_trunc_fft, RT_lsgdual, RT_gdual, RT_trunc_dir, RT_trunc_fft
# def stability_experiment(
#         Lambda  = 10 * np.array([0.0257, 0.1163, 0.2104, 0.1504, 0.0428]).reshape(-1, 1),
#         Delta   = np.array([0.2636, 0.2636, 0.2636, 0.2636]).reshape(-1, 1),
#         Rho     = 0.5 * np.ones(5),
#         epsilon = 1e-5, # allowable error in truncated fa
#         n_reps  = 10,   # number of times to repeat the experiment
#         N_init  = 0,    # initial value of N_max to use in each experiment. if N_init < max(y), it will be set to max(y)
#         N_LIMIT = 25000, # hard cap on the max value for the truncated algorithm
#         silent  = False,
#         arrival = 'poisson',
#         branch  = 'bernoulli'
#         ):

def stability_experiment_suite(experiment = 'demo'):
    silent = False
    if experiment == 'demo':
        # vary Y via a scaling parameter on all entries of Lambda

        # Lambda_scale = [10, 50, 100, 300, 500, 1000, 1500, 3000]
        Lambda_scale = [10, 50, 100, 500, 1500, 3000]
        n_experiments = len(Lambda_scale)

        Lambda_base = np.array([0.0257, 0.1163, 0.2104, 0.1504, 0.0428]).reshape(-1, 1)
        Delta = np.array([0.2636, 0.2636, 0.2636, 0.2636]).reshape(-1, 1)
        Rho = 0.5 * np.ones(5)
        epsilon = 1e-5
        n_reps = 3
        N_init = 0
        N_LIMIT = 1500
        arrival = 'poisson'
        branch = 'bernoulli'

        # RMSE records
        RMSE_LL_lsgdual     = np.zeros(n_experiments)
        RMSE_LL_gdual     = np.zeros(n_experiments)
        # RMSE_LL_trfwd_dir = np.zeros(n_experiments)
        RMSE_LL_trfwd_fft = np.zeros(n_experiments)

        for i_experiment in range(n_experiments):
            result = stability_experiment(Lambda_scale[i_experiment] * Lambda_base,
                                          Delta,
                                          Rho,
                                          epsilon,
                                          n_reps,
                                          N_init,
                                          N_LIMIT,
                                          silent,
                                          arrival,
                                          branch)
            LL_lsgdual, LL_gdual, LL_trunc_dir, LL_trunc_fft, y, N_max_trunc_dir, N_max_trunc_fft, RT_lsgdual, RT_gdual, RT_trunc_dir, RT_trunc_fft = result[:]

            RMSE_LL_lsgdual[i_experiment] = np.sqrt(np.nanmean((LL_lsgdual - LL_trunc_dir) ** 2))
            RMSE_LL_gdual[i_experiment] = np.sqrt(np.nanmean((LL_gdual - LL_trunc_dir) ** 2))
            RMSE_LL_trfwd_fft[i_experiment] = np.sqrt(np.nanmean((LL_trunc_fft - LL_trunc_dir) ** 2))

            # RMSE_LL_gdual[i_experiment]     = np.sqrt(np.nanmean((LL_gdual     - LL_lsgdual) ** 2))
            # RMSE_LL_trfwd_dir[i_experiment] = np.sqrt(np.nanmean((LL_trunc_dir - LL_lsgdual) ** 2))
            # RMSE_LL_trfwd_fft[i_experiment] = np.sqrt(np.nanmean((LL_trunc_fft - LL_lsgdual) ** 2))

        x_axis_data  = Lambda_scale
        x_axis_label = r'$\Lambda$'
        y_axis_data  = [RMSE_LL_lsgdual, RMSE_LL_gdual, RMSE_LL_trfwd_fft]
        y_axis_label = r'RMSE'
    elif experiment == 'gen_vs_eval: delta':
        # vary the survival parameter used for evaluation of LL

        # Lambda_scale = [10, 50, 100, 300, 500, 1000, 1500, 3000]
        Delta_true = 0.5
        Delta_eval = np.linspace(0, 1.0, 11)
        n_experiments = len(Delta_eval)

        Lambda_scale = 500
        Lambda = Lambda_scale * np.array([0.0257, 0.1163, 0.2104, 0.1504, 0.0428]).reshape(-1, 1)
        Delta_gen = (Delta_true * np.ones(4)).reshape(-1, 1)
        Rho = 0.5 * np.ones(5)
        epsilon = 1e-5
        n_reps = 3
        N_init = 0
        N_LIMIT = 1500
        arrival = 'poisson'
        branch = 'bernoulli'

        # RMSE records
        RMSE_LL_lsgdual     = np.zeros(n_experiments)
        RMSE_LL_gdual     = np.zeros(n_experiments)
        # RMSE_LL_trfwd_dir = np.zeros(n_experiments)
        RMSE_LL_trfwd_fft = np.zeros(n_experiments)

        # Runtime records
        meanRT_lsgdual   = np.zeros(n_experiments)
        meanRT_gdual     = np.zeros(n_experiments)
        meanRT_trfwd_dir = np.zeros(n_experiments)
        meanRT_trfwd_fft = np.zeros(n_experiments)

        for i_experiment in range(n_experiments):
            Delta_eval_iter = (Delta_eval[i_experiment] * np.ones(4)).reshape(-1, 1)

            result = stability_experiment(Lambda,
                                          Delta_gen,
                                          Rho,
                                          epsilon,
                                          n_reps,
                                          N_init,
                                          N_LIMIT,
                                          silent,
                                          arrival,
                                          branch,
                                          Delta_eval=Delta_eval_iter)
            LL_lsgdual, LL_gdual, LL_trunc_dir, LL_trunc_fft, y, N_max_trunc_dir, N_max_trunc_fft, RT_lsgdual, RT_gdual, RT_trunc_dir, RT_trunc_fft = result[:]

            RMSE_LL_lsgdual[i_experiment] = np.sqrt(np.nanmean((LL_lsgdual - LL_trunc_dir) ** 2))
            RMSE_LL_gdual[i_experiment] = np.sqrt(np.nanmean((LL_gdual - LL_trunc_dir) ** 2))
            RMSE_LL_trfwd_fft[i_experiment] = np.sqrt(np.nanmean((LL_trunc_fft - LL_trunc_dir) ** 2))

            meanRT_lsgdual[i_experiment]   = np.mean(RT_lsgdual)
            meanRT_gdual[i_experiment]     = np.mean(RT_gdual)
            meanRT_trfwd_dir[i_experiment] = np.mean(RT_trunc_dir)
            meanRT_trfwd_fft[i_experiment] = np.mean(RT_trunc_fft)

            # RMSE_LL_gdual[i_experiment]     = np.sqrt(np.nanmean((LL_gdual     - LL_lsgdual) ** 2))
            # RMSE_LL_trfwd_dir[i_experiment] = np.sqrt(np.nanmean((LL_trunc_dir - LL_lsgdual) ** 2))
            # RMSE_LL_trfwd_fft[i_experiment] = np.sqrt(np.nanmean((LL_trunc_fft - LL_lsgdual) ** 2))
        elif experiment == 'fixedY':
            None
        elif experiment == 'poissonbranching_RT':
            None
        elif experiment == 'RT wrt Y directly'

        x_axis_data  = Delta_eval
        x_axis_label = r'$\delta$'
        y_axis_data  = [RMSE_LL_lsgdual, RMSE_LL_gdual, RMSE_LL_trfwd_fft]
        y_axis_label = r'RMSE'


    # plot results
    fig = plt.figure()
    for data in y_axis_data:
        plt.plot(x_axis_data, data)

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    fig.savefig('test.png')



# def runtime_hmm(
#         Lambda  = 10 * np.array([0.0257, 0.1163, 0.2104, 0.1504, 0.0428]),
#         Delta   = np.array([0.2636, 0.2636, 0.2636, 0.2636]),
#         Rho     = 0.5 * np.ones(5),
#         epsilon = 1e-10, # allowable error in truncated fa
#         n_reps  = 10,   # number of times to repeat the experiment
#         N_LIMIT = 1000, # hard cap on the max value for the truncated algorithm
#         verbose = True,
#         arrival = 'poisson',
#         branch  = 'bernoulli',
#         observ  = 'binomial'
#         ):
#
#     K = len(Lambda)
#
#     # sample record
#     N = np.zeros((n_reps, K), dtype=np.int32)
#     y = np.zeros((n_reps, K), dtype=np.int32)
#
#     # runtime record
#     runtime_trunc_final = np.zeros(n_reps)
#     runtime_trunc_total = np.zeros(n_reps)
#     runtime_pgffa       = np.zeros(n_reps)
#     runtime_utppgffa    = np.zeros(n_reps)
#
#     # truncated fa final truncation value
#     n_max = np.zeros(n_reps).astype(int)
#
#     # organize parameters for pgffa, utppgffa and trunfa
#     Theta  = {'arrival': Lambda.reshape((-1, 1)),
#               'branch':  Delta.reshape((-1, 1)),
#               'observ':  Rho}
#
#     Lambda_trunc = Lambda.reshape((-1, 1))
#     Delta_trunc  = Delta.reshape((-1, 1))
#
#     # configure distributions
#     if arrival == 'poisson':
#         arrival_pmf = stats.poisson
#         arrival_pgf = poisson_utppgf_cython
#         arrival_pgf_name = 'poisson'
#     elif arrival == 'logser':
#         arrival_pmf = stats.logser
#         arrival_pgf = logarithmic_utppgf_cython
#     elif arrival == 'geom':
#         arrival_pmf = stats.geom
#         # arrival_pgf = lambda s, theta: geometric_pgf(s, theta)
#         arrival_pgf = geometric_utppgf_cython
#
#     if branch  == 'bernoulli':
#         branch_fun  = truncatedfa.binomial_branching
#         # branch_pgf  = lambda s, theta: bernoulli_pgf(s, theta)
#         branch_pgf = bernoulli_utppgf_cython
#         branch_pgf_name = 'bernoulli'
#     elif branch == 'poisson':
#         branch_fun  = truncatedfa.poisson_branching
#         # branch_pgf  = lambda s, theta: poisson_pgf(s, theta)
#         branch_pgf = poisson_utppgf_cython
#         branch_pgf_name = 'poisson'
#
#     if observ  == 'binomial':
#         observ_pgf  = None
#
#     for iter in range(0, n_reps):
#         if verbose == "full": print "Iteration %d of %d" % (iter, n_reps)
#
#         attempt = 1
#         while True:
#             # try:
#                 # sample data
#                 for i in range(0, K):
#                     if i == 0:
#                         N[iter, i] = arrival_pmf.rvs(Lambda[i])
#                     else:
#                         if branch == 'binomial':
#                             N[iter, i] = arrival_pmf.rvs(Lambda[i]) + stats.binom.rvs(N[iter, i-1], Delta[i-1])
#                         elif branch == 'poisson':
#                             N[iter, i] = arrival_pmf.rvs(Lambda[i]) + stats.poisson.rvs(N[iter, i - 1] * Delta[i - 1])
#                     y[iter, i] = stats.binom.rvs(N[iter, i], Rho[i])
#
#                 if verbose == "full": print y[iter,:]
#
#                 # likelihood from UTPPGFFA
#                 t_start = time.clock()
#                 # Alpha_utppgffa, logZ_utppgffa = UTPPGFFA.utppgffa(y[iter, :],
#                 #                                                Theta,
#                 #                                                arrival_pgf,
#                 #                                                branch_pgf,
#                 #                                                observ_pgf,
#                 #                                                d=1,
#                 #                                                normalized=True)
#                 alpha_utppgffa, logZ_utppgffa = UTPPGFFA_cython.utppgffa_cython(y[iter, :],
#                                                                   arrival_pgf_name,
#                                                                   Lambda_trunc,
#                                                                   branch_pgf_name,
#                                                                   Delta_trunc,
#                                                                   Rho,
#                                                                   d=3)
#                 # loglikelihood_utppgffa = np.log(Alpha_utppgffa[-1][0]) + np.sum(logZ_utppgffa)
#                 loglikelihood_utppgffa = np.log(alpha_utppgffa[0]) + np.sum(logZ_utppgffa)
#                 runtime_utppgffa[iter] = time.clock() - t_start
#                 if verbose == "full": print "UTPPGFFA: %0.4f" % runtime_utppgffa[iter]
#
#                 # likelihood from PGFFA
#                 t_start = time.clock()
#                 a, b, f = pgffa.pgf_forward(Lambda,
#                                                 Rho,
#                                                 Delta,
#                                                 y[iter, :])
#                 runtime_pgffa[iter] = time.clock() - t_start
#                 if verbose == "full": print "PGFFA: %0.4f" % runtime_pgffa[iter]
#
#                 # likelihood from truncated forward algorithm
#                 n_max[iter] = max(y[iter, :])
#                 t_start = time.clock()
#                 loglikelihood_trunc = float('inf')
#                 loglikelihood_diff  = float('inf')
#                 while abs(loglikelihood_trunc - loglikelihood_utppgffa) >= epsilon and \
#                       loglikelihood_diff >= CONV_LIMIT and \
#                       n_max[iter] < N_LIMIT:
#                 # while abs(1 - (loglikelihood_trunc / loglikelihood_utppgffa)) >= epsilon and n_max[iter] < N_LIMIT:
#                     n_max[iter] += 1
#                     t_loop = time.clock()
#                     Alpha_trunc, z = truncatedfa.truncated_forward(arrival_pmf,
#                                                                    Lambda_trunc,
#                                                                    branch_fun,
#                                                                    Delta_trunc,
#                                                                    Rho,
#                                                                    y[iter, :],
#                                                                    n_max=n_max[iter])
#                     loglikelihood_iter = truncatedfa.likelihood(z, log=True)
#                     loglikelihood_diff = abs(loglikelihood_trunc - loglikelihood_iter)
#                     loglikelihood_trunc = loglikelihood_iter
#                     runtime_trunc_final[iter] = time.clock() - t_loop
#                 runtime_trunc_total[iter] = time.clock() - t_start
#
#                 if verbose == "full": print "Trunc: %0.4f last run @%d, %0.4f total" % (runtime_trunc_final[iter], n_max[iter], runtime_trunc_total[iter])
#
#                 if n_max[iter] >= N_LIMIT:
#                     print "Attempt #%d, trunc failed to converge." % attempt
#                     attempt += 1
#                 else:
#                     break
#             # except Exception as inst:
#             #     print "Attempt #%d failed, Error: " % attempt, inst
#             #     attempt += 1
#     return runtime_utppgffa, runtime_pgffa, runtime_trunc_final, runtime_trunc_total, n_max, y, N


if __name__ == "__main__":
    stability_experiment_suite('gen_vs_eval: delta')