import os
import sys
from glob import glob
import pwd
import time
import datetime
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
        Rho_eval    = None,
        y_given     = None   # if provided, will override all sampling with this y instead
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
        if y_given is not None:
            y[rep, :] = np.copy(y_given[:])
        else:
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
    resultdir = '/Users/kwinner/Work/Data/Results'

    silent = False
    if experiment == 'demo':
        # vary Y via a scaling parameter on all entries of Lambda
        experiment_folder_prefix = 'Lambda_scale'
        experiment_timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S%f")
        experiment_folder = os.path.join(resultdir, experiment_folder_prefix + experiment_timestamp)

        if (not os.path.exists(experiment_folder)):
            os.makedirs(experiment_folder)

        # Lambda_scale = [10, 50, 100, 300, 500, 1000, 1500, 3000]
        Lambda_scale = [10, 50, 100, 500, 1500, 3000]
        n_experiments = len(Lambda_scale)

        Lambda_base = np.array([0.0257, 0.1163, 0.2104, 0.1504, 0.0428]).reshape(-1, 1)
        Delta = np.array([0.2636, 0.2636, 0.2636, 0.2636]).reshape(-1, 1)
        Rho = 0.5 * np.ones(5)
        epsilon = 1e-5
        n_reps = 5
        N_init = 0
        N_LIMIT = 1500
        arrival = 'poisson'
        branch = 'poisson'

        # write meta file
        meta_file = open(os.path.join(experiment_folder, 'meta.txt'), 'w')
        meta_file.write('experiment: ' + experiment + '\n')
        meta_file.write('Lambda_gen: ' + str(Lambda_base) + '\n')
        meta_file.write('Delta_gen: ' + str(Lambda_scale) + '\n')
        meta_file.write('Delta_eval: ' + str(Delta) + '\n')
        meta_file.write('Rho: ' + str(Rho) + '\n')
        meta_file.write('epsilon: ' + str(epsilon) + '\n')
        meta_file.write('n_reps: ' + str(n_reps) + '\n')
        meta_file.write('N_init: ' + str(N_init) + '\n')
        meta_file.write('arrival: ' + arrival + '\n')
        meta_file.write('branch: ' + branch + '\n')
        meta_file.close()

        # RMSE records
        RMSE_LL_lsgdual     = np.zeros(n_experiments)
        RMSE_LL_gdual     = np.zeros(n_experiments)
        # RMSE_LL_trfwd_dir = np.zeros(n_experiments)
        RMSE_LL_trfwd_fft = np.zeros(n_experiments)
        # RMSE records
        RMSE_LL_lsgdual = np.zeros(n_experiments)
        RMSE_LL_gdual = np.zeros(n_experiments)
        # RMSE_LL_trfwd_dir = np.zeros(n_experiments)
        RMSE_LL_trfwd_fft = np.zeros(n_experiments)

        # Runtime records
        meanRT_lsgdual = np.zeros(n_experiments)
        meanRT_gdual = np.zeros(n_experiments)
        meanRT_trfwd_dir = np.zeros(n_experiments)
        meanRT_trfwd_fft = np.zeros(n_experiments)

        # LL records
        mean_LL_lsgdual   = np.zeros(n_experiments)
        mean_LL_gdual     = np.zeros(n_experiments)
        mean_LL_trfwd_dir = np.zeros(n_experiments)
        mean_LL_trfwd_fft = np.zeros(n_experiments)

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

            pickle.dump(result, open(os.path.join(experiment_folder, 'result'+str(i_experiment)+'.pickle'), 'wb'))

            LL_lsgdual, LL_gdual, LL_trunc_dir, LL_trunc_fft, y, N_max_trunc_dir, N_max_trunc_fft, RT_lsgdual, RT_gdual, RT_trunc_dir, RT_trunc_fft = result[:]

            RMSE_LL_lsgdual[i_experiment] = np.sqrt(np.nanmean((LL_lsgdual - LL_trunc_dir) ** 2))
            RMSE_LL_gdual[i_experiment] = np.sqrt(np.nanmean((LL_gdual - LL_trunc_dir) ** 2))
            RMSE_LL_trfwd_fft[i_experiment] = np.sqrt(np.nanmean((LL_trunc_fft - LL_trunc_dir) ** 2))

            meanRT_lsgdual[i_experiment] = np.mean(RT_lsgdual)
            meanRT_gdual[i_experiment] = np.mean(RT_gdual)
            meanRT_trfwd_dir[i_experiment] = np.mean(RT_trunc_dir)
            meanRT_trfwd_fft[i_experiment] = np.mean(RT_trunc_fft)

            mean_LL_lsgdual[i_experiment] = np.nanmean(LL_lsgdual)
            mean_LL_gdual[i_experiment] = np.nanmean(LL_gdual)
            mean_LL_trfwd_dir[i_experiment] = np.nanmean(LL_trunc_dir)
            mean_LL_trfwd_fft[i_experiment] = np.nanmean(LL_trunc_fft)

            # RMSE_LL_gdual[i_experiment]     = np.sqrt(np.nanmean((LL_gdual     - LL_lsgdual) ** 2))
            # RMSE_LL_trfwd_dir[i_experiment] = np.sqrt(np.nanmean((LL_trunc_dir - LL_lsgdual) ** 2))
            # RMSE_LL_trfwd_fft[i_experiment] = np.sqrt(np.nanmean((LL_trunc_fft - LL_lsgdual) ** 2))

        pickle.dump([RMSE_LL_lsgdual, RMSE_LL_gdual, RMSE_LL_trfwd_fft, meanRT_lsgdual, meanRT_gdual, meanRT_trfwd_dir,
                     meanRT_trfwd_fft, mean_LL_lsgdual, mean_LL_gdual, mean_LL_trfwd_dir, mean_LL_trfwd_fft],
                    open(os.path.join(experiment_folder, 'result_means.pickle'), 'wb'))

        x_axis_label = r'$\Lambda$'
        RMSE_y_axis_label = r'RMSE'
        RT_y_axis_label = r'RT'
        LL_y_axis_label  = r'LL (average)'
        method_names = ['LSGDual', 'GDual', 'Trunc w/ FFT', 'Trunc w/ Direct Conv']
        method_filenames = ['lsgd', 'gdual', 'trfft', 'trdir']
        RMSEsuffix = '_rmse.png'
        RTsuffix = '_rt.png'
        LLsuffix = '_ll.png'

        # plot RMSE results
        RMSE_data = [RMSE_LL_lsgdual, RMSE_LL_gdual, RMSE_LL_trfwd_fft]
        for i_method in range(3):
            fig = plt.figure()
            plt.plot(Lambda_scale, RMSE_data[i_method])

            plt.xlabel(x_axis_label)
            plt.ylabel(RMSE_y_axis_label)

            plt.title(method_names[i_method])

            fig.savefig(os.path.join(experiment_folder, method_filenames[i_method] + RMSEsuffix))

        RT_data = [meanRT_lsgdual, meanRT_gdual, meanRT_trfwd_fft, meanRT_trfwd_dir]
        for i_method in range(4):
            fig = plt.figure()
            plt.plot(Lambda_scale, RT_data[i_method])

            plt.xlabel(x_axis_label)
            plt.ylabel(RT_y_axis_label)

            plt.title(method_names[i_method])

            fig.savefig(os.path.join(experiment_folder, method_filenames[i_method] + RTsuffix))

        LL_data = [mean_LL_lsgdual, mean_LL_gdual, mean_LL_trfwd_fft, mean_LL_trfwd_dir]
        for i_method in range(4):
            fig = plt.figure()
            plt.plot(Lambda_scale, LL_data[i_method])

            plt.xlabel(x_axis_label)
            plt.ylabel(LL_y_axis_label)

            plt.title(method_names[i_method])

            fig.savefig(os.path.join(experiment_folder, method_filenames[i_method] + LLsuffix))

        # all RTs on same axes
        method_names = ['Trunc w/ Direct Conv', 'Trunc w/ FFT', 'GDual', 'LSGDual']
        fig = plt.figure()
        plt.plot(Lambda_scale, meanRT_trfwd_dir)
        plt.plot(Lambda_scale, meanRT_trfwd_fft)
        plt.plot(Lambda_scale, meanRT_gdual)
        plt.plot(Lambda_scale, meanRT_lsgdual)
        plt.xlabel(x_axis_label)
        plt.ylabel(RT_y_axis_label)
        plt.title(r'Runtime vs $\delta$, all methods')
        plt.legend(method_names)
        fig.savefig(os.path.join(experiment_folder, 'RT_all.png'))

        # all LLs on same axes
        fig = plt.figure()
        plt.plot(Lambda_scale, mean_LL_trfwd_dir)
        plt.plot(Lambda_scale, mean_LL_trfwd_fft)
        plt.plot(Lambda_scale, mean_LL_gdual)
        plt.plot(Lambda_scale, mean_LL_lsgdual)
        plt.xlabel(x_axis_label)
        plt.ylabel(LL_y_axis_label)
        plt.title(r'LL vs $\delta$, all methods')
        plt.legend(method_names)
        fig.savefig(os.path.join(experiment_folder, 'LL_all.png'))
    elif experiment == 'gen_vs_eval: delta':
        fix_y = True

        if fix_y:
            # experiment_folder_prefix = 'genvseval_delta_fixedy'
            experiment_folder_prefix = 'poiss_fixedy'
        else:
            experiment_folder_prefix = 'poiss'
        experiment_timestamp     = datetime.datetime.now().strftime("%y%m%d%H%M%S%f")
        experiment_folder        = os.path.join(resultdir, experiment_folder_prefix + experiment_timestamp)

        if (not os.path.exists(experiment_folder)):
            os.makedirs(experiment_folder)

        # vary the survival parameter used for evaluation of LL

        # Lambda_scale = [10, 50, 100, 300, 500, 1000, 1500, 3000]
        Delta_true = 0.5
        # Delta_eval = np.linspace(0, 1.0, 21)
        Delta_eval = np.linspace(0.25, 5.0, 20)
        # Delta_eval = np.linspace(0, 1.0, 11)
        n_experiments = len(Delta_eval)

        Lambda_scale = 500
        Lambda = Lambda_scale * np.array([0.0257, 0.1163, 0.2104, 0.1504, 0.0428]).reshape(-1, 1)
        Delta_gen = (Delta_true * np.ones(4)).reshape(-1, 1)
        Rho = 0.5 * np.ones(5)
        epsilon = 1e-5
        n_reps = 5
        N_init = 0
        N_LIMIT = 2000
        arrival = 'poisson'
        branch = 'poisson'

        # write meta file
        meta_file = open(os.path.join(experiment_folder, 'meta.txt'), 'w')
        meta_file.write('experiment: ' + experiment + '\n')
        meta_file.write('Lambda_gen: ' + str(Lambda) + '\n')
        meta_file.write('Delta_gen: ' + str(Delta_gen) + '\n')
        meta_file.write('Delta_eval: ' + str(Delta_eval) + '\n')
        meta_file.write('Rho: ' + str(Rho) + '\n')
        meta_file.write('epsilon: ' + str(epsilon) + '\n')
        meta_file.write('n_reps: ' + str(n_reps) + '\n')
        meta_file.write('N_init: ' + str(N_init) + '\n')
        meta_file.write('arrival: ' + arrival + '\n')
        meta_file.write('branch: ' + branch + '\n')
        meta_file.write('fix-y: ' + str(fix_y) + '\n')
        meta_file.close()

        # RMSE records
        RMSE_LL_lsgdual     = np.zeros(n_experiments)
        RMSE_LL_gdual     = np.zeros(n_experiments)
        # RMSE_LL_trfwd_dir = np.zeros(n_experiments)
        RMSE_LL_trfwd_fft = np.zeros(n_experiments)

        # LL records
        mean_LL_lsgdual   = np.zeros(n_experiments)
        mean_LL_gdual     = np.zeros(n_experiments)
        mean_LL_trfwd_dir = np.zeros(n_experiments)
        mean_LL_trfwd_fft = np.zeros(n_experiments)


        # Runtime records
        meanRT_lsgdual   = np.zeros(n_experiments)
        meanRT_gdual     = np.zeros(n_experiments)
        meanRT_trfwd_dir = np.zeros(n_experiments)
        meanRT_trfwd_fft = np.zeros(n_experiments)

        # Sample data
        if fix_y:
            K = len(Lambda)
            N = np.zeros(K, dtype=np.int64)
            y_given = np.zeros(K, dtype=np.int64)
            for i in range(0, K):
                if i == 0:
                    N[i] = stats.poisson.rvs(Lambda[i])
                else:
                    if branch == 'bernoulli':
                        N[i] = stats.poisson.rvs(Lambda[i]) + stats.binom.rvs(N[i - 1], Delta_gen[i - 1])
                    elif branch == 'poisson':
                        N[i] = stats.poisson.rvs(Lambda[i]) + stats.poisson.rvs(N[i - 1] * Delta_gen[i - 1])
                y_given[i] = stats.binom.rvs(N[i], Rho[i])

        for i_experiment in range(n_experiments):
            Delta_eval_iter = (Delta_eval[i_experiment] * np.ones(4)).reshape(-1, 1)

            if fix_y:
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
                                              Delta_eval=Delta_eval_iter,
                                              y_given = y_given)
            else:
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
            pickle.dump(result, open(os.path.join(experiment_folder, 'result' + str(i_experiment) + '.pickle'), 'wb'))
            LL_lsgdual, LL_gdual, LL_trunc_dir, LL_trunc_fft, y, N_max_trunc_dir, N_max_trunc_fft, RT_lsgdual, RT_gdual, RT_trunc_dir, RT_trunc_fft = result[:]

            RMSE_LL_lsgdual[i_experiment] = np.sqrt(np.nanmean((LL_lsgdual - LL_trunc_dir) ** 2))
            RMSE_LL_gdual[i_experiment] = np.sqrt(np.nanmean((LL_gdual - LL_trunc_dir) ** 2))
            RMSE_LL_trfwd_fft[i_experiment] = np.sqrt(np.nanmean((LL_trunc_fft - LL_trunc_dir) ** 2))

            meanRT_lsgdual[i_experiment]   = np.mean(RT_lsgdual)
            meanRT_gdual[i_experiment]     = np.mean(RT_gdual)
            meanRT_trfwd_dir[i_experiment] = np.mean(RT_trunc_dir)
            meanRT_trfwd_fft[i_experiment] = np.mean(RT_trunc_fft)

            mean_LL_lsgdual[i_experiment] = np.nanmean(LL_lsgdual)
            mean_LL_gdual[i_experiment] = np.nanmean(LL_gdual)
            mean_LL_trfwd_dir[i_experiment] = np.nanmean(LL_trunc_dir)
            mean_LL_trfwd_fft[i_experiment] = np.nanmean(LL_trunc_fft)

            # RMSE_LL_gdual[i_experiment]     = np.sqrt(np.nanmean((LL_gdual     - LL_lsgdual) ** 2))
            # RMSE_LL_trfwd_dir[i_experiment] = np.sqrt(np.nanmean((LL_trunc_dir - LL_lsgdual) ** 2))
            # RMSE_LL_trfwd_fft[i_experiment] = np.sqrt(np.nanmean((LL_trunc_fft - LL_lsgdual) ** 2))

        pickle.dump([RMSE_LL_lsgdual, RMSE_LL_gdual, RMSE_LL_trfwd_fft, meanRT_lsgdual, meanRT_gdual, meanRT_trfwd_dir,
                     meanRT_trfwd_fft, mean_LL_lsgdual, mean_LL_gdual, mean_LL_trfwd_dir, mean_LL_trfwd_fft], open(os.path.join(experiment_folder, 'result_means.pickle'), 'wb'))

        x_axis_label     = r'$\delta$'
        RMSE_y_axis_label = r'RMSE'
        RT_y_axis_label  = r'RT'
        LL_y_axis_label  = r'LL (average)'
        method_names     = ['LSGDual', 'GDual', 'Trunc w/ FFT', 'Trunc w/ Direct Conv']
        method_filenames = ['lsgd', 'gdual', 'trfft', 'trdir']
        RMSEsuffix       = '_rmse.png'
        RTsuffix         = '_rt.png'
        LLsuffix         = '_ll.png'

        # plot RMSE results
        RMSE_data = [RMSE_LL_lsgdual, RMSE_LL_gdual, RMSE_LL_trfwd_fft]
        for i_method in range(3):
            fig = plt.figure()
            plt.plot(Delta_eval, RMSE_data[i_method])

            plt.xlabel(x_axis_label)
            plt.ylabel(RMSE_y_axis_label)

            plt.title(method_names[i_method])

            fig.savefig(os.path.join(experiment_folder, method_filenames[i_method] + RMSEsuffix))

        RT_data = [meanRT_lsgdual, meanRT_gdual, meanRT_trfwd_fft, meanRT_trfwd_dir]
        for i_method in range(4):
            fig = plt.figure()
            plt.plot(Delta_eval, RT_data[i_method])

            plt.xlabel(x_axis_label)
            plt.ylabel(RT_y_axis_label)

            plt.title(method_names[i_method])

            fig.savefig(os.path.join(experiment_folder, method_filenames[i_method] + RTsuffix))

        LL_data = [mean_LL_lsgdual, mean_LL_gdual, mean_LL_trfwd_fft, mean_LL_trfwd_dir]
        for i_method in range(4):
            fig = plt.figure()
            plt.plot(Delta_eval, LL_data[i_method])

            plt.xlabel(x_axis_label)
            plt.ylabel(LL_y_axis_label)

            plt.title(method_names[i_method])

            fig.savefig(os.path.join(experiment_folder, method_filenames[i_method] + LLsuffix))

        # all RTs on same axes
        method_names = ['Trunc w/ Direct Conv', 'Trunc w/ FFT', 'GDual', 'LSGDual']
        fig = plt.figure()
        plt.plot(Delta_eval, meanRT_trfwd_dir)
        plt.plot(Delta_eval, meanRT_trfwd_fft)
        plt.plot(Delta_eval, meanRT_gdual)
        plt.plot(Delta_eval, meanRT_lsgdual)
        plt.xlabel(x_axis_label)
        plt.ylabel(RT_y_axis_label)
        plt.title(r'Runtime vs $\delta$, all methods')
        plt.legend(method_names)
        fig.savefig(os.path.join(experiment_folder, 'RT_all.png'))

        # all LLs on same axes
        fig = plt.figure()
        plt.plot(Delta_eval, mean_LL_trfwd_dir)
        plt.plot(Delta_eval, mean_LL_trfwd_fft)
        plt.plot(Delta_eval, mean_LL_gdual)
        plt.plot(Delta_eval, mean_LL_lsgdual)
        plt.xlabel(x_axis_label)
        plt.ylabel(LL_y_axis_label)
        plt.title(r'LL vs $\delta$, all methods')
        plt.legend(method_names)
        fig.savefig(os.path.join(experiment_folder, 'LL_all.png'))
    elif experiment == 'poissonbranching_RT':
        fix_y = False

        if fix_y:
            experiment_folder_prefix = 'poisson_branch_fixedy'
        else:
            experiment_folder_prefix = 'poisson_branch_delta'

        experiment_timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S%f")
        experiment_folder = os.path.join(resultdir, experiment_folder_prefix + experiment_timestamp)

        if (not os.path.exists(experiment_folder)):
            os.makedirs(experiment_folder)

        # vary the survival parameter used for evaluation of LL

        # Lambda_scale = [10, 50, 100, 300, 500, 1000, 1500, 3000]
        Delta_true = 0.5
        Delta_eval = np.linspace(0, 1.0, 21)
        # Delta_eval = np.linspace(0, 1.0, 11)
        n_experiments = len(Delta_eval)

        Lambda_scale = 500
        Lambda = Lambda_scale * np.array([0.0257, 0.1163, 0.2104, 0.1504, 0.0428]).reshape(-1, 1)
        Delta_gen = (Delta_true * np.ones(4)).reshape(-1, 1)
        Rho = 0.5 * np.ones(5)
        epsilon = 1e-5
        n_reps = 20
        N_init = 0
        N_LIMIT = 2000
        arrival = 'poisson'
        branch = 'poisson'

        # write meta file
        meta_file = open(os.path.join(experiment_folder, 'meta.txt'), 'w')
        meta_file.write('experiment: ' + experiment + '\n')
        meta_file.write('Lambda_gen: ' + str(Lambda) + '\n')
        meta_file.write('Delta_gen: ' + str(Delta_gen) + '\n')
        meta_file.write('Delta_eval: ' + str(Delta_eval) + '\n')
        meta_file.write('Rho: ' + str(Rho) + '\n')
        meta_file.write('epsilon: ' + str(epsilon) + '\n')
        meta_file.write('n_reps: ' + str(n_reps) + '\n')
        meta_file.write('N_init: ' + str(N_init) + '\n')
        meta_file.write('arrival: ' + arrival + '\n')
        meta_file.write('branch: ' + branch + '\n')
        meta_file.write('fix-y: ' + str(fix_y) + '\n')
        meta_file.close()

        # RMSE records
        RMSE_LL_lsgdual = np.zeros(n_experiments)
        RMSE_LL_gdual = np.zeros(n_experiments)
        # RMSE_LL_trfwd_dir = np.zeros(n_experiments)
        RMSE_LL_trfwd_fft = np.zeros(n_experiments)

        # Runtime records
        meanRT_lsgdual = np.zeros(n_experiments)
        meanRT_gdual = np.zeros(n_experiments)
        meanRT_trfwd_dir = np.zeros(n_experiments)
        meanRT_trfwd_fft = np.zeros(n_experiments)

        # Sample data
        if fix_y:
            K = len(Lambda)
            N = np.zeros(K, dtype=np.int64)
            y_given = np.zeros(K, dtype=np.int64)
            for i in range(0, K):
                if i == 0:
                    N[i] = stats.poisson.rvs(Lambda[i])
                else:
                    if branch == 'bernoulli':
                        N[i] = stats.poisson.rvs(Lambda[i]) + stats.binom.rvs(N[i - 1], Delta_gen[i - 1])
                    elif branch == 'poisson':
                        N[i] = stats.poisson.rvs(Lambda[i]) + stats.poisson.rvs(N[i - 1] * Delta_gen[i - 1])
                y_given[i] = stats.binom.rvs(N[i], Rho[i])

        for i_experiment in range(n_experiments):
            Delta_eval_iter = (Delta_eval[i_experiment] * np.ones(4)).reshape(-1, 1)

            if fix_y:
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
                                              Delta_eval=Delta_eval_iter,
                                              y_given = y_given)
            else:
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
            pickle.dump(result, open(os.path.join(experiment_folder, 'result' + str(i_experiment) + '.pickle'), 'wb'))
            LL_lsgdual, LL_gdual, LL_trunc_dir, LL_trunc_fft, y, N_max_trunc_dir, N_max_trunc_fft, RT_lsgdual, RT_gdual, RT_trunc_dir, RT_trunc_fft = result[
                                                                                                                                                      :]

            RMSE_LL_lsgdual[i_experiment] = np.sqrt(np.nanmean((LL_lsgdual - LL_trunc_dir) ** 2))
            RMSE_LL_gdual[i_experiment] = np.sqrt(np.nanmean((LL_gdual - LL_trunc_dir) ** 2))
            RMSE_LL_trfwd_fft[i_experiment] = np.sqrt(np.nanmean((LL_trunc_fft - LL_trunc_dir) ** 2))

            meanRT_lsgdual[i_experiment] = np.mean(RT_lsgdual)
            meanRT_gdual[i_experiment] = np.mean(RT_gdual)
            meanRT_trfwd_dir[i_experiment] = np.mean(RT_trunc_dir)
            meanRT_trfwd_fft[i_experiment] = np.mean(RT_trunc_fft)

            # RMSE_LL_gdual[i_experiment]     = np.sqrt(np.nanmean((LL_gdual     - LL_lsgdual) ** 2))
            # RMSE_LL_trfwd_dir[i_experiment] = np.sqrt(np.nanmean((LL_trunc_dir - LL_lsgdual) ** 2))
            # RMSE_LL_trfwd_fft[i_experiment] = np.sqrt(np.nanmean((LL_trunc_fft - LL_lsgdual) ** 2))

        pickle.dump([RMSE_LL_lsgdual, RMSE_LL_gdual, RMSE_LL_trfwd_fft, meanRT_lsgdual, meanRT_gdual, meanRT_trfwd_dir,
                     meanRT_trfwd_fft], open(os.path.join(experiment_folder, 'result_means.pickle'), 'wb'))

        x_axis_label = r'$\delta$'
        RMSE_y_axis_label = r'RMSE'
        RT_y_axis_label = r'RT'
        method_names = ['LSGDual', 'GDual', 'Trunc w/ FFT', 'Trunc w/ Direct Conv']
        method_filenames = ['lsgd', 'gdual', 'trfft', 'trdir']
        RMSEsuffix = '_rmse.png'
        RTsuffix = '_rt.png'

        # plot RMSE results
        RMSE_data = [RMSE_LL_lsgdual, RMSE_LL_gdual, RMSE_LL_trfwd_fft]
        for i_method in range(3):
            fig = plt.figure()
            plt.plot(Delta_eval, RMSE_data[i_method])

            plt.xlabel(x_axis_label)
            plt.ylabel(RMSE_y_axis_label)

            plt.title(method_names[i_method])

            fig.savefig(os.path.join(experiment_folder, method_filenames[i_method] + RMSEsuffix))

        RT_data = [meanRT_lsgdual, meanRT_gdual, meanRT_trfwd_fft, meanRT_trfwd_dir]
        for i_method in range(4):
            fig = plt.figure()
            plt.plot(Delta_eval, RT_data[i_method])

            plt.xlabel(x_axis_label)
            plt.ylabel(RT_y_axis_label)

            plt.title(method_names[i_method])

            fig.savefig(os.path.join(experiment_folder, method_filenames[i_method] + RTsuffix))
    # elif experiment == 'RT wrt Y directly':
    #     # vary Y via a scaling parameter on all entries of Lambda
    #     experiment_folder_prefix = 'RTvsY'
    #     experiment_timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S%f")
    #     experiment_folder = os.path.join(resultdir, experiment_folder_prefix + experiment_timestamp)
    #
    #     if (not os.path.exists(experiment_folder)):
    #         os.makedirs(experiment_folder)
    #
    #     Y_scale = [1, 10, 50, 100, 150, 200, 250, 300]
    #     n_experiments = len(Y_scale)
    #
    #     Lambda_eval = np.array([1., 1., 1., 1., 1.]).reshape(-1, 1)
    #     K = len(Lambda_eval)
    #     Delta_eval = 0.25 * np.zeros(K - 1).reshape(-1, 1)
    #     Rho_eval = 0.5 * np.ones(5)
    #     epsilon = 1e-5
    #     n_reps = 20
    #     N_init = 0
    #     N_LIMIT = 1500
    #     arrival = 'poisson'
    #     branch = 'bernoulli'
    #
    #     # write meta file
    #     meta_file = open(os.path.join(experiment_folder, 'meta.txt'), 'w')
    #     meta_file.write('experiment: ' + experiment + '\n')
    #     meta_file.write('Lambda_gen: ' + str(Lambda_base) + '\n')
    #     meta_file.write('Delta_gen: ' + str(Lambda_scale) + '\n')
    #     meta_file.write('Delta_eval: ' + str(Delta) + '\n')
    #     meta_file.write('Rho: ' + str(Rho) + '\n')
    #     meta_file.write('epsilon: ' + str(epsilon) + '\n')
    #     meta_file.write('n_reps: ' + str(n_reps) + '\n')
    #     meta_file.write('N_init: ' + str(N_init) + '\n')
    #     meta_file.write('arrival: ' + arrival + '\n')
    #     meta_file.write('branch: ' + branch + '\n')
    #     meta_file.close()
    #
    #     # RMSE records
    #     RMSE_LL_lsgdual     = np.zeros(n_experiments)
    #     RMSE_LL_gdual     = np.zeros(n_experiments)
    #     # RMSE_LL_trfwd_dir = np.zeros(n_experiments)
    #     RMSE_LL_trfwd_fft = np.zeros(n_experiments)
    #     # RMSE records
    #     RMSE_LL_lsgdual = np.zeros(n_experiments)
    #     RMSE_LL_gdual = np.zeros(n_experiments)
    #     # RMSE_LL_trfwd_dir = np.zeros(n_experiments)
    #     RMSE_LL_trfwd_fft = np.zeros(n_experiments)
    #
    #     # Runtime records
    #     meanRT_lsgdual = np.zeros(n_experiments)
    #     meanRT_gdual = np.zeros(n_experiments)
    #     meanRT_trfwd_dir = np.zeros(n_experiments)
    #     meanRT_trfwd_fft = np.zeros(n_experiments)
    #
    #     for i_experiment in range(n_experiments):
    #         result = stability_experiment(Lambda_scale[i_experiment] * Lambda_base,
    #                                       Delta,
    #                                       Rho,
    #                                       epsilon,
    #                                       n_reps,
    #                                       N_init,
    #                                       N_LIMIT,
    #                                       silent,
    #                                       arrival,
    #                                       branch)
    #         pickle.dump(result, open(os.path.join(experiment_folder, 'result' + str(i_experiment) + '.pickle'), 'wb'))
    #         LL_lsgdual, LL_gdual, LL_trunc_dir, LL_trunc_fft, y, N_max_trunc_dir, N_max_trunc_fft, RT_lsgdual, RT_gdual, RT_trunc_dir, RT_trunc_fft = result[:]
    #
    #         RMSE_LL_lsgdual[i_experiment] = np.sqrt(np.nanmean((LL_lsgdual - LL_trunc_dir) ** 2))
    #         RMSE_LL_gdual[i_experiment] = np.sqrt(np.nanmean((LL_gdual - LL_trunc_dir) ** 2))
    #         RMSE_LL_trfwd_fft[i_experiment] = np.sqrt(np.nanmean((LL_trunc_fft - LL_trunc_dir) ** 2))
    #
    #         meanRT_lsgdual[i_experiment] = np.mean(RT_lsgdual)
    #         meanRT_gdual[i_experiment] = np.mean(RT_gdual)
    #         meanRT_trfwd_dir[i_experiment] = np.mean(RT_trunc_dir)
    #         meanRT_trfwd_fft[i_experiment] = np.mean(RT_trunc_fft)
    #
    #         # RMSE_LL_gdual[i_experiment]     = np.sqrt(np.nanmean((LL_gdual     - LL_lsgdual) ** 2))
    #         # RMSE_LL_trfwd_dir[i_experiment] = np.sqrt(np.nanmean((LL_trunc_dir - LL_lsgdual) ** 2))
    #         # RMSE_LL_trfwd_fft[i_experiment] = np.sqrt(np.nanmean((LL_trunc_fft - LL_lsgdual) ** 2))
    #
    #     x_axis_label = r'$\Lambda$'
    #     RMSE_y_axis_label = r'RMSE'
    #     RT_y_axis_label = r'RT'
    #     method_names = ['LSGDual', 'GDual', 'Trunc w/ FFT', 'Trunc w/ Direct Conv']
    #     method_filenames = ['lsgd', 'gdual', 'trfft', 'trdir']
    #     RMSEsuffix = '_rmse.png'
    #     RTsuffix = '_rt.png'
    #
    #     # plot RMSE results
    #     RMSE_data = [RMSE_LL_lsgdual, RMSE_LL_gdual, RMSE_LL_trfwd_fft]
    #     for i_method in range(3):
    #         fig = plt.figure()
    #         plt.plot(Lambda_scale, RMSE_data[i_method])
    #
    #         plt.xlabel(x_axis_label)
    #         plt.ylabel(RMSE_y_axis_label)
    #
    #         plt.title(method_names[i_method])
    #
    #         fig.savefig(os.path.join(experiment_folder, method_filenames[i_method] + RMSEsuffix))
    #
    #     RT_data = [meanRT_lsgdual, meanRT_gdual, meanRT_trfwd_fft, meanRT_trfwd_dir]
    #     for i_method in range(4):
    #         fig = plt.figure()
    #         plt.plot(Lambda_scale, RT_data[i_method])
    #
    #         plt.xlabel(x_axis_label)
    #         plt.ylabel(RT_y_axis_label)
    #
    #         plt.title(method_names[i_method])
    #
    #         fig.savefig(os.path.join(experiment_folder, method_filenames[i_method] + RTsuffix))


if __name__ == "__main__":
    stability_experiment_suite('demo')