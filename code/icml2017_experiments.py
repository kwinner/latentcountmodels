import os
import sys
from glob import glob
import pwd
import time
import cProfile
import pickle
import itertools
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats, integrate

# from distributions import *
from distributions_cython import *

import UTPPGFFA
import pgffa_kev as pgffa
# import pgffa
import truncatedfa


def runtime_hmm(
        Lambda  = 10 * np.array([0.0257, 0.1163, 0.2104, 0.1504, 0.0428]),
        Delta   = np.array([0.2636, 0.2636, 0.2636, 0.2636]),
        Rho     = 0.5 * np.ones(5),
        epsilon = 1e-10, # allowable error in truncated fa
        n_reps  = 10,   # number of times to repeat the experiment
        N_LIMIT = 1000, # hard cap on the max value for the truncated algorithm
        verbose = True,
        arrival = 'poisson',
        branch  = 'binomial',
        observ  = 'binomial',
        CONV_LIMIT = 1e-10
        ):

    K = len(Lambda)

    # sample record
    N = np.zeros((n_reps, K)).astype(int)
    y = np.zeros((n_reps, K)).astype(int)

    # runtime record
    runtime_trunc_final = np.zeros(n_reps)
    runtime_trunc_total = np.zeros(n_reps)
    runtime_pgffa       = np.zeros(n_reps)
    runtime_utppgffa    = np.zeros(n_reps)

    # truncated fa final truncation value
    n_max = np.zeros(n_reps).astype(int)

    # organize parameters for pgffa, utppgffa and trunfa
    Theta  = {'arrival': Lambda.reshape((-1, 1)),
              'branch':  Delta.reshape((-1, 1)),
              'observ':  Rho}

    Lambda_trunc = Lambda.reshape((-1, 1))
    Delta_trunc  = Delta.reshape((-1, 1))

    # configure distributions
    if arrival == 'poisson':
        arrival_pmf = stats.poisson
        # arrival_pgf = lambda s, theta: poisson_pgf(s, theta)
        arrival_pgf = poisson_utppgf_cython
    elif arrival == 'negbin':
        arrival_pmf = stats.nbinom
        # arrival_pgf = lambda s, theta: negbin_pgf(s, theta)
        arrival_pgf = negbin_utppgf_cython
    elif arrival == 'logser':
        arrival_pmf = stats.logser
        # arrival_pgf = lambda s, theta: logarithmic_pgf(s, theta)
        arrival_pgf = logarithmic_utppgf_cython
    elif arrival == 'geom':
        arrival_pmf = stats.geom
        # arrival_pgf = lambda s, theta: geometric_pgf(s, theta)
        arrival_pgf = geometric_utppgf_cython

    if branch  == 'binomial':
        branch_fun  = truncatedfa.binomial_branching
        # branch_pgf  = lambda s, theta: bernoulli_pgf(s, theta)
        branch_pgf = bernoulli_utppgf_cython
    elif branch == 'poisson':
        branch_fun  = truncatedfa.poisson_branching
        # branch_pgf  = lambda s, theta: poisson_pgf(s, theta)
        branch_pgf = poisson_utppgf_cython

    if observ  == 'binomial':
        observ_pgf  = None

    for iter in range(0, n_reps):
        if verbose == "full": print "Iteration %d of %d" % (iter, n_reps)

        attempt = 1
        while True:
            try:
                # sample data
                for i in range(0, K):
                    if i == 0:
                        N[iter, i] = arrival_pmf.rvs(Lambda[i])
                    else:
                        N[iter, i] = arrival_pmf.rvs(Lambda[i]) + stats.binom.rvs(N[iter, i-1], Delta[i-1])
                    y[iter, i] = stats.binom.rvs(N[iter, i], Rho[i])

                if verbose == "full": print y[iter,:]

                # likelihood from UTPPGFFA
                t_start = time.clock()
                Alpha_utppgffa, logZ_utppgffa = UTPPGFFA.utppgffa(y[iter, :],
                                                               Theta,
                                                               arrival_pgf,
                                                               branch_pgf,
                                                               observ_pgf,
                                                               d=1,
                                                               normalized=True)
                loglikelihood_utppgffa = np.log(Alpha_utppgffa[-1][0]) + np.sum(logZ_utppgffa)
                runtime_utppgffa[iter] = time.clock() - t_start
                if verbose == "full": print "UTPPGFFA: %0.4f" % runtime_utppgffa[iter]

                # likelihood from PGFFA
                t_start = time.clock()
                a, b, f = pgffa.pgf_forward(Lambda,
                                                Rho,
                                                Delta,
                                                y[iter, :])
                runtime_pgffa[iter] = time.clock() - t_start
                if verbose == "full": print "PGFFA: %0.4f" % runtime_pgffa[iter]

                # likelihood from truncated forward algorithm
                n_max[iter] = max(y[iter, :])
                t_start = time.clock()
                loglikelihood_trunc = float('inf')
                loglikelihood_diff  = float('inf')
                while abs(loglikelihood_trunc - loglikelihood_utppgffa) >= epsilon and \
                      loglikelihood_diff >= CONV_LIMIT and \
                      n_max[iter] < N_LIMIT:
                # while abs(1 - (loglikelihood_trunc / loglikelihood_utppgffa)) >= epsilon and n_max[iter] < N_LIMIT:
                    n_max[iter] += 1
                    t_loop = time.clock()
                    Alpha_trunc, z = truncatedfa.truncated_forward(arrival_pmf,
                                                                   Lambda_trunc,
                                                                   branch_fun,
                                                                   Delta_trunc,
                                                                   Rho,
                                                                   y[iter, :],
                                                                   n_max=n_max[iter])
                    loglikelihood_iter = truncatedfa.likelihood(z, log=True)
                    loglikelihood_diff = abs(loglikelihood_trunc - loglikelihood_iter)
                    loglikelihood_trunc = loglikelihood_iter
                    runtime_trunc_final[iter] = time.clock() - t_loop
                runtime_trunc_total[iter] = time.clock() - t_start

                if verbose == "full": print "Trunc: %0.4f last run @%d, %0.4f total" % (runtime_trunc_final[iter], n_max[iter], runtime_trunc_total[iter])

                if n_max[iter] >= N_LIMIT:
                    print "Attempt #%d, trunc failed to converge." % attempt
                    attempt += 1
                else:
                    break
            except Exception as inst:
                print "Attempt #%d failed, Error: " % attempt, inst
                attempt += 1
    return runtime_utppgffa, runtime_pgffa, runtime_trunc_final, runtime_trunc_total, n_max, y, N


def runtime_nmix(
        lmbda   = 20,   # mean population size
        R       = 10,   # number of counts (equiv to K elsewhere)
        rho     = 0.8,  # detection probability
        epsilon = 1e-4, # allowable error in truncated fa
        n_reps  = 10,   # number of times to repeat the experiment
        N_LIMIT = 1000, # hard cap on the max value for the truncated algorithm
        verbose = "silent",
        arrival = 'poisson',
        observ  = 'binomial'
        ):
    # organize parameters for pgffa, utppgffa and trunfa
    Lambda = np.concatenate(([lmbda], np.zeros(R)))
    Delta  = np.ones(R - 1)
    Rho    = rho * np.ones(R)

    return runtime_hmm(Lambda, Delta, Rho, epsilon, n_reps, N_LIMIT, verbose, arrival, 'binomial', observ)


def runtime_hmm_zonn(
        mu      = 8.0,               # mean arrival time
        sigma   = 4.0,               # SD of arrival
        omega   = 3.0,               # exponential survival param
        N       = 200,                # superpopulation size
        T       = np.arange(1,20,4), # survey times
        rho     = 0.8,               # detection probability
        epsilon = 1e-4,              # error tolerance in truncated fa
        n_reps  = 10,                # number of times to repeat the experiment
        N_LIMIT = 1000,              # hard cap on the max value for the truncated algorithm
        verbose = "silent",
        arrival = 'poisson',
        branch  = 'binomial',
        observ  = 'binomial'
        ):
    K = len(T)

    Lambda, Delta = zonn_params(mu, sigma, omega, T, N)

    Rho = rho * np.ones(K)

    return runtime_hmm(Lambda, Delta, Rho, epsilon, n_reps, N_LIMIT, verbose, arrival, branch, observ)


def zonn_params(mu, sigma, omega, T, N):
    K = len(T)

    #append -inf, inf to T
    T_all = np.concatenate(([-float("inf")], T.copy(), [float("inf")]))

    # compute Lambda, Delta from mu, sigma, omega, N parameters
    Lambda = np.zeros(K)
    for i in range(0, K):
        Lambda[i] = N * integrate.quad(
            lambda (t): stats.norm.pdf(t, mu, sigma) * stats.expon.sf(T_all[i + 1] - t, scale=omega),
            T_all[i],
            T_all[i + 1])[0]
    Delta = np.zeros(K - 1)
    for i in range(0, K - 1):
        Delta[i] = stats.expon.sf(T[i + 1] - T[i], scale=omega)

    return Lambda, Delta


# compute a timestamped directory to put the results of this experiment into
def default_result_directory():
    # full path if you're Kevin, relative path otherwise (feel free to add your own full paths!)
    username = pwd.getpwuid(os.getuid())[0]
    if username == 'kwinner':
        base = '/Users/kwinner/Work/Data/Results'
    else:
        base = 'Results'

    timestamp = time.strftime("%Y%m%dT%H%M%S") + ("%03d" % int(round((time.time() * 1000) % 1000)))

    return os.path.join(base, timestamp)


def runtime_hmm_shannon_wrapper(
        Lambda,
        Delta,
        Rho,
        epsilon,
        n_reps,
        N_LIMIT,
        verbose,
        arrival,
        branch,
        observ,
        N_val,
        rho_val,
        K_val,
        resultdir
        ):
    runtime_utppgffa, runtime_pgffa, runtime_trunc_final, runtime_trunc_total, n_max, y, N = runtime_hmm(Lambda,Delta,Rho,epsilon,n_reps,N_LIMIT,verbose,arrival,branch,observ)

    pickle_method = 'dict'
    if pickle_method == 'array':
        #compute average runtimes
        record = np.array([
            N_val,
            rho_val,
            K_val,
            np.mean(runtime_utppgffa),
            np.mean(runtime_pgffa),
            np.mean(runtime_trunc_final),
            np.mean(y),
            np.mean(N)
        ])

        # print record[3:]

        filename = "N%dR%0.2fK%d" % (N_val, rho_val, K_val)
        np.save(os.path.join(resultdir, filename), record)
    elif pickle_method == 'dict':
        record = {
            "N_val": N_val,
            "rho_val": rho_val,
            "K_val": K_val,
            "runtime_utppgffa": runtime_utppgffa,
            "runtime_pgffa": runtime_pgffa,
            "runtime_trunc": runtime_trunc_final,
            "y": y,
            "N": N,
            "n_max": n_max
        }

        if verbose != "silent": print "N: %d, rho: %0.2f, K: %d\nutppgffa: %0.4f\npgffa: %0.4f\ntrunc: %0.4f\n" % (N_val, rho_val, K_val, np.mean(runtime_utppgffa), np.mean(runtime_pgffa), np.mean(runtime_trunc_final))

        filename = "N%dR%0.2fK%d.pickle" % (N_val, rho_val, K_val)
        pickle.dump(record, open(os.path.join(resultdir, filename), 'wb'))


def runtime_experiment_plot(resultdir):
    # read in experiment metadata
    metafile = open(os.path.join(resultdir, "meta.txt"))
    for line in metafile:
        if line.startswith("Repetitions:"):
            nReps = int(line[len("Repetitions:"):].strip())
        elif line.startswith("epsilon:"):
            epsilon = float(line[len("epsilon:"):].strip())
        elif line.startswith("Arrivals:"):
            arrival = line[len("Arrivals:"):].strip()
        elif line.startswith("Branching:"):
            branch = line[len("Branching:"):].strip()
        elif line.startswith("Observations:"):
            observ = line[len("Observations:"):].strip()
    metafile.close()

    # read in the results in resultdir
    resultlist = glob(os.path.join(resultdir, "*.pickle"))
    nResults = len(resultlist)

    # collect results from all experiments
    N_val            = np.ndarray((nResults))
    rho_val          = np.ndarray((nResults))
    K_val            = np.ndarray((nResults))
    runtime_utppgffa = np.ndarray((nResults, nReps))
    runtime_pgffa    = np.ndarray((nResults, nReps))
    runtime_trunc    = np.ndarray((nResults, nReps))
    y_sum            = np.ndarray((nResults, nReps))
    y_mean           = np.ndarray((nResults, nReps))
    y_var            = np.ndarray((nResults, nReps))
    n_sum            = np.ndarray((nResults, nReps))
    n_mean           = np.ndarray((nResults, nReps))
    n_var            = np.ndarray((nResults, nReps))
    n_max            = np.ndarray((nResults, nReps))

    for iResult in xrange(0,nResults):
        result = pickle.load(open(resultlist[iResult], 'rb'))
        N_val[iResult]   = result['N_val']
        rho_val[iResult] = result['rho_val']
        K_val[iResult]   = result['K_val']
        runtime_utppgffa[iResult, :] = result['runtime_utppgffa']
        runtime_pgffa[iResult, :]    = result['runtime_pgffa']
        runtime_trunc[iResult, :]    = result['runtime_trunc']
        n_max[iResult, :]            = result['n_max']
        y_sum[iResult, :]  = np.sum(result['y'], axis=1)
        y_mean[iResult, :] = np.mean(result['y'], axis=1)
        y_var[iResult, :]  = np.var(result['y'], axis=1)
        n_sum[iResult, :]  = np.sum(result['N'], axis=1)
        n_mean[iResult, :] = np.mean(result['N'], axis=1)
        n_var[iResult, :]  = np.var(result['N'], axis=1)
        # if N_val[iResult] == 200 and K_val[iResult] == 5 and rho_val[iResult] == 0.8:
        #     True

    # take the mean over all repetitions of the experiment
    mean_runtime_utppgffa = np.mean(runtime_utppgffa, axis=1)
    mean_runtime_pgffa    = np.mean(runtime_pgffa, axis=1)
    mean_runtime_trunc    = np.mean(runtime_trunc, axis=1)

    # setup latex
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    ### plot n_max vs Y
    # plt.scatter(np.ravel(y_sum), np.ravel(n_max))
    # plt.xlabel(r'$Y = \sum y$')
    # plt.ylabel(r'$n_{max}$')
    # plt.title('Total observed counts vs truncation parameter')

    ### plot n_max vs rho
    # plt.scatter(np.ravel(rho_val), np.ravel(np.mean(n_max, axis=1)))
    # plt.xlabel(r'$Y = \sum y$')
    # plt.ylabel(r'$\rho$')
    # plt.title('Rho vs truncation parameter')

    ### plot n_max vs runtime of trunc
    # handle_trunc = plt.scatter(np.ravel(n_max), np.ravel(runtime_trunc), color="#352A87", label="Trunc", alpha=0.1)
    # plt.xlabel(r'$n_{max}$')
    # plt.title('Runtime as a function of truncation parameter')
    # plt.xlim(np.min(n_max), np.max(n_max))
    # plt.ylabel('Runtime (s)')
    # plt.ylim(np.min(runtime_trunc), np.max(runtime_trunc))

    ### plot runtime vs Y
    # handle_utppgffa = plt.scatter(np.ravel(y_sum), np.ravel(runtime_utppgffa), color="#352A87", label="UTPPGFFA", alpha=0.25)
    # handle_pgffa    = plt.scatter(np.ravel(y_sum), np.ravel(runtime_pgffa), color="#33B8A1", label="PGFFA", alpha=0.25)
    # handle_trunc    = plt.scatter(np.ravel(y_sum), np.ravel(runtime_trunc), color="#F9FB0E", label="Trunc", alpha=0.25)
    #
    # plt.legend(handles=[handle_trunc, handle_pgffa, handle_utppgffa], loc=2)
    # plt.ylabel('Runtime (s)')
    # plt.xlabel(r'$Y = \sum y$')
    # plt.title('Runtime as a function of total observed counts')
    # plt.xlim(np.min(y_sum), np.max(y_sum))
    # plt.ylim(0, max((np.max(runtime_utppgffa), np.max(runtime_pgffa), np.max(runtime_trunc))))

    ### plot runtime vs rho
    # plotalpha = 0.5
    # handle_utppgffa = plt.scatter(np.ravel(rho_val), np.ravel(mean_runtime_utppgffa), color="#352A87", label="UTPPGFFA", alpha=plotalpha)
    # handle_pgffa = plt.scatter(np.ravel(rho_val), np.ravel(mean_runtime_pgffa), color="#33B8A1", label="PGFFA", alpha=plotalpha)
    # handle_trunc = plt.scatter(np.ravel(rho_val), np.ravel(mean_runtime_trunc), color="#F9FB0E", label="Trunc", alpha=plotalpha)
    #
    # plt.legend(handles=[handle_trunc, handle_pgffa, handle_utppgffa], loc=2)
    # plt.ylabel('Runtime (s)')
    # plt.xlabel(r'$\rho$')
    # plt.title('Runtime as a function of rho')
    # plt.xlim(0, 1)
    # plt.ylim(0, max((np.max(runtime_utppgffa), np.max(runtime_pgffa), np.max(runtime_trunc))))

    ### plot runtime vs rho*Lambda
    # plotalpha = 1.0
    # rhoLambda = np.multiply(rho_val, N_val)
    # handle_utppgffa = plt.scatter(np.ravel(rhoLambda), np.ravel(mean_runtime_utppgffa), color="#352A87", label="UTPPGFFA", alpha=plotalpha)
    # handle_pgffa = plt.scatter(np.ravel(rhoLambda), np.ravel(mean_runtime_pgffa), color="#33B8A1", label="PGFFA", alpha=plotalpha)
    # handle_trunc = plt.scatter(np.ravel(rhoLambda), np.ravel(mean_runtime_trunc), color="#F9FB0E", label="Trunc", alpha=plotalpha)
    #
    # plt.legend(handles=[handle_trunc, handle_pgffa, handle_utppgffa], loc=2)
    # plt.ylabel('Runtime (s)')
    # plt.xlabel(r'$\rho\Lambda$')
    # plt.title(r'Runtime as a function of $\rho\Lambda$')
    # plt.xlim(np.min(rhoLambda), np.max(rhoLambda))
    # plt.ylim(0, max((np.max(runtime_utppgffa), np.max(runtime_pgffa), np.max(runtime_trunc))))

    ### plot runtime vs rho*Lambda (only for K = 5)
    # plotalpha = 0.5
    # rhoLambda = np.multiply(rho_val, N_val)
    # handle_utppgffa = plt.scatter(np.ravel(rhoLambda[K_val == 5]), np.ravel(mean_runtime_utppgffa[K_val == 5]), color="#352A87",
    #                               label="UTPPGFFA", alpha=plotalpha)
    # handle_pgffa = plt.scatter(np.ravel(rhoLambda[K_val == 5]), np.ravel(mean_runtime_pgffa[K_val == 5]), color="#33B8A1", label="PGFFA",
    #                            alpha=plotalpha)
    # handle_trunc = plt.scatter(np.ravel(rhoLambda[K_val == 5]), np.ravel(mean_runtime_trunc[K_val == 5]), color="#F9FB0E", label="Trunc",
    #                            alpha=plotalpha)
    #
    # plt.legend(handles=[handle_trunc, handle_pgffa, handle_utppgffa], loc=2)
    # plt.ylabel('Runtime (s)')
    # plt.xlabel(r'$\rho\Lambda$')
    # plt.title(r'Runtime as a function of $\rho\Lambda$')
    # plt.xlim(np.min(rhoLambda[K_val == 5]), np.max(rhoLambda[K_val == 5]))
    # plt.ylim(0, max((np.max(runtime_utppgffa[K_val == 5]), np.max(runtime_pgffa[K_val == 5]), np.max(runtime_trunc[K_val == 5]))))

    ### plot difference in runtime between trunc, utppgffa vs rho
    # plt.scatter(np.ravel(rho_val), np.ravel(mean_runtime_utppgffa - mean_runtime_trunc), color="#352A87")
    # plt.ylabel('T(utppgffa) - T(trunc)')
    # plt.xlabel(r'$\rho$')
    # plt.title(r'Runtime difference as a function of $\rho$')

    ### plot runtime vs Lambda (for some fixed rho)
    rho_target = 0.5
    plotalpha = 1.0

    N_val                 = N_val[rho_val == rho_target]
    mean_runtime_utppgffa = mean_runtime_utppgffa[rho_val == rho_target]
    mean_runtime_pgffa    = mean_runtime_pgffa[rho_val == rho_target]
    mean_runtime_trunc    = mean_runtime_trunc[rho_val == rho_target]

    lists = sorted(itertools.izip(*[N_val, mean_runtime_utppgffa, mean_runtime_pgffa, mean_runtime_trunc]))
    N_val, mean_runtime_utppgffa, mean_runtime_pgffa, mean_runtime_trunc = list(itertools.izip(*lists))

    handle_trunc    = plt.plot(np.ravel(N_val),
                               np.ravel(mean_runtime_trunc),
                               color="#fabf0e", label="Trunc", alpha=plotalpha)
    handle_pgffa    = plt.plot(np.ravel(N_val),
                               np.ravel(mean_runtime_pgffa),
                               color="#33B8A1", label="PGFFA", alpha=plotalpha)
    handle_utppgffa = plt.plot(np.ravel(N_val),
                               np.ravel(mean_runtime_utppgffa),
                               color="#352A87", label="UTPPGFFA", alpha=plotalpha)

    plt.legend(loc=2)
    plt.ylabel('Runtime (s)')
    plt.xlabel(r'$\Lambda$')
    plt.title(r'Runtime vs $\Lambda$ for $\rho = %s$' % str(rho_target))
    plt.xlim(np.min(N_val), np.max(N_val))
    plt.ylim(0, max((np.max(runtime_utppgffa), np.max(runtime_pgffa), np.max(runtime_trunc))))

    plt.show(block=True)

def runtime_experiment_gen(N_space   = np.arange(10,100,10),
                           rho_space = np.arange(0.05, 1.00, 0.05),
                           Lambda    = np.array([0.0257, 0.1163, 0.2104, 0.1504, 0.0428], dtype=np.float64), # unscaled Lambda
                           Delta     = np.array([0.2636, 0.2636, 0.2636, 0.2636], dtype=np.float64),
                           epsilon = 1e-6,              # error tolerance in truncated fa
                           n_reps  = 10,                # number of times to repeat the experiment
                           N_LIMIT = 1000,              # hard cap on the max value for the truncated algorithm
                           verbose = "silent",
                           arrival = 'poisson',
                           branch  = 'binomial',
                           observ  = 'binomial'
                           ):
    assert Lambda.shape[0] == Delta.shape[0] + 1
    K = Lambda.shape[0]

    resultdir = default_result_directory()
    os.mkdir(resultdir)

    f = open(os.path.join(resultdir, "meta.txt"),'w')
    f.write("Experiment parameters:\n")
    f.write("Arrivals: %s\n" % arrival)
    f.write("Branching: %s\n" % branch)
    f.write("Observations: %s\n" % observ)
    f.write("Repetitions: %d\n" % n_reps)
    f.write("N values: %s\n" % str(N_space))
    f.write("Rho values: %s\n" % str(rho_space))
    f.write("Lambda: %s\n" % str(Lambda))
    f.write("Delta: %s\n" % str(Delta))
    f.write("epsilon: %f\n" % epsilon)
    f.close()

    for iN in range(0, len(N_space)):
        N = N_space[iN]
        Lambda_iter = N * Lambda
        for iRho in range(0, len(rho_space)):
            rho = rho_space[iRho]
            Rho = np.full(K, rho)

            runtime_hmm_shannon_wrapper(Lambda_iter,
                                        Delta,
                                        Rho,
                                        epsilon,
                                        n_reps,
                                        N_LIMIT,
                                        verbose,
                                        arrival,
                                        branch,
                                        observ,
                                        N,
                                        rho,
                                        K,
                                        resultdir)
    return resultdir


def runtime_experiment_zonn(N_space   = np.arange(10,100,10),
                            rho_space = np.arange(0.05, 1.00, 0.05),
                            K_space   = np.array([5,8]),
                            mu      = 8.0,               # mean arrival time
                            sigma   = 4.0,               # SD of arrival
                            omega   = 3.0,               # exponential survival param
                            T_min   = 1.,
                            T_max   = 17.,
                            epsilon = 1e-6,              # error tolerance in truncated fa
                            n_reps  = 10,                # number of times to repeat the experiment
                            N_LIMIT = 1000,              # hard cap on the max value for the truncated algorithm
                            verbose = "silent",
                            arrival = 'poisson',
                            branch  = 'binomial',
                            observ  = 'binomial'
                            ):
    resultdir = default_result_directory()
    os.mkdir(resultdir)

    f = open(os.path.join(resultdir, "meta.txt"),'w')
    f.write("Experiment parameters:\n")
    f.write("Arrivals: %s\n" % arrival)
    f.write("Branching: %s\n" % branch)
    f.write("Observations: %s\n" % observ)
    f.write("Repetitions: %d\n" % n_reps)
    f.write("N values: %s\n" % str(N_space))
    f.write("Rho values: %s\n" % str(rho_space))
    f.write("K values: %s\n" % str(K_space))
    f.write("mu: %f\n" % mu)
    f.write("sigma: %f\n" % sigma)
    f.write("omega: %f\n" % omega)
    f.write("epsilon: %f\n" % epsilon)
    f.close()

    for iK in range(0, len(K_space)):
        K = K_space[iK]
        T = np.linspace(T_min, T_max, K)
        for iN in range(0, len(N_space)):
            N = N_space[iN]
            Lambda, Delta = zonn_params(mu, sigma, omega, T, N)
            for iRho in range(0, len(rho_space)):
                rho = rho_space[iRho]
                Rho = np.full(K, rho)

                runtime_hmm_shannon_wrapper(Lambda,
                                            Delta,
                                            Rho,
                                            epsilon,
                                            n_reps,
                                            N_LIMIT,
                                            verbose,
                                            arrival,
                                            branch,
                                            observ,
                                            N,
                                            rho,
                                            K,
                                            resultdir)
    return resultdir


def nmax_vs_runtime(nmax_space = np.arange(42, 250),
                    y = np.array([5, 21, 42, 36, 15]),
                    nReps = 10):
    arrival_pmf = stats.poisson
    branch_fun = truncatedfa.binomial_branching
    Lambda, Delta = zonn_params(8., 4., 3., np.linspace(1,17,5),50)
    Rho = 0.8 * np.ones(5)

    runtime_trunc = np.zeros_like(nmax_space, dtype=float)
    for in_max in xrange(0,len(nmax_space)):
        for i in xrange(0,nReps):
            t_loop = time.clock()
            Alpha_trunc, z = truncatedfa.truncated_forward(arrival_pmf,
                                                           Lambda.reshape((-1, 1)),
                                                           branch_fun,
                                                           Delta.reshape((-1, 1)),
                                                           Rho,
                                                           y,
                                                           n_max=nmax_space[in_max])
            loglikelihood_trunc = truncatedfa.likelihood(z, log=True)
            runtime_trunc[in_max] += time.clock() - t_loop
        runtime_trunc[in_max] /= float(nReps)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    handle_trunc = plt.scatter(np.ravel(nmax_space), np.ravel(runtime_trunc), color="#352A87", label="Trunc", alpha=1.0)
    plt.xlabel(r'$n_{max}$')
    plt.title('Runtime as a function of truncation parameter')
    plt.xlim(np.min(nmax_space), np.max(nmax_space))
    plt.ylabel('Runtime (s)')
    plt.ylim(np.min(runtime_trunc), np.max(runtime_trunc))

    plt.show(block=True)


if __name__ == "__main__":
    # runtime_utppgffa, runtime_pgffa, runtime_trunc_final, runtime_trunc_total, n_max, y, N = runtime_hmm_zonn(verbose="full")
    # runtime_nmix()

    # resultdir = runtime_experiment_zonn(verbose="partial",
    #                                     sigma = 2.,
    #                                     # N_space=np.append(np.arange(10,101,10), np.arange(125, 501, 25)),
    #                                     N_space=np.arange(25,501,50),
    #                                     K_space=np.array([5]),
    #                                     rho_space=np.arange(0.05,0.95,0.05),
    #                                     n_reps=20,
    #                                     epsilon=1e-5)
    # resultdir = runtime_experiment_gen(verbose="partial",
    #                                    Lambda=np.array([1.,0,0,0,0],dtype=np.float64),
    #                                    Delta=0.4 * np.ones(4,dtype=np.float64),
    #                                    N_space=np.arange(25,101,25),
    #                                    rho_space=np.arange(0.1,0.96,0.4),
    #                                    n_reps=50,
    #                                    epsilon=1e-5,
    #                                    )
    # print resultdir

    # resultdir = "/Users/kwinner/Work/Data/Results/20170215T115245506"
    # resultdir = "/Users/kwinner/Work/Data/Results/20170213T232631920"
    # resultdir = "/Users/kwinner/Work/Data/Results/20170217T111957694"
    # resultdir = "/Users/kwinner/Work/Data/Results/20170219T004331426"
    # resultdir = "/Users/kwinner/Work/Data/Results/20170219T011553514"
    # resultdir = "/Users/kwinner/Work/Data/Results/20170219T151119276"
    resultdir = "/Users/kwinner/Work/Data/Results/20170219T153853790"
    runtime_experiment_plot(resultdir)

    # nmax_vs_runtime()

    # def runtime_profile():
    #     for i in range(0,100):
    #         runtime_hmm_zonn(verbose="silent")
    # cProfile.run('runtime_profile()','utppgffa-vec+affine.stats')