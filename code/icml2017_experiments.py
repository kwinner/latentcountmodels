import os
import pwd
import time
import cProfile

import numpy as np
from scipy import stats, integrate

from distributions import *

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
        silent  = True,
        arrival = 'poisson',
        branch  = 'binomial',
        observ  = 'binomial'
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
    Theta  = {'arrival': Lambda,
              'branch':  Delta,
              'observ':  Rho}

    Lambda_trunc = Lambda.reshape((-1, 1))
    Delta_trunc  = Delta.reshape((-1, 1))

    # configure distributions
    if arrival == 'poisson':
        arrival_pmf = stats.poisson
        arrival_pgf = lambda s, theta: poisson_pgf(s, theta)
    elif arrival == 'negbin':
        arrival_pmf = stats.nbinom
        arrival_pgf = lambda s, theta: negbin_pgf(s, theta)
    elif arrival == 'logser':
        arrival_pmf = stats.logser
        arrival_pgf = lambda s, theta: logarithmic_pgf(s, theta)
    elif arrival == 'geom':
        arrival_pmf = stats.geom
        arrival_pgf = lambda s, theta: geometric_pgf(s, theta)

    if branch  == 'binomial':
        branch_fun  = truncatedfa.binomial_branching
        branch_pgf  = lambda s, theta: bernoulli_pgf(s, theta)
    elif branch == 'poisson':
        branch_fun  = truncatedfa.poisson_branching
        branch_pgf  = lambda s, theta: poisson_pgf(s, theta)

    if observ  == 'binomial':
        observ_pgf  = None

    for iter in range(0, n_reps):
        if not silent: print "Iteration %d of %d" % (iter, n_reps)

        # sample data
        for i in range(0, K):
            if i == 0:
                N[iter, i] = arrival_pmf.rvs(Lambda[i])
            else:
                N[iter, i] = arrival_pmf.rvs(Lambda[i]) + stats.binom.rvs(N[iter, i-1], Delta[i-1])
            y[iter, i] = stats.binom.rvs(N[iter, i], Rho[i])

        if not silent: print y[iter,:]

        # likelihood from UTPPGFFA
        t_start = time.clock()
        Alpha_utppgffa = UTPPGFFA.utppgffa(y[iter, :],
                                           Theta,
                                           arrival_pgf,
                                           branch_pgf,
                                           observ_pgf,
                                           d=3)
        likelihood_utppgffa = Alpha_utppgffa[-1][0]
        # likelihood_utppgffa = Alpha_utppgffa[-1].data[0,0]
        runtime_utppgffa[iter] = time.clock() - t_start
        if not silent: print "UTPPGFFA: %0.4f" % runtime_utppgffa[iter]

        # # likelihood from PGFFA
        # t_start = time.clock()
        # a, b, f = pgffa.pgf_forward(Lambda,
        #                                 Rho,
        #                                 Delta,
        #                                 y[iter, :])
        # runtime_pgffa[iter] = time.clock() - t_start
        # if not silent: print "PGFFA: %0.4f" % runtime_pgffa[iter]

        # # likelihood from truncated forward algorithm
        # n_max[iter] = max(y[iter, :])
        # t_start = time.clock()
        # likelihood_trunc = float('inf')
        # while abs(likelihood_trunc - likelihood_utppgffa) >= epsilon and n_max[iter] < N_LIMIT:
        #     n_max[iter] += 1
        #     t_loop = time.clock()
        #     Alpha_trunc, z = truncatedfa.truncated_forward(arrival_pmf,
        #                                                    Lambda_trunc,
        #                                                    branch_fun,
        #                                                    Delta_trunc,
        #                                                    Rho,
        #                                                    y[iter, :],
        #                                                    n_max=n_max[iter])
        #     likelihood_trunc = truncatedfa.likelihood(z, log=False)
        #     runtime_trunc_final[iter] = time.clock() - t_loop
        # runtime_trunc_total[iter] = time.clock() - t_start
        # if not silent: print "Trunc: %0.4f last run @%d, %0.4f total" % (runtime_trunc_final[iter], n_max[iter], runtime_trunc_total[iter])

    return runtime_utppgffa, runtime_pgffa, runtime_trunc_final, runtime_trunc_total, n_max, y


def runtime_nmix(
        lmbda   = 20,   # mean population size
        R       = 10,   # number of counts (equiv to K elsewhere)
        rho     = 0.8,  # detection probability
        epsilon = 1e-4, # allowable error in truncated fa
        n_reps  = 10,   # number of times to repeat the experiment
        N_LIMIT = 1000, # hard cap on the max value for the truncated algorithm
        silent  = True,
        arrival = 'poisson',
        observ  = 'binomial'
        ):
    # organize parameters for pgffa, utppgffa and trunfa
    Lambda = np.concatenate(([lmbda], np.zeros(R)))
    Delta  = np.ones(R - 1)
    Rho    = rho * np.ones(R)

    return runtime_hmm(Lambda, Delta, Rho, epsilon, n_reps, N_LIMIT, silent, arrival, 'binomial', observ)


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
        silent  = True,
        arrival = 'poisson',
        branch  = 'binomial',
        observ  = 'binomial'
        ):
    K = len(T)

    Lambda, Delta = zonn_params(mu, sigma, omega, T, N)

    Rho = rho * np.ones(K)

    return runtime_hmm(Lambda, Delta, Rho, epsilon, n_reps, N_LIMIT, silent, arrival, branch, observ)


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
        silent,
        arrival,
        branch,
        observ,
        N_val,
        rho_val,
        K_val,
        resultdir
        ):
    runtime_utppgffa, runtime_pgffa, runtime_trunc_final, runtime_trunc_total, n_max, y = runtime_hmm(Lambda,Delta,Rho,epsilon,n_reps,N_LIMIT,silent,arrival,branch,observ)

    # compute average runtimes
    record = np.array([
        N_val,
        rho_val,
        K_val,
        np.mean(runtime_utppgffa),
        np.mean(runtime_pgffa),
        np.mean(runtime_trunc_final)
    ])

    filename = "N%dR%0.2fK%d" % (N_val, rho_val, K_val)
    np.save(os.path.join(resultdir, filename), record)


def runtime_experiment_zonn(N_space   = np.arange(10,100,10),
                            rho_space = np.arange(0.05, 1.00, 0.05),
                            K_space   = np.array([5]),
                            mu      = 7.0,               # mean arrival time
                            sigma   = 4.0,               # SD of arrival
                            omega   = 3.0,               # exponential survival param
                            T_min   = 0,
                            T_max   = 19,
                            epsilon = 1e-4,              # error tolerance in truncated fa
                            n_reps  = 10,                # number of times to repeat the experiment
                            N_LIMIT = 1000,              # hard cap on the max value for the truncated algorithm
                            silent  = True,
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
        T = np.arange(T_min, T_max, (T_max - T_min) / (K-1))
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
                                            silent,
                                            arrival,
                                            branch,
                                            observ,
                                            N,
                                            rho,
                                            K,
                                            resultdir)


if __name__ == "__main__":
    # runtime_utppgffa, runtime_pgffa, runtime_trunc_final, runtime_trunc_total, n_max, y = runtime_hmm_zonn(silent=False)
    # runtime_nmix()

    # runtime_experiment_zonn(silent=False)

    def runtime_profile():
        for i in range(0,100):
            runtime_hmm_zonn(silent=True)
    cProfile.run('runtime_profile()','utppgffa-vec+affine.stats')