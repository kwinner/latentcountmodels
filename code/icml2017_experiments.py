import os
import pwd
import time
import cProfile

import numpy as np
from scipy import stats, integrate

from distributions import *

import UTPPGFFA
import pgffa
import truncatedfa


def runtime_nmix_poisson(
        lmbda   = 20,   # mean population size
        R       = 10,   # number of counts (equiv to K elsewhere)
        rho     = 0.8,  # detection probability
        epsilon = 1e-4, # allowable error in truncated fa
        n_reps  = 10,   # number of times to repeat the experiment
        N_LIMIT = 1000  # hard cap on the max value for the truncated algorithm
):
    # setup output directory
    result_dir = default_result_directory()
    # os.mkdir(result_dir)

    # sample record
    N = np.zeros(n_reps).astype(int)
    y = np.zeros((n_reps, R)).astype(int)

    # runtime record
    runtime_trunc_final = np.zeros(n_reps)
    runtime_trunc_total = np.zeros(n_reps)
    runtime_pgffa       = np.zeros(n_reps)
    runtime_utppgffa    = np.zeros(n_reps)

    # truncated fa final truncation value
    n_max = np.zeros(n_reps).astype(int)

    # organize parameters for pgffa, utppgffa and trunfa
    Lambda = np.concatenate(([lmbda], np.zeros(R)))
    Delta  = np.ones(R - 1)
    Rho    = rho * np.ones(R)
    Theta  = {'arrival': Lambda,
              'branch':  Delta,
              'observ':  Rho}

    Lambda_trunc = Lambda.reshape((-1, 1))
    Delta_trunc  = Delta.reshape((-1, 1))

    arrival_pmf = stats.poisson
    arrival_pgf = lambda s, theta: poisson_pgf(s, theta)
    branch_fun  = truncatedfa.binomial_branching
    branch_pgf  = lambda s, theta: bernoulli_pgf(s, theta)
    observ_pgf  = None

    for iter in range(0, n_reps):
        print "Iteration %d of %d" % (iter, n_reps)

        # sample data
        N[iter] = stats.poisson.rvs(lmbda, size=1)
        y[iter, :] = stats.binom.rvs(N[iter], rho, size=R)

        # likelihood from UTPPGFFA
        t_start = time.clock()
        Alpha_utppgffa = UTPPGFFA.utppgffa(y[iter, :],
                                           Theta,
                                           arrival_pgf,
                                           branch_pgf,
                                           observ_pgf,
                                           d=3)
        likelihood_utppgffa = Alpha_utppgffa[-1].data[0,0]
        runtime_utppgffa[iter] = time.clock() - t_start
        print "UTPPGFFA: %0.2f" % runtime_utppgffa[iter]

        # likelihood from PGFFA
        t_start = time.clock()
        a, b, f = pgffa.pgf_forward(Lambda,
                                        Rho,
                                        Delta,
                                        y[iter, :])
        runtime_pgffa[iter] = time.clock() - t_start
        print "PGFFA: %0.2f" % runtime_pgffa[iter]

        # likelihood from truncated forward algorithm
        n_max[iter] = max(y[iter, :])
        t_start = time.clock()
        likelihood_trunc = float('inf')
        while abs(likelihood_trunc - likelihood_utppgffa) >= epsilon and n_max[iter] < N_LIMIT:
            n_max[iter] += 1
            t_loop = time.clock()
            Alpha_trunc, z = truncatedfa.truncated_forward(arrival_pmf,
                                                           Lambda.reshape((-1, 1)),
                                                           branch_fun,
                                                           Delta.reshape((-1, 1)),
                                                           Rho,
                                                           y[iter, :],
                                                           n_max=n_max[iter])
            likelihood_trunc = truncatedfa.likelihood(z, log=False)
            runtime_trunc_final[iter] = time.clock() - t_loop
        runtime_trunc_total[iter] = time.clock() - t_start
        print "Trunc: %0.3f last run @%d, %0.3f total" % (runtime_trunc_final[iter], n_max[iter], runtime_trunc_total[iter])

    return runtime_utppgffa, runtime_pgffa, runtime_trunc_final, runtime_trunc_total, n_max


def runtime_hmm_zonn(
        mu      = 8.0,               # mean arrival time
        sigma   = 4.0,               # SD of arrival
        omega   = 3.0,               # exponential survival param
        N       = 200,                # superpopulation size
        T       = np.arange(1,20,4), # survey times
        rho     = 0.8,               # detection probability
        epsilon = 1e-4,              # error tolerance in truncated fa
        n_reps  = 10,                # number of times to repeat the experiment
        N_LIMIT = 1000               # hard cap on the max value for the truncated algorithm
):
    K = len(T)

    Lambda, Delta = zonn_params(mu, sigma, omega, T, N)

    Rho = rho * np.ones(K)

    return runtime_hmm_poisson(Lambda, Delta, Rho, epsilon, n_reps, N_LIMIT)


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


def runtime_hmm_poisson(
        Lambda  = 10 * np.array([0.0257, 0.1163, 0.2104, 0.1504, 0.0428]),
        Delta   = np.array([0.2636, 0.2636, 0.2636, 0.2636]),
        Rho     = 0.5 * np.ones(5),
        epsilon = 1e-10, # allowable error in truncated fa
        n_reps  = 10,   # number of times to repeat the experiment
        N_LIMIT = 1000  # hard cap on the max value for the truncated algorithm
):
    # setup output directory
    result_dir = default_result_directory()
    # os.mkdir(result_dir)

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

    arrival_pmf = stats.poisson
    arrival_pgf = lambda s, theta: poisson_pgf(s, theta)
    branch_fun  = truncatedfa.binomial_branching
    branch_pgf  = lambda s, theta: bernoulli_pgf(s, theta)
    observ_pgf  = None

    for iter in range(0, n_reps):
        print "Iteration %d of %d" % (iter, n_reps)

        # sample data
        for i in range(0, K):
            if i == 0:
                N[iter, i] = arrival_pmf.rvs(Lambda[i])
            else:
                N[iter, i] = arrival_pmf.rvs(Lambda[i]) + stats.binom.rvs(N[iter, i-1], Delta[i-1])
            y[iter, i] = stats.binom.rvs(N[iter, i], Rho[i])

        print y[iter,:]

        # likelihood from UTPPGFFA
        t_start = time.clock()
        Alpha_utppgffa = UTPPGFFA.utppgffa(y[iter, :],
                                           Theta,
                                           arrival_pgf,
                                           branch_pgf,
                                           observ_pgf,
                                           d=3)
        likelihood_utppgffa = Alpha_utppgffa[-1].data[0,0]
        runtime_utppgffa[iter] = time.clock() - t_start
        print "UTPPGFFA: %0.2f" % runtime_utppgffa[iter]

        # likelihood from PGFFA
        t_start = time.clock()
        a, b, f = pgffa.pgf_forward(Lambda,
                                        Rho,
                                        Delta,
                                        y[iter, :])
        runtime_pgffa[iter] = time.clock() - t_start
        print "PGFFA: %0.2f" % runtime_pgffa[iter]

        # likelihood from truncated forward algorithm
        n_max[iter] = max(y[iter, :])
        t_start = time.clock()
        likelihood_trunc = float('inf')
        while abs(likelihood_trunc - likelihood_utppgffa) >= epsilon and n_max[iter] < N_LIMIT:
            n_max[iter] += 1
            t_loop = time.clock()
            Alpha_trunc, z = truncatedfa.truncated_forward(arrival_pmf,
                                                           Lambda_trunc,
                                                           branch_fun,
                                                           Delta_trunc,
                                                           Rho,
                                                           y[iter, :],
                                                           n_max=n_max[iter])
            likelihood_trunc = truncatedfa.likelihood(z, log=False)
            runtime_trunc_final[iter] = time.clock() - t_loop
        runtime_trunc_total[iter] = time.clock() - t_start
        print "Trunc: %0.3f last run @%d, %0.3f total" % (runtime_trunc_final[iter], n_max[iter], runtime_trunc_total[iter])

    return runtime_utppgffa, runtime_pgffa, runtime_trunc_final, runtime_trunc_total, n_max


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

# runtime_hmm_zonn()
# runtime_nmix_poisson()

# cProfile.run('runtime_hmm_zonn()', 'trunc.stats')