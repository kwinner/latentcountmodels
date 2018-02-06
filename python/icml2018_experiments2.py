import os
import time
import datetime
import pickle
import uuid
from glob import glob
import sys

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import gdual as gd
import forward as gdfwd
import truncatedfa as trfwd

RESULT_BASE_DIR = os.path.expanduser('~/Work/Data/Results')
SHANNON_RESULTS_DIR = os.path.expanduser('~/shannon-results')

# just made this a global for simplicity
SILENT = False
TRFWD_SILENT = True

COUNT_DTYPE = np.int32
LL_DTYPE    = np.float64
RT_DTYPE    = np.float32
GD_RECORD_DTYPE    = np.dtype([                        ('LL', LL_DTYPE), ('RT', RT_DTYPE)])
TRFWD_RECORD_DTYPE = np.dtype([('N_max', COUNT_DTYPE), ('LL', LL_DTYPE), ('RT', RT_DTYPE)])

N_REPS_DEFAULT        = 20
THETA_ARRIVAL_DEFAULT = 5 * np.array([2.5, 11, 21, 15, 4]).reshape(-1, 1)
THETA_BRANCH_DEFAULT  = np.array([0.5, 0.5, 0.5, 0.5]).reshape(-1, 1)
THETA_OBSERV_DEFAULT  = 0.5 * np.ones(5)
DIST_ARRIVAL_DEFAULT  = 'poisson'
DIST_BRANCH_DEFAULT   = 'bernoulli'
N_INIT_DEFAULT        = 'y_max' # initial value of N_max. integer or {'y_max', 'Y'}
N_LIMIT_DEFAULT       = 10000 # hard cap on N_max. integer
EPSILON_DEFAULT       = 1e-5
FIXED_Y_DEFAULT       = True

THETA_BRANCH_EXPERIMENT_DEFAULT  = np.linspace(0., 1.0, 5)
THETA_ARRIVAL_EXPERIMENT_DEFAULT = np.array([5, 10, 25, 50])
# THETA_BRANCH_EXPERIMENT_DEFAULT  = np.linspace(0., 1.0, 21)
# THETA_ARRIVAL_EXPERIMENT_DEFAULT = np.array([5, 10, 25, 50, 75, 100, 150, 200, 250])

TRFWD_RESULT_INDEX_DEFAULT = -1 # default to last entry (largest N_max attempted) for plotting trfwd results
TRFWD_LL_RESULT_INDEX      = -1
TRFWD_RT_RESULT_INDEX      = -1

# plotting parameters
BRANCHING_PARAM_LABEL    = r'$\delta$'
ARRIVAL_PARAM_LABEL      = r'$\Lambda$'
METHOD_NAMES             = ['Trfwd', 'Trfwd w/ FFT', 'GDual', 'LSGDual']
METHOD_FILENAME_SUFFIXES = ['trdir', 'trfft', 'gd', 'lsgd']
Y_LABEL_DICT             = {'RT': r'mean RT (s)', 'LL': r'LL', 'nan': 'nan frequency'}

# pmfs (for trfwd)
ARRIVAL_PMF_DICT = {
    'poisson': stats.poisson,
    'negbin':  stats.nbinom
}

# branching (truncated) pmfs (for trfwd)
BRANCH_TRPMF_DICT = {
    'poisson':   trfwd.poisson_branching,
    'bernoulli': trfwd.binomial_branching,
    'negbin':    trfwd.nbinom_branching
}

# pgfs (for arrival and branching in GD/LSGD)
PGF_DICT = {
    'poisson':   gdfwd.poisson_pgf,
    'bernoulli': gdfwd.bernoulli_pgf,
    'negbin':    gdfwd.negbin_pgf
}

# function to sample offspring RVs
# function prototype:
#   N_k = lambda(N_{k-1}, theta_branch)
BRANCH_SAMPLING_DICT = {
    'poisson':   lambda N, theta_branch: stats.poisson.rvs(N * theta_branch),
    'bernoulli': lambda N, theta_branch: stats.binom.rvs(N, theta_branch)
}

# mostly standardizing the timestamp formatting used in a few places
def experiment_name(
        prefix = '',
        suffix = '',
        timestamp = True # whether to add a timestamp
        ):
    if timestamp:
        timestamp_str = datetime.datetime.now().strftime("%y%m%d%H%M%S%f")
    else:
        timestamp_str = ''
    return prefix + timestamp_str + suffix

def sample_data(
        theta_arrival = THETA_ARRIVAL_DEFAULT,
        theta_branch  = THETA_BRANCH_DEFAULT,
        theta_observ  = THETA_OBSERV_DEFAULT,
        dist_arrival  = DIST_ARRIVAL_DEFAULT,
        dist_branch   = DIST_BRANCH_DEFAULT
        ):
    K = len(theta_arrival)
    arrival_rvs = lambda i: ARRIVAL_PMF_DICT[dist_arrival].rvs(theta_arrival[i])
    branch_rvs  = lambda N, i: BRANCH_SAMPLING_DICT[dist_branch](N, theta_branch[i])
    observ_rvs  = lambda N, i: stats.binom.rvs(N, theta_observ[i])

    N = np.empty(K, dtype=COUNT_DTYPE)
    y = np.empty(K, dtype=COUNT_DTYPE)

    # initial population
    N[0] = arrival_rvs(0)
    y[0] = observ_rvs(N[0], 0)
    # sample chain
    for i in range(1, K):
        N[i] = arrival_rvs(i) + branch_rvs(N[i - 1], i - 1)
        y[i] = observ_rvs(N[i], i)

    return {'y': y, 'N': N}

def stability_experiment(
        results_folder,
        control_variable, # not used in experiment, but saved alongside the results as the control variable (on the x-axis)
        experiment_id      = str(uuid.uuid4()), # a string identifying this experiment. used as filename for pickling
        n_reps             = N_REPS_DEFAULT,
        theta_arrival_eval = THETA_ARRIVAL_DEFAULT,
        theta_branch_eval  = THETA_BRANCH_DEFAULT,
        theta_observ_eval  = THETA_OBSERV_DEFAULT,
        theta_arrival_gen  = None, # data sampling parameters (if different from evaluation params)
        theta_branch_gen   = None, # if None, eval params used instead
        theta_observ_gen   = None,
        y_given            = None, # observed values. if None, y will be sampled w/ the generating params
        fixed_y            = FIXED_Y_DEFAULT, # if False, resample y every repetition
        dist_arrival       = DIST_ARRIVAL_DEFAULT,
        dist_branch        = DIST_BRANCH_DEFAULT,
        N_init             = N_INIT_DEFAULT,
        N_limit            = N_LIMIT_DEFAULT,
        epsilon            = EPSILON_DEFAULT
        ):
    if(theta_arrival_gen is None): theta_arrival_gen = theta_arrival_eval
    if(theta_branch_gen  is None): theta_branch_gen  = theta_branch_eval
    if(theta_observ_gen  is None): theta_observ_gen  = theta_observ_eval

    K = len(theta_arrival_eval)
    y_record = np.empty([n_reps, K]) # observed data record

    # measurement records
    trfwd_dir_result = [np.array([], dtype = TRFWD_RECORD_DTYPE) for i in range(n_reps)]
    trfwd_fft_result = [np.array([], dtype = TRFWD_RECORD_DTYPE) for i in range(n_reps)]
    gdual_result     = np.empty(n_reps, dtype = GD_RECORD_DTYPE)
    lsgdual_result   = np.empty(n_reps, dtype = GD_RECORD_DTYPE)

    # y_fixed -> y shared across all repetitions
    if fixed_y:
        if y_given is not None:
            y = y_given
        else:
            # sample data
            y = sample_data(theta_arrival_gen, theta_branch_gen, theta_observ_gen, dist_arrival, dist_branch)['y']

    for i_rep in range(n_reps):
        if not SILENT:
            print("Iteration %d of %d" % (i_rep + 1, n_reps))

        # !y_fixed -> y differs for each repetition
        if not fixed_y:
            if y_given is not None:
                if y.ndim > 1:
                    y = y_given[i_rep, :] # handle y_given with multiple y's
                else:
                    y = y_given
            else:
                # sample data
                y = sample_data(theta_arrival_gen, theta_branch_gen, theta_observ_gen, dist_arrival, dist_branch)['y']
        y_record[i_rep, :] = y
        if not SILENT:
            print("y: %s" % str(y))

        # initial value of N_max this iteration
        if N_init == 'Y':
            N_init = np.sum(y)
        elif N_init == 'y_max' or N_init < np.max(y):
            N_init = np.max(y)

        # lsgdual test
        start_time = time.process_time()
        try:
            lsgdual_result[i_rep]['LL'], _, _ = gdfwd.forward(y,
                                                              PGF_DICT[dist_arrival],
                                                              theta_arrival_eval,
                                                              PGF_DICT[dist_branch],
                                                              theta_branch_eval,
                                                              theta_observ_eval,
                                                              GDualType=gd.LSGDual,
                                                              d=0)
        except Exception as e:
            lsgdual_result[i_rep]['LL'] = np.nan
            if not SILENT:
                print("Error in lsgdual. LL set to nan")
        finally:
            lsgdual_result[i_rep]['RT'] = time.process_time() - start_time
            if not SILENT:
                print("LL (lsgd): %f, time = %f" % (lsgdual_result[i_rep]['LL'], lsgdual_result[i_rep]['RT']))

        # gdual test
        start_time = time.process_time()
        try:
            gdual_result[i_rep]['LL'], _, _ = gdfwd.forward(y,
                                                            PGF_DICT[dist_arrival],
                                                            theta_arrival_eval,
                                                            PGF_DICT[dist_branch],
                                                            theta_branch_eval,
                                                            theta_observ_eval,
                                                            GDualType=gd.GDual,
                                                            d=0)
        except Exception:
            gdual_result[i_rep]['LL'] = np.nan
            if not SILENT:
                print("Error in gdual. LL set to nan")
        finally:
            gdual_result[i_rep]['RT'] = time.process_time() - start_time
            if not SILENT:
                print("LL (gd): %f, time = %f" % (gdual_result[i_rep]['LL'], gdual_result[i_rep]['RT']))

        # trfwd w/ direct convolution
        N_max = N_init
        diff  = np.inf # diff is the difference in LL between the last two iterations of trfwd
        while N_max <= N_limit and diff > epsilon:
            if not SILENT:
                print("Trfwd.dir trial, N_max = %d" % N_max, end='', flush=True)
            # run the actual trial
            start_time = time.process_time()
            try:
                _, ll = trfwd.truncated_forward(ARRIVAL_PMF_DICT[dist_arrival],
                                                  theta_arrival_eval,
                                                  BRANCH_TRPMF_DICT[dist_branch],
                                                  theta_branch_eval,
                                                  theta_observ_eval,
                                                  y,
                                                  N_max,
                                                  silent=TRFWD_SILENT,
                                                  conv_method='direct')
            except:
                ll = np.nan
                if not SILENT:
                    print("Error in trfwd.dir. LL set to nan")
            finally:
                rt = time.process_time() - start_time

            # record the attempt
            trfwd_dir_result[i_rep] = np.append(trfwd_dir_result[i_rep],
                                                np.array((N_max, ll, rt), dtype=TRFWD_RECORD_DTYPE))
            if not SILENT:
                print(", LL = %f, RT = %f" % (ll, rt), end='')

            if np.isnan(ll):
                if not SILENT:
                    print("\nLL in trfwd dir is nan, breaking")
                break

            # compute diff, the convergence criteria
            if trfwd_dir_result[i_rep].size == 1:
                diff = np.inf
            else:
                diff = np.abs(trfwd_dir_result[i_rep][-1]['LL'] - trfwd_dir_result[i_rep][-2]['LL'])
            if not SILENT:
                print(", diff = %f" % diff)

            # double N_max for the next iteration
            if N_max < N_limit:
                N_max = N_max * 2

                # if this brings N_max over the limit, try one last time at the limiting value
                if N_max > N_limit:
                    N_max = N_limit
            else:
                if not SILENT:
                    print("N_max limit reached, breaking")
                break

        # trfwd w/ FFT convolution
        N_max = N_init
        diff = np.inf  # diff is the difference in LL between the last two iterations of trfwd
        while N_max <= N_limit and diff > epsilon:
            if not SILENT:
                print("Trfwd.fft trial, N_max = %d" % N_max, end='', flush=True)
            # run the actual trial
            start_time = time.process_time()
            try:
                _, ll = trfwd.truncated_forward(ARRIVAL_PMF_DICT[dist_arrival],
                                                  theta_arrival_eval,
                                                  BRANCH_TRPMF_DICT[dist_branch],
                                                  theta_branch_eval,
                                                  theta_observ_eval,
                                                  y,
                                                  N_max,
                                                  silent=TRFWD_SILENT,
                                                  conv_method='fft')
            except:
                ll = np.nan
                if not SILENT:
                    print("Error in trfwd.fft. LL set to nan")
            finally:
                rt = time.process_time() - start_time

            # record the attempt
            trfwd_fft_result[i_rep] = np.append(trfwd_fft_result[i_rep],
                                                np.array((N_max, ll, rt), dtype=TRFWD_RECORD_DTYPE))
            if not SILENT:
                print(", LL = %f, RT = %f" % (ll, rt), end='')

            if np.isnan(ll):
                if not SILENT:
                    print("\nLL in trfwd fft is nan, breaking")
                break

            # compute diff, the convergence criteria
            if trfwd_fft_result[i_rep].size == 1:
                diff = np.inf
            else:
                diff = np.abs(trfwd_fft_result[i_rep][-1]['LL'] - trfwd_fft_result[i_rep][-2]['LL'])
            if not SILENT:
                print(", diff = %f" % diff)

            # double N_max for the next iteration
            if N_max < N_limit:
                N_max = N_max * 2

                # if this brings N_max over the limit, try one last time at the limiting value
                if N_max > N_limit:
                    N_max = N_limit
            else:
                if not SILENT:
                    print("N_max limit reached, breaking")
                break

    # save results to file
    results_path = os.path.join(results_folder, (str(experiment_id) + '.pickle'))
    results_file = open(results_path, 'wb')
    pickle.dump({'control_variable': control_variable,
                 'experiment_id': experiment_id,
                 'trfwd_dir_result': trfwd_dir_result,
                 'trfwd_fft_result': trfwd_fft_result,
                 'gdual_result': gdual_result,
                 'lsgdual_result': lsgdual_result,
                 'n_reps': n_reps,
                 'theta_arrival_eval': theta_arrival_eval,
                 'theta_branch_eval': theta_branch_eval,
                 'theta_observ_eval': theta_observ_eval,
                 'theta_arrival_gen': theta_arrival_gen,
                 'theta_branch_gen': theta_branch_gen,
                 'theta_observ_gen': theta_observ_gen,
                 'y_record': y_record,
                 'dist_arrival': dist_arrival,
                 'dist_branch': dist_branch,
                 'N_init': N_init,
                 'N_limit': N_limit,
                 'epsilon': epsilon},
                results_file)
    results_file.close()

    return results_file


def vary_branching_params(
        theta_branch_experiment = THETA_BRANCH_EXPERIMENT_DEFAULT,
        results_folder          = None,
        n_reps                  = N_REPS_DEFAULT,
        theta_arrival_eval      = THETA_ARRIVAL_DEFAULT,
        theta_observ_eval       = THETA_OBSERV_DEFAULT,
        theta_arrival_gen       = None,                 # data sampling parameters (if different from evaluation params)
        theta_branch_gen        = THETA_BRANCH_DEFAULT, # if None, eval params used instead
        theta_observ_gen        = None,
        dist_arrival            = DIST_ARRIVAL_DEFAULT,
        dist_branch             = DIST_BRANCH_DEFAULT,
        N_init                  = N_INIT_DEFAULT,
        N_limit                 = N_LIMIT_DEFAULT,
        epsilon                 = EPSILON_DEFAULT,
        fixed_y                 = FIXED_Y_DEFAULT
        ):
    if results_folder is None:
        experiment_timestep = experiment_name(suffix='_branch')
        results_folder = os.path.join(RESULT_BASE_DIR, experiment_timestep)
    if(theta_arrival_gen is None): theta_arrival_gen = theta_arrival_eval
    # branch_gen shouldn't be None since there is no good way to select one
    # if(theta_branch_gen  is None): theta_branch_gen  = theta_branch_eval
    if(theta_observ_gen  is None): theta_observ_gen  = theta_observ_eval

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    n_experiments = len(theta_branch_experiment)
    K = len(theta_arrival_eval)

    meta_file = open(os.path.join(results_folder, 'meta.txt'), 'w')
    meta_file.write('Delta_values: ' + str(theta_branch_experiment) + '\n')
    meta_file.write('Lambda_gen: ' + str(theta_arrival_gen) + '\n')
    meta_file.write('Lambda_eval: ' + str(theta_arrival_eval) + '\n')
    meta_file.write('Delta_gen: ' + str(theta_branch_gen) + '\n')
    meta_file.write('Rho_gen: ' + str(theta_observ_gen) + '\n')
    meta_file.write('Rho_eval: ' + str(theta_observ_eval) + '\n')
    meta_file.write('epsilon: ' + str(epsilon) + '\n')
    meta_file.write('n_reps: ' + str(n_reps) + '\n')
    meta_file.write('N_init: ' + str(N_init) + '\n')
    meta_file.write('N_limit: ' + str(N_limit) + '\n')
    meta_file.write('arrival: ' + dist_arrival + '\n')
    meta_file.write('branch: ' + dist_branch + '\n')
    meta_file.write('fix-y: ' + str(fixed_y) + '\n')
    meta_file.write('x_label: ' + BRANCHING_PARAM_LABEL)
    meta_file.close()

    if fixed_y:
        # sample data
        y = sample_data(theta_arrival_gen, theta_branch_gen, theta_observ_gen, dist_arrival, dist_branch)['y']
    else:
        y = None

    for theta_branch_constant in theta_branch_experiment:
        theta_branch_eval = theta_branch_constant * np.ones((K - 1, 1))

        results_path = stability_experiment(results_folder,
                                            control_variable   = theta_branch_constant,
                                            experiment_id      = str(theta_branch_constant),  # a string identifying this experiment. used as filename for pickling
                                            n_reps             = n_reps,
                                            theta_arrival_eval = theta_arrival_eval,
                                            theta_branch_eval  = theta_branch_eval,
                                            theta_observ_eval  = theta_observ_eval,
                                            theta_arrival_gen  = theta_arrival_gen,
                                            theta_branch_gen   = theta_branch_gen,
                                            theta_observ_gen   = theta_observ_gen,
                                            y_given            = y,
                                            fixed_y            = fixed_y,
                                            dist_arrival       = dist_arrival,
                                            dist_branch        = dist_branch,
                                            N_init             = N_init,
                                            N_limit            = N_limit,
                                            epsilon            = epsilon)

    return results_folder


def vary_arrival_params(
        theta_arrival_experiment = THETA_ARRIVAL_EXPERIMENT_DEFAULT,
        results_folder          = None,
        n_reps                  = N_REPS_DEFAULT,
        theta_branch_eval       = THETA_BRANCH_DEFAULT,
        theta_observ_eval       = THETA_OBSERV_DEFAULT,
        theta_arrival_gen       = THETA_ARRIVAL_DEFAULT,
        theta_branch_gen        = None,
        theta_observ_gen        = None,
        dist_arrival            = DIST_ARRIVAL_DEFAULT,
        dist_branch             = DIST_BRANCH_DEFAULT,
        N_init                  = N_INIT_DEFAULT,
        N_limit                 = N_LIMIT_DEFAULT,
        epsilon                 = EPSILON_DEFAULT,
        fixed_y                 = FIXED_Y_DEFAULT
        ):
    if results_folder is None:
        experiment_timestep = experiment_name(suffix='_arrival')
        results_folder = os.path.join(RESULT_BASE_DIR, experiment_timestep)
    # arrival_gen shouldn't be None since there is no good way to select one
    # if(theta_arrival_gen is None): theta_arrival_gen = theta_arrival_eval
    if(theta_branch_gen  is None): theta_branch_gen  = theta_branch_eval
    if(theta_observ_gen  is None): theta_observ_gen  = theta_observ_eval

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    n_experiments = len(theta_arrival_experiment)
    K = len(theta_observ_eval)

    meta_file = open(os.path.join(results_folder, 'meta.txt'), 'w')
    meta_file.write('Lambda_values: ' + str(theta_arrival_experiment) + '\n')
    meta_file.write('Lambda_gen: ' + str(theta_arrival_gen) + '\n')
    meta_file.write('Delta_gen: ' + str(theta_branch_gen) + '\n')
    meta_file.write('Delta_eval: ' + str(theta_branch_eval) + '\n')
    meta_file.write('Rho_gen: ' + str(theta_observ_gen) + '\n')
    meta_file.write('Rho_eval: ' + str(theta_observ_eval) + '\n')
    meta_file.write('epsilon: ' + str(epsilon) + '\n')
    meta_file.write('n_reps: ' + str(n_reps) + '\n')
    meta_file.write('N_init: ' + str(N_init) + '\n')
    meta_file.write('N_limit: ' + str(N_limit) + '\n')
    meta_file.write('arrival: ' + dist_arrival + '\n')
    meta_file.write('branch: ' + dist_branch + '\n')
    meta_file.write('fix-y: ' + str(fixed_y) + '\n')
    meta_file.write('x_label: ' + ARRIVAL_PARAM_LABEL)
    meta_file.close()

    if fixed_y:
        # sample data
        y = sample_data(theta_arrival_gen, theta_branch_gen, theta_observ_gen, dist_arrival, dist_branch)['y']
    else:
        y = None

    for theta_arrival_constant in theta_arrival_experiment:
        theta_arrival_eval = theta_arrival_constant * np.ones((K, 1))

        results_path = stability_experiment(results_folder,
                                            control_variable   = theta_arrival_constant,
                                            experiment_id      = str(theta_arrival_constant),  # a string identifying this experiment. used as filename for pickling
                                            n_reps             = n_reps,
                                            theta_arrival_eval = theta_arrival_eval,
                                            theta_branch_eval  = theta_branch_eval,
                                            theta_observ_eval  = theta_observ_eval,
                                            theta_arrival_gen  = theta_arrival_gen,
                                            theta_branch_gen   = theta_branch_gen,
                                            theta_observ_gen   = theta_observ_gen,
                                            y_given            = y,
                                            fixed_y            = fixed_y,
                                            dist_arrival       = dist_arrival,
                                            dist_branch        = dist_branch,
                                            N_init             = N_init,
                                            N_limit            = N_limit,
                                            epsilon            = epsilon)

    return results_folder


def plot_result(
        results_folder,
        response_variable  = 'LL',                      # one of {'LL', 'RT', 'nan'}
        trfwd_result_index = TRFWD_RESULT_INDEX_DEFAULT # which N_max attempt to plot (typically -1 or -2)
        ):
    # read some metadata
    meta_file = open(os.path.join(results_folder, "meta.txt"))
    for line in meta_file:
        if line.startswith("n_reps:"):
            n_reps = int(line[len("n_reps:"):].strip())
        elif line.startswith("x_label:"):
            x_label = line[len("x_label:"):].strip()

    # glob is basically unix find
    results_glob = glob(os.path.join(results_folder, "*.pickle"))
    n_results = len(results_glob)

    # read all the results
    x_vals            = np.empty(n_results)
    trfwd_dir_results = np.empty((n_results, n_reps))
    trfwd_fft_results = np.empty((n_results, n_reps))
    gdual_results     = np.empty((n_results, n_reps))
    lsgdual_results   = np.empty((n_results, n_reps))

    for i_result in range(n_results):
        result = pickle.load(open(results_glob[i_result], 'rb'))

        x_vals[i_result]               = result['control_variable']

        if response_variable is not 'nan':
            trfwd_dir_results[i_result, :] = np.array(list(map(lambda x: x[max(-len(x), trfwd_result_index)][response_variable], result['trfwd_dir_result'])))
            trfwd_fft_results[i_result, :] = np.array(list(map(lambda x: x[max(-len(x), trfwd_result_index)][response_variable], result['trfwd_fft_result'])))
            gdual_results[i_result, :]     = result['gdual_result'][response_variable]
            lsgdual_results[i_result, :]   = result['lsgdual_result'][response_variable]
        elif response_variable is 'nan':
            trfwd_dir_results[i_result, :] = np.array(list(map(lambda x: x[max(-len(x), -1)]['LL'], result['trfwd_dir_result'])))
            trfwd_fft_results[i_result, :] = np.array(list(map(lambda x: x[max(-len(x), -1)]['LL'], result['trfwd_fft_result'])))
            gdual_results[i_result, :]     = result['gdual_result']['LL']
            lsgdual_results[i_result, :]   = result['lsgdual_result']['LL']

    if response_variable is not 'nan':
        # average over all reps
        trfwd_dir_mean = np.nanmean(trfwd_dir_results, axis=1)
        trfwd_dir_var  = np.nanvar (trfwd_dir_results, axis=1)
        trfwd_fft_mean = np.nanmean(trfwd_fft_results, axis=1)
        trfwd_fft_var  = np.nanvar (trfwd_fft_results, axis=1)
        gdual_mean     = np.nanmean(gdual_results,     axis=1)
        gdual_var      = np.nanvar (gdual_results,     axis=1)
        lsgdual_mean   = np.nanmean(lsgdual_results,   axis=1)
        lsgdual_var    = np.nanvar (lsgdual_results,   axis=1)
    elif response_variable is 'nan':
        trfwd_dir_mean = np.sum(np.isnan(trfwd_dir_results), axis=1)
        trfwd_fft_mean = np.sum(np.isnan(trfwd_fft_results), axis=1)
        gdual_mean     = np.sum(np.isnan(gdual_results), axis=1)
        lsgdual_mean   = np.sum(np.isnan(lsgdual_results), axis=1)


    # sort x_vals, then order the means, vars accordingly
    idx = np.argsort(x_vals)
    x_vals         = x_vals[idx]
    trfwd_dir_mean = trfwd_dir_mean[idx]
    trfwd_fft_mean = trfwd_fft_mean[idx]
    gdual_mean     = gdual_mean[idx]
    lsgdual_mean   = lsgdual_mean[idx]
    if response_variable is not 'nan':
        trfwd_dir_var  = trfwd_dir_var[idx]
        trfwd_fft_var  = trfwd_fft_var[idx]
        gdual_var      = gdual_var[idx]
        lsgdual_var    = lsgdual_var[idx]

    fig = plt.figure()
    plt.plot(x_vals, trfwd_dir_mean)
    plt.plot(x_vals, trfwd_fft_mean)
    plt.plot(x_vals, gdual_mean)
    plt.plot(x_vals, lsgdual_mean)
    plt.xlabel(x_label)
    plt.ylabel(Y_LABEL_DICT[response_variable])
    plt.title(r'All methods')
    plt.legend(METHOD_NAMES)
    fig.savefig(os.path.join(results_folder, response_variable + str(int(trfwd_result_index)) + "_all.png"))


def plot_all_results(result_collection_folder):
    experiments_list = os.listdir(result_collection_folder)

    experiments_list.remove('.DS_Store')

    for experiment_folder in experiments_list:
        plot_result(os.path.join(result_collection_folder, experiment_folder), 'LL',  TRFWD_LL_RESULT_INDEX)
        plot_result(os.path.join(result_collection_folder, experiment_folder), 'RT',  TRFWD_RT_RESULT_INDEX)
        plot_result(os.path.join(result_collection_folder, experiment_folder), 'nan', TRFWD_RT_RESULT_INDEX)



if __name__ == "__main__":
    if os.uname()[1] == 'kwinn':
        # plot_all_results(SHANNON_RESULTS_DIR)
            vary_branching_params(n_reps=1, theta_branch_experiment=np.linspace(0., 1.0, 41), dist_branch='bernoulli')
    elif 'shannon' in os.uname()[1]:
        if len(sys.argv) == 1:
            vary_branching_params()
            vary_arrival_params()
        elif int(sys.argv[1]) == 1:
            vary_branching_params(n_reps=20, theta_branch_experiment=np.linspace(0., 1.0, 41), dist_branch='bernoulli')
        elif int(sys.argv[1]) == 2:
            vary_branching_params(n_reps=20, theta_branch_experiment=np.linspace(0., 15.0, 31), dist_branch='poisson')
        elif int(sys.argv[1]) == 3:
            vary_arrival_params(n_reps=20, theta_arrival_experiment=np.array([5, 10, 25, 50, 75, 100, 150, 200, 250]), dist_branch='bernoulli')
        elif int(sys.argv[1]) == 4:
            vary_arrival_params(n_reps=20, theta_arrival_experiment=np.array([5, 10, 25, 50, 75, 100, 150, 200, 250]), dist_branch='poisson')
