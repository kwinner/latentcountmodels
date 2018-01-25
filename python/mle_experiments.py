import os, sys
import numpy as np

from mle_distributions import *
from mle import *

def mean2p(mu, size):
    return float(size)/(size+mu)

def run_experiment(mode, params, n, n_reps, max_attempts, out_dir, out_mode):
    lmbda, v, min_delta, max_delta, step, rho = params
    T = np.arange(7) # vector of observation times

    # Distributions
    arrival_idx = 0 if mode < 3 else 1
    branch_idx = mode % 3

    arrival = [constant_poisson_arrival, constant_nbinom_arrival][arrival_idx]
    branch = [constant_binom_branch, constant_poisson_branch, constant_nbinom_branch][branch_idx]
    observ = fix_binom_observ
    print(arrival['pgf'], branch['pgf'])

    # Arrival params
    p = mean2p(lmbda, v) # ignored in Poisson arrival cases

    # Branching params
    deltas = np.arange(min_delta, max_delta + 0.1, step)

    # True params dict
    if mode < 3:
        true_params = {'arrival': lmbda, 'observ': rho}
    else:
        true_params = {'arrival': [v, p], 'observ': rho}
    
    # Setup outdir
    dist_str = ['pois_bin', 'pois_pois', 'pois_nb',
                'nb_bin'  , 'nb_pois'  , 'nb_nb'  ][mode]

    if mode in range(3):
        fixed_params = [lmbda, rho]
    elif mode in range(3, 6):
        fixed_params = [v, p, rho]
    fixed_param_str = '_'.join([str(round(d, 2)) for d in fixed_params])
    
    path = '/'.join([out_dir, dist_str, fixed_param_str])
    if not os.path.exists(path):
        os.makedirs(path)

    # MLE for each delta
    for delta in deltas:
        print('delta =', delta)
        true_params['branch'] = delta
        out = '{}/{}.csv'.format(path, delta)
        log = '{}/warnings.log'.format(path)
        print(out)

        run_mle(T, arrival, branch, observ, log=log, n=n, n_reps=n_reps,
                max_attempts=max_attempts, out=out, out_mode=out_mode,
                true_params=true_params)

if __name__ == "__main__":
    """
    Experiments:
    1. Poisson arrival, binomial branching
    2. Poisson arrival, Poisson branching
    3. Poisson arrival, negative binomial branching
    4. Negative binomial arrival, binomial branching
    5. Negative binomial arrival, Poisson branching
    6. Negative binomial arrival, negative binomial branching
    """
    mode = int(sys.argv[1]) - 1   # experiment number
    n = 10                        # number of estimates
    n_reps = 10                   # number of replicates for each estimate
    max_attempts = 10             # max number of random restarts (incl first attempt)
    out_dir = '../data/mle_out3/'
    out_mode = 'w'                # 'a' for append, 'w' for write

    # Arrival params
    lmbda = 5                     # mean
    v = 10                        # dispersion (ignored in Poisson arrival cases)

    # Branching params
    min_delta = 0.2
    max_delta = 0.9 if mode in [0, 3] else 1.6 # 0.9 if binomial branching, else 1.6
    step = 0.1 if mode in [0, 3] else 0.2      # 0.1 if binomial branching, else 0.2

    # Observation params
    rho = 0.6
    
    assert mode in range(6), 'Choose an experiment 1-6'
    params = [lmbda, v, min_delta, max_delta, step, rho]
    run_experiment(mode, params, n, n_reps, max_attempts, out_dir, out_mode)
