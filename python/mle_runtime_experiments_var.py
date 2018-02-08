import os, sys

from mle_distributions import *
from mle import *

def run_experiment(mode, lmbda, v, delta, rho, n, n_reps, max_attempts,
                   out_dir, out_mode, Ks):
    # Distributions
    arrival_idx = 0 if mode < 3 else 1
    branch_idx = mode % 3

    arrival = [constant_poisson_arrival, fix_nbinom_arrival][arrival_idx]
    branch = [var_binom_branch, var_poisson_branch, var_nbinom_branch][branch_idx]
    observ = fix_binom_observ
    print(arrival['pgf'], branch['pgf'])

    # Arrival params
    p = mean2p(lmbda, v) # ignored in Poisson arrival cases

    # True params dict
    if mode < 3:
        true_params = {'arrival': lmbda, 'branch': None, 'observ': rho}
    else:
        true_params = {'arrival': [v, p], 'branch': None, 'observ': rho}
    
    # Setup outdir
    mdl_str = ['pois_bin', 'pois_pois', 'pois_nb',
               'nb_bin'  , 'nb_pois'  , 'nb_nb'  ][mode]

    #if mode in range(3):
    #    ctrl_params = [lmbda, delta, rho]
    #elif mode in range(3, 6):
    #    ctrl_params = [v, p, delta, rho]
    #ctrl_param_str = '_'.join([str(d) for d in ctrl_params])
    #if grad: ctrl_param_str += '_grad'
    
    path = '/'.join([out_dir, mdl_str])
    if not os.path.exists(path):
        os.makedirs(path)

    # MLE for each T
    for K in Ks:
        T = np.arange(K)
        out = '{}/{}.csv'.format(path, K)
        log = '{}/warnings.log'.format(path)

        print('K =', K)
        true_params['branch'] = delta[:K-1]
        run_mle(T, arrival, branch, observ, log=log, n=n, n_reps=n_reps,
                max_attempts=max_attempts, out=out, out_mode=out_mode,
                true_params=true_params, grad='both')

if __name__ == "__main__":
    """
    Models:
    1. Poisson arrival, binomial branching
    2. Poisson arrival, Poisson branching
    3. Poisson arrival, negative binomial branching
    4. Negative binomial arrival, binomial branching
    5. Negative binomial arrival, Poisson branching
    6. Negative binomial arrival, negative binomial branching
    """
    mode = int(sys.argv[1]) - 1   # model number
    assert mode in range(6), 'Choose a model 1-6'

    # Experimental settings (local version - for testing purposes)
    Ks = range(2, 11, 1)          # if len(Ks) > 11, must modify delta below!
    n = 3                        # rounds of experiments
    n_reps = 10                   # number of independent chains for each round
    max_attempts = 10             # max number of attempts for each round
    out_dir = '../data/mle_bprop_rt_var1' # output directory
    out_mode = 'w'                # 'a' for append, 'w' for write

    # True arrival params
    lmbda = 5                     # mean
    v = 10                        # dispersion (ignored in Poisson arrival cases)

    # True branching params
    if mode in [0, 3]: # Bernoulli branching
        delta = np.array([ 0.91925101,  0.19911554,  0.90281847,  0.40939839,  0.83666886,
                           0.29030967,  0.81204456,  0.74217406,  0.07162307,  0.50924052])
    else:              # Poisson or geometric branching
        delta = np.array([ 0.20989965,  0.39263032,  1.13094169,  0.14849046,  0.44552288,
                           0.87396244,  0.18968949,  2.45750465,  2.59874096,  1.70924293])

    # True observation params
    rho = 0.6

    # Run experiments
    run_experiment(mode, lmbda, v, delta, rho, n, n_reps, max_attempts,
                   out_dir, out_mode, Ks)
