import pgffa, truncatedfa, UTPPGFFA_cython as utppgffa
import os, sys, warnings

from mle import *
from mle_distributions import *

warnings.filterwarnings('error')

def mean2p(mu, size):
    return float(size)/(size+mu)

def run_experiment(mode, min_delta, n, out_dir, out_mode):
    fa = utppgffa           # forward algorithm
    T = np.arange(7)        # vector of observation times

    # Distributions
    arrival_idx = 0 if mode in range(3) else 1
    branch_idx = mode % 3

    arrival = [constant_poisson_arrival, constant_nbinom_arrival][arrival_idx]
    branch = [constant_binom_branch, constant_poisson_branch, constant_nbinom_branch][branch_idx]
    observ = constant_binom_observ
    print arrival['pgf'], branch['pgf']

    # Arrival params
    lmbda = 5
    v = 10
    p = mean2p(lmbda, v)
    arrival_params = [lmbda, [v, p]]

    # Branching params
    max_delta = 1 if mode in [0, 3] else 1.6
    deltas = np.arange(min_delta, max_delta + 0.1, 0.2)
    
    # Observation params
    rho = 0.6

    if mode in range(3):
        true_params = {'arrival': lmbda, 'observ': rho}
    elif mode in range(3, 6):
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
        print 'delta =', delta
        true_params['branch'] = delta
        out = '{}/{}.csv'.format(path, delta)
        log = '{}/warnings.log'.format(path)
        print out

        run_mle(T, arrival, branch, observ, fa, log=log, max_iters=n,
                n=n, out=out, out_mode=out_mode, true_params=true_params)

if __name__ == "__main__":
    """
    Modes:
    1. Poisson arrival, binomial branching
    2. Poisson arrival, Poisson branching
    3. Poisson arrival, negative binomial branching
    4. Negative binomial arrival, binomial branching
    5. Negative binomial arrival, Poisson branching
    6. Negative binomial arrival, negative binomial branching
    """
    mode = int(sys.argv[1]) - 1
    min_delta = 0.3
    n = 50
    out_dir = '../data/mle_cython_gtol1e-15/'
    out_mode = 'w'
    
    run_experiment(mode, min_delta, n, out_dir, out_mode)
