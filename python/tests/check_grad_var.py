import sys
from scipy import optimize

from mle import *
from mle_distributions import *

def grad(theta, y, T, arrival, branch, observ, fixed_params, log):
    return objective_grad(theta, y, T, arrival, branch, observ, fixed_params, log)[1]

def run_check_grad(mode):
    K = 5
    n_samples = 5
    
    # Parameter values
    lmbda = 5                              # arrival mean
    v = 10                                 # arrival dispersion (ignored in Poisson arrival cases)
    rho = 0.6                              # detection probability
    
    # Time-varying branching parameters
    if mode in [0, 3]: # Bernoulli branching
        delta = np.array([ 0.91925101,  0.19911554,  0.90281847,  0.40939839,  0.83666886,
                           0.29030967,  0.81204456,  0.74217406,  0.07162307,  0.50924052])[:K-1]
    else:              # Poisson or geometric branching
        delta = np.array([ 0.20989965,  0.39263032,  1.13094169,  0.14849046,  0.44552288,
                           0.87396244,  0.18968949,  2.45750465,  2.59874096,  1.70924293])[:K-1]

    # Distributions
    arrival_idx = 0 if mode < 3 else 1
    branch_idx = mode % 3
    
    arrival = [fix_poisson_arrival, fix_nbinom_arrival][arrival_idx]
    branch = [var_binom_branch, var_poisson_branch, var_nbinom_branch][branch_idx]
    observ = fix_binom_observ
    
    # Generate data
    p = mean2p(lmbda, v) # ignored in Poisson arrival cases
    if mode < 3:
        true_params = {'arrival': lmbda, 'branch': delta, 'observ': rho}
    else:
        true_params = {'arrival': [v, p], 'branch': delta, 'observ': rho}
    T = np.arange(K)
    y = generate_data(true_params, T, arrival, branch, observ, n_samples)
    #print(y)
    
    # Check gradients
    fixed_params = get_fixed_params(arrival, branch, observ, true_params, T)
    theta0 = unpack('init', y, arrival, branch, observ, T)
    theta = delta

    objective_args = (y, T, arrival, branch, observ, fixed_params, False)
    error0 = optimize.check_grad(objective, grad, theta0, *objective_args)
    error = optimize.check_grad(objective, grad, theta, *objective_args)
    print(error0, error, error0 < 1e-4 and error < 1e-4)

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
    if len(sys.argv) == 1:
        print('Checking gradients for all models')
        for mode in range(6):
            print('Model', mode + 1)
            run_check_grad(mode)
    else:
        mode = int(sys.argv[1]) - 1
        run_check_grad(mode)
