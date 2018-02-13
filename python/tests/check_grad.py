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
    delta = 0.3 if mode in [0, 3] else 1.2 # 0.3 if binomial branching, else 1.2
    rho = 0.6                              # detection probability
    
    # Distributions
    arrival_idx = 0 if mode < 3 else 1
    branch_idx = mode % 3
    
    arrival = [fix_poisson_arrival, fix_nbinom_arrival][arrival_idx]
    branch = [constant_binom_branch, constant_poisson_branch, constant_nbinom_branch][branch_idx]
    observ = fix_binom_observ
    
    # Generate data
    p = mean2p(lmbda, v) # ignored in Poisson arrival cases
    if mode < 3:
        true_params = {'arrival': lmbda, 'branch': delta, 'observ': rho}
    else:
        true_params = {'arrival': [v, p], 'branch': delta, 'observ': rho}
    T = np.arange(K)
    y = generate_data(true_params, T, arrival, branch, observ, n_samples)
    
    # Check gradients
    fixed_params = get_fixed_params(arrival, branch, observ, true_params, T)
    theta0 = unpack('init', y, arrival, branch, observ, T)
    theta = np.array([delta])

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
