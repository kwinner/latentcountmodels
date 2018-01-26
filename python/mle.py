from scipy import optimize
from forward import forward
from forward_grad import Parameter, forward_grad

import csv, sys, warnings
import numpy as np

warnings.filterwarnings('error')
np.seterr(divide='ignore')

def run_mle(T, arrival, branch, observ, y=None, true_params=None, n=1, n_reps=1,
            out=None, out_mode='w', max_attempts=3, log=None, grad=False):
    # If y is None true_params must be provided to generate samples,
    # otherwise (y is provided) true_params will be ignored if provided
    if y is None:
        assert true_params is not None, 'Must provide true_params'
    if y is not None and true_params is not None:
        print('true_params will be ignored')

    # Save results if out is not None
    if out:
        fout = open(out, out_mode)
        writer = csv.writer(fout)

    # Log RuntimeWarnings and AssertionErrors if log is not None
    if log:
        flog = open(log, 'a')
        if true_params: flog.write('True params: ' + str(true_params))
    else:
        flog = None

    n_successes = 0
    for i in range(n):
        # Generate data
        if true_params is not None: y = generate_data(true_params, T, arrival, branch, observ, n_reps)
        print(y)

        success, n_attempts = False, 0
        while (not success) and n_attempts < max_attempts:
            if n_attempts > 0: print('Restarting')
            res = mle(y, T, arrival, branch, observ, flog, grad=grad)
            success = res.success
            n_attempts += 1

        if success:
            theta_hat = res.x
    
            # Calculate CI
            ci_width = np.absolute(1.96*np.sqrt(np.diagonal(res.hess_inv.todense())))
            ci_left = theta_hat - ci_width
            ci_right = theta_hat + ci_width
            #print(theta_hat, ci_width)
    
            # Write to out
            if out: writer.writerow(np.concatenate((y[0], theta_hat, ci_left, ci_right)))
            n_successes += 1

        print(res.success, res.message, n_successes, 'successes out of', i + 1, 'trials')

    print('Number of successes:', n_successes)
    print('Number of trials:', n)

    if out: fout.close()
    if log:
        flog.write('Number of successes: ' + str(n_successes) + '\n')
        flog.write('Number of trials: ' + str(n) + '\n')
        flog.close()

    try:
        return theta_hat, ci_left, ci_right
    except:
        return None, None, None

def mle(y, T, arrival, branch, observ, log, grad=False):
    theta0 = unpack('init', y, arrival, branch, observ)
    bounds = unpack('bounds', y, arrival, branch, observ)
    print(theta0)

    # Call the optimizer
    objective_args = (y, T, arrival, branch, observ, log)

    if grad:
        obj = objective_grad
        jac = True
    else:
        obj = objective
        jac = False
        
    return optimize.minimize(obj, theta0, args=objective_args,
                             method='L-BFGS-B', jac=jac, bounds=bounds,
                             options={'gtol': 1e-15})
    #options={'disp': True})#, 'eps': 1e-12, 'ftol': 1e-15, 'gtol': 1e-15})
    
def objective(theta, y, T, arrival, branch, observ, log):
    # Make sure all values in theta are valid
    if not np.all(np.isfinite(theta)): return float('inf')

    # Turn theta from np array into dictionary
    arrival_theta, branch_theta, observ_theta = theta_unpack(theta, T, arrival, branch, observ)
    #print('arrival', arrival_theta)
    #print('branch', branch_theta)
    #print('observ', observ_theta)

    # Compute log likelihood and return negative log likelihood
    arrival_pgf = arrival['pgf']
    branch_pgf = branch['pgf']
    #print(arrival_pgf, branch_pgf)
    
    ll = 0
    for i in range(len(y)):
        try:
            ll_i, _, _ = forward(y[i], arrival_pgf, arrival_theta,
                                 branch_pgf, branch_theta, observ_theta)
        except RuntimeWarning as w:
            if log:
                line = ' '.join([str(x) for x in [y, theta, ll, w]])
                log.write(line + '\n')
            ll_i = -1e12
        except AssertionError as e:
            if log:
                line = ' '.join([str(x) for x in [y, theta, ll, e]])
                log.write(line + '\n')
            ll_i = -1e12

        ll += ll_i

    print(theta, ll)
    return -ll

def objective_grad(theta, y, T, arrival, branch, observ, log):
    # Make sure all values in theta are valid
    if not np.all(np.isfinite(theta)): return float('inf')

    # Turn theta from np array into dictionary
    arrival_theta, branch_theta, observ_theta = theta_unpack(theta, T, arrival, branch, observ)

    arrival_mask = arrival['need_grad'](T)
    branch_mask = branch['need_grad'](T)
    observ_mask = observ['need_grad'](T)

    arrival_theta = Parameter.wrap(arrival_theta, need_grad = arrival_mask)
    branch_theta  = Parameter.wrap( branch_theta, need_grad =  branch_mask)
    observ_theta  = Parameter.wrap( observ_theta, need_grad=   observ_mask)
        
    #print('arrival', arrival_theta)
    #print('branch', branch_theta)
    #print('observ', observ_theta)

    # Compute log likelihood and return negative log likelihood
    arrival_pgf = arrival['pgf_grad']
    branch_pgf = branch['pgf_grad']
    #print(arrival_pgf, branch_pgf)
    
    nll = 0
    grad = np.zeros_like(theta)
    for i in range(len(y)):
        try:
            (ll_i,
             arrival_grad,
             branch_grad,
             observ_grad) = forward_grad(y[i],
                                         arrival_pgf,
                                         arrival_theta,
                                         branch_pgf,
                                         branch_theta,
                                         observ_theta)
        except RuntimeWarning as w:
            if log:
                line = ' '.join([str(x) for x in [y, theta, ll, w]])
                log.write(line + '\n')
            ll_i = -1e12
        except AssertionError as e:
            if log:
                line = ' '.join([str(x) for x in [y, theta, ll, e]])
                log.write(line + '\n')
            ll_i = -1e12


        grad -= recover_grad(T, arrival_grad, branch_grad, observ_grad,
                             arrival, branch, observ)
        nll -= ll_i

    print(theta, nll, grad)
    return nll, grad


# Flatten 2d list
def flatten(l):
    return [i for row in l for i in row]

def recover_grad(T, arrival_grad, branch_grad, observ_grad, arrival, branch, observ):
    
    arrival_hyperparam_grad = arrival['backprop'](arrival_grad)
    branch_hyperparam_grad  = branch['backprop'](branch_grad)
    observ_hyperparam_grad  = observ['backprop'](observ_grad)

    out = np.concatenate((arrival_hyperparam_grad, branch_hyperparam_grad, observ_hyperparam_grad))

    return out

def theta_unpack(theta_array, T, arrival, branch, observ):
    n_arrival_params = count_params(arrival)
    n_observ_params = count_params(observ)

    arrival_params = theta_array[:n_arrival_params]
    if n_observ_params > 0:
        branch_params = theta_array[n_arrival_params:-n_observ_params]
        observ_params = theta_array[-n_observ_params:]
    else:
        branch_params = theta_array[n_arrival_params:]
        observ_params = []

    theta_arrival = arrival['hyperparam2param'](arrival_params, T).astype(np.float64)
    theta_branch = branch['hyperparam2param'](branch_params, T).astype(np.float64)
    theta_observ = observ['hyperparam2param'](observ_params, T).astype(np.float64)
    
    return theta_arrival, theta_branch, theta_observ

def generate_data(theta, T, arrival, branch, observ, n_reps):
    K = len(T)
    arrival_params = arrival['hyperparam2param'](theta['arrival'], T)
    branch_params = branch['hyperparam2param'](theta['branch'], T)
    observ_params = observ['hyperparam2param'](theta['observ'], T)

    if arrival_params.ndim == 1: arrival_params = arrival_params.reshape((-1, 1))
    m = arrival['sample'](*arrival_params.T, size=(n_reps, K))
    n = np.zeros((n_reps, K), dtype=int)
    y = np.zeros((n_reps, K), dtype=int)
    z = np.zeros((n_reps, K-1), dtype=int)

    for k in range(K):
        n[:, k] = m[:, k] + z[:, k-1] if k > 0 else m[:, k]

        # Some hack to avoid domain error for n[i, k] = 0
        n_tmp = n[:, k]
        mask = np.array(n_tmp <= 0)
        n_tmp[mask] = 1
        if k < K-1:
            z_tmp = branch['sample'](n_tmp, branch_params[k])
            z_tmp[mask] = 0
            z[:, k] = z_tmp

    y = observ['sample'](n, observ_params)

    return y.astype(np.int32)

def count_params(dist):
    learn_mask = dist['learn_mask']
    if np.isscalar(learn_mask): learn_mask = [learn_mask]
    return np.sum(learn_mask)

def unpack(k, y, arrival, branch, observ):
    tmp = [d[k](y[0]) for d in [arrival, branch, observ]]
    tmp = [v if isinstance(v, list) else [v] for v in tmp]
    return [v for l in tmp for v in l]
    
