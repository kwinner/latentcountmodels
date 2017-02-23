from scipy import optimize

import csv, sys, warnings, traceback
import numpy as np
import truncatedfa, pgffa, UTPPGFFA_cython as utppgffa

def run_mle(T, arrival, branch, observ, fa, y=None, true_params=None, n=20,
            out=None, out_mode='w', max_iters=100, log=None):
    # If y is None true_params must be provided,
    # otherwise (y is provided) true_params must be None
    if y is None: assert true_params is not None, 'Must provide true_params'
    if y is not None: assert true_params is None, 'Cannot provide true_params'

    # For PGFFA, make sure Poisson arrival and binomial branching
    if fa is pgffa:
        assert self.arrival_dist is arrival.poisson and \
               self.branching_dist is branching.binom, \
               'Error: PGFFA only supports Poisson arrival and binomial branching'

    # FA args
    if fa is pgffa:
        fa_args = tuple()
    elif fa is utppgffa:
        fa_args = (arrival['pgf'], branch['pgf'], observ['pgf'])
    elif fa is truncatedfa:
        fa_args = (arrival['pmf'], branch['pmf'], observ['pmf'])
    else:
        print 'Invalid FA'
        sys.exit(0)

    # MLE, and save results if out is not None
    if out:
        fout = open(out, out_mode)
        writer = csv.writer(fout)

    if log:
        flog = open(log, 'a')

    i = 0       # total number of reps needed to get n estimates
    n_conv = 0  # number of convergent estimates
    while n_conv < n and i < max_iters:
        # Generate data
        if true_params is not None: y = generate_data(true_params, T, arrival, branch, observ)
        #print y

        res = mle(y, T, arrival, branch, observ, fa, fa_args, flog)
        if res.success:
            #print res.success, res.message
            theta_hat = res.x
    
            # Calculate CI
            ci_width = np.absolute(1.96*np.sqrt(np.diagonal(res.hess_inv.todense())))
            ci_left = theta_hat - ci_width
            ci_right = theta_hat + ci_width
            
            #print theta_hat, ci_width
    
            # Write to out
            if out: writer.writerow(np.concatenate((y, theta_hat, ci_left, ci_right)))
            n_conv += 1
        else:
            print n_conv, 'out of', i, res.success, res.message

        i += 1

    print 'Number of iters:', i
    print 'Number of successes:', n_conv

    if out: fout.close()
    if log: flog.close()

    try:
        return theta_hat, ci_left, ci_right
    except:
        return None, None, None

def mle(y, T, arrival, branch, observ, fa, fa_args, log):
    theta0 = unpack('init', y, arrival, branch, observ)
    bounds = unpack('bounds', y, arrival, branch, observ)

    # Call the optimizer
    objective_args = (y, T, fa, fa_args, arrival, branch, observ, log)
    return optimize.minimize(objective, theta0, args=objective_args,
                             method='L-BFGS-B', jac=False, bounds=bounds,
                             options={'gtol': 1e-15})
                             #options={'disp': True})#, 'eps': 1e-12, 'ftol': 1e-15, 'gtol': 1e-15})

def objective(theta, y, T, fa, fa_args, arrival, branch, observ, log):
    # Make sure there is no nan in theta
    if np.any(np.isnan(theta)): return float('inf')

    # Turn theta from np array into dictionary
    arrival_theta, branch_theta, observ_theta = theta_array2dict(theta, T, arrival, branch, observ)
    #print 'arrival', arrival_theta
    #print 'branch', branch_theta
    #print 'observ', observ_theta

    # Compute log likelihood and return negative log likelihood
    arrival_pgf, branch_pgf, observ_pgf = fa_args
    #print arrival_pgf, branch_pgf
    
    ll = None
    try:
        alpha, z = utppgffa.utppgffa_cython(y, arrival_pgf, arrival_theta,
                                            branch_pgf, branch_theta, observ_theta)
        ll = np.log(alpha[0]) + np.sum(z)
    except RuntimeWarning as w:
        if log:
            line = ' '.join([str(x) for x in [arrival_pgf, branch_pgf, y, theta, ll, w]])
            log.write(line + '\n')
        ll = -1e12

    if ll and (np.isnan(ll) or np.isneginf(ll)):
        if log:
            w = 'invalid value encountered in double_scalars'
            line = ' '.join([str(x) for x in [arrival_pgf, branch_pgf, y, theta, ll, w]])
            log.write(line + '\n')
        ll = -1e12
    
    return -ll

def theta_array2dict(theta_array, T, arrival, branch, observ):
    n_arrival_params = count_params(arrival)
    n_observ_params = count_params(observ)

    arrival_params = theta_array[:n_arrival_params]
    if n_observ_params > 0:
        branch_params = theta_array[n_arrival_params:-n_observ_params]
        observ_params = theta_array[-n_observ_params:]
    else:
        branch_params = theta_array[n_arrival_params:]
        observ_params = []

    return (arrival['hyperparam2param'](arrival_params, T).astype(np.float64),
            branch['hyperparam2param'](branch_params, T).astype(np.float64),
            observ['hyperparam2param'](observ_params, T).astype(np.float64))

def generate_data(theta, T, arrival, branch, observ):
    K = len(T)
    arrival_params = arrival['hyperparam2param'](theta['arrival'], T)
    branch_params = branch['hyperparam2param'](theta['branch'], T)
    observ_params = observ['hyperparam2param'](theta['observ'], T)

    if arrival_params.ndim == 1: arrival_params = arrival_params.reshape((-1, 1))
    m = arrival['sample'](*arrival_params.T)
    n, y, z = np.zeros(K, dtype=int), np.zeros(K, dtype=int), np.zeros(K-1, dtype=int)
    for k in range(K):
        n[k] = m[k] + z[k-1] if k > 0 else m[k]
        if k < K-1: z[k] = branch['sample'](n[k], branch_params[k]) if n[k] > 0 else 0
    y = observ['sample'](n, observ_params)

    return y.astype(np.int32)

def count_params(dist):
    learn_mask = dist['learn_mask']
    if np.isscalar(learn_mask): learn_mask = [learn_mask]
    return np.sum(learn_mask)

def unpack(k, y, arrival, branch, observ):
    tmp = [d[k](y) for d in [arrival, branch, observ]]
    tmp = [v if isinstance(v, list) else [v] for v in tmp]
    return [v for l in tmp for v in l]
    