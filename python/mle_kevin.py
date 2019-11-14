from scipy import optimize
from forward import forward
from forward_grad import Parameter, forward_grad

import csv, sys, warnings, time
import numpy as np

warnings.filterwarnings('error')
np.seterr(divide='ignore')

KEVIN_ONLY = True
Nfeval = 1
def optimizer_progress_report_2(Xi):
    global Nfeval
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}'.format(Nfeval, Xi[0], Xi[1]))
    Nfeval += 1

def run_mle(T, arrival, branch, observ, y=None, true_params=None, n=1, n_reps=1,
            out=None, out_mode='w', max_attempts=3, log=None,
            grad=False):
    """
    A couple of notes about the arguments:
    - If y is None true_params must be provided to generate samples,
      otherwise (y is provided) true_params will be ignored if provided
    - grad accepts True (for exact gradient), False (for numerical gradient),
      or 'both' (for both!)
    """

    if y is None:
        assert true_params is not None, 'Must provide true_params'
    if y is not None and true_params is not None:
        print('true_params will be ignored')

    grads = [True, False] if grad == 'both' else [grad]

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

    # Fixed params
    fixed_params = get_fixed_params(arrival, branch, observ, true_params, T)

    if True in grads: n_successes1 = 0
    if False in grads: n_successes0 = 0

    for i in range(n):
        # Generate data
        if true_params is not None: y = generate_data(true_params, T, arrival, branch, observ, n_reps)
        #print(y)

        for g in grads:
            for n_attempts in range(max_attempts):
                if n_attempts > 0: print('Restarting')
                res, runtime = mle(y, T, arrival, branch, observ, fixed_params, flog, grad=g)
                if res.success: break
    
            if res.success:
                theta_hat = res.x

                #print(res)
                
                # Write to out
                #if out: writer.writerow(np.concatenate((y[0], theta_hat, ci_left, ci_right, runtime)))
                if out:
                    writer.writerow(np.concatenate(([g, len(T), runtime, res.nit, res.fun], theta_hat, y[0])))
                    fout.flush()
                
                if g is True: n_successes1 += 1
                if g is False: n_successes0 += 1
    
            #print(res.success, res.message, n_successes, 'successes out of', i + 1, 'trials')

    if True in grads:
        print('Exact gradient', n_successes1, 'successes out of', n, 'trials')
    if False in grads:
        print('Numerical gradient', n_successes0, 'successes out of', n, 'trials')

    if out: fout.close()
    if log:
        if True in grads:
            flog.write('Exact gradient ' + str(n_successes1) + \
                       ' successes out of ' + str(n) + ' trials\n')
        if False in grads:
            flog.write('Numerical gradient ' + str(n_successes0) + \
                       ' successes out of ' + str(n) + ' trials\n')
        flog.close()

    return theta_hat

def mle(y, T, arrival, branch, observ, fixed_params, log,
        grad=False, trace=False, disp=0):
    theta0 = unpack('init', y, arrival, branch, observ, T)
    bounds = unpack('bounds', y, arrival, branch, observ, T)
    #print(theta0)
    #print(bounds)

    # Call the optimizer
    objective_args = (y, T, arrival, branch, observ, fixed_params, log)

    if grad:
        obj = smooth_objective_grad
        jac = True
    else:
        obj = objective
        jac = False
        


    theta_trace = []
    obj_trace = []
    time_trace = []
    if not trace:
        callback=None
    else:
        def callback(theta):
            z = objective(theta, *objective_args)
            theta_trace.append(theta)
            obj_trace.append(z)
            time_trace.append(time.process_time() - start)
    def optimizer_progress_report_str(Xi):
        global Nfeval
        print('{0:4d}   {1:s} {2:s}'.format(Nfeval, str(Xi), str(objective(Xi, *objective_args))))
        Nfeval += 1

    if KEVIN_ONLY:
        callback = optimizer_progress_report_str
        print('{0:4s}   {1:9s}'.format('Iter', ' theta'))

    start = time.process_time()
    res = optimize.minimize(obj, theta0, args=objective_args,
                            method='L-BFGS-B', jac=jac, bounds=bounds,
                            options={'disp': disp, 'iprint': 1}, callback=callback)
    #options={'disp': True})#, 'eps': 1e-12, 'ftol': 1e-15, 'gtol': 1e-15})
    end = time.process_time()
    runtime = end - start

    if trace:
        return res, runtime, obj_trace, time_trace
    else:
        return res, runtime
    
def objective(theta, y, T, arrival, branch, observ, fixed_params, log):
    # Make sure all values in theta are valid
    if not np.all(np.isfinite(theta)): return float('inf')

    # Turn theta from np array into dictionary
    arrival_theta, branch_theta, observ_theta = theta_unpack(theta, T, arrival, branch, observ, fixed_params)
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

    #print(theta, ll)
    return -ll

def objective_grad(theta, y, T, arrival, branch, observ, fixed_params, log):
    # Make sure all values in theta are valid
    if not np.all(np.isfinite(theta)): return float('inf')

    # Turn theta from np array into dictionary
    tmp = theta_unpack(theta, T, arrival, branch, observ, fixed_params, return_hyperparams=True)
    arrival_theta, branch_theta, observ_theta = tmp[:3]
    arrival_hyperparam, branch_hyperparam, observ_hyperparam = tmp[3:]

    arrival_mask = arrival['need_grad'](T)
    branch_mask = branch['need_grad'](T)
    observ_mask = observ['need_grad'](T)

    arrival_theta = Parameter.wrap(arrival_theta, need_grad=arrival_mask)
    branch_theta  = Parameter.wrap( branch_theta, need_grad=branch_mask)
    observ_theta  = Parameter.wrap( observ_theta, need_grad=observ_mask)

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
                             arrival, branch, observ,
                             arrival_hyperparam, branch_hyperparam, observ_hyperparam)
        nll -= ll_i

    #print(theta, nll, grad)
    return nll, grad

def smooth_objective_grad(theta, y, T, arrival, branch, observ, fixed_params, log):
    # Make sure all values in theta are valid
    if not np.all(np.isfinite(theta)): return float('inf')

    # Turn theta from np array into dictionary
    tmp = theta_unpack(theta, T, arrival, branch, observ, fixed_params, return_hyperparams=True)
    arrival_theta, branch_theta, observ_theta = tmp[:3]
    arrival_hyperparam, branch_hyperparam, observ_hyperparam = tmp[3:]

    arrival_mask = arrival['need_grad'](T)
    branch_mask = branch['need_grad'](T)
    observ_mask = observ['need_grad'](T)

    arrival_theta = Parameter.wrap(arrival_theta, need_grad=arrival_mask)
    branch_theta = Parameter.wrap(branch_theta, need_grad=branch_mask)
    observ_theta = Parameter.wrap(observ_theta, need_grad=observ_mask)

    # print('arrival', arrival_theta)
    # print('branch', branch_theta)
    # print('observ', observ_theta)

    # Compute log likelihood and return negative log likelihood
    arrival_pgf = arrival['pgf_grad']
    branch_pgf = branch['pgf_grad']
    # print(arrival_pgf, branch_pgf)

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
                             arrival, branch, observ,
                             arrival_hyperparam, branch_hyperparam, observ_hyperparam)

        nll -= ll_i

    nll += branch['log_prior'](branch_theta)

    # print(theta, nll, grad)
    return nll, grad

# Flatten 2d list
def flatten(l):
    return [i for row in l for i in row]

def recover_grad(T, arrival_grad, branch_grad, observ_grad,
                 arrival, branch, observ,
                 arrival_hyperparam, branch_hyperparam, observ_hyperparam):
    arrival_hyperparam_grad = arrival['backprop'](arrival_grad, arrival_hyperparam)
    branch_hyperparam_grad  = branch['backprop'](branch_grad, branch_hyperparam) - branch['log_prior_grad'](branch_hyperparam)
    # branch_hyperparam_grad = branch['backprop'](branch_grad, branch_hyperparam)
    observ_hyperparam_grad  = observ['backprop'](observ_grad, observ_hyperparam)

    out = np.concatenate((arrival_hyperparam_grad, branch_hyperparam_grad, observ_hyperparam_grad))
    return out

def theta_unpack(theta_array, T, arrival, branch, observ, fixed_params,
                 return_hyperparams=False):
    n_arrival_params = arrival['n_params'](T)
    n_branch_params = branch['n_params'](T)
    n_observ_params = observ['n_params'](T)

    idx = 0
    arrival_params = theta_array[idx:idx+n_arrival_params] if n_arrival_params > 0 else fixed_params['arrival']
    idx += n_arrival_params
    branch_params = theta_array[idx:idx+n_branch_params] if n_branch_params > 0 else fixed_params['branch']
    idx += n_branch_params
    observ_params = theta_array[idx:idx+n_observ_params] if n_observ_params > 0 else fixed_params['observ']
    idx += n_observ_params
    
    assert idx == len(theta_array)

    theta_arrival = arrival['hyperparam2param'](arrival_params, T).astype(np.float64)
    theta_branch = branch['hyperparam2param'](branch_params, T).astype(np.float64)
    theta_observ = observ['hyperparam2param'](observ_params, T).astype(np.float64)
    
    if return_hyperparams:
        return theta_arrival, theta_branch, theta_observ, \
               arrival_params, branch_params, observ_params
    else:
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

def unpack(k, y, arrival, branch, observ, T):
    tmp = [d[k](y, T) for d in [arrival, branch, observ]]
    tmp = [v if isinstance(v, list) else [v] for v in tmp]
    return [v for l in tmp for v in l]

def mean2p(mu, size):
    return float(size)/(size+mu)

def get_fixed_params(arrival, branch, observ, true_params, T):
    fixed_params = {}
    if arrival['n_params'](T) == 0: fixed_params['arrival'] = true_params['arrival']
    if branch['n_params'](T) == 0: fixed_params['branch'] = true_params['branch']
    if observ['n_params'](T) == 0: fixed_params['observ'] = true_params['observ']

    return fixed_params