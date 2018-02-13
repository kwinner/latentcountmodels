import csv, os
import numpy as np

import pickle

from mle_distributions import *
from mle_kevin import *

import matplotlib.pyplot as plt

if os.uname()[1] == 'kwinn':
    # RESULTS_DIR = os.path.expanduser('~/Work/Data/Results/')
    RESULTS_DIR = os.path.expanduser('~/shannon-results/case_study2/')
    # RESULTS_DIR = os.path.expanduser('~/shannon-results/case_study_Beta50/')
else:
    RESULTS_DIR = os.path.expanduser('~/Results/')

FIG_SIZE = (6.4, 4.8)
ADD_TITLE = False

LINESTYLE = '-'
MARKERSTYLE = 's'

LINE_WIDTH            = 4.5  # default 1.5
MARKER_SIZE           = 12 # default 6
LEGEND_FONTSIZE       = 16
XAXES_FONTSIZE        = 14
YAXES_FONTSIZE        = 12
TITLE_FONTSIZE        = 30
XLABEL_FONTSIZE       = 20
YLABEL_FONTSIZE       = 20

# Data source: https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html
# One-hundredth of H1N1 cases in the US during 2009 outbreak
# First wave
#y = np.array([ 1, 14, 25, 18, 28, 32, 40, 43, 46, 33, 27, 28, 22, 20, 20, 15])
# First half of the first wave
# y = np.array([[ 1, 14, 25, 18, 28, 32, 40, 43, 46]])

# All H1N1 cases in the US, first half of the first wave
#y = np.array([14, 1354, 2484, 1741, 2796, 3144, 3967, 4203, 4587])

# All H1N1 cases in New England, first half of the first wave
# y = np.array([[1, 82, 104, 111, 391, 499, 655]])
# y = np.array([[1, 82, 104]])

# All flu cases in NE, 2014-2015 season, first half
# y = np.array([[3,4,4,5,11,31,49,106,185,320,346,466,691,831]])

# All H3 cases in NE, 2014-2015 season, first half
# y = np.array([[2,9,22,32,63,112,181,186,251,346,377]])

# All H1N1 cases in NE, 2013-2014 season, first half
# y = np.array([[1,3,4,8,23,21,56,138,132,207,244]])

# y2013h1n1 = np.array([[1,3,4,8,23,21,56,138,132,207,244,199,182,134,115,89,55,27,34,22,18,10,5,8,7,1]])
# y2009h1n1 = np.array([[1,82,104,111,391,499,665,360,192,167,142,126,69,29,30,17,39,17,8,6]])
# y = y2009h1n1
# y = (0.1 * y).astype(int)

def flatten(l):
    return [i for row in l for i in row]

''' 
 "hyperparameters": learned parameters
 "parameters": exploded parameters (full lambda, delta, rho arrays)


GENERAL

T              vector of time steps

theta_arrival  2d array of arrival parameters: size K x p, where p = number of arrival pgf parameters

theta_branch   2d array of branching parameters: size K x p, where p = number of branching pgf parameters

theta_observ   1d array (double-check that all code expects 1d array)



DICT

learn_mask     (existing) number of True values here is the number of hyperparameters
               could change to just be a function that accepts T and returns number of hyperparameters

pmf            (existing) function to compute *log* pmf, defined in truncatedfa.py. used only by trunc
               (NOTE: if this is used outside trunc alg., it needs to be updated to accept logpmf)

pgf            (existing) pgf from forward.py. passed to forward(). computes only pgf value

pgf_grad       (new) pgf from forward_grad.py. passed to forward_grad(). computes pgf + grad

need_grad      (new) function that, given T, returns an array of Booleans the same size size
               as the corresponding _expanded_ parameter vector indicating which parameter partial
               derivatives are needed

hyperparam2param  (existing) maps from hyperparameters to full parameter array (2d/1d depending on component)

backprop       (new) takes gradient wrt expanded parameters, and returns gradient wrt hyperparameters.
               gradient dtheta has same shape as theta; entries that are not needed are None. It is
               a list, not a numpy array

init           (existing) accepts y, and returns array of initial hyperparameter values

bounds         (existing) accepts y, and returns array of bounds for hyperparameters

'''

# TODO: eliminate references to K within this dictionary

# var_poisson_branch = {
#     'learn_mask': [True] * (K - 1),   # TODO: 'num_hyperparams': lambda K: K-1,  (return number of parameters)
#     'pmf': poisson_branching,
#     'pgf': poisson_pgf,
#     'pgf_grad': poisson_pgf_grad,  # new: version of branching that supports backprop
#     'need_grad': lambda T: [[True]] * (len(T)-1),
#     'sample': lambda n, gamma: stats.poisson.rvs(n * gamma),
#     'hyperparam2param': lambda x, T: x.reshape((-1, 1)),
#     'backprop': lambda dtheta, x: flatten(dtheta),
#     'init': lambda y: [1.5] * (K - 1),
#     'bounds': lambda y: [(0.1, None)] * (K - 1)
# }
#
# var_nbinom_branch = {
#     'learn_mask': [True] * (K - 1),
#     'pmf': nbinom_branching,
#     'pgf': 'geometric',
#     'sample': stats.nbinom.rvs,
#     'hyperparam2param': lambda x, T: 1/(x.reshape((-1, 1)) + 1),
#     'init': lambda y: [1] * (K - 1),
#     'bounds': lambda y: [(1e-6, None)] * (K - 1)
# }
#
# grr_nbinom_branch = {
#     'learn_mask': [True] * (K + 1),
#     'pmf': nbinom_branching,
#     'pgf': 'geometric',
#     'sample': stats.nbinom.rvs,
#     'hyperparam2param': lambda x, T: 1/(x[2:].reshape((-1, 1)) + 1),
#     'init': lambda y: [1/np.var(y)] + [0.1] + [1] * (K - 1),
#     'bounds': lambda y: [(None, None)] + [(1e-6, None)] * K # [sigma, R_0, R_t's]
# }
#
# lmbda = 30
# fixed_poisson_arrival = {
#     'learn_mask': False,
#     'pmf': stats.poisson.pmf,
#     'pgf': poisson_pgf,
#     'sample': stats.poisson.rvs,
#     'hyperparam2param': lambda x, T: np.concatenate(([lmbda], np.zeros(len(T) - 1))).reshape((-1, 1)),
#     'init': lambda y: [],
#     'bounds': lambda y: []
# }

def case_study(year = None):
    # All H1N1 cases in New England, first half of the first wave
    # y = np.array([[1, 82, 104, 111, 391, 499, 655]])
    # y = np.array([[1, 82, 104]])

    # All H1N1 cases in NE, 2013-2014 season, first half
    # y = np.array([[1,3,4,8,23,21,56,138,132,207,244]])
    # y2013h1n1 = np.array([[1, 3, 4, 8, 23, 21, 56, 138, 132, 207, 244, 199, 182, 134, 115, 89, 55, 27, 34, 22, 18, 10, 5, 8, 7, 1]])
    # y2009h1n1 = np.array([[1, 82, 104, 111, 391, 499, 665, 360, 192, 167, 142, 126, 69, 29, 30, 17, 39, 17, 8, 6]])
    # y = y2009h1n1
    # y = (0.1 * y).astype(int)

    if year == None:
        y = np.array([[1, 82, 104]])
    elif year == '2009':
        y = np.array([[1, 82, 104, 111, 391, 499, 665, 360, 192, 167, 142, 126, 69, 29, 30, 17, 39, 17, 8, 6]])
    elif year == '2009.50pct':
        y = (0.5 * np.array([[1, 82, 104, 111, 391, 499, 665, 360, 192, 167, 142, 126, 69, 29, 30, 17, 39, 17, 8, 6]])).astype(int)
    elif year == '2009.firsthalf':
        y = np.array([[1, 82, 104, 111, 391, 499, 665]])
    elif year == '2013':
        y = np.array([[1, 3, 4, 8, 23, 21, 56, 138, 132, 207, 244, 199, 182, 134, 115, 89, 55, 27, 34, 22, 18, 10, 5, 8, 7, 1]])
    elif year == '2013.50pct':
        y = (0.5 * np.array([[1, 3, 4, 8, 23, 21, 56, 138, 132, 207, 244, 199, 182, 134, 115, 89, 55, 27, 34, 22, 18, 10, 5, 8, 7, 1]])).astype(int)
    elif year == '2013.firsthalf':
        y = np.array([[1, 3, 4, 8, 23, 21, 56, 138, 132, 207, 244]])

    K = len(y[0])
    print(K)

    # Varying across time
    Beta = 50.00
    var_poisson_branch_logprior = {
        'n_params': lambda T: (len(T) - 1),
        'pgf': poisson_pgf,
        'pgf_grad': poisson_pgf_grad,
        'log_prior': lambda delta: 0.5 * Beta * np.sum(np.diff(delta)**2),       # log prior on delta
        'log_prior_grad': lambda delta: Beta * np.array([delta[0] - delta[1]] + \
                                                 [2*delta[j] - delta[j-1] - delta[j+1] for j in range(1,len(delta)-1)] + \
                                                 [delta[-1] - delta[-2]]), # gradient of log prior on delta
        'sample': lambda n, gamma: stats.poisson.rvs(n * gamma),
        'hyperparam2param': lambda x, T: np.reshape(x, (-1, 1)),
        'need_grad': lambda T: [[True]] * len(T),
        'backprop': lambda dtheta, x: np.reshape(dtheta, -1),
        'init': lambda y, T: [1.0] * (len(T) - 1),
        'bounds': lambda y, T: [(1e-6, None)] * (len(T) - 1)
    }

    # Distributions
    # arrival = nmixture_poisson_arrival
    # arrival = constant_poisson_arrival
    arrival = fix_poisson_arrival

    #branch = grr_nbinom_branch
    #branch = var_nbinom_branch
    # branch = var_poisson_branch
    #branch = constant_nbinom_branch
    # branch = constant_binom_branch
    # branch = constant_poisson_branch
    branch = var_poisson_branch_logprior

    # observ = constant_binom_observ
    #observ = full_binom_observ
    observ = fix_binom_observ

    T = np.arange(K) # vector of observation times

    # theta_hat = run_mle(T, arrival, branch, observ,
    #                     y=y.astype(np.int32), grad=True)
    res = mle(y, T, arrival, branch, observ,
                    fixed_params={'arrival': 50, 'branch': None, 'observ': 0.05},
                    log=None, grad=True)[0]
    print(res)
    print(str(y))

    with open(os.path.expanduser(RESULTS_DIR) + 'case_study' + str(year) + '.pickle', 'wb') as result_file:
        pickle.dump({'res': res, 'y': y},
                    result_file)
        result_file.close()

    return(res['x'], y)


def plot_result(year):
    result_file = open(os.path.expanduser(RESULTS_DIR) + 'case_study' + str(year) + '.pickle', 'rb')
    data = pickle.load(result_file)
    theta = data['res']['x']
    y = data['y'].reshape(-1)
    result_file.close()

    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(theta, LINESTYLE + MARKERSTYLE, linewidth=LINE_WIDTH)
    # plt.xlabel('Running Time (s)', fontsize=XLABEL_FONTSIZE)
    plt.ylabel(r'$R_0$', fontsize=YLABEL_FONTSIZE)
    # if ADD_TITLE:
    #     plt.title('NB Immigration', fontsize=TITLE_FONTSIZE)
    plt.savefig(os.path.expanduser(RESULTS_DIR + 'case_study' + str(year) + 'delta.pdf'))
    plt.close(fig)

    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(y, LINESTYLE + MARKERSTYLE, linewidth=LINE_WIDTH)
    # plt.xlabel('Running Time (s)', fontsize=XLABEL_FONTSIZE)
    plt.ylabel(r'$y$', fontsize=YLABEL_FONTSIZE)
    # if ADD_TITLE:
    #     plt.title('NB Immigration', fontsize=TITLE_FONTSIZE)
    plt.savefig(os.path.expanduser(RESULTS_DIR + 'case_study' + str(year) + 'y.pdf'))
    plt.close(fig)


if __name__ == "__main__":
    if os.uname()[1] == 'kwinn':
        # case_study('2009')
        # plot_result(None)
        # plot_result('2009')
        # plot_result('2013')
        plot_result('2009.50pct')
        plot_result('2013.50pct')
        # plot_result('2009.firsthalf')
        # plot_result('2013.firsthalf')
    elif 'shannon' in os.uname()[1]:
        if len(sys.argv) == 1:
            year = None
        else:
            year = sys.argv[1]

        case_study(year)
"""
# Clean output format
r_hat = theta_hat[3:-1]
r_ci_left = ci_left[3:-1]
r_ci_right = ci_right[3:-1]

f = open('../data/case_study/pois_pois_esc_7wk_feb23.csv', 'w')
writer = csv.writer(f)
writer.writerow(['k', 'r_hat', 'ci_left', 'ci_right'])
for k, (r_k, ci1_k, ci2_k) in enumerate(zip(r_hat, r_ci_left, r_ci_right)):
	writer.writerow([k + 1, r_k, ci1_k, ci2_k])
f.close()


# Vary fixed arrival rate
path = '../data/case_study_pois_fix_lambda/'
if not os.path.exists(path):
    os.makedirs(path)

for lmbda in range(10, 101, 10):
    print lmbda
    fixed_poisson_arrival = {
        'learn_mask': False,
        'pmf': stats.poisson.pmf,
        'pgf': 'poisson',
        'sample': stats.poisson.rvs,
        'hyperparam2param': lambda x, T: np.concatenate(([lmbda], np.zeros(len(T) - 1))).reshape((-1, 1)),
        'init': lambda y: [],
        'bounds': lambda y: []
    }
    arrival = fixed_poisson_arrival
    out = None #path + str(lmbda) +'.csv'
    log = None #path + 'warnings.log'

    run_mle(T, arrival, branch, observ, fa, out=None, log=log,
            y=y.astype(np.int32), n=1, max_iters=5)
"""
