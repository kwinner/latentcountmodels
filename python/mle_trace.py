import os, sys
import matplotlib.pyplot as plt
import pickle

from mle_distributions import *
from mle import *

EXACT_LINESTYLE       = '-'
# EXACT_MARKERSTYLE     = 'o' #circle
NUMER_LINESTYLE     = ':'
# NUMER_MARKERSTYLE   = '^' #triangle up

FIG_SIZE = (6.4, 4.8)
ADD_TITLE = False

LINE_WIDTH            = 3  # default 1.5
# MARKER_SIZE           = None # default 6
LEGEND_FONTSIZE       = 16
XAXES_FONTSIZE        = 14
YAXES_FONTSIZE        = 12
TITLE_FONTSIZE        = 30
XLABEL_FONTSIZE       = 20
YLABEL_FONTSIZE       = 20

def trace_plots():
    with open(os.path.expanduser('~/Work/Data/Results/bad_mle_trace.pickle'), 'rb') as file:
        results = pickle.load(file)
        t_numer = results['t_numer']
        z_numer = results['z_numer']
        t_exact = results['t_exact']
        z_exact = results['z_exact']

    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(t_numer, z_numer, NUMER_LINESTYLE, linewidth=LINE_WIDTH)
    plt.plot(t_exact[:120], z_exact[:120], EXACT_LINESTYLE, linewidth=LINE_WIDTH)
    # plt.plot(t_exact, z_exact, EXACT_LINESTYLE, linewidth=LINE_WIDTH)
    plt.legend(('Numerical', 'Exact'), title='Gradient', fontsize=LEGEND_FONTSIZE)
    plt.xlabel('Running Time (s)', fontsize=XLABEL_FONTSIZE)
    plt.ylabel('Objective (NLL)', fontsize=YLABEL_FONTSIZE)
    if ADD_TITLE:
        plt.title('NB Immigration', fontsize=TITLE_FONTSIZE)
    plt.savefig(os.path.expanduser('~/Work/Data/Results/bad_mle_trace_rt.pdf'))
    plt.close(fig)

    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(z_numer, NUMER_LINESTYLE, linewidth=LINE_WIDTH)
    plt.plot(z_exact[:120], EXACT_LINESTYLE, linewidth=LINE_WIDTH)
    # plt.plot(z_exact, EXACT_LINESTYLE, linewidth=LINE_WIDTH)
    plt.legend(('Numerical', 'Exact'), title='Gradient', fontsize=LEGEND_FONTSIZE)
    plt.xlabel('Iteration', fontsize=XLABEL_FONTSIZE)
    plt.ylabel('Objective (NLL)', fontsize=YLABEL_FONTSIZE)
    if ADD_TITLE:
        plt.title('NB Immigration', fontsize=TITLE_FONTSIZE)
    plt.savefig(os.path.expanduser('~/Work/Data/Results/bad_mle_trace_it.pdf'))
    plt.close(fig)

    with open(os.path.expanduser('~/Work/Data/Results/good_mle_trace.pickle'), 'rb') as file:
        results = pickle.load(file)
        t_numer = results['t_numer']
        z_numer = results['z_numer']
        t_exact = results['t_exact']
        z_exact = results['z_exact']

    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(t_numer, z_numer, NUMER_LINESTYLE, linewidth=LINE_WIDTH)
    # plt.plot(t_exact[:120], z_exact[:120])
    plt.plot(t_exact, z_exact, EXACT_LINESTYLE, linewidth=LINE_WIDTH)
    plt.legend(('Numerical', 'Exact'), title='Gradient', fontsize=LEGEND_FONTSIZE)
    plt.xlabel('Running Time (s)', fontsize=XLABEL_FONTSIZE)
    plt.ylabel('Objective (NLL)', fontsize=YLABEL_FONTSIZE)
    if ADD_TITLE:
        plt.title('Poisson Immigration', fontsize=TITLE_FONTSIZE)
    plt.savefig(os.path.expanduser('~/Work/Data/Results/good_mle_trace_rt.pdf'))
    plt.close(fig)

    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(z_numer, NUMER_LINESTYLE, linewidth=LINE_WIDTH)
    # plt.plot(z_exact[:120])
    plt.plot(z_exact, EXACT_LINESTYLE, linewidth=LINE_WIDTH)
    plt.legend(('Numerical', 'Exact'), title='Gradient', fontsize=LEGEND_FONTSIZE)
    plt.xlabel('Iteration', fontsize=XLABEL_FONTSIZE)
    plt.ylabel('Objective (NLL)', fontsize=YLABEL_FONTSIZE)
    if ADD_TITLE:
        plt.title('NB Immigration', fontsize=TITLE_FONTSIZE)
    plt.savefig(os.path.expanduser('~/Work/Data/Results/good_mle_trace_it.pdf'))
    plt.close(fig)

if __name__ == "__main__":
    #
    # arrival = constant_nbinom_arrival
    # branch = var_binom_branch
    # observ = constant_binom_observ
    #
    # # y = np.array([[ 1,  0,  4,  2,  4,  9,  4,  8,  5,  2],
    # #               [ 3,  3,  2,  7,  4,  8,  3,  7,  6,  3],
    # #               [ 4,  6,  3,  4,  4,  8,  6,  9, 13,  4],
    # #               [ 4,  3,  4,  6,  9,  6,  7,  6,  8,  4],
    # #               [ 1,  5,  3,  4,  6,  4,  7,  7,  9,  1],
    # #               [ 2,  7,  8, 11,  6,  3,  1,  5,  6,  3],
    # #               [ 2,  4,  3,  4,  1,  5,  4,  3,  6,  1],
    # #               [ 3,  9,  6,  7,  9,  7,  2,  5,  2,  4],
    # #               [ 3,  3,  1,  5,  9,  7,  4,  7,  9,  2],
    # #               [ 1,  4,  5,  3,  4,  6,  2,  6,  8,  4]])
    #
    #
    # y = np.array([[ 1,  1,  4,  2,  4,  9,  4,  8,  5,  2],
    #               [ 1,  4,  5,  3,  4,  6,  2,  6,  8,  4]])
    #
    #
    # K = len(y[0])
    # T = np.arange(K)
    #
    # with open('foo.txt', 'w') as log:
    #     res_exact, runtime_exact, z_exact, t_exact = mle(y, T, arrival, branch, observ, fixed_params=None, log = log, grad=True, trace=True, disp=1)
    #     res_numer, runtime_numer, z_numer, t_numer = mle(y, T, arrival, branch, observ, fixed_params=None, log = log, grad=False, trace=True, disp=1)
    #
    # pickle.dump({'t_exact': t_exact,
    #              't_numer': t_numer,
    #              'z_exact': z_exact,
    #              'z_numer': z_numer,
    #              'y': y},
    #             open(os.path.expanduser('~/Work/Data/Results/bad_mle_trace.pickle'), 'wb'))
    #
    #
    #
    #
    # arrival = constant_poisson_arrival
    # with open('foo.txt', 'w') as log:
    #     res_exact, runtime_exact, z_exact, t_exact = mle(y, T, arrival, branch, observ, fixed_params=None, log = log, grad=True, trace=True, disp=1)
    #     res_numer, runtime_numer, z_numer, t_numer = mle(y, T, arrival, branch, observ, fixed_params=None, log = log, grad=False, trace=True, disp=1)
    #
    # pickle.dump({'t_exact': t_exact,
    #              't_numer': t_numer,
    #              'z_exact': z_exact,
    #              'z_numer': z_numer,
    #              'y': y},
    #             open(os.path.expanduser('~/Work/Data/Results/good_mle_trace.pickle'), 'wb'))

    trace_plots()