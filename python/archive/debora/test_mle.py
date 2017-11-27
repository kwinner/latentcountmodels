import time, sys
import numpy as np
from lib import pgffa, truncatedfa, arrival, branching
from lib.model import NMixture, Zonneveld

"""
Modes:
1.  PGF, N-mixture, Poisson arrival
2.  PGF, Zonneveld, Poisson arrival, binomial branching
3.  Truncated, N-mixture, Poisson arrival
4.  Truncated, N-mixture, NB arrival
5.  Truncated, N-mixture, geom arrival
6.  Truncated, Zonneveld, Poisson arrival, binomial branching
7.  Truncated, Zonneveld, Poisson arrival, Poisson branching
8.  Truncated, Zonneveld, NB arrival, binomial branching
9.  Truncated, Zonneveld, NB arrival, Poisson branching
10. Truncated, Zonneveld, geom arrival, binomial branching
11. Truncated, Zonneveld, geom arrival, Poisson branching
"""

models = [(NMixture, (arrival.poisson, )),
          (Zonneveld, (arrival.poisson, branching.binom)),
          (NMixture, (arrival.poisson, )),
          (NMixture, (arrival.nbinom, )),
          (NMixture, (arrival.geom, )),
          (Zonneveld, (arrival.poisson, branching.binom)),
          (Zonneveld, (arrival.poisson, branching.poisson)),
          (Zonneveld, (arrival.nbinom, branching.binom)),
          (Zonneveld, (arrival.nbinom, branching.poisson)),
          (Zonneveld, (arrival.geom, branching.binom)),
          (Zonneveld, (arrival.geom, branching.poisson))]

data = [np.array([21, 25, 15, 21, 22, 21, 28, 22, 15, 19]),
        np.array([12, 78, 85, 88, 74, 58, 44, 40, 26, 15]),
        np.array([21, 25, 15, 21, 22, 21, 28, 22, 15, 19]),
        np.array([11, 16, 13, 15, 12,  9, 12, 12, 14, 11]),
        np.array([203, 205, 207, 196, 201, 210, 212, 198, 202, 210]),
        np.array([12, 78, 85, 88, 74, 58, 44, 40, 26, 15]),
        np.array([2, 34, 27, 43, 43, 66, 44, 57, 84, 80]),
        np.array([20, 77, 73, 76, 64, 47, 41, 32, 23, 19]),
        np.array([19, 65, 69, 70, 70, 75, 102, 122, 105, 109]),
        np.array([12, 17, 51, 43, 46, 35, 30, 24, 16, 15]),
        np.array([5, 5, 18, 14, 37, 52, 74, 96, 113, 164])]

true_params = [{'lambda': 76, 'rho': 0.3},
               {'mu': 2, 'c': 300, 'sigma': 1, 'rho': 0.6, 'lambda': 1.5},
               {'rho': 0.3, 'lambda': 76},
               {'p': 0.15, 'r': 15, 'rho': 0.3},
               {'p': 0.15, 'rho': 0.3},
               {'mu': 2, 'c': 300, 'sigma': 1, 'rho': 0.6, 'lambda': 1.5},
               {'mu': 2, 'c': 100, 'sigma': 1, 'rho': 0.6, 'lambda': 1.2},
               {'c': 300, 'mu': 2, 'r': 3, 'rho': 0.8, 'sigma': 1, 'lambda': 1.5},
               {'c': 200, 'mu': 2, 'r': 3, 'rho': 0.8, 'sigma': 1, 'lambda': 1.1},
               {'mu': 2, 'c': 300, 'sigma': 1, 'rho': 0.8, 'lambda': 1.5},
               {'mu': 2, 'c': 100, 'sigma': 2, 'rho': 0.7, 'lambda': 1.4}]

def run_mle(y, T, model, fa, true_params):
    t_start = time.clock()
    params, fmin, info = model.mle(y, T, fa=fa)
    t_end = time.clock()
    print 'learned params =', params
    print 'true params =', true_params
    print 'fmin =', fmin
    print 'info =', info
    print 'time =', t_end - t_start

def main(mode):
    fa = pgffa if mode in {1, 2} else truncatedfa
    T = None if mode in {1, 3, 4, 5} else range(0, 10, 2) + range(9, 14)
    i = mode - 1
    model, model_args = models[i]
    run_mle(data[i], T, model(*model_args), fa, true_params[i])

main(int(sys.argv[1]))
