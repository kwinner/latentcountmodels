import os, sys
import matplotlib.pyplot as plt

from mle_distributions import *
from mle import *


if __name__ == "__main__":

    arrival = constant_nbinom_arrival
    branch = var_binom_branch
    observ = constant_binom_observ

    y = np.array([[ 1,  0,  4,  2,  4,  9,  4,  8,  5,  2],
                  [ 3,  3,  2,  7,  4,  8,  3,  7,  6,  3],
                  [ 4,  6,  3,  4,  4,  8,  6,  9, 13,  4],
                  [ 4,  3,  4,  6,  9,  6,  7,  6,  8,  4],
                  [ 1,  5,  3,  4,  6,  4,  7,  7,  9,  1],
                  [ 2,  7,  8, 11,  6,  3,  1,  5,  6,  3],
                  [ 2,  4,  3,  4,  1,  5,  4,  3,  6,  1],
                  [ 3,  9,  6,  7,  9,  7,  2,  5,  2,  4],
                  [ 3,  3,  1,  5,  9,  7,  4,  7,  9,  2],
                  [ 1,  4,  5,  3,  4,  6,  2,  6,  8,  4]])


    y = np.array([[ 1,  1,  4,  2,  4,  9,  4,  8,  5,  2],
                  [ 1,  4,  5,  3,  4,  6,  2,  6,  8,  4]])

    
    K = len(y[0])
    T = np.arange(K)

    with open('foo.txt', 'w') as log:
        res_exact, runtime_exact, z_exact, t_exact = mle(y, T, arrival, branch, observ, log, grad=True, trace=True, disp=1)
        res_numer, runtime_numer, z_numer, t_numer = mle(y, T, arrival, branch, observ, log, grad=False, trace=True, disp=1)
        
    plt.figure()
    plt.plot(t_exact, z_exact)
    plt.plot(t_numer, z_numer)
    plt.legend(('exact', 'numerical'))
    plt.xlabel('Time')
    plt.ylabel('Objective')
    plt.savefig('mle_trace.pdf')
    
