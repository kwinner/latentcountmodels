import numpy as np
import scipy.misc
from algopy import UTPM

from UTPPGF_util import *

def utppgffa(y, Theta, arrival_pgf, branch_pgf, observ_pgf, d=1):
    K = len(y)

    Alpha = [None] * K

    # define the recursive function to compute the Alpha messages
    def lift_A(s, k, d_k):  # returns < A_k(s), ds >_{d_k}

        if k < 0:
            alpha = UTPM(np.zeros((d_k, 1)))
            alpha.data[0, 0] = 1.
            alpha.data[1, 0] = 0.

            Alpha[k] = alpha
            return alpha

        # print "k=%01d, y=%02d, d=%02d" % (k, y[k], d)

        F = lambda u: branch_pgf(u, Theta['branch'][k - 1])  # branching PGF
        G = lambda u: arrival_pgf(u, Theta['arrival'][k])  # arrival PGF

        u = s * (1 - Theta['observ'][k])
        s_prev = F(u)

        u_du = new_utp(u, d_k + y[k])
        beta = utp_compose(lift_A(s_prev, k - 1, d_k + y[k]), F(u_du)) * G(u_du)

        s_ds = new_utp(s, d_k)
        alpha = utp_compose(utp_deriv(beta, y[k]), s_ds * (1 - Theta['observ'][k])) / scipy.misc.factorial(y[k]) * np.power(
            s_ds * Theta['observ'][k], y[k])

        Alpha[k] = alpha
        return alpha

    # call the top level lift_A (which records all the Alpha messages as it goes)
    lift_A(1, K-1, d)

    return Alpha