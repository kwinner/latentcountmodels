import numpy as np
import scipy.misc, scipy.special
from algopy import UTPM

from UTPPGF_util import *
from UTPPGF_cython import utp_compose_cython


# def utppgffa(y, Theta, arrival_pgf, branch_pgf, observ_pgf, d=1):
#     K = len(y)
#
#     Alpha = [None] * K
#
#     # define the recursive function to compute the Alpha messages
#     def lift_A(s, k, d_k):  # returns < A_k(s), ds >_{d_k}
#
#         if k < 0:
#             alpha = UTPM(np.zeros((d_k, 1)))
#             alpha.data[0, 0] = 1.
#             alpha.data[1, 0] = 0.
#
#             Alpha[k] = alpha
#             return alpha
#
#         # print "k=%01d, y=%02d, d=%02d" % (k, y[k], d)
#
#         F = lambda u: branch_pgf(u, Theta['branch'][k - 1])  # branching PGF
#         G = lambda u: arrival_pgf(u, Theta['arrival'][k])  # arrival PGF
#
#         u = s * (1 - Theta['observ'][k])
#         s_prev = F(u)
#
#         u_du = new_utp(u, d_k + y[k])
#         # beta = utp_compose(lift_A(s_prev, k - 1, d_k + y[k]), F(u_du)) * G(u_du)
#         beta = utp_compose_utpm(lift_A(s_prev, k - 1, d_k + y[k]), F(u_du)) * G(u_du)
#         # beta = utp_compose(lift_A(s_prev, k - 1, d_k + y[k]).data.squeeze(axis=(1,)), F(u_du).data.squeeze(axis=(1,))) * G(u_du)
#
#         s_ds = new_utp(s, d_k)
#         # alpha = utp_compose(utp_deriv(beta, y[k]), (s_ds * (1 - Theta['observ'][k]))) / scipy.misc.factorial(y[k]) * np.power(s_ds * Theta['observ'][k], y[k])
#         alpha = utp_compose_utpm(utp_deriv(beta, y[k]), (s_ds * (1 - Theta['observ'][k]))) / scipy.misc.factorial(y[k]) * np.power(s_ds * Theta['observ'][k], y[k])
#         # alpha = utp_compose(utp_deriv(beta, y[k]).data.squeeze(axis=(1,)), (s_ds * (1 - Theta['observ'][k])).data.squeeze(axis=(1,))) / scipy.misc.factorial(y[k]) * np.power(s_ds * Theta['observ'][k], y[k])
#
#         Alpha[k] = alpha
#         return alpha
#
#     # call the top level lift_A (which records all the Alpha messages as it goes)
#     lift_A(1, K-1, d)
#
#     return Alpha


# def utppgffa_vec(y, Theta, arrival_pgf, branch_pgf, observ_pgf, d=1):
def utppgffa(y, Theta, arrival_pgf, branch_pgf, observ_pgf, d=1):
    #print y, Theta, arrival_pgf, branch_pgf, observ_pgf
    K = len(y)

    Alpha = [None] * K

    # define the recursive function to compute the Alpha messages
    def lift_A(s, k, d_k):  # returns < A_k(s), ds >_{d_k}

        if k < 0:
            # new utp for f = 1
            alpha = np.zeros(d_k)
            alpha[0] = 1.
            # alpha[1] = 0.

            Alpha[k] = alpha
            return alpha

        F = lambda u: lift_generating_function_utpm(branch_pgf, u, Theta['branch'][k - 1])  # branching PGF
        G = lambda u: lift_generating_function_utpm(arrival_pgf, u, Theta['arrival'][k])  # arrival PGF

        # scalar mul
        u = s * (1 - Theta['observ'][k])
        # lifted GF
        s_prev = F(u)

        # init vector utp
        u_du = new_utp_vec(u, d_k + y[k])

        # recurse
        beta = utp_compose_vec(lift_A(s_prev, k - 1, d_k + y[k]), F(u_du))

        # utp mul
        beta = utp_mul_vec(beta, G(u_du))

        s_ds = new_utp_vec(s, d_k)
        # derivative, scalar mul, and compose
        alpha = utp_compose_affine(utp_deriv_vec(beta, y[k]), (s_ds * (1 - Theta['observ'][k])))
        # scalar mul
        #alpha /= scipy.misc.factorial(y[k])
        alpha /= scipy.special.gamma(y[k] + 1)
        alpha = utp_mul_vec(alpha, utp_pow_vec(s_ds * Theta['observ'][k], y[k]))

        Alpha[k] = alpha
        return alpha

    # call the top level lift_A (which records all the Alpha messages as it goes)
    lift_A(1, K-1, d)

    return Alpha

# def utppgffa(y, Theta, arrival_pgf, branch_pgf, observ_pgf, d=1):
#     K = len(y)
#
#     Alpha = [None] * K
#
#     # define the recursive function to compute the Alpha messages
#     def log_lift_A(s, k, d_k):  # returns < A_k(s), ds >_{d_k}
#
#         if k < 0:
#             # new utp for f = 1
#             lalpha = np.zeros(d_k)
#             lalpha[0] = 0.
#             # alpha[1] = 0.
#
#             Alpha[k] = lalpha
#             return lalpha
#
#         F = lambda u: lift_generating_function_utpm(branch_pgf, u, Theta['branch'][k - 1])  # branching PGF
#         G = lambda u: lift_generating_function_utpm(arrival_pgf, u, Theta['arrival'][k])  # arrival PGF
#         # G = lambda u: utp_log_vec(lift_generating_function_utpm(arrival_pgf, u, Theta['arrival'][k]))  # arrival PGF
#
#         # scalar mul
#         u = s * (1 - Theta['observ'][k])
#         # lifted GF
#         s_prev = F(u)
#
#         # init vector utp
#         u_du = new_utp_vec(u, d_k + y[k])
#
#         # recurse
#         beta = utp_compose_vec(log_lift_A(s_prev, k - 1, d_k + y[k]), F(u_du))
#
#         # utp mul
#         # beta = utp_mul_vec(beta, G(u_du))
#         # beta = beta + utp_log_vec(G(u_du))
#         beta = beta + np.log(G(u_du))
#
#         s_ds = new_utp_vec(s, d_k)
#         # derivative, scalar mul, and compose
#         alpha = utp_log_vec(utp_compose_vec(utp_deriv_vec(utp_exp_vec(beta), y[k]), s_ds * (1 - Theta['observ'][k])))
#         # scalar mul
#         # alpha /= scipy.misc.factorial(y[k])
#         # alpha = utp_mul_vec(alpha, utp_pow_vec(s_ds * Theta['observ'][k], y[k]))
#         alpha = alpha + utp_log_vec(utp_pow_vec(s_ds * Theta['observ'][k], y[k]) / scipy.misc.factorial(y[k]))
#
#         Alpha[k] = alpha
#         return alpha
#
#     # call the top level lift_A (which records all the Alpha messages as it goes)
#     log_lift_A(1, K-1, d)
#
#     return Alpha

def log_likelihood(y, Theta, arrival_pgf, branch_pgf, observ_pgf, d=2):
    alpha = utppgffa(y, Theta, arrival_pgf, branch_pgf, observ_pgf, d)
    lik = alpha[-1][0]
    ll = np.log(lik) if lik > 0 else -np.inf
    return ll
