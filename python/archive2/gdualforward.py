import numpy as np
from scipy.special import gammaln
from scipy.misc import factorial

import gdual_impl as gd
from generatingfunctions import *

def gdualforward(y,
                 arrival_pgf_gdual,
                 theta_arrival,
                 branch_pgf_gdual,
                 theta_branch,
                 theta_observ,
                 d = 1):

    K = len(y)

    Alpha = [None] * K
    logZ  = np.zeros(K)

    def lift_A(s, k, q_k):
        # base case
        if k < 0:
            # special type of new gdual for f = 1
            alpha = np.zeros(q_k)
            alpha[0] = 1.

            # Alpha[k] = alpha
            return alpha

        u = s.copy()
        u *= (1 - theta_observ[k])

        u_du = gd.new(u, q_k + y[k])

        assert np.all(np.isfinite(u_du))

        F = branch_pgf_gdual(u_du, theta_branch[k-1,:])

        assert np.all(np.isfinite(F))

        s_prev = gd.new(F[0], 1)

        # recurse
        beta = lift_A(s_prev,
                      k - 1,
                      q_k + y[k])
        logZ[k], beta = gd.normalize(beta)

        beta = gd.compose(beta,
                                   F)

        assert np.all(np.isfinite(beta))

        G = arrival_pgf_gdual(u_du, theta_arrival[k,:])

        assert np.all(np.isfinite(G))

        G = gd.adjust_Z(G, 0, logZ[k])

        beta = gd.mul(beta, G)

        logZ[k], beta = gd.renormalize(beta, logZ[k])

        assert np.all(np.isfinite(beta))

        # observe
        s_ds = gd.new(s, q_k)
        # alpha = gd.deriv(beta, y[k])
        logZ_alpha, alpha = gd.deriv_safe(beta, y[k])
        logZ[k] += logZ_alpha

        # logZ[k], alpha = gd.renormalize(alpha, logZ[k])

        assert np.all(np.isfinite(alpha))

        alpha = gd.compose_affine(alpha, (s_ds * (1 - theta_observ[k])))

        # normalize the current alpha message
        logZ[k], alpha = gd.renormalize(alpha, logZ[k])

        assert np.all(np.isfinite(alpha))

        # build the UTP for (s * rho)^{y_k}
        scalar = gd.pow(s_ds * theta_observ[k], y[k])
        # normalize that UTP
        logZ_scalar, scalar = gd.normalize(scalar)

        assert np.all(np.isfinite(scalar))

        # if logZ_scalar > logZ[k]:
        #     # normalize both UTPs to logZ_scalar
        #     alpha = gd.adjust_Z(alpha, logZ[k], logZ_scalar)
        #     logZ[k] = logZ_scalar
        #     print "renorm"
        # elif logZ_scalar < logZ[k]:
        #     scalar = gd.adjust_Z(scalar, logZ_scalar, logZ[k])
        #     print "renorm"

        alpha = gd.mul(alpha, scalar)
        logZ[k] = logZ[k] + logZ_scalar
        logZ[k], alpha = gd.renormalize(alpha, logZ[k])
        # logZ[k] = logZ[k] + logZ_scalar

        assert np.all(np.isfinite(alpha))

        # divide by y_k! (incorporating directly into logZ)
        logZ[k] += -gammaln(y[k] + 1)

        assert np.isfinite(logZ[k])

        Alpha[k] = alpha
        return alpha

    lift_A(gd.new(1., 1), K - 1, d)

    return Alpha, logZ


def gdualforward_original(y,
                          arrival_pgf_gdual,
                          theta_arrival,
                          branch_pgf_gdual,
                          theta_branch,
                          theta_observ,
                          d=1):

    K = len(y)

    Alpha = [None] * K
    logZ = np.zeros(K)

    def lift_A(s, k, q_k):
        # base case
        if k < 0:
            # special type of new gdual for f = 1
            alpha = np.zeros(q_k)
            alpha[0] = 1.

            # Alpha[k] = alpha
            return alpha

        u = s.copy()
        u *= (1 - theta_observ[k])

        u_du = gd.new(u, q_k + y[k])

        assert np.all(np.isfinite(u_du))

        F = branch_pgf_gdual(u_du, theta_branch[k - 1, :])

        assert np.all(np.isfinite(F))

        s_prev = gd.new(F[0], 1)

        # recurse
        beta = gd.compose(lift_A(s_prev,
                                          k - 1,
                                          q_k + y[k]),
                                   F)

        assert np.all(np.isfinite(beta))

        G = arrival_pgf_gdual(u_du, theta_arrival[k, :])

        assert np.all(np.isfinite(G))

        beta = gd.mul(beta, G)

        assert np.all(np.isfinite(beta))

        # observe
        s_ds = gd.new(s, q_k)
        alpha = gd.deriv(beta, y[k])

        assert np.all(np.isfinite(alpha))

        alpha = gd.compose_affine(alpha, (s_ds * (1 - theta_observ[k])))

        assert np.all(np.isfinite(alpha))

        # normalize the alpha messages
        if np.any(alpha) and not np.any(np.isinf(alpha)):
            Z = np.max(np.abs(alpha))
            logZ[k] += np.log(Z)
            alpha /= Z

        assert np.all(np.isfinite(alpha))

        alpha = gd.mul(alpha, gd.pow(s_ds * theta_observ[k], y[k]))

        assert np.all(np.isfinite(alpha))

        # divide by y_k! (incorporating directly into logZ)
        logZ[k] = -gammaln(y[k] + 1)

        assert np.isfinite(logZ[k])

        # normalize the alpha messages again
        if np.any(alpha) and not np.any(np.isinf(alpha)):
            Z = np.max(np.abs(alpha))
            logZ[k] += np.log(Z)
            alpha /= Z

        assert np.all(np.isfinite(alpha))
        assert np.isfinite(logZ[k])

        Alpha[k] = alpha
        return alpha

    lift_A(gd.new(1., 1), K - 1, d)

    return Alpha, logZ


def gdualforward_unnormalized(y,
                          arrival_pgf_gdual,
                          theta_arrival,
                          branch_pgf_gdual,
                          theta_branch,
                          theta_observ,
                          d=1):

    K = len(y)

    Alpha = [None] * K

    def lift_A(s, k, q_k):
        # base case
        if k < 0:
            # special type of new gdual for f = 1
            alpha = np.zeros(q_k)
            alpha[0] = 1.

            # Alpha[k] = alpha
            return alpha

        u = s.copy()
        u *= (1 - theta_observ[k])

        u_du = gd.new(u, q_k + y[k])

        assert np.all(np.isfinite(u_du))

        F = branch_pgf_gdual(u_du, theta_branch[k - 1, :])

        assert np.all(np.isfinite(F))

        s_prev = gd.new(F[0], 1)

        # recurse
        beta = gd.compose(lift_A(s_prev,
                                          k - 1,
                                          q_k + y[k]),
                                   F)

        assert np.all(np.isfinite(beta))

        G = arrival_pgf_gdual(u_du, theta_arrival[k, :])

        assert np.all(np.isfinite(G))

        beta = gd.mul(beta, G)

        assert np.all(np.isfinite(beta))

        # observe
        s_ds = gd.new(s, q_k)
        alpha = gd.deriv(beta, y[k])

        assert np.all(np.isfinite(alpha))

        alpha = gd.compose_affine(alpha, (s_ds * (1 - theta_observ[k])))

        assert np.all(np.isfinite(alpha))

        alpha = gd.mul(alpha, gd.pow(s_ds * theta_observ[k], y[k]))

        assert np.all(np.isfinite(alpha))

        # divide by y_k! (incorporating directly into logZ)
        alpha /= factorial(y[k])

        assert np.all(np.isfinite(alpha))

        Alpha[k] = alpha
        return alpha

    lift_A(gd.new(1., 1), K - 1, d)

    return Alpha

# def gdualforward_broken(y,
#                  arrival_pgf_gdual,
#                  theta_arrival,
#                  branch_pgf_gdual,
#                  theta_branch,
#                  theta_observ,
#                  d = 1):
#
#     K = len(y)
#
#     Alpha = [None] * K
#     logZ  = np.zeros(K)
#
#     def lift_A(s, k, q_k):
#         # base case
#         if k < 0:
#             # special type of new gdual for f = 1
#             alpha = np.zeros(q_k)
#             alpha[0] = 1.
#
#             # Alpha[k] = alpha
#             return alpha
#
#         u = s.copy()
#         u *= (1 - theta_observ[k])
#
#         u_du = gd.new(u, q_k + y[k])
#
#         assert np.all(np.isfinite(u_du))
#
#         F = branch_pgf_gdual(u_du, theta_branch[k-1,:])
#
#         assert np.all(np.isfinite(F))
#
#         s_prev = gd.new(F[0], 1)
#
#         # recurse
#         beta = lift_A(s_prev,
#                       k - 1,
#                       q_k + y[k])
#         logZ[k], beta = gd.normalize(beta)
#
#         beta = gd.compose(beta,
#                                    F)
#
#         assert np.all(np.isfinite(beta))
#
#         G = arrival_pgf_gdual(u_du, theta_arrival[k,:])
#
#         assert np.all(np.isfinite(G))
#
#         G = gd.adjust_Z(G, 0, logZ[k])
#
#         beta = gd.mul(beta, G)
#
#         logZ[k], beta = gd.renormalize(beta, logZ[k])
#
#         assert np.all(np.isfinite(beta))
#
#         # observe
#         s_ds = gd.new(s, q_k)
#         # alpha = gd.deriv(beta, y[k])
#         logZ_alpha, alpha = gd.deriv_safe(beta, y[k])
#         logZ[k] += logZ_alpha
#
#         # logZ[k], alpha = gd.renormalize(alpha, logZ[k])
#
#         assert np.all(np.isfinite(alpha))
#
#         alpha = gd.compose_affine(alpha, (s_ds * (1 - theta_observ[k])))
#
#         logZ[k], alpha = gd.renormalize(alpha, logZ[k])
#
#         assert np.all(np.isfinite(alpha))
#
#         scalar = gd.pow(s_ds * theta_observ[k], y[k])
#         logZ_scalar, scalar = gd.normalize(scalar)
#
#         if logZ_scalar > logZ[k]:
#             # normalize both UTPs to logZ_scalar
#             alpha = gd.adjust_Z(alpha, logZ[k], logZ_scalar)
#             logZ[k] = logZ_scalar
#         elif logZ_scalar < logZ[k]:
#             scalar = gd.adjust_Z(scalar, logZ_scalar, logZ[k])
#
#         alpha = gd.mul(alpha, scalar)
#         # logZ[k] += logZ_scalar
#         logZ[k], alpha = gd.renormalize(alpha, logZ[k])
#
#         assert np.all(np.isfinite(alpha))
#
#         # divide by y_k! (incorporating directly into logZ)
#         logZ[k] += -gammaln(y[k] + 1)
#
#         assert np.isfinite(logZ[k])
#
#         Alpha[k] = alpha
#         return alpha
#
#     lift_A(gd.new(1., 1), K - 1, d)
#
#     return Alpha, logZ

if __name__ == "__main__":
    
    y     = np.array([2, 5, 3])
    lmbda = np.array([  10. ,  0.  , 0.  ]).reshape(-1, 1)
    delta = np.array([ 1.0 ,  1.0 , 1.0 ]).reshape(-1, 1)
    rho   = np.array([ 0.25,  0.25, 0.25]).reshape(-1, 1)

    Alpha = gdualforward_unnormalized(y,
                               poisson_gdual,
                               lmbda,
                               bernoulli_gdual,
                               delta,
                               rho,
                               d = 1)

    lik = Alpha[-1][0]
#    lik = np.exp(np.log(Alpha[-1][0]) + np.sum(logZ))

    print lik
