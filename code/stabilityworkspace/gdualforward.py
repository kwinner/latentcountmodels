import numpy as np
from scipy.special import gammaln

import gdual
import generatingfunctions

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

        u_du = gdual.gdual_new(u, q_k + y[k])

        assert np.all(np.isfinite(u_du))

        F = branch_pgf_gdual(u_du, theta_branch[k-1,:])

        assert np.all(np.isfinite(F))

        s_prev = gdual.gdual_new(F[0], 1)

        # recurse
        beta = gdual.gdual_compose(lift_A(s_prev,
                                          k - 1,
                                          q_k + y[k]),
                                   F)

        assert np.all(np.isfinite(beta))

        G = arrival_pgf_gdual(u_du, theta_arrival[k,:])

        assert np.all(np.isfinite(G))

        beta = gdual.gdual_mul(beta, G)

        assert np.all(np.isfinite(beta))

        # observe
        s_ds = gdual.gdual_new(s, q_k)
        alpha = gdual.gdual_deriv(beta, y[k])

        assert np.all(np.isfinite(alpha))

        alpha = gdual.gdual_compose_affine(alpha, (s_ds * (1 - theta_observ[k])))

        assert np.all(np.isfinite(alpha))

        # normalize the alpha messages
        if np.any(alpha) and not np.any(np.isinf(alpha)):
            Z = np.max(np.abs(alpha))
            logZ[k] += np.log(Z)
            alpha /= Z

        assert np.all(np.isfinite(alpha))

        alpha = gdual.gdual_mul(alpha, gdual.gdual_pow(s_ds * theta_observ[k], y[k]))

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

    lift_A(gdual.gdual_new(1., 1), K - 1, d)

    return Alpha, logZ

def gdualforward2(y,
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

        u_du = gdual.gdual_new(u, q_k + y[k])

        assert np.all(np.isfinite(u_du))

        F = branch_pgf_gdual(u_du, theta_branch[k-1,:])

        assert np.all(np.isfinite(F))

        s_prev = gdual.gdual_new(F[0], 1)

        # recurse
        beta = lift_A(s_prev,
                      k - 1,
                      q_k + y[k])
        logZ[k], beta = gdual.gdual_normalize(beta)

        beta = gdual.gdual_compose(beta,
                                   F)

        assert np.all(np.isfinite(beta))

        G = arrival_pgf_gdual(u_du, theta_arrival[k,:])

        assert np.all(np.isfinite(G))

        G = gdual.gdual_adjust_Z(G, 0, logZ[k])

        beta = gdual.gdual_mul(beta, G)

        logZ[k], beta = gdual.gdual_renormalize(beta, logZ[k])

        assert np.all(np.isfinite(beta))

        # observe
        s_ds = gdual.gdual_new(s, q_k)
        alpha = gdual.gdual_deriv(beta, y[k])

        logZ[k], alpha = gdual.gdual_renormalize(alpha, logZ[k])

        assert np.all(np.isfinite(alpha))

        alpha = gdual.gdual_compose_affine(alpha, (s_ds * (1 - theta_observ[k])))

        logZ[k], alpha = gdual.gdual_renormalize(alpha, logZ[k])

        assert np.all(np.isfinite(alpha))

        scalar = gdual.gdual_pow(s_ds * theta_observ[k], y[k])
        scalar = gdual.gdual_adjust_Z(scalar, 0, logZ[k])

        alpha = gdual.gdual_mul(alpha, scalar)
        logZ[k], alpha = gdual.gdual_renormalize(alpha, logZ[k])

        assert np.all(np.isfinite(alpha))

        # divide by y_k! (incorporating directly into logZ)
        logZ[k] += -gammaln(y[k] + 1)

        assert np.isfinite(logZ[k])

        Alpha[k] = alpha
        return alpha

    lift_A(gdual.gdual_new(1., 1), K - 1, d)

    return Alpha, logZ