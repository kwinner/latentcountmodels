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

        F = branch_pgf_gdual(u_du, theta_branch[k-1,:])

        s_prev = gdual.gdual_new(F[0], 1)

        # recurse
        beta = gdual.gdual_compose(lift_A(s_prev,
                                          k - 1,
                                          q_k + y[k]),
                                   F)

        G = arrival_pgf_gdual(u_du, theta_arrival[k,:])

        beta = gdual.gdual_mul(beta, G)

        # observe
        s_ds = gdual.gdual_new(s, q_k)
        alpha = gdual.gdual_deriv(beta, y[k])

        alpha = gdual.gdual_compose_affine(alpha, (s_ds * (1 - theta_observ[k])))

        # normalize the alpha messages
        if np.any(alpha) and not np.any(np.isinf(alpha)):
            Z = np.max(np.abs(alpha))
            logZ[k] += np.log(Z)
            alpha /= Z

        alpha = gdual.gdual_mul(alpha, gdual.gdual_pow(s_ds * theta_observ[k], y[k]))

        # divide by y_k! (incorporating directly into logZ)
        logZ[k] = -gammaln(y[k] + 1)

        # normalize the alpha messages again
        if np.any(alpha) and not np.any(np.isinf(alpha)):
            Z = np.max(np.abs(alpha))
            logZ[k] += np.log(Z)
            alpha /= Z

        Alpha[k] = alpha
        return alpha

    lift_A(gdual.gdual_new(1., 1), K - 1, d)

    return Alpha, logZ