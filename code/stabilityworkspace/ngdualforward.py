import numpy as np
from scipy.special import gammaln
from copy import deepcopy

import ngdual


def ngdualforward(y,
                  arrival_pgf_ngdual,
                  theta_arrival,
                  branch_pgf_ngdual,
                  theta_branch,
                  theta_observ,
                  d = 1):
    K = len(y)

    Alpha = [None] * K

    def lift_A(s, k, q_k):
        # base case
        if k < 0:
            # special type of new ngdual for f = 1
            # alpha_utp = np.zeros(q_k)
            # alpha_utp[0] = 1.
            # alpha = (0, alpha_utp)
            alpha = ngdual.ngdual_new_c_dx(1.0, q_k)

            return alpha

        u = deepcopy(s)
        u = ngdual.ngdual_scalar_mul(u, (1 - theta_observ[k]))

        u_du = ngdual.ngdual_new_x_dx(u, q_k + y[k])

        assert np.isfinite(u_du[0])
        assert np.all(np.isfinite(u_du[1]))

        F = branch_pgf_ngdual(u_du, theta_branch[k-1, :])

        assert np.isfinite(F[0])
        assert np.all(np.isfinite(F[1]))

        s_prev = ngdual.ngdual_new_x_dx(F, 1)

        assert np.isfinite(s_prev[0])
        assert np.all(np.isfinite(s_prev[1]))

        # recurse

        beta = ngdual.ngdual_compose(lift_A(s_prev,
                                            k - 1,
                                            q_k + y[k]),
                                     F)

        assert np.isfinite(beta[0])
        assert np.all(np.isfinite(beta[1]))

        G = arrival_pgf_ngdual(u_du, theta_arrival[k, :])

        assert np.isfinite(G[0])
        assert np.all(np.isfinite(G[1]))

        beta = ngdual.ngdual_mul(beta, G)

        assert np.isfinite(beta[0])
        assert np.all(np.isfinite(beta[1]))

        alpha = ngdual.ngdual_deriv(beta, y[k])

        assert np.isfinite(alpha[0])
        assert np.all(np.isfinite(alpha[1]))

        s_ds = ngdual.ngdual_new_x_dx(s, q_k)

        assert np.isfinite(s_ds[0])
        assert np.all(np.isfinite(s_ds[1]))

        alpha = ngdual.ngdual_compose_affine(alpha, ngdual.ngdual_scalar_mul(s_ds, 1 - theta_observ[k]))

        assert np.isfinite(alpha[0])
        assert np.all(np.isfinite(alpha[1]))

        # scalar = ngdual.ngdual_scalar_mul(s_ds, theta_observ[k])
        #
        # assert np.isfinite(scalar[0])
        # assert np.all(np.isfinite(scalar[1]))
        #
        # scalar = ngdual.ngdual_pow(scalar, y[k])
        #
        # assert np.isfinite(scalar[0])
        # assert np.all(np.isfinite(scalar[1]))
        #
        # alpha = ngdual.ngdual_mul(alpha, scalar)
        #
        # assert np.isfinite(alpha[0])
        # assert np.all(np.isfinite(alpha[1]))

        alpha = ngdual.ngdual_mul(alpha,
                                  ngdual.ngdual_pow(ngdual.ngdual_scalar_mul(s_ds, theta_observ[k]),
                                                    y[k]))

        assert np.isfinite(alpha[0])
        assert np.all(np.isfinite(alpha[1]))

        alpha = ngdual.ngdual_scalar_mul_log(alpha, -gammaln(y[k] + 1))

        assert np.isfinite(alpha[0])
        assert np.all(np.isfinite(alpha[1]))

        Alpha[k] = alpha
        return alpha

    lift_A(ngdual.ngdual_new_x_dx(1., 1), K - 1, d)

    return Alpha