import cygdual as cygd
import logsign as ls
import lsgdual as lsgd
import gdual as gd
import numpy as np
import copy
import lsgeneratingfunctions as lsgf
from scipy.special import gammaln


def compose(G, F):
    assert G.shape == F.shape

    """compose two gduals as G(F)"""
    q = len(F)
    H = lsgd.lsgdual_cdx(0, q)

    # cache first terms of G, F and then clear same
    G_0 = copy.deepcopy(G[0])
    F_0 = copy.deepcopy(F[0])
    G[0] = ls.real2ls(0)
    F[0] = ls.real2ls(0)

    H[0] = G[q - 1]
    for i in range(q - 2, -1, -1):
        H = cygd.mul(H, F)
        H[0] = ls.add(H[0], G[i])

    # restore cached values and copy G[0] to output
    H[0] = copy.deepcopy(G_0)
    G[0] = G_0
    F[0] = F_0

    return H


def compose_affine(G, F):
    """compose two gduals as G(F)"""
    if F.shape[0] <= 1:
        # composition with a constant F
        return copy.deepcopy(G)

    q = G.shape[0]

    # no need for Horner's method, utp composition uses only the 2nd and higher
    # coefficients, of which F has only 1 nonzero in this case
    H = ls.mul(G, ls.pow(F[1], np.arange(0, q)))

    return H


def lsgdualforward(y,
                   arrival_pgf_lsgdual,
                   theta_arrival,
                   branch_pgf_lsgdual,
                   theta_branch,
                   theta_observ,
                   d = 1):
    K = len(y) # K = length of chain/number of observations

    Alpha = [None] * K # Alpha = list of gdual objects for each alpha message

    def lift_A(s, k, q_k):
        # recursively compute alpha messages
        # k = observation indices, used to index into theta objects
        # q_k = length of gduals for index k (varies due to observation)

        # base case, alpha = 1
        if k < 0:
            alpha = lsgd.lsgdual_1dx(q_k)

            return alpha

        s_ds = lsgd.lsgdual_xdx(s, q_k)

        # unroll to recurse to the next layer of lift_A
        u_du = lsgd.lsgdual_xdx(copy.deepcopy(s_ds[0]), q_k + y[k])
        u_du = cygd.mul(u_du, lsgd.lsgdual_cdx(1 - theta_observ[k], q_k + y[k]))

        F = branch_pgf_lsgdual(u_du, theta_branch[k - 1, :])

        s_prev = copy.deepcopy(F[0])

        # recurse
        beta = lift_A(s_prev,
                      k - 1,
                      q_k + y[k])

        beta = compose(beta, F)

        # construct the arrival pgf, then mix with beta
        G = arrival_pgf_lsgdual(u_du, theta_arrival[k,:])
        beta = cygd.mul(beta, G)

        # observe
        alpha = lsgd.deriv(beta, y[k])
        alpha = compose_affine(alpha,
                               cygd.mul(s_ds, lsgd.lsgdual_cdx(1 - theta_observ[k], q_k)))
        # UTP for (s * rho)^{y_k} (the conditioning correction)
        corr_ds = cygd.mul(s_ds, lsgd.lsgdual_cdx(theta_observ[k], q_k))
        corr_ds = cygd.pow(corr_ds, y[k])
        alpha = cygd.mul(alpha, corr_ds)

        # divide by y[k]! (in log space)
        alpha[0]['mag'] -= gammaln(y[k] + 1)

        Alpha[k] = alpha
        return alpha

    lift_A(1.0, K - 1, d)

    return Alpha


if __name__ == "__main__":
    # F = lsgd.lsgdual_xdx(4, 7)
    # F = lsgd.log(F)
    #
    # # print(lsgd.lsgd2gd(F))
    #
    # # F_gd = gd.gdual_new(4, 7)
    # # F_gd = gd.gdual_log(F_gd)
    #
    # # print(F_gd)
    #
    # G = lsgd.lsgdual_xdx(-2, 7)
    # # G = cygd.exp(G)
    # # G = lsgd.add_scalar(G, 3)
    #
    # # GF = compose(G, F)
    # GF = compose_affine(F, G)
    #
    # # print(lsgd.lsgd2gd(GF))
    #
    # F_gd = lsgd.lsgd2gd(F)
    # G_gd = lsgd.lsgd2gd(G)
    # GF_gd = gd.gdual_compose(G_gd, F_gd)
    #
    # print(GF_gd)

    lsgdualforward(np.array([1,2,3]),
                   lsgf.poisson,
                   np.array([1,2,3]).reshape(-1,1),
                   lsgf.bernoulli,
                   np.array([0.1, 0.1, 0.1]).reshape(-1,1),
                   np.array([0.8, 0.8, 0.8]),
                   d = 1)