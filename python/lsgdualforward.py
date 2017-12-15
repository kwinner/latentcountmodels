import cygdual as cygd
import logsign as ls
import lsgdual as lsgd
import gdual as gd
import numpy as np
import copy


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
    q = G.shape[0]

    # no need for Horner's method, utp composition uses only the 2nd and higher
    # coefficients, of which F has only 1 nonzero in this case
    H = ls.mul(G, ls.pow(F[1], np.arange(0, q)))

    return H


if __name__ == "__main__":
    F = lsgd.lsgdual_xdx(4, 7)
    F = lsgd.log(F)

    # print(lsgd.lsgd2gd(F))

    # F_gd = gd.gdual_new(4, 7)
    # F_gd = gd.gdual_log(F_gd)

    # print(F_gd)

    G = lsgd.lsgdual_xdx(-2, 7)
    # G = cygd.exp(G)
    # G = lsgd.add_scalar(G, 3)

    # GF = compose(G, F)
    GF = compose_affine(F, G)

    print(lsgd.lsgd2gd(GF))

    F_gd = lsgd.lsgd2gd(F)
    G_gd = lsgd.lsgd2gd(G)
    GF_gd = gd.gdual_compose(G_gd, F_gd)

    print(GF_gd)