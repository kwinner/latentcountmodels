import numpy as np
from scipy.special import gammaln

import gdual

class ngdual(object):
    __slots__ = ['logZ', 'x']

    # define a dual by specifying its degree
    def __init__(self, k):
        self.logZ = 0
        self.x = np.zeros(k)
        self.x[0] = 1

    # define an lgdual by normalizing a gdual
    @classmethod
    def fromgdual(cls, x):
        inst = cls(x.shape[0])
        inst.x = x

        Z = np.max(inst.x)
        inst.logZ = np.log(Z)
        inst.x /= Z
        return inst

    def tolin(self):
        return np.exp(self.logZ) * self.x

def lgdual_mul(F, G):
    q = F.shape[0]

    F_tilde, logZ_F = lgdual_normalize(F)
    G_tilde, logZ_G = lgdual_normalize(G)

    H = np.convolve(F_tilde, G_tilde)[:q]

    # TODO: how to correct for the logZs when converting back to lgduals
    # hypothesis: "renormalize" one of F or G so both have the same logZ
    #             that logZ would factor out
    #             after conv. do the following:
    #                 convert to logs
    #                 remove factorials
    #                 reapply normalizing constant

    return H

def lgdual_wrapper(gdual_function, F):
