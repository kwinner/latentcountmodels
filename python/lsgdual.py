import numpy as np
import ngdual as ngd
import gdual as gd
import logsign as ls
import copy

from util import *

from scipy.special import logsumexp

#TODO: Implement lsgdual, ngdual, gdual with single-dispatch methods? https://www.python.org/dev/peps/pep-0443/

"""
In general, the following naming conventions are used for parameters/outputs of these methods:
    F (and G): input lsgduals for unary (and binary) operations
    H:         output lsgdual
    q:         length of lsgdual (positive int)
    c:         real-valued constant (any real)
    k:         integer-valued constant (non-negative int)
    x:         name for the RV with which derivatives in UTP are taken
               as a constructor parameter: the value of this variable (i.e. the first entry of the UTP)
"""


def _lsgdual_empty(q):
    """instantiate an empty lsgdual 'object'"""
    assert q > 0

    return ls.ls(q)


def lsgdual_1dx(q):
    """construct a new lsgdual for <1, dx>_q"""
    assert q > 0

    F = _lsgdual_empty(q)

    with np.errstate(divide='ignore'):
        F['mag'] = np.append([0], np.tile(-np.inf, q - 1))
        F['sgn'] = np.append([1], np.tile(0,       q - 1))

    return F


def lsgdual_cdx(c, q):
    """construct a new lsgdual for <c, dx>_q"""
    assert np.isreal(c) and (not hasattr(c, "__len__") or len(c) == 1)
    assert q > 0

    F = _lsgdual_empty(q)

    with np.errstate(divide='ignore'):
        F['mag'] = np.append([np.log(np.abs(c))], np.tile(-np.inf, q - 1))
        F['sgn'] = np.append([np.sign(c)],        np.tile(0,       q - 1))

    return F


def lsgdual_xdx(x, q):
    """construct a new lsgdual for <x, dx>_q"""
    assert np.isreal(x) and (not hasattr(x, "__len__") or len(x) == 1)
    assert q > 0

    F = _lsgdual_empty(q)

    with np.errstate(divide='ignore'):
        if q == 1:
            F['mag'] = np.array(np.log(np.abs(x)))
            F['sgn'] = np.array(np.sign(x))
        else:
            F['mag'] = np.append([np.log(np.abs(x)), 0], np.tile(-np.inf, q - 2))
            F['sgn'] = np.append([np.sign(x),        1], np.tile(0,       q - 2))

    return F


def islsgdual(F):
    """test that F is a numpy array with the structure of an lsgdual"""
    return ls.isls(F) and F.ndim == 1


def lsgd2gd(F):
    """convert an lsgdual to an (unnormalized) gdual"""
    assert(islsgdual(F))

    H = F['sgn'] * np.exp(F['mag'])

    return H


def lsgd2ngd(F):
    """convert an lsgdual to an ngdual"""
    assert(islsgdual(F))

    logZ = np.max(F['mag'])
    utp  = F['sgn'] * np.exp(F['mag'] - logZ)

    return (logZ, utp)


def gd2lsgd(F):
    """convert an (unnormalized) gdual to an lsgdual"""
    assert(isinstance(F, np.ndarray))

    q = len(F)
    H = _lsgdual_empty(q)

    with np.errstate(divide='ignore'):
        H['mag'] = np.log(np.abs(F))
        H['sgn'] = np.sign(F)

    return H


def ngd2lsgd(F):
    """convert an ngdual to an lsgdual"""
    assert(isinstance(F, tuple))

    q = len(F[1])
    H = _lsgdual_empty(q)

    with np.errstate(divide='ignore'):
        H['mag'] = F[0] + np.log(np.abs(F[1]))
        H['sgn'] = np.sign(F[1])

    return H


def add_scalar(F, c):
    """compute <f + c, dx>_q, c \in R"""
    assert islsgdual(F)
    assert np.isreal(c) and (not hasattr(c, "__len__") or len(c) == 1)

    # convert c to ls-space
    with np.errstate(divide='ignore'):
        c_mag = np.log(np.abs(c))
        c_sgn = np.sign(c)

    # take care of the simplest case
    if c_sgn == 0:
        return F

    H = copy.deepcopy(F)

    H[0] = logsumexp(    [F[0]['mag'], c_mag],
                     b = [F[0]['sgn'], c_sgn],
                     return_sign = True)

    return H

def add(F, G):
    """compute <f + g, dx>_q from <f, dx>_q and <g, dx>_q"""
    assert islsgdual(F)
    assert islsgdual(G)
    assert F.shape == G.shape

    H = ls.sum(np.array([F, G]), axis = 0)

    return H

def mul_scalar(F, c):
    """compute <c * f, dx>_q, c \in R"""
    assert islsgdual(F)
    assert np.isreal(c) and (not hasattr(c, "__len__") or len(c) == 1)

    H = copy.deepcopy(F)

    if c == 0:
        # special case mul by zero to avoid log(0)
        H[0]['mag'] = -np.inf
        H[0]['sgn'] = 0
    else:
        H[0]['mag'] += np.log(np.abs(c))
        H[0 ]['sgn'] *= np.sign(c)

    return H

def mul_fast(F, G):
    """compute <f * g, dx>_q from <f, dx>_q and <g, dx>_q
       lsgduals are first converted to ngduals, then multiplied using ngdual_mul
       ngduals are relatively stable, and can still use the FFT to do convolution"""
    assert islsgdual(F)
    assert islsgdual(G)
    assert F.shape == G.shape

    q = len(F)
    H = _lsgdual_empty(q)

    F_ngd = lsgd2ngd(F)
    G_ngd = lsgd2ngd(G)

    H_ngd = ngd.ngdual_mul(F_ngd, G_ngd)

    H     = ngd2lsgd(H_ngd)

    return H


def mul(F, G):
    """compute <f * g, dx>_q from <f, dx>_q and <g, dx>_q
       convolution of lsgduals is performed "in ls-space" using logsumexp
       will be significantly slower than mul_fast which uses ngduals and FFT
       but should be more stable"""
    assert islsgdual(F)
    assert islsgdual(G)
    assert F.shape == G.shape

    q = len(F)
    H = _lsgdual_empty(q)

    for k in range(0, q):
        # convoluted way to do convolution in ls-space
        # H[k] = \log(\sum_{j=0}^k F[j]G[k-j])
        H[k] = logsumexp(    F[0:k+1]['mag'] + G[k::-1]['mag'],
                         b = F[0:k+1]['sgn'] * G[k::-1]['sgn'],
                         return_sign = True)

    return H


def deriv(F, k):
    """compute d^k/dx^k <f, dx>_q = <d^k/dx^k f, dx>_{q-k}, k \in Z^+"""
    assert islsgdual(F)
    assert isinstance(k, int) or float(k).is_integer()

    q = len(F)

    # drop the lowest order terms from F
    H = F[k:]

    # compute the vector of falling factorial terms (from the chain rule)
    logff = logfallingfactorial(k, np.arange(k, q))

    # ls-mult H by logff
    H['mag'] += logff

    return H


def exp(F):
    """compute <exp(f), dx>_q"""
    assert islsgdual(F)

    q = len(F)
    H = _lsgdual_empty(q)

    # define f_i \in \tilde{F} = i * f_i
    F_tilde = F[1:]
    F_tilde['mag'] += np.log(np.arange(1, q))

    # first term is the simple exp
    H[0] = ls.exp(F[0])

    for i in range(1, q):
        H[i] = ls.sum(ls.mul(H[:i][::-1], F_tilde[:i]))
        H[i]['mag'] -= np.log(i)

    return H


def log(F):
    """compute <log(f), dx>_q"""
    assert islsgdual(F)

    q = len(F)
    H = _lsgdual_empty(q)

    ind = np.arange(q)
    ind[0] = 1
    ls_ind = ls.real2ls(ind)

    # define f_i \in \tilde{F} = i * f_i
    F_tilde = ls.mul(F, ls_ind)

    # first term is the simple log (note: if F[0]['sgn'] <= 0, this will still "fail"
    H[0] = ls.log(F[0])

    for i in range(1, q):
        if i == 1:
            #H_inner = 0 by construction

            H[i] = ls.div(F_tilde[i], F[0])
        else:
            H_inner = ls.sum(ls.mul(H[1:i], F[1:i][::-1]))
            # mul by -1
            H_inner['sgn'] *= -1

            H[i] = ls.div(ls.add(F_tilde[i], H_inner), F[0])
    # actually computed H_tilde (H_tilde[i] = H[i] * i). correct for that here
    H = ls.div(H, ls_ind)

    return H

def pow(F, k):
    """compute <f^k, dx>_q"""
    return

def inv(F):
    """compute <1/f, dx>_q"""
    return

def compose(G, F):
    """compose two lsgduals as G(F)"""
    assert islsgdual(F) and islsgdual(G)
    assert F.shape == G.shape

    q = len(F)
    H = lsgdual_cdx(0, q)

    # cache first terms of G, F and then clear first terms of F, G
    G_0 = copy.deepcopy(G[0])
    F_0 = copy.deepcopy(F[0])
    G[0] = ls.real2ls(0)
    F[0] = ls.real2ls(0)

    H[0] = G[q - 1]
    for i in range(q - 2, -1, -1):
        H = mul(H, F)
        H[0] = ls.add(H[0], G[i])

    # restore cached value for G as first value of H
    H[0] = G_0
    G[0] = G_0
    F[0] = F_0

    return H

def compose_affine(G, F):
    """comopse two lsgduals as G(F) where len(F) <= 2"""
    return

#optional methods (in ngdual, don't know yet if we need/want them)
#add_logscalar (add a scalar as log(C))
#mul_logscalar (mul by a scalar as log(C))

if __name__ == "__main__":
    F = lsgdual_xdx(4, 7)
    F = log(F)

    G = lsgdual_xdx(-2, 7)
    G = exp(G)
    G = add_scalar(G, 3)

    print(F)
    print(G)

    GF = compose(G, F)

    print(F)
    print(G)

    print("")

    print(lsgd2gd(GF))

    F_gd = lsgd2gd(F)
    G_gd = lsgd2gd(G)
    GF_gd = gd.gdual_compose(G_gd, F_gd)

    print(GF_gd)
