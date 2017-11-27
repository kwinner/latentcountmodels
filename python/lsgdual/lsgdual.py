import numpy as np
import ngdual
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
    """method to standardize the data format of all new lsgduals"""
    assert q > 0

    return np.empty(q, dtype=[('mag', '<f16'), ('sgn', 'i1')])


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
    return isinstance(F, np.ndarray)            and \
           F.ndim == 1                          and \
           F.dtype.names == ('mag', 'sgn')      and \
           np.issubdtype(F.dtype['mag'], float) and \
           np.issubdtype(F.dtype['sgn'], int)


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

    H = np.copy(F)

    H[0] = logsumexp(    [F[0]['mag'], c_mag],
                     b = [F[0]['sgn'], c_sgn],
                     return_sign = True)

    return H

def add(F, G):
    """compute <f + g, dx>_q from <f, dx>_q and <g, dx>_q"""
    assert islsgdual(F)
    assert islsgdual(G)
    assert F.shape == G.shape

    q = len(F)
    H = _lsgdual_empty(q)

    # stack the two lsgds
    FG = np.array([F, G])

    # sum elementwise over the stacked F and G, using the signs as weights
    H['mag'], H['sgn'] = logsumexp(FG['mag'], b = FG['sgn'], axis = 0, return_sign = True)

    return H

def mul_scalar(F, c):
    """compute <c * f, dx>_q, c \in R"""
    assert islsgdual(F)
    assert np.isreal(c) and (not hasattr(c, "__len__") or len(c) == 1)

    H = np.copy(F)

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

    H_ngd = ngdual.ngdual_mul(F_ngd, G_ngd)

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
        H[k] = logsumexp(    F[range(0, k+1)]['mag'] + G[range(k, -1, -1)]['mag'],
                         b = F[range(0, k+1)]['sgn'] * G[range(k, -1, -1)]['sgn'],
                         return_sign = True)

    return H


def deriv(F, k):
    """compute d^k/dx^k <f, dx>_q = <d^k/dx^k f, dx>_{q-k}, k \in Z^+"""
    return

def exp(F):
    """compute <exp(f), dx>_q"""
    return

def log(F):
    """compute <log(f), dx>_q"""
    return

def pow(F, k):
    """compute <f^k, dx>_q"""
    return

def reciprocal(F):
    """compute <1/f, dx>_q"""
    return

def compose(G, F):
    """compose two lsgduals as G(F)"""
    return

def compose_affine(G, F):
    """comopse two lsgduals as G(F) where len(F) <= 2"""
    return

#optional methods (in ngdual, don't know yet if we need/want them)
#add_logscalar (add a scalar as log(C))
#mul_logscalar (mul by a scalar as log(C))

if __name__ == "__main__":
    F = lsgdual_xdx(5, 2)
    G = lsgdual_xdx(4, 2)
    FG = np.array([F,G])

    H = add(F, G)

    print(H)