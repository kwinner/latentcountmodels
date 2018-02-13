import numpy as np
import gdual as gd
import logsign as ls
import copy
import cygdual

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


def _lsgdual_zeros(q):
    """instantiate an lsgdual 'object' with all entries equiv to zero"""
    assert q > 0

    F = np.zeros(q, dtype=ls.DTYPE)
    F['mag'] = -np.inf

    return F


def lsgdual_1dx(q):
    """construct a new lsgdual for <1, dx>_q"""
    assert q > 0

    F = _lsgdual_zeros(q)

    with np.errstate(divide='ignore'):
        F['mag'] = np.append([0], np.tile(-np.inf, q - 1))
        F['sgn'] = np.append([1], np.tile(0,       q - 1))

    return F


def lsgdual_cdx(c, q):
    """construct a new lsgdual for <c, dx>_q"""
    assert q > 0
    if ls.isls(c):
        F = _lsgdual_zeros(q)

        F[0] = copy.deepcopy(c)
    else:
        assert np.isreal(c) and (not hasattr(c, "__len__") or len(c) == 1)

        F = _lsgdual_zeros(q)

        with np.errstate(divide='ignore'):
            F['mag'] = np.append([np.log(np.abs(c))], np.tile(-np.inf, q - 1))
            F['sgn'] = np.append([np.sign(c)],        np.tile(0,       q - 1))

    return F


def lsgdual_xdx(x, q):
    """construct a new lsgdual for <x, dx>_q"""
    assert q > 0
    if ls.isls(x):
        F = _lsgdual_zeros(q)

        F[0] = copy.deepcopy(x)

        if q > 1:
            F[1] = ls.real2ls(1.0)
    else:
        assert np.isreal(x) and (not hasattr(x, "__len__") or len(x) == 1)

        F = _lsgdual_zeros(q)

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


def gd2lsgd(F):
    """convert an (unnormalized) gdual to an lsgdual"""
    assert(isinstance(F, np.ndarray))

    q = len(F)
    H = _lsgdual_empty(q)

    with np.errstate(divide='ignore'):
        H['mag'] = np.log(np.abs(F))
        H['sgn'] = np.sign(F)

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

def get_derivatives(F):
    q = len(F)
    H = copy.deepcopy(F)
    log_factorial = gammaln(1 + np.arange(q))
    H['mag'] += log_factorial
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

def pow(F, y):
    """Compute F^y"""

    q = len(F)
    
    # Determine index k of first nonzero
    zero = (F['mag'] == -np.inf) | (F['sgn'] == 0)
    k = np.flatnonzero( ~zero )[0]
    
    # Compute (F/x^k)^y
    out = cygdual.pow( F[k:], y )

    # Multiply by x^{k*y}
    out = np.append(ls.zeros(k*y), out)

    # Truncate
    out = out[:q]
    
    return out

def inv(F):
    """compute <1/f, dx>_q"""
    return


def compose_brent_kung(G, F):
    q = G.shape[0]
    
    # cache first terms of F, G and then clear
    # cache first terms of G, F and then clear same
    G_0 = copy.deepcopy(G[0])
    F_0 = copy.deepcopy(F[0])
    G[0] = ls.real2ls(0.0)
    F[0] = ls.real2ls(0.0)

    k         = int(np.ceil(np.sqrt(q)))
    n_chunks  = int(np.ceil(q / k))

    B = ls.zeros((n_chunks, k)) # holds chunks of G of len k
    A = ls.zeros((k, q))        # holds powers of F

    # Fill rows of B with chunks of G
    n_full_chunks, rem = int(q / k), q % k
    for i in range(n_full_chunks):
        B[i,:] = G[i*k:(i+1)*k]
    # There may be a final partial row
    if rem > 0:
        start = (n_chunks-1)*k
        B[-1,:rem] = G[start:(start+rem)]

    # Fill rows of A with powers of F
    A[0,0] = ls.real2ls(1.0)
    for i in np.arange(1,k):
        A[i,:] = cygdual.mul(A[i-1], F)

    # Multiplication: the ith row of C now contains
    # the coefficients of G_i(F(t)) where G_i is the
    # ith segment of G divided by t^ki
    C = ls.dot(B, A)
    
    # Now we need to compute \sum_i G_i(F(t))(F(t))^ki
    # This can be viewed as block "composition", where
    # the rows of C are the "coefficients" of a polynomial
    # to be evaluated at (F(t)^k). We will use Horner's
    # method to do this block composition

    H = C[-1,:];        # last "coefficient"
    val = cygdual.mul(A[-1,:], F) # F^k
    
    for i in range(n_chunks-2, -1, -1):
        tmp = cygdual.mul(H, val)
        H = cygdual.add(tmp, C[i,:])

    # restore cached values
    H[0] = copy.deepcopy(G_0)
    G[0] = G_0
    F[0] = F_0

    return H

def compose(G, F):
    assert G.shape == F.shape

    """compose two gduals as G(F)"""
    q = len(F)
    H = lsgdual_cdx(0, q)

    # cache first terms of G, F and then clear same
    G_0 = copy.deepcopy(G[0])
    F_0 = copy.deepcopy(F[0])
    G[0] = ls.real2ls(0.0)
    F[0] = ls.real2ls(0.0)

    H[0] = G[q - 1]
    for i in range(q - 2, -1, -1):
        H = cygdual.mul(H, F, truncation_order=q)
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
