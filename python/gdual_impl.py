import numpy as np
import util as util

# todo function descriptions
# todo assert shape match
def new(x, q):
    # todo this is really a special case of new gdual for new RVs
    out = np.zeros(q, dtype=np.longdouble)

    out[0] = x
    if q > 1:
        out[1] = 1

    return out

def compose_brent_kung(G, F):
    q = G.shape[0]
    
    # cache first terms of F, G and then clear
    G_0_cache = G[0]
    F_0_cache = F[0]

    G[0] = 0
    F[0] = 0

    k         = int(np.ceil(np.sqrt(q)))
    n_chunks  = int(np.ceil(q / k))

    B = np.zeros((n_chunks, k)) # holds chunks of G of len k
    A = np.zeros((k, q))        # holds powers of F

    # Fill rows of B with chunks of G
    n_full_chunks, rem = int(q / k), q % k
    for i in range(n_full_chunks):
        B[i,:] = G[i*k:(i+1)*k]
    # There may be a final partial row
    if rem > 0:
        start = (n_chunks-1)*k
        B[-1,:rem] = G[start:(start+rem)]

    # Fill rows of A with powers of F
    A[0,0] = 1.0
    for i in np.arange(1,k):
        A[i,:] = mul(A[i-1], F)

    # Multiplication: the ith row of C now contains
    # the coefficients of G_i(F(t)) where G_i is the
    # ith segment of G divided by t^ki
    C = B.dot(A)

    # Now we need to compute \sum_i G_i(F(t))(F(t))^ki
    # This can be viewed as block "composition", where
    # the rows of C are the "coefficients" of a polynomial
    # to be evaluated at (F(t)^k). We will use Horner's
    # method to do this block composition

    res = C[-1,:];        # last "coefficient"
    val = mul(A[-1,:], F) # F^k

    for i in range(n_chunks-2, -1, -1):
        tmp = mul(res, val)
        res = tmp + C[i,:]

    # restore cached values
    res[0] = G_0_cache
    G[0]   = G_0_cache
    F[0]   = F_0_cache
    
    return res
    

def compose(G, F):
    out = np.zeros_like(G)
    q = out.shape[0]

    # cache first terms of F, G and then clear
    G_0_cache = G[0]
    F_0_cache = F[0]

    G[0] = 0
    F[0] = 0

    # Horner's method truncated to q
    out[0] = G[q - 1]
    for i in range(q - 2, -1, -1):
        out = np.convolve(out, F)[:q]
        out[0] += G[i]

    # restore cached values
    out[0] = G_0_cache
    G[0]   = G_0_cache
    F[0]   = F_0_cache

    return out


def compose_affine(G, F):
    if F.shape[0] <= 1:
        # handling composition with a constant F
        return G.copy()
    else:
        q = G.shape[0]

        # no need for Horner's method, utp composition uses only the 2nd and higher
        # coefficients, of which F has only 1 nonzero in this case
        return G * np.power(F[1], np.arange(0, q))


def deriv(F, k):
    q = F.shape[0]

    fact = util.fallingfactorial(k, np.arange(q))
    out = F * fact

    out = out[k:]

    return out

def integrate(F):
    out = np.zeros_like(F)
    q = len(F)
    k = np.arange(1, q).astype(float)
    out[1:] = (1/k) * F[0:q-1]
    return out
    
def div(x, y):
    return mul(x, inv(y))
    
# def log1p(F):
#     q = len(F)
#     F_prime = np.zeros_like(F)
#     F_prime[:q-1] = deriv(F, 1)
    
#     F_plus_one = F.copy();
#     F_plus_one[0] += 1

#     integrand = div( F_prime, F_plus_one )
#     out = integrate( integrand )
#     out[0] = np.log1p(F[0])

#     return out
    
# def log_(F):
#     F_minus_one = F.copy()
#     F_minus_one[0] -= 1.0
#     return log1p( F_minus_one )

# def log_(F):
#     q = len(F)
#     F_prime = np.zeros_like(F)
#     F_prime[:q-1] = deriv(F, 1)

#     out = integrate( div( F_prime, F) )
#     out[0] = np.log(F[0])
#     return out

def deriv_safe(F, k):
    q = F.shape[0]

    factln = util.logfallingfactorial(k, np.arange(q))

    # truncate both vectors
    factln = factln[k:]
    out = F[k:]

    if np.any(out == 0):
        None

    # cache negative signs before dividing by factorial
    outsigns = np.sign(out)
    out = np.log(np.abs(out))

    # normalize out * factln before performing the division
    out = out + factln
    logZ = np.max(out)

    if ~np.isfinite(logZ):
        None

    out -= logZ

    # return out to linear space, restore signs
    return logZ, outsigns * np.exp(out)


def mul(F, G):
    q = max(F.shape[0], G.shape[0])

    return np.convolve(F, G)[:q]

def mul2(F, G):
    H = np.zeros_like(F)
    for k in range(0, F.shape[0]):
        for i in range(0, k+1):
            H[k] += F[i] * G[k-i]

    return H

import scipy.signal
def mul3(F, G):
    q = max(F.shape[0], G.shape[0])

    return scipy.signal.fftconvolve(F, G)[:q]

def exp(F):
    out = np.empty_like(F)
    q   = out.shape[0]
    Ftilde = F[1:].copy()

    out[0] = np.exp(F[0])
    Ftilde *= np.arange(1, q)
    for i in range(1, q):
        out[i] = np.sum(out[:i][::-1]*Ftilde[:i], axis=0) / i

    return out


def log(F):
    out = np.empty_like(F)
    q   = out.shape[0]

    out[0] = np.log(F[0])

    for i in range(1, q):
        out[i] = (F[i]*i - np.sum(F[1:i][::-1]*out[1:i], axis=0))
        out[i] /= F[0]
    for i in range(1, q):
        out[i] /= i

    return out


#TODO: broken if F[0] < 0, need to handle
def pow(F, k):
    return exp(k * log(F))


def inv(F):
    out = np.zeros_like(F)
    q   = out.shape[0]

    out[0] = 1. / F[0]
    for i in range(1, q):
        out[i] = 1. / F[0] * (-np.sum(out[:i] * F[i:0:-1], axis=0))

    return out


def mean(F):
    return F[1]


def var(F):
    return (2 * F[2]) - np.power(F[1], 2) + F[1]


def normalize(F):
    Z = np.max(F)
    F /= Z
    logZ = np.log(Z)
    return logZ, F


def renormalize(F, old_logZ):
    Z = np.max(F)
    F /= Z
    logZ = old_logZ + np.log(Z)
    return logZ, F


def adjust_Z(F, old_logZ, new_logZ):
    if ~np.isfinite(new_logZ):
        None
    adjustment = np.exp(old_logZ - new_logZ)
    F *= adjustment
    return F
