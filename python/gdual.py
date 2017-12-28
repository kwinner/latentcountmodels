import numpy as np
import util as util

# todo function descriptions
# todo assert shape match
def gdual_new(x, q):
    # todo this is really a special case of new gdual for new RVs
    out = np.zeros(q, dtype=np.double)

    out[0] = x
    if q > 1:
        out[1] = 1

    return out


def gdual_compose(G, F):
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


def gdual_compose_affine(G, F):
    if F.shape[0] <= 1:
        # handling composition with a constant F
        return G.copy()
    else:
        q = G.shape[0]

        # no need for Horner's method, utp composition uses only the 2nd and higher
        # coefficients, of which F has only 1 nonzero in this case
        return G * np.power(F[1], np.arange(0, q))


def gdual_deriv(F, k):
    q = F.shape[0]

    fact = util.fallingfactorial(k, np.arange(q))
    out = F * fact

    out = out[k:]

    return out


def gdual_deriv_safe(F, k):
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


def gdual_mul(F, G):
    q = max(F.shape[0], G.shape[0])

    return np.convolve(F, G)[:q]


def gdual_mul2(F, G):
    H = np.zeros_like(F)
    for k in range(0, F.shape[0]):
        for i in range(0, k+1):
            H[k] += F[i] * G[k-i]

    return H


def gdual_exp(F):
    out = np.empty_like(F)
    q   = out.shape[0]
    Ftilde = F[1:].copy()

    out[0] = np.exp(F[0])
    Ftilde *= np.arange(1, q)
    for i in range(1, q):
        out[i] = np.sum(out[:i][::-1]*Ftilde[:i], axis=0) / i

    return out


def gdual_log(F):
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
def gdual_pow(F, k):
    return gdual_exp(k * gdual_log(F))


def gdual_reciprocal(F):
    out = np.zeros_like(F)
    q   = out.shape[0]

    out[0] = 1. / F[0]
    for i in range(1, q):
        out[i] = 1. / F[0] * (-np.sum(out[:i] * F[i:0:-1], axis=0))

    return out


def gdual_mean(F):
    return F[1]


def gdual_var(F):
    return (2 * F[2]) - np.power(F[1], 2) + F[1]


def gdual_normalize(F):
    Z = np.max(F)
    F /= Z
    logZ = np.log(Z)
    return logZ, F


def gdual_renormalize(F, old_logZ):
    Z = np.max(F)
    F /= Z
    logZ = old_logZ + np.log(Z)
    return logZ, F


def gdual_adjust_Z(F, old_logZ, new_logZ):
    if ~np.isfinite(new_logZ):
        None
    adjustment = np.exp(old_logZ - new_logZ)
    F *= adjustment
    return F