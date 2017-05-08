import numpy as np
import util
import gdual


# construct a new ngdual tuple for e^logZ * <x/Z, dx>_q
def ngdual_new_x_dx(x, q):
    if isinstance(x, tuple):
        # take the first instance of x.utp as x
        utp = np.zeros(q, dtype=np.double)
        utp[0] = np.exp(x[0]) * x[1][0]
        if q > 1:
            utp[1] = 1.0

        # normalize the utp
        Z = np.max([utp[0], 1.0])
        logZ = np.log(Z)
        utp /= Z

        return logZ, utp
    elif isinstance(x, np.ndarray):
        # take the first instance of x[:] as x
        utp = np.zeros(q, dtype=np.double)
        utp[0] = x[0]
        if q > 1:
            utp[1] = 1.0

        # normalize the utp
        Z = np.max([x[0], 1.0])
        logZ = np.log(Z)
        utp /= Z

        return logZ, utp
    else:
        # construct a new utp array of length q
        utp = np.zeros(q, dtype=np.double)
        utp[0] = x
        if q > 1:
            utp[1] = 1.0

        # normalize the utp
        Z = np.max([x, 1.0])
        logZ = np.log(Z)
        utp /= Z

        return logZ, utp


def ngdual_new_c_dx(c, q):
    # construct a new utp array of length q
    utp = np.zeros(q, dtype=np.double)
    utp[0] = c

    # normalize the utp
    Z = np.max([c, 1.0])
    logZ = np.log(Z)
    utp /= Z

    return logZ, utp

# compose two ngduals as G(F)
# note: F will need to be "denormalized", may be unstable
def ngdual_compose(G, F):
    out_utp = np.zeros_like(G[1])
    q       = out_utp.shape[0]

    # unnormalize F
    F_utp = np.copy(F[1])
    F_utp *= np.exp(F[0])

    # G doesn't need to be unnormalized
    G_utp = np.copy(G[1])

    # cache first term of G and then clear first terms of F,G
    G_0_cache = G_utp[0]
    G_utp[0]  = 0
    F_utp[0]  = 0

    # Horner's method truncated to q
    out_utp[0] = G_utp[q - 1]
    for i in range(q - 2, -1, -1):
        out_utp = np.convolve(out_utp, F_utp)[:q]
        out_utp[0] += G_utp[i]

    # restore cached values
    out_utp[0] = G_0_cache

    # handle normalization
    out_Z    = np.max(out_utp)
    out_logZ = np.log(out_Z) + G[0]
    out_utp  /= out_Z

    return out_logZ, out_utp


# compose two ngduals as G(F) where |F| = 1 or 2
def ngdual_compose_affine(G, F):
    if F[1].shape[0] <= 1:
        # composition with a constant F
        return G[0], np.copy(G[1])
    else:
        out_utp = np.copy(G[1])
        q       = out_utp.shape[0]

        # unnormalize F
        F_utp = np.copy(F[1])
        F_utp *= np.exp(F[0])

        # no need for Horner's method, utp composition uses only the 2nd and higher
        # coefficients, of which F has only 1 nonzero in this case
        out_utp *= np.power(F_utp[1], np.arange(0, q))

        # handle normalization
        out_Z    = np.max(out_utp)
        out_logZ = np.log(out_Z) + G[0]
        out_utp  /= out_Z

        return out_logZ, out_utp


# compute d^k/dx^k <f, dx>_q
# note: if |F| = q, then |out| = q - k
def ngdual_deriv(F, k):
    out_utp = np.copy(F[1])
    q       = out_utp.shape[0]

    # compute the vector of falling factorial terms in log space
    factln = util.fallingfactorialln(k, np.arange(q))

    # drop the lowest order terms of both vectors
    factln  = factln[k:]
    out_utp = out_utp[k:]

    # cache negative signs before switching to logspace
    out_utp_signs = np.sign(out_utp)
    out_utp       = np.log(np.abs(out_utp))

    out_utp = out_utp + factln # mul in logspace

    # handle normalization
    out_logZ = np.max(out_utp)
    out_utp  -= out_logZ
    out_logZ += F[0]

    # restore utp to linear space
    out_utp = out_utp_signs * np.exp(out_utp)

    return out_logZ, out_utp


def ngdual_scalar_add(F, c):
    out_utp = np.copy(F[1])

    # TODO: if F_utp[0] + (c / np.exp(F[0])) > 1, then renormalization can be handled explicitly
    out_utp[0] = out_utp[0] + (c / np.exp(F[0]))

    out_Z = np.max(out_utp)
    out_logZ = np.log(out_Z) + F[0]
    out_utp /= out_Z

    return out_logZ, out_utp


# compute <f * g, dx>_q from <f, dx>_q and <g, dx>_q
def ngdual_mul(F, G):
    assert F[1].shape[0] == G[1].shape[0]

    F_utp = np.copy(F[1])
    G_utp = np.copy(G[1])
    q     = F_utp.shape[0]

    out_utp = np.convolve(F_utp, G_utp)[:q]

    # handle normalization
    out_Z    = np.max(out_utp)
    out_logZ = np.log(out_Z) + F[0] + G[0]
    out_utp  /= out_Z

    return out_logZ, out_utp


def ngdual_scalar_mul(F, c):
    # push c into logZ
    return (F[0] + np.log(c), np.copy(F[1]))


def ngdual_scalar_mul_log(F, logc):
    # push logc into logZ
    return (F[0] + logc, np.copy(F[1]))


# compute <exp(f), dx>_q from <f, dx>_q
# note: F will currently be unnormalized first
def ngdual_exp(F):
    out_utp = np.empty_like(F[1])
    q       = out_utp.shape[0]

    # unnormalize F
    F_utp = np.copy(F[1])
    F_utp *= np.exp(F[0])

    # compute the first term of exp(f)
    out_utp[0] = np.exp(F_utp[0])

    # copy the non scalar terms of F
    F_utp_tilde = np.copy(F_utp[1:])

    F_utp_tilde *= np.arange(1, q)
    for i in xrange(1, q):
        out_utp[i] = np.sum(out_utp[:i][::-1] * F_utp_tilde[:i], axis=0) / i

    # handle normalization
    out_Z    = np.max(out_utp)
    out_logZ = np.log(out_Z)
    out_utp  /= out_Z

    return out_logZ, out_utp


# compute <log(f), dx>_q from <f, dx>_q
# note: F will currently be unnormalized first
def ngdual_log(F):
    out_utp = np.empty_like(F[1])
    q       = out_utp.shape[0]

    # unnormalize F
    F_utp = np.copy(F[1])
    F_utp *= np.exp(F[0])

    # compute the first term of log(f)
    out_utp[0] = np.log(F_utp[0])

    for i in xrange(1, q):
        out_utp[i] = (F_utp[i] * i - np.sum(F_utp[1:i][::-1] * out_utp[1:i]))
        out_utp[i] /= F_utp[0]
    out_utp[1:q] /= np.arange(1, q)

    # handle normalization
    out_Z    = np.max(out_utp)
    out_logZ = np.log(out_Z)
    out_utp  /= out_Z

    return out_logZ, out_utp


# compute <f^k, dx>_q from <f, dx>_q
# note: F will currently be unnormalized first
#TODO: broken if F_utp[0] < 0, need special case
def ngdual_pow(F, k):
    # unnormalize F
    F_utp = np.copy(F[1])
    F_utp *= np.exp(F[0])

    out_utp = gdual.gdual_exp(k * gdual.gdual_log(F_utp))

    # handle normalization
    out_Z    = np.max(out_utp)
    out_logZ = np.log(out_Z)
    out_utp  /= out_Z

    return out_logZ, out_utp


#compute <1/f, dx>_q from <f, dx>_q
# note: f will currently be unnormalized first
def ngdual_reciprocal(F):
    out_utp = np.zeros_like(F[1])
    q       = out_utp.shape[0]

    # unnormalize F
    F_utp = np.copy(F[1])
    F_utp *= np.exp(F[0])

    out_utp[0] = 1. / F_utp[0]
    for i in xrange(1, q):
        out_utp[i] = 1. / F_utp[0] * (-np.sum(out_utp[:i] * F_utp[i:0:-1], axis=0))

    # handle normalization
    out_Z    = np.max(out_utp)
    out_logZ = np.log(out_Z)
    out_utp  /= out_Z

    return out_logZ, out_utp