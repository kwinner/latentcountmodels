import numpy as np
import scipy.signal
import util
# import gdual


# construct a new ngdual tuple for e^logZ * <x/Z, dx>_q
def ngdual_new_x_dx(x, q):
    if isinstance(x, tuple):
        # take the first instance of x.utp as x
        utp = np.zeros(q, dtype=np.longdouble)
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
        utp = np.zeros(q, dtype=np.longdouble)
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
        utp = np.zeros(q, dtype=np.longdouble)
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
    utp = np.zeros(q, dtype=np.longdouble)
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
        # out_utp = np.convolve(out_utp, F_utp)[:q]
        out_utp = scipy.signal.convolve(out_utp, F_utp)[:q]
        out_utp[0] += G_utp[i]

    # restore cached values
    out_utp[0] = G_0_cache

    # handle normalization
    out_Z    = max([1, np.max(np.abs(out_utp))])
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
        out_Z    = max([1, np.max(np.abs(out_utp))])
        out_logZ = np.log(out_Z) + G[0]
        out_utp  /= out_Z

        return out_logZ, out_utp


# compute d^k/dx^k <f, dx>_q
# note: if |F| = q, then |out| = q - k
def ngdual_deriv(F, k):
    out_utp = np.copy(F[1])
    q       = out_utp.shape[0]

    # compute the vector of falling factorial terms in log space
    factln = util.logfallingfactorial(k, np.arange(q))

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

    out_Z    = max([1, np.max(np.abs(out_utp))])
    out_logZ = np.log(out_Z) + F[0]
    out_utp /= out_Z

    return out_logZ, out_utp


# compute <f * g, dx>_q from <f, dx>_q and <g, dx>_q
def ngdual_mul(F, G):
    assert F[1].shape[0] == G[1].shape[0]

    F_utp = np.copy(F[1])
    G_utp = np.copy(G[1])
    q     = F_utp.shape[0]

    # out_utp = np.convolve(F_utp, G_utp)[:q]
    out_utp = scipy.signal.convolve(F_utp, G_utp)[:q]

    # handle normalization
    out_Z    = max([1, np.max(np.abs(out_utp))])
    out_logZ = np.log(out_Z) + F[0] + G[0]
    out_utp  /= out_Z

    return out_logZ, out_utp


def ngdual_scalar_mul(F, c):
    if c > 0:
        # push c into logZ
        return (F[0] + np.log(c), np.copy(F[1]))
    elif c < 0:
        return (F[0] + np.log(-c), -1 * np.copy(F[1]))
    else: #c = 0
        return 0, np.zeros_like(F[1])


def ngdual_scalar_mul_log(F, logc):
    # push logc into logZ
    return (F[0] + logc, np.copy(F[1]))


# compute <exp(f), dx>_q from <f, dx>_q
# note: in safe version, F will first be unnormalized
def ngdual_exp_very_safe(F):
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
    for i in range(1, q):
        out_utp[i] = np.sum(out_utp[:i][::-1] * F_utp_tilde[:i], axis=0) / i

    # handle normalization
    out_Z    = max([1, np.max(np.abs(out_utp))])
    out_logZ = np.log(out_Z)
    out_utp  /= out_Z

    return out_logZ, out_utp


# compute <exp(f), dx>_q from <f, dx>_q
def ngdual_exp_safe(F):
    out_utp = np.empty_like(F[1])
    q       = out_utp.shape[0]

    # compute Z
    Z = np.exp(F[0])

    # compute the first term of exp(f)
    out_utp[0] = np.power(np.exp(F[1][0]), Z)

    # copy the non scalar terms of F
    F_utp_tilde = np.copy(F[1][1:])

    F_utp_tilde *= np.arange(1, q)
    for i in range(1, q):
        out_utp[i] = Z * np.sum(out_utp[:i][::-1] * F_utp_tilde[:i], axis=0) / i

    # handle normalization
    out_Z    = max([1, np.max(np.abs(out_utp))])
    out_logZ = np.log(out_Z)
    out_utp  /= out_Z

    return out_logZ, out_utp


# compute <exp(c) * exp(f - c), dx>_q from <f, dx>_q
def ngdual_exp(F):
    # take off c
    # correction = np.exp(F[0])
    correction = np.exp(F[0]) * F[1][0]
    F_prime = ngdual_scalar_add(F, -correction)

    out_utp = np.empty_like(F_prime[1])
    q       = out_utp.shape[0]

    # compute Z
    Z = np.exp(F_prime[0])

    # compute the first term of exp(f)
    out_utp[0] = np.power(np.exp(F_prime[1][0]), Z)

    # copy the non scalar terms of F
    F_utp_tilde = np.copy(F_prime[1][1:])

    F_utp_tilde *= np.arange(1, q)
    for i in range(1, q):
        out_utp[i] = Z * np.sum(out_utp[:i][::-1] * F_utp_tilde[:i], axis=0) / i

    # handle normalization
    out_Z    = max([1, np.max(np.abs(out_utp))])
    out_logZ = correction + np.log(out_Z)
    out_utp  /= out_Z

    return out_logZ, out_utp


# tried to reduce the magnitude of the intermediate values in the inner loop
# didn't significantly improve stability.
def ngdual_exp_experimental(F):
    # take off c
    # correction = np.exp(F[0])
    correction = np.exp(F[0]) * F[1][0]
    F_prime = ngdual_scalar_add(F, -correction)

    out_utp = np.empty_like(F_prime[1])
    q       = out_utp.shape[0]

    # compute Z
    Z = np.exp(F_prime[0])

    # compute the first term of exp(f)
    log_out_val = Z * F_prime[1][0]

    # for stability, use out_utp[0] = 1/Z instead of out_val for now
    out_utp[0] = 1.0 / Z

    # copy the non scalar terms of F
    F_utp_tilde = np.copy(F_prime[1][1:])

    F_utp_tilde *= np.arange(1, q)
    for i in range(1, q):
        out_utp[i] = Z * np.sum(out_utp[:i][::-1] * F_utp_tilde[:i], axis=0) / i

    # handle normalization
    out_Z    = max([1, np.max(np.abs(out_utp))])
    out_logZ = correction + np.log(out_Z) + F_prime[0] + log_out_val
    out_utp  /= out_Z

    return out_logZ, out_utp

# def ngdual_exp_logspace(F):
#     # take off c
#     # correction = np.exp(F[0])
#     # correction = np.exp(F[0]) * F[1][0]
#     # F_prime = ngdual_scalar_add(F, -correction)
#
#     q           = F[1].shape[0]
#     log_out_utp = np.empty_like(F[1])
#
#     # compute the first term of exp(f)
#     Z = np.exp(F[0])
#     log_out_val = Z * F[1][0]
#     log_out_utp[0] = 0 # use 0 for the first term and push the log_out_val into out_logZ below
#
#     # construct \tilde{F}
#     F_utp_tilde = np.copy(F[1][1:]) * np.arange(1, q)
#
#     for i in range(1, q):
#         # preslice the vector for this iteration
#         F_iter = F_utp_tilde[i-1::-1]
#         log_out_utp_iter = log_out_utp[:i]
#
#         # remove entries where F is zero (will become zero in logsumexp anyways, this avoids warnings)
#         log_out_utp_iter = log_out_utp_iter[F_iter != 0.]
#         F_iter           = F_iter[F_iter != 0.]
#
#         log_out_utp[i] = F[0] - np.log(i) + scipy.misc.logsumexp(log_out_utp_iter + np.log(F_iter))
#
#     # handle normalization
#     out_logZ = np.max([np.e, np.max(log_out_utp)])
#     log_out_utp -= out_logZ
#     out_logZ += log_out_val
#
#     # convert utp out of logspace
#     out_utp = np.exp(log_out_utp)
#
#     return out_logZ, out_utp


def ngdual_exp_logspace(F):
    q            = F[1].shape[0]
    H_utp_logabs = np.empty_like(F[1]) # the log absvalue of the output UTP
    H_utp_sign   = np.empty_like(F[1]) # the sign bit of each term of the output UTP

    # Ft = \tilde{F} = F[2:q] * (1:q-1)
    Ft_utp_logabs = np.log(np.abs(F[1][1:])) + np.log(np.arange(1, q))
    Ft_utp_sign   = np.sign(F[1])

    # compute the first term of the output utp
    H_utp_logabs[0] = np.exp(F[0]) * F[1][0]
    H_utp_sign[0]   = 1

    for i in range(1, q):
        # slice the vectors for this iteration
        Ft_utp_logabs_iter = Ft_utp_logabs[i-1::-1]
        Ft_utp_sign_iter   = Ft_utp_sign[i-1::-1]
        H_utp_logabs_iter  = H_utp_logabs[:i]
        H_utp_sign_iter    = H_utp_sign[:i]

        # G = H_iter * Ft.iter
        G_utp_logabs = H_utp_logabs_iter + Ft_utp_logabs_iter
        G_utp_sign   = H_utp_sign_iter   * Ft_utp_sign_iter

        # logsumexp with sign
        (mag, sgn) = scipy.special.logsumexp(G_utp_logabs, b = G_utp_sign, return_sign = True)
        H_utp_logabs[i] = F[0] - np.log(i) + mag
        H_utp_sign[i]   = sgn

    H_logZ = np.max([np.e, np.max(H_utp_logabs)])
    H_utp_logabs -= H_logZ * H_utp_sign

    # convert utp out of logspace
    H_utp = np.exp(H_utp_logabs) * H_utp_sign

    return H_logZ, H_utp


# compute <log(f), dx>_q from <f, dx>_q
# note: in safe version, F will first be unnormalized
def ngdual_log_safe(F):
    out_utp = np.empty_like(F[1])
    q       = out_utp.shape[0]

    # unnormalize F
    F_utp = np.copy(F[1])
    F_utp *= np.exp(F[0])

    # compute the first term of log(f)
    out_utp[0] = np.log(F_utp[0])

    for i in range(1, q):
        out_utp[i] = (F_utp[i] * i - np.sum(F_utp[1:i][::-1] * out_utp[1:i]))
        out_utp[i] /= F_utp[0]
    out_utp[1:q] /= np.arange(1, q)

    # handle normalization
    out_Z    = max([1, np.max(np.abs(out_utp))])
    out_logZ = np.log(out_Z)
    out_utp  /= out_Z

    return out_logZ, out_utp


# compute <log(f), dx>_q from <f, dx>_q
def ngdual_log(F):
    out_utp = np.empty_like(F[1])
    q       = out_utp.shape[0]

    # compute the first term of log(f)
    out_utp[0] = np.log(F[1][0])
    # correct for log Z of F
    out_utp[0] += F[0]

    for i in range(1, q):
        out_utp[i] = (F[1][i] * i - np.sum(F[1][1:i][::-1] * out_utp[1:i]))
        out_utp[i] /= F[1][0]
    out_utp[1:q] /= np.arange(1, q)

    # handle normalization
    out_Z    = max([1, np.max(np.abs(out_utp))])
    out_logZ = np.log(out_Z)
    out_utp  /= out_Z

    return out_logZ, out_utp


# compute <f^k, dx>_q from <f, dx>_q
# note: F will currently be unnormalized first
#TODO: broken if F_utp[0] < 0, need special case
def ngdual_pow_safe(F, k):
    # unnormalize F
    F_utp = np.copy(F[1])
    F_utp *= np.exp(F[0])

    out_utp = gdual.gdual_exp(k * gdual.gdual_log(F_utp))

    # handle normalization
    out_Z    = max([1, np.max(np.abs(out_utp))])
    out_logZ = np.log(out_Z)
    out_utp  /= out_Z

    return out_logZ, out_utp


def ngdual_pow(F, k):
    return ngdual_exp(ngdual_scalar_mul(ngdual_log(F), k))


#compute <1/f, dx>_q from <f, dx>_q
# note: f will currently be unnormalized first
def ngdual_reciprocal(F):
    out_utp = np.zeros_like(F[1])
    q       = out_utp.shape[0]

    # unnormalize F
    F_utp = np.copy(F[1])
    F_utp *= np.exp(F[0])

    out_utp[0] = 1. / F_utp[0]
    for i in range(1, q):
        out_utp[i] = 1. / F_utp[0] * (-np.sum(out_utp[:i] * F_utp[i:0:-1], axis=0))

    # handle normalization
    out_Z    = max([1, np.max(np.abs(out_utp))])
    out_logZ = np.log(out_Z)
    out_utp  /= out_Z

    return out_logZ, out_utp


if __name__ == "__main__":
    F = ngdual_new_x_dx(6,4)
    G = ngdual_new_x_dx(4,4)

    H = ngdual_compose_affine(F, G)

    print(np.exp(H[0]) * H[1])