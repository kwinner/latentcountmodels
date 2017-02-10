import numpy as np
import scipy as sp
from algopy import UTPM
from UTPPGF_cython import utp_compose_cython

# i!/(i-k)!
def falling_factorial(k, i):
    return sp.special.poch(i - k + 1, k)


def new_utp(x, d):
    x_utp = UTPM(np.zeros((d, 1)))
    x_utp.data[0,0] = x
    if d > 1:
        x_utp.data[1,0] = 1
    return x_utp


def new_utp_vec(x, d):
    x_utp = np.zeros(d)
    x_utp[0] = x
    if d > 1:
        x_utp[1] = 1
    return x_utp


def utp_deriv(x, k):
    dx = x.data.ravel()
    n = len(dx)
    fact = falling_factorial(k, np.arange(n))
    dx = dx * fact

    # shift by k places
    dy = np.zeros(n - k)
    dy[:n - k] = dx[k:]
    y = UTPM( dy.reshape(n-k, 1))
    return y


def utp_deriv_vec(x, k):
    dx = x.ravel()
    n = len(dx)
    fact = falling_factorial(k, np.arange(n))
    dx = dx * fact

    #shift by k places
    dy = np.zeros(n - k)
    dy[:n - k] = dx[k:]
    return dy


def utppgf_mean(F):
    return F.data[1, 0]


def utppgf_var(F):
    return (2 * F.data[2, 0]) - np.power(F.data[1, 0], 2) + F.data[1, 0]


def utp_compose(G, F):
    # require G and F to have same dimension, and always output something of same size
    assert G.data.shape[0] == F.data.shape[0]

    g = G.data.copy().squeeze(axis=(1,))
    f = F.data.copy().squeeze(axis=(1,))
    g_scalar = g[0]
    g[0], f[0] = 0, 0

    d = len(g)

    # Horner's method truncated to d
    res = np.array([g[d - 1]])
    for i in range(d - 2, -1, -1):
        res = np.convolve(res, f)[:d]
        res[0] += g[i]

    res[0] = g_scalar

    H = UTPM(res.reshape(-1, 1))
    return H


def utp_compose_utpm(G, F):
    assert G.data.shape[0] == F.data.shape[0]

    # return UTPM(np.asarray(utp_compose_vec(G.data[:,0], F.data[:,0])).reshape(-1,1))
    return UTPM(utp_compose_cython(G.data[:,0], F.data[:,0]).reshape(-1,1))


def utp_compose_vec(G, F):
    assert G.shape[0] == F.shape[0]

    return utp_compose_cython(G, F)


def utp_mul_vec(F, G):
    d = max(F.shape[0], G.shape[0])

    return np.convolve(F, G)[:d]


def utp_exp_vec(F):
    d = F.shape[0]

    out = np.empty_like(F)

    out[0] = np.exp(F[0])
    Ftilde = F[1:].copy()
    for i in range(1, d):
        Ftilde[i-1] *= i
    for i in range(1, d):
        out[i] = np.sum(out[:i][::-1]*Ftilde[:i], axis=0)/i

    return out

def utp_log_vec(F):
    d = F.shape[0]

    out = np.empty_like(F)

    out[0] = np.log(F[0])

    for i in range(1,d):
        out[i] = (F[i]*i - np.sum(F[1:i][::-1]*out[1:i],axis=0))
        out[i] /= F[0]
    for i in range(1,d):
        out[i] /= i

    return out


def utp_pow_vec(F, k):
    d = F.shape[0]

    return utp_exp_vec(k * utp_log_vec(F))


def utp_pow_utpm(F, k):
    # return np.pow(F,k)

    return np.exp(k * np.log(F))


# wrapper to call pgf(s, theta) with s as a temporary utpm
def lift_generating_function_utpm(pgf, s, theta):
    s_utpm = UTPM(s.reshape(-1,1))
    out_utpm = pgf(s_utpm, theta)

    return out_utpm.data[:,0]

def compose_poly_horner_special(f, g):
    K = len(f)
    if K == 1:  # special case if f is constant
        return f
    res = f[K - 1]
    for i in range(K - 2, -1, -1):
        res = np.append(res * g[0], 0) + np.append(f[i], res * g[1])

    return res


def compose_poly_horner(f, g):
    K = len(f)
    if K == 1:  # special case if f is constant
        return f
    res = f[K - 1]
    for i in range(K - 2, -1, -1):
        res = np.convolve(res, g)
        res[0] = res[0] + f[i]
    return res


def poly_der(f):
    fprime = f[1:]  # discard constant term
    fprime = fprime * np.arange(1, len(fprime)+1)

    return fprime


if __name__ == "__main__":
    f = np.array([1, 2, 3])
    g = np.array([5, 8, 16])
    h = compose_poly_horner(f, g)
    print h

    f = np.array([5, 8, 16])
    g = np.array([9, 11])
    h = compose_poly_horner_special(f, g)
    print h