import numpy as np
import scipy as sp
from algopy import UTPM

# i!/(i-k)!
def falling_factorial(k, i):
    return sp.special.poch(i - k + 1, k)


def new_utp(x, d):
    x_utp = UTPM(np.zeros((d, 1)))
    x_utp.data[0,0] = x
    if d > 1:
        x_utp.data[1,0] = 1
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