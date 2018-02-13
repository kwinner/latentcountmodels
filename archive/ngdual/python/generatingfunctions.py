import numpy as np
import gdual_impl as gd
import ngdual
from copy import deepcopy

def poisson_pgf(s, theta):
    lmbda = theta[0]
    return np.exp(lmbda * (s - 1))


def poisson_gdual(s, theta):
    lmbda = theta[0]
    out = s.copy()
    out[0] = out[0] - 1
    return gd.exp(lmbda * out)


def poisson_ngdual(s, theta):
    lmbda = theta[0]
    return ngdual.ngdual_exp(ngdual.ngdual_scalar_mul(ngdual.ngdual_scalar_add(s, -1.0), lmbda))


def bernoulli_pgf(s, theta):
    p = theta[0]
    return (1 - p) + (p * s)


def bernoulli_gdual(s, theta):
    p = theta[0]
    out = s.copy()

    out *= p
    out[0] += (1 - p)
    return out


def bernoulli_ngdual(s, theta):
    p = theta[0]

    return ngdual.ngdual_scalar_add(ngdual.ngdual_scalar_mul(s, p),
                                    1 - p)


def binomial_pgf(s, theta):
    n, p = theta[:]
    return np.power((1 - theta[1]) + (p * s), n)


def binomial_gdual(s, theta):
    n, p = theta[:]
    out = s.copy()

    out *= p
    out[0] += (1 - p)
    return gd.pow(out, n)


def binomial_ngdual(s, theta):
    n, p = theta[:]

    return ngdual.ngdual_pow(ngdual.ngdual_scalar_add(ngdual.ngdual_scalar_mul(s, p),
                                                      1 - p),
                             n)


def negbin_pgf(s, theta):
    r, p = theta[:]

    return np.power(p / (1 - ((1 - p) * s)), r)


def negbin_gdual(s, theta):
    r, p = theta[:]
    out = s.copy()

    out *= (p - 1)
    out[0] += 1
    out = gd.inv(out)
    out = p * out
    out = gd.pow(out, r)

    return out


def negbin_ngdual(s, theta):
    r, p = theta[:]

    out = ngdual.ngdual_scalar_mul(s, p - 1)
    out = ngdual.ngdual_scalar_add(out, 1)
    out = ngdual.ngdual_reciprocal(out)
    out = ngdual.ngdual_scalar_mul(out, p)
    out = ngdual.ngdual_pow(out, r)

    return out

    # return ngdual.ngdual_pow(ngdual.ngdual_scalar_mul(ngdual.ngdual_reciprocal(ngdual.ngdual_scalar_add(ngdual.ngdual_scalar_mul(s, p - 1),
    #                                                                                                     1)),
    #                                                   p),
    #                          r)


def logarithmic_pgf(s, theta):
    p = theta[0]
    return np.log(1 - (p * s)) / np.log(1 - p)


def logarithmic_gdual(s, theta):
    p = theta[0]
    out = s.copy()

    out *= -p
    out[0] += 1
    out = gd.log(out)
    return (out / np.log(1 - p))


def logarithmic_ngdual(s, theta):
    p = theta[0]

    return ngdual.ngdual_scalar_mul(ngdual.ngdual_log(ngdual.ngdual_scalar_add(ngdual.ngdual_scalar_mul(s, -p),
                                                                               1)),
                                    1 / np.log(1 - p))


# PGF for geometric with support 0, 1, 2, ...
def geometric_pgf(s, theta):
    p = theta[0]
    return p / (1 - ((1 - p) * s))


def geometric_gdual(s, theta):
    p = theta[0]
    out = s.copy()

    out *= (p - 1)
    out[0] += 1
    return (p * gd.inv(out))


def geometric_ngdual(s, theta):
    p = theta[0]

    return ngdual.ngdual_scalar_mul(ngdual.ngdual_reciprocal(ngdual.ngdual_scalar_add(ngdual.ngdual_scalar_mul(s, p - 1),
                                                                                      1)),
                                    p)


# PGF for geometric with support 1, 2, ...
def geometric2_pgf(s, theta):
    p = theta[0]
    return (p * s) / (1 - ((1 - p) * s))


def geometric2_gdual(s, theta):
    p = theta[0]
    out = s.copy()

    out *= (p - 1)
    out[0] += 1
    out = gd.inv(out)
    return gd.mul(p * s, out)


def geometric2_ngdual(s, theta):
    p = theta[0]

    return ngdual.ngdual_mul(ngdual.ngdual_scalar_mul(ngdual.ngdual_reciprocal(ngdual.ngdual_scalar_add(ngdual.ngdual_scalar_mul(s, p - 1),
                                                                                                        1)),
                                                      p),
                             s)
