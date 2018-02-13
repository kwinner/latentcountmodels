import cygdual as cygd
import lsgdual as lsgd
import logsign as ls
import numpy as np


def poisson(s, theta):
    q = s.shape[0]
    lmbda = theta[0]

    # s.add(-1).mul(lmbda).exp()
    # cygd.exp(lambda*(s-1))

    return cygd.exp(cygd.mul(cygd.add(s, lsgd.lsgdual_cdx(-1.0, q)),
                             lsgd.lsgdual_cdx(lmbda, q)))


def bernoulli(s, theta):
    q = s.shape[0]
    p = theta[0]

    return cygd.add(cygd.mul(s, lsgd.lsgdual_cdx(p, q)),
                    lsgd.lsgdual_cdx(1 - p, q))


def binomial(s, theta):
    q = s.shape[0]
    n, p = theta[:]

    return cygd.pow(cygd.add(cygd.mul(s, lsgd.lsgdual_cdx(p, q)),
                             lsgd.lsgdual_cdx(1 - p, q)),
                    n)


def negbin(s, theta):
    q = s.shape[0]
    r, p = theta[:]

    return cygd.pow(cygd.mul(cygd.inv(cygd.add(cygd.mul(s, lsgd.lsgdual_cdx(p - 1, q)),
                                               lsgd.lsgdual_1dx(q))),
                             lsgd.lsgdual_cdx(p, q)),
                    r)


def logarithmic(s, theta):
    q = s.shape[0]
    p = theta[0]

    return cygd.mul(cygd.log(cygd.add(cygd.mul(s, lsgd.lsgdual_cdx(-p, q)),
                                      lsgd.lsgdual_1dx(q))),
                    lsgd.lsgdual_cdx(1.0 / np.log(1 - p), q))


def geometric(s, theta):
    q = s.shape[0]
    p = theta[0]

    return cygd.mul(cygd.inv(cygd.add(cygd.mul(s, lsgd.lsgdual_cdx(p - 1, q)),
                                      lsgd.lsgdual_1dx(q))),
                    lsgd.lsgdual_cdx(p, q))


def geometric2(s, theta):
    q = s.shape[0]
    p = theta[0]

    return cygd.mul(cygd.mul(cygd.inv(cygd.add(cygd.mul(s, lsgd.lsgdual_cdx(p - 1, q)),
                                               lsgd.lsgdual_1dx(q))),
                             lsgd.lsgdual_cdx(p, q)),
                    s)
