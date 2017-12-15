import cygdual as cygd
import logsign as ls
import numpy as np


def poisson(s, theta):
    lmbda = theta[0]

    return cygd.exp(cygd.mul(cygd.add(s, ls.real2ls(-1.0)),
                             ls.real2ls(lmbda)))


def bernoulli(s, theta):
    p = theta[0]

    return cygd.add(cygd.mul(s, ls.real2ls(p)),
                    ls.real2ls(1 - p))


def binomial(s, theta):
    n, p = theta[:]

    return cygd.pow(cygd.add(cygd.mul(s, ls.real2ls(p)),
                             ls.real2ls(1 - p)),
                    n)


def negbin(s, theta):
    r, p = theta[:]

    return cygd.pow(cygd.mul(cygd.inv(cygd.add(cygd.mul(s, ls.real2ls(p - 1)),
                                               ls.real2ls(1))),
                             ls.real2ls(p)),
                    r)


def logarithmic(s, theta):
    p = theta[0]

    return cygd.mul(cygd.log(cygd.add(cygd.mul(s, ls.real2ls(-p)),
                                      ls.real2ls(1))),
                    ls.real2ls(1.0 / np.log(1 - p)))


def geometric(s, theta):
    p = theta[0]

    return cygd.mul(cygd.inv(cygd.add(cygd.mul(s, ls.real2ls(p - 1)),
                                      ls.real2ls(1))),
                    ls.real2ls(p))


def geometric2(s, theta):
    p = theta[0]

    return cygd.mul(cygd.mul(cygd.inv(cygd.add(cygd.mul(s, ls.real2ls(p - 1)),
                                               ls.real2ls(1))),
                             ls.real2ls(p)),
                    s)