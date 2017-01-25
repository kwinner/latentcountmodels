import numpy

#pgfs not implemented in scipy, should be compatible with UTPM

def poisson_pgf(s, theta):
    lmbda = theta
    return numpy.exp(lmbda * (s - 1))


def bernoulli_pgf(s, theta):
    p = theta
    return (1 - p) + (p * s)


def binomial_pgf(s, theta):
    n, p = theta[:]
    return numpy.power((1 - theta[1]) + (p * s), n)


def negbin_pgf(s, theta):
    r, p = theta[:]
    return numpy.power(p / (1 - ((1 - p) * s)), r)


def logarithmic_pgf(s, theta):
    p = theta
    return numpy.log(1 - (p * s)) / numpy.log(1 - p)


def geometric_pgf(s, theta):
    p = theta
    return (p * s) / (1 - ((1 - p) * s))