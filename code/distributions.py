import numpy

#pgfs not implemented in scipy, should be compatible with UTPM

def poisson_pgf(s, lmbda):
    return numpy.exp(lmbda * (s - 1))


def bernoulli_pgf(s, p):
    return (1 - p) + (p * s)


def binomial_pgf(s, n, p):
    return numpy.power((1 - p) + (p * s), n)


def negbin_pgf(s, r, p):
    return numpy.power((1 - p) / (1 - (p * s)), r)
