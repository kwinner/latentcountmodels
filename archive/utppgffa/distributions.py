import numpy as np

# pgfs not implemented in scipy, should be compatible with UTPM


def poisson_pgf(s, theta):
    lmbda = theta
    return np.exp(lmbda * (s - 1))


def bernoulli_pgf(s, theta):
    p = theta
    return (1 - p) + (p * s)


def binomial_pgf(s, theta):
    n, p = theta[:]
    return np.power((1 - theta[1]) + (p * s), n)


def negbin_pgf(s, theta):
    r, p = theta[:]

    # out = s * (1 - p)
    #
    # # print '1. %s' % out
    # out = 1 - out
    #
    # # print '2. %s' % out
    # out = 1/ out
    # # print '3. %s' % out
    # out *= p
    # # print '4. %s' % out
    # out = np.power(out, r)
    # # print '5. %s' % out
    # return out
    return np.power(p / (1 - ((1 - p) * s)), r)


def logarithmic_pgf(s, theta):
    p = theta
    return np.log(1 - (p * s)) / np.log(1 - p)


# PGF for geometric with support 0, 1, 2, ...
def geometric_pgf(s, theta):
    p = theta
    return p / (1 - ((1 - p) * s))


# PGF for geometric with support 1, 2, ...
def geometric_pgf2(s, theta):
    p = theta
    return (p * s) / (1 - ((1 - p) * s))