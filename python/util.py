import numpy as np
from scipy.special import gammaln

def poch(x, n):
    return np.exp(gammaln(x + n) - gammaln(x))


def fallingfactorial(k, i):
    return poch(i - k + 1, k)


def logpoch(x, n):
    """compute the log of the pochhammer symbol := \Gamma(x+n) / \Gamma(x)"""
    return gammaln(x + n) - gammaln(x)


def logfallingfactorial(k, i):
    """compute the log of the falling factorial := k(k-1)...(k-i+1)"""
    return logpoch(i - k + 1, k)