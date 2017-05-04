import numpy as np
from scipy.special import gammaln

def poch(x, n):
    return np.exp(gammaln(x + n) - gammaln(x))

def fallingfactorial(k, i):
    return poch(i - k + 1, k)

def pochln(x, n):
    return gammaln(x + n) - gammaln(x)

def fallingfactorialln(k, i):
    return pochln(i - k + 1, k)