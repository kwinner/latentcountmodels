import numpy
import scipy


# i!/(i-k)!
def falling_factorial(k, i):
    return scipy.special.poch(i - k + 1, k)


def UTP_deriv(x, k):
    n = len(x)
    fact = falling_factorial(k, numpy.arange(n))
    x = x * fact

    # shift by k places
    y = numpy.zeros(n - k)
    y[:n - k] = x[k:]

    return y