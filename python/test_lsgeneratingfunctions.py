import gdual as gd
import generatingfunctions as gf
import lsgdual as lsgd
import cygdual as cygd
import logsign as ls
import lsgeneratingfunctions as lsgf
import numpy as np
import traceback

C = 5
Q = 4

def test_poisson(c, q):
    try:
        theta = np.array([3.0])

        F = lsgd.lsgdual_xdx(c, q)
        F = lsgd.add_scalar(F, 1.0)
        F = cygd.inv(F)
        F = cygd.exp(F)

        F_gd = lsgd.lsgd2gd(F)

        print(lsgd.lsgd2gd(lsgf.poisson(F, theta)))
        print(gf.poisson_gdual(F_gd, theta))
    except AssertionError:
        print(traceback.format_exc())

        return False
    else:
        return True


def test_bernoulli(c, q):
    try:
        print()
    except AssertionError:
        print(traceback.format_exc())

        return False
    else:
        return True


def test_binomial(c, q):
    try:
        print()
    except AssertionError:
        print(traceback.format_exc())

        return False
    else:
        return True


def test_negbin(c, q):
    try:
        print()
    except AssertionError:
        print(traceback.format_exc())

        return False
    else:
        return True


def test_logarithmic(c, q):
    try:
        print()
    except AssertionError:
        print(traceback.format_exc())

        return False
    else:
        return True


def test_geometric(c, q):
    try:
        print()
    except AssertionError:
        print(traceback.format_exc())

        return False
    else:
        return True


def test_geometric2(c, q):
    try:
        print()
    except AssertionError:
        print(traceback.format_exc())

        return False
    else:
        return True


if __name__ == "__main__":
    print("test_poisson: %s" % test_poisson(C, Q))