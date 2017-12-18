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

        H    = lsgf.poisson(F, theta)
        H_gd = gf.poisson_gdual(F_gd, theta)

        assert np.allclose(lsgd.lsgd2gd(H), H_gd)
    except AssertionError:
        print(traceback.format_exc())

        return False
    else:
        return True


def test_bernoulli(c, q):
    try:
        theta = np.array([0.6])

        F = lsgd.lsgdual_xdx(c, q)
        F = lsgd.add_scalar(F, 1.0)
        F = cygd.inv(F)
        F = cygd.exp(F)

        F_gd = lsgd.lsgd2gd(F)

        H    = lsgf.bernoulli(F, theta)
        H_gd = gf.bernoulli_gdual(F_gd, theta)

        assert np.allclose(lsgd.lsgd2gd(H), H_gd)
    except AssertionError:
        print(traceback.format_exc())

        return False
    else:
        return True


def test_binomial(c, q):
    try:
        theta = np.array([8.0, 0.6])

        F = lsgd.lsgdual_xdx(c, q)
        F = lsgd.add_scalar(F, 1.0)
        F = cygd.inv(F)
        F = cygd.exp(F)

        F_gd = lsgd.lsgd2gd(F)

        H    = lsgf.binomial(F, theta)
        H_gd = gf.binomial_gdual(F_gd, theta)

        assert np.allclose(lsgd.lsgd2gd(H), H_gd)
    except AssertionError:
        print(traceback.format_exc())

        return False
    else:
        return True


def test_negbin(c, q):
    try:
        theta = np.array([8.0, 0.6])

        F = lsgd.lsgdual_xdx(c, q)
        F = lsgd.add_scalar(F, 1.0)
        F = cygd.inv(F)
        F = cygd.exp(F)

        F_gd = lsgd.lsgd2gd(F)

        H    = lsgf.negbin(F, theta)
        H_gd = gf.negbin_gdual(F_gd, theta)

        assert np.allclose(lsgd.lsgd2gd(H), H_gd)
    except AssertionError:
        print(traceback.format_exc())

        return False
    else:
        return True


def test_logarithmic(c, q):
    try:
        theta = np.array([0.6])

        F = lsgd.lsgdual_xdx(c, q)
        F = lsgd.add_scalar(F, 1.0)
        F = cygd.inv(F)
        F = cygd.exp(F)

        F_gd = lsgd.lsgd2gd(F)

        H    = lsgf.logarithmic(F, theta)
        H_gd = gf.logarithmic_gdual(F_gd, theta)

        assert np.allclose(lsgd.lsgd2gd(H), H_gd)
    except AssertionError:
        print(traceback.format_exc())

        return False
    else:
        return True


def test_geometric(c, q):
    try:
        theta = np.array([8.0, 0.6])

        F = lsgd.lsgdual_xdx(c, q)
        F = lsgd.add_scalar(F, 1.0)
        F = cygd.inv(F)
        F = cygd.exp(F)

        F_gd = lsgd.lsgd2gd(F)

        H    = lsgf.geometric(F, theta)
        H_gd = gf.geometric_gdual(F_gd, theta)

        assert np.allclose(lsgd.lsgd2gd(H), H_gd)
    except AssertionError:
        print(traceback.format_exc())

        return False
    else:
        return True


def test_geometric2(c, q):
    try:
        theta = np.array([8.0, 0.6])

        F = lsgd.lsgdual_xdx(c, q)
        F = lsgd.add_scalar(F, 1.0)
        F = cygd.inv(F)
        F = cygd.exp(F)

        F_gd = lsgd.lsgd2gd(F)

        H    = lsgf.geometric2(F, theta)
        H_gd = gf.geometric2_gdual(F_gd, theta)

        assert np.allclose(lsgd.lsgd2gd(H), H_gd)
    except AssertionError:
        print(traceback.format_exc())

        return False
    else:
        return True


if __name__ == "__main__":
    print("test_poisson:     %s" % test_poisson(C, Q))
    print("test_bernoulli:   %s" % test_bernoulli(C, Q))
    print("test_binomial:    %s" % test_binomial(C, Q))
    print("test_negbin:      %s" % test_negbin(C, Q))
    print("test_logarithmic: %s" % test_logarithmic(C, Q))
    print("test_geometric:   %s" % test_geometric(C, Q))
    print("test_geometric2:  %s" % test_geometric2(C, Q))