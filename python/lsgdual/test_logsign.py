import numpy as np
from scipy.special import logsumexp
import traceback

from logsign import *

def test_test_ls():
    return True


def test_real2ls():
    try:
        x = real2ls(4.0)
        assert x['mag'] == np.log(4)
        assert x['sgn'] == 1

        x = real2ls(-3)
        assert x['mag'] == np.log(3)
        assert x['sgn'] == -1

        x = real2ls(0.0)
        assert x['mag'] == -np.inf
        assert x['sgn'] == 0
    except AssertionError:
        print(traceback.format_exc())

        return False
    else:
        return True


def test_ls2real():
    try:
        x = real2ls(4.0)
        z = ls2real(x)
        assert np.isclose(z, 4.0)

        x = real2ls(-3)
        z = ls2real(x)
        assert np.isclose(z, -3)

        x = real2ls(0.0)
        z = ls2real(x)
        assert np.isclose(z, 0.0)
    except AssertionError:
        print(traceback.format_exc())

        return False
    else:
        return True


def test_isls():
    return True


def test_ls_add():
    try:
        print()
    except AssertionError:
        print(traceback.format_exc())

        return False
    else:
        return True


def test_ls_sum():
    try:
        print()
    except AssertionError:
        print(traceback.format_exc())

        return False
    else:
        return True


def test_ls_mul(x, y):
    """multiply two numbers (or vectors of numbers) in log-sign space"""
    assert isls(x) and isls(y)
    assert x.shape == y.shape

    z = np.empty_like(x)
    z['mag'] = x['mag'] + y['mag']
    z['sgn'] = x['sgn'] * y['sgn']

    return z


def test_ls_exp(x):
    """exponentiate a number (or vector of numbers) in log-sign space"""
    assert isls(x)

    z = np.empty_like(x)
    z['mag'] = x['sgn'] * np.exp(x['mag'])
    z['sgn'] = 1

    return z


def test_ls_log(x):
    """compute the log of a number (or vector) in log-sign space"""
    assert isls(x)

    z = np.empty_like(x)
    if x['sgn'] == 1:
        # x is already the desired quantity in linear space, we just need to go one layer deeper
        z['mag'] = np.log(x['mag'])
        z['sgn'] = np.sign(x['mag'])
    elif x['sgn'] == 0:
        # log(0) = -inf, but in ls-space
        z['mag'] = np.inf
        z['sgn'] = -1
    else:  # x['sgn'] == -1:
        # log(x \in Z^-) = nan
        # TODO: warning?
        z['mag'] = np.nan
        z['sgn'] = np.nan

    return z


if __name__ == "__main__":
    print("test_real2ls: %s" % test_real2ls())
    print("test_ls2real: %s" % test_ls2real())