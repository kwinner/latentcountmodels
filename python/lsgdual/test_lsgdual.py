import numpy as np
from lsgdual import *
import traceback

# TODO: tests currently don't explicitly test fail cases and may not be thorough on edge cases
# TODO: add tests with more varied derivative terms

def test_lsgdual_1dx():
    F = lsgdual_1dx(4)
    assert len(F) == 4
    assert np.array_equal(F['mag'], np.array([0, -np.inf, -np.inf, -np.inf]))
    assert np.array_equal(F['sgn'], np.array([1,       0,       0,       0]))
    assert np.issubdtype(F.dtype['mag'], float)
    assert np.issubdtype(F.dtype['sgn'], int)

    F = lsgdual_1dx(1)
    assert len(F) == 1
    assert np.array_equal(F['mag'], np.array([0]))
    assert np.array_equal(F['sgn'], np.array([1]))

    return True


def test_lsgdual_cdx():
    F = lsgdual_cdx(7, 4)
    assert len(F) == 4
    assert np.array_equal(F['mag'], np.array([np.log(7), -np.inf, -np.inf, -np.inf]))
    assert np.array_equal(F['sgn'], np.array([        1,       0,       0,       0]))
    assert np.issubdtype(F.dtype['mag'], float)
    assert np.issubdtype(F.dtype['sgn'], int)

    F = lsgdual_cdx(-5, 3)
    assert len(F) == 3
    assert np.array_equal(F['mag'], np.array([np.log(5), -np.inf, -np.inf]))
    assert np.array_equal(F['sgn'], np.array([       -1,       0,       0]))

    F = lsgdual_cdx(0, 2)
    assert len(F) == 2
    assert np.array_equal(F['mag'], np.array([-np.inf, -np.inf]))
    assert np.array_equal(F['sgn'], np.array([      0,       0]))

    F = lsgdual_cdx(3, 1)
    assert len(F) == 1
    assert np.array_equal(F['mag'], np.array([np.log(3)]))
    assert np.array_equal(F['sgn'], np.array([1]))

    return True


def test_lsgdual_xdx():
    F = lsgdual_xdx(6, 4)
    assert len(F) == 4
    assert np.array_equal(F['mag'], np.array([np.log(6), 0, -np.inf, -np.inf]))
    assert np.array_equal(F['sgn'], np.array([        1, 1,       0,       0]))
    assert np.issubdtype(F.dtype['mag'], float)
    assert np.issubdtype(F.dtype['sgn'], int)

    F = lsgdual_xdx(-4, 3)
    assert len(F) == 3
    assert np.array_equal(F['mag'], np.array([np.log(4), 0, -np.inf]))
    assert np.array_equal(F['sgn'], np.array([       -1, 1,       0]))

    F = lsgdual_xdx(0, 2)
    assert len(F) == 2
    assert np.array_equal(F['mag'], np.array([-np.inf, 0]))
    assert np.array_equal(F['sgn'], np.array([      0, 1]))

    F = lsgdual_xdx(-3, 1)
    assert len(F) == 1
    assert np.array_equal(F['mag'], np.array([np.log(3)]))
    assert np.array_equal(F['sgn'], np.array([-1]))

    return True


def test_islsgdual():
    F = lsgdual_1dx(4)
    assert islsgdual(F)

    F = lsgdual_cdx(6, 1)
    assert islsgdual(F)

    F = lsgdual_xdx(-8, 3)
    assert islsgdual(F)

    F = np.array([1, 2, 3])
    assert not islsgdual(F)

    return True


def test_lsgd2gd():
    F    = lsgdual_xdx(7, 4)
    F_gd = lsgd2gd(F)
    assert np.allclose(F_gd, np.array([7.0, 1.0, 0.0, 0.0]))

    F    = lsgdual_xdx(3, 1)
    F_gd = lsgd2gd(F)
    assert np.allclose(F_gd, np.array([3.0]))

    F    = lsgdual_xdx(-4, 2)
    F_gd = lsgd2gd(F)
    assert np.allclose(F_gd, np.array([-4.0, 1.0]))

    return True


def test_lsgd2ngd():
    F    = lsgdual_xdx(7, 4)
    F_gd = lsgd2ngd(F)
    assert F_gd[0] == np.log(7)
    assert np.allclose(F_gd[1], np.array([1.0, 1.0/7.0, 0.0, 0.0]))

    F    = lsgdual_xdx(3, 1)
    F_gd = lsgd2ngd(F)
    assert F_gd[0] == np.log(3)
    assert np.allclose(F_gd[1], np.array([1.0]))

    F    = lsgdual_xdx(-4, 2)
    F_gd = lsgd2ngd(F)
    assert F_gd[0] == np.log(4)
    assert np.allclose(F_gd[1], np.array([-1.0, 1.0/4.0]))

    return True


def test_gd2lsgd():
    F      = np.array([7.0, 1.0, 0.0, 0.0])
    F_lsgd = gd2lsgd(F)
    assert np.allclose(F_lsgd['mag'], np.array([np.log(7), 0, -np.inf, -np.inf]))
    assert np.allclose(F_lsgd['sgn'], np.array([        1, 1,       0,       0]))

    F      = np.array([3.0])
    F_lsgd = gd2lsgd(F)
    assert np.allclose(F_lsgd['mag'], np.array([np.log(3)]))
    assert np.allclose(F_lsgd['sgn'], np.array([        1]))

    F      = np.array([-4.0, 1.0])
    F_lsgd = gd2lsgd(F)
    assert np.allclose(F_lsgd['mag'], np.array([np.log(4), 0]))
    assert np.allclose(F_lsgd['sgn'], np.array([       -1, 1]))

    return True


def test_ngd2lsgd():
    F      = (np.log(7), np.array([1.0, 1.0/7.0, 0.0, 0.0]))
    F_lsgd = ngd2lsgd(F)
    assert np.allclose(F_lsgd['mag'], np.array([np.log(7), 0, -np.inf, -np.inf]))
    assert np.allclose(F_lsgd['sgn'], np.array([        1, 1,       0,       0]))

    F      = (np.log(3), np.array([1.0]))
    F_lsgd = ngd2lsgd(F)
    assert np.allclose(F_lsgd['mag'], np.array([np.log(3)]))
    assert np.allclose(F_lsgd['sgn'], np.array([        1]))

    F      = (np.log(4), np.array([-1.0, 1.0/4.0]))
    F_lsgd = ngd2lsgd(F)
    assert np.allclose(F_lsgd['mag'], np.array([np.log(4), 0]))
    assert np.allclose(F_lsgd['sgn'], np.array([       -1, 1]))

    return True

def test_add_scalar():
    F = lsgdual_xdx(5, 2)
    c = 2
    H = add_scalar(F, c)
    assert np.allclose(   H['mag'], np.array([np.log(7), 0]))
    assert np.array_equal(H['sgn'], np.array([1,         1]))

    F = lsgdual_xdx(5, 2)
    c = -7
    H = add_scalar(F, c)
    assert np.allclose(   H['mag'], np.array([np.log(2), 0]))
    assert np.array_equal(H['sgn'], np.array([-1,        1]))

    F = lsgdual_xdx(-5, 2)
    c = 2
    H = add_scalar(F, c)
    assert np.allclose(   H['mag'], np.array([np.log(3), 0]))
    assert np.array_equal(H['sgn'], np.array([-1,        1]))

    F = lsgdual_xdx(5, 2)
    c = 0
    H = add_scalar(F, c)
    assert np.allclose(   H['mag'], np.array([np.log(5), 0]))
    assert np.array_equal(H['sgn'], np.array([1,         1]))

    F = lsgdual_xdx(0, 2)
    c = 5
    H = add_scalar(F, c)
    assert np.allclose(   H['mag'], np.array([np.log(5), 0]))
    assert np.array_equal(H['sgn'], np.array([1,         1]))

    F = lsgdual_xdx(5, 2)
    c = -5
    H = add_scalar(F, c)
    assert np.allclose(   H['mag'], np.array([-np.inf, 0]))
    assert np.array_equal(H['sgn'], np.array([0,       1]))

    return True


def test_add():
    F = lsgdual_xdx(5, 2)
    G = lsgdual_xdx(4, 2)
    H = add(F, G)
    assert np.allclose(   H['mag'], np.array([np.log(9), np.log(2)]))
    assert np.array_equal(H['sgn'], np.array([1,         1]))

    F = lsgdual_xdx(5, 2)
    G = lsgdual_xdx(-7, 2)
    H = add(F, G)
    assert np.allclose(   H['mag'], np.array([np.log(2), np.log(2)]))
    assert np.array_equal(H['sgn'], np.array([-1,        1]))

    F = lsgdual_xdx(-5, 2)
    G = lsgdual_xdx(2, 2)
    H = add(F, G)
    assert np.allclose(   H['mag'], np.array([np.log(3), np.log(2)]))
    assert np.array_equal(H['sgn'], np.array([-1,        1]))

    F = lsgdual_xdx(-5, 2)
    G = lsgdual_cdx(0, 2)
    H = add(F, G)
    assert np.allclose(H['mag'], np.array([np.log(5), 0]))
    assert np.array_equal(H['sgn'], np.array([-1, 1]))

    F = lsgdual_xdx(-5, 2)
    G = lsgdual_cdx(-3, 2)
    H = add(F, G)
    assert np.allclose(H['mag'], np.array([np.log(8), 0]))
    assert np.array_equal(H['sgn'], np.array([-1, 1]))

    F = lsgdual_xdx(5, 2)
    G = lsgdual_cdx(-5, 2)
    H = add(F, G)
    assert np.allclose(   H['mag'], np.array([-np.inf, 0]))
    assert np.array_equal(H['sgn'], np.array([0,       1]))

    F = lsgdual_xdx(3, 2)
    G = lsgdual_xdx(-2, 2)
    G[1]['sgn'] = -1 # flip to -x
    H = add(F, G)
    assert np.allclose(   H['mag'], np.array([np.log(1), -np.inf]))
    assert np.array_equal(H['sgn'], np.array([1,         0]))

    return True


def test_mul_scalar():
    F = lsgdual_xdx(5, 2)
    c = 2
    H = mul_scalar(F, c)
    assert np.allclose(   H['mag'], np.array([np.log(10), 0]))
    assert np.array_equal(H['sgn'], np.array([1,         1]))

    F = lsgdual_xdx(5, 2)
    c = -7
    H = mul_scalar(F, c)
    assert np.allclose(   H['mag'], np.array([np.log(35), 0]))
    assert np.array_equal(H['sgn'], np.array([-1,        1]))

    F = lsgdual_xdx(-5, 2)
    c = 2
    H = mul_scalar(F, c)
    assert np.allclose(   H['mag'], np.array([np.log(10), 0]))
    assert np.array_equal(H['sgn'], np.array([-1,        1]))

    F = lsgdual_xdx(5, 2)
    c = 0
    H = mul_scalar(F, c)
    assert np.allclose(   H['mag'], np.array([-np.inf, 0]))
    assert np.array_equal(H['sgn'], np.array([0,         1]))

    F = lsgdual_xdx(0, 2)
    c = 5
    H = mul_scalar(F, c)
    assert np.allclose(   H['mag'], np.array([-np.inf, 0]))
    assert np.array_equal(H['sgn'], np.array([0,         1]))

    return True


def test_mul_fast():
    try:
        F = lsgdual_xdx(5, 2)
        G = lsgdual_xdx(4, 2)
        H = mul_fast(F, G)
        assert np.allclose(   H['mag'], np.array([np.log(20), np.log(9)]))
        assert np.array_equal(H['sgn'], np.array([1,          1]))

        F = lsgdual_xdx(5, 2)
        G = lsgdual_xdx(-7, 2)
        H = mul_fast(F, G)
        assert np.allclose(   H['mag'], np.array([np.log(35), np.log(2)]))
        assert np.array_equal(H['sgn'], np.array([-1,         -1]))

        F = lsgdual_xdx(-5, 2)
        G = lsgdual_xdx(2, 2)
        H = mul_fast(F, G)
        assert np.allclose(   H['mag'], np.array([np.log(10), np.log(3)]))
        assert np.array_equal(H['sgn'], np.array([-1,         -1]))

        F = lsgdual_xdx(-5, 2)
        G = lsgdual_cdx(-3, 2)
        H = mul_fast(F, G)
        assert np.allclose(   H['mag'], np.array([np.log(15), 3]))
        assert np.array_equal(H['sgn'], np.array([1, -1]))

        F = lsgdual_xdx(3, 2)
        G = lsgdual_xdx(-2, 2)
        G[1]['sgn'] = -1 # flip to -x
        H = mul_fast(F, G)
        assert np.allclose(   H['mag'], np.array([np.log(6), 0]))
        assert np.array_equal(H['sgn'], np.array([-1,        -1]))

        F = lsgdual_xdx(-5, 2)
        G = lsgdual_cdx(0, 2)
        H = mul_fast(F, G)
        assert np.allclose(H['mag'], np.array([-np.inf, -np.inf]))
        assert np.array_equal(H['sgn'], np.array([0, 0]))
    except AssertionError:
        print(traceback.format_exc())

        return False
    else:
        return True


def test_compose():
    return True

def test_compose_affine():
    return True

def test_deriv():
    return True

def test_exp():
    return True

def test_log():
    return True

def test_pow():
    return True

def test_reciprocal():
    return True


if __name__ == "__main__":
    print("test_lsgdual_1dx: %s" % test_lsgdual_1dx())
    print("test_lsgdual_cdx: %s" % test_lsgdual_cdx())
    print("test_lsgdual_xdx: %s" % test_lsgdual_xdx())
    print("test_islsgdual: %s"   % test_islsgdual())
    print("test_lsgd2gd: %s"     % test_lsgd2gd())
    print("test_lsgd2ngd: %s"    % test_lsgd2ngd())
    print("test_gd2lsgd: %s"     % test_gd2lsgd())
    print("test_ngd2lsgd: %s"    % test_ngd2lsgd())
    print("test_add_scalar: %s"  % test_add_scalar())
    print("test_add: %s"         % test_add())
    print("test_mul_scalar: %s"  % test_mul_scalar())
    print("test_mul_fast: %s"    % test_mul_fast())