import numpy as np
from scipy.special import logsumexp


LS_DTYPE = [('mag', '<f16'), ('sgn', 'i1')]


def ls(shape = 0):
    """instantiate an empty log-sign array with given shape"""
    return np.empty(shape, dtype = LS_DTYPE)


def real2ls(x):
    """convert a number in linear space to log-sign space"""
    z = np.array(x)
    z = ls(shape = z.shape)
    with np.errstate(divide='ignore'):
        z['mag'] = np.log(np.abs(x))
        z['sgn'] = np.sign(x)

    return z


def ls2real(x):
    """convert a number in log-sign space to linear space"""
    return x['sgn'] * np.exp(x['mag'])


def isls(x):
    """test that x is a number in log-sign space"""
    return isinstance(x, np.ndarray)            and \
           x.dtype.names == ('mag', 'sgn')      and \
           np.issubdtype(x.dtype['mag'], float) and \
           np.issubdtype(x.dtype['sgn'], int)


def ls_add(x, y):
    """add two numbers in log-sign space"""
    assert isls(x) and isls(y)

    return logsumexp(x['mag'] + y['mag'],
                     b = x['sgn'] * y['sgn'],
                     return_sign = True)


def ls_sum(x, axis=None):
    """sum all values in the vector of numbers in ls-space along some axis"""
    mag, sgn = logsumexp(x['mag'], b = x['sgn'], axis = axis, return_sign = True)

    z = ls(mag.shape)
    z['mag'] = mag
    z['sgn'] = sgn

    return z


def ls_mul(x, y):
    """multiply two numbers (or vectors of numbers) in log-sign space"""
    assert isls(x) and isls(y)
    assert x.shape == y.shape

    z = np.empty_like(x)
    z['mag'] = x['mag'] + y['mag']
    z['sgn'] = x['sgn'] * y['sgn']

    return z


def ls_exp(x):
    """exponentiate a number (or vector of numbers) in log-sign space"""
    assert isls(x)

    z = np.empty_like(x)
    z['mag'] = x['sgn'] * np.exp(x['mag'])
    z['sgn'] = 1

    return z


def ls_log(x):
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
    else: #x['sgn'] == -1:
        # log(x \in Z^-) = nan
        # TODO: warning?
        z['mag'] = np.nan
        z['sgn'] = np.nan

    return z