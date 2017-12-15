import numpy as np
from scipy.special import logsumexp


LS_DTYPE = np.dtype([('mag', np.float64), ('sgn', np.int32)], align=True)

#TODO: add underscore versions of methods w/o assertions for internal use?

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
    return (isinstance(x, np.ndarray) or isinstance(x, np.void)) and \
           x.dtype.names == ('mag', 'sgn')                       and \
           np.issubdtype(x.dtype['mag'], float)                  and \
           np.issubdtype(x.dtype['sgn'], int)


def add(x, y):
    """add two numbers in log-sign space"""
    assert isls(x) and isls(y)

    # logsumexp doesn't handle these cases well, especially for scalar numbers
    if x['sgn'] == 0:
        return y
    elif y['sgn'] == 0:
        return x

    return logsumexp(x['mag'] + y['mag'],
                     b = x['sgn'] * y['sgn'],
                     return_sign = True)


def sum(x, axis=None):
    """sum all values in the vector of numbers in ls-space along some axis"""
    mag, sgn = logsumexp(x['mag'], b = x['sgn'], axis = axis, return_sign = True)

    z = ls(mag.shape)
    z['mag'] = mag
    z['sgn'] = sgn

    return z


def inv(x):
    """compute 1/x"""
    assert isls(x)

    z = np.empty_like(x)
    z['mag'] = -x['mag']
    z['sgn'] = x['sgn']

    return z


def mul(x, y):
    """multiply two numbers (or vectors of numbers) in log-sign space"""
    assert isls(x) and isls(y)
    assert x.shape == y.shape

    z = np.empty_like(x)
    z['mag'] = x['mag'] + y['mag']
    z['sgn'] = x['sgn'] * y['sgn']

    return z


def div(x, y):
    """divide x by y in log-sign space"""
    assert isls(x) and isls(y)
    assert x.shape == y.shape

    z = mul(x, inv(y))

    return z


def exp(x):
    """exponentiate a number (or vector of numbers) in log-sign space"""
    assert isls(x)

    z = np.empty_like(x)
    z['mag'] = x['sgn'] * np.exp(x['mag'])
    z['sgn'] = 1

    return z


def log(x):
    """compute the log of a number (or vector) in log-sign space"""
    assert isls(x)

    return real2ls(x['mag'] + np.log(x['sgn']))


def pow(x, k):
    q = max(len(x), len(k))

    z = np.empty_like(x)
    z['sgn'] = np.sign(np.power(x['sgn'], k))
    z['mag']  = k * x['mag']

    return z