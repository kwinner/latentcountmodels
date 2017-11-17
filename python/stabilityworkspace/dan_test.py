import ngdual
import numpy as np
import scipy
import scipy.special
import gdual

def logsign(x):
    """
    Convert from floating-point to log-sign number system
    :param x:   numpy array
    :return:    tuple of log(abs(x)), sign(x)
    """
    log_mag = np.log(np.abs(x))
    sign    = np.sign(x)
    return (log_mag, sign)


def invlogsign(z):
    """
    Convert from log-sign number system to floating point
    :param z:  type with z[0] = log(|x|), z[1] = sign(x)
    :return:   x = z[1] * exp(z[0])
    """
    return z[1] * np.exp(z[0])

def ls_mult(x, y):
    """
    Multiply two numbers in log-sign number system
    :param x:   log-sign representation of x
    :param y:   log-sign representation of y
    :return:    log-sign representation of x * y
    """
    log_mag  = x[0] + y[0]
    sign     = x[1] * y[1]
    return log_mag, sign

def ls_add(x, y):
    """
    Add two numbers in log-sign number system
    :param x:   log-sign representation of x
    :param y:   log-sign representation of y
    :return:    log-sign representation of x * y
    """
    log_mag_xy = np.array([x[0], y[0]])
    sign_xy    = np.array([x[1], y[1]])
    log_mag, sign = scipy.special.logsumexp(log_mag_xy, b=sign_xy, return_sign = True )
    return log_mag, sign

def ls_sum(x, axis=None):
    lz, sz = scipy.special.logsumexp(x[0], b=x[1], axis=axis, return_sign=True )
    return lz, sz

def ls_exp(x):
    l = np.exp(x[0])*x[1]
    s = 1
    return l, s

def ls_allclose(x, y, **kwargs):
    return np.allclose(x[0], y[0]) and np.allclose(x[1], y[1], **kwargs)


def test_logsign():

    x = np.array([0.1, -0.4, -100])

    # Test transform and inverse transform
    z = logsign(x)
    x_prime = invlogsign(z)
    assert( np.allclose(x, x_prime, rtol=1e-10, atol=1e-10) )

    # Test ls_add
    for pair in [(4.0, -100.0), (0.1, 0.9), (-1.0, -100.0)]:
        x = pair[0]
        y = pair[1]
        z1 = logsign(x + y)
        z2 = ls_add(logsign(x), logsign(y))
        assert( ls_allclose(z1, z2) )

    # Test ls_mult
    for pair in [(4.0, -100.0), (0.1, 0.9), (-1.0, -100.0)]:
        x = pair[0]
        y = pair[1]
        z1 = logsign(x * y)
        z2 = ls_mult(logsign(x), logsign(y))
        assert( ls_allclose (z1, z2) )


    # Test ls_sum
    z1 = logsign(np.sum(x))
    z2 = ls_sum( logsign(x) )
    assert( ls_allclose (z1, z2) )

    print 'test_logsign: success'


def some_gdual():
    t1 = gdual.gdual_new(4, 7)
    t2 = gdual.gdual_reciprocal(t1)
    t3 = gdual.gdual_log(t2)
    t4 = gdual.gdual_compose(t3, t1)
    return t4


def dan_gdual_exp(F):
    q = F.shape[0]
    out = (np.empty_like(F), np.empty_like(F))

    Ftilde = logsign(F[1:].copy())

    out[0][0], out[1][0] = ls_exp(logsign(F[0]))

    Ftilde = ls_mult(Ftilde, logsign(np.arange(1, q)))
    for i in xrange(1, q):

        Ftilde_slice = (Ftilde[0][:i], Ftilde[1][:i])
        out_slice_rev = (out[0][:i][::-1], out[1][:i][::-1])
        prod = ls_mult(Ftilde_slice, out_slice_rev)
        out[0][i], out[1][i] = ls_mult(ls_sum(prod, axis=0) , logsign(1./i) )

    return invlogsign(out)

test_logsign()

t = some_gdual()

e2 = dan_gdual_exp(t)
e1 = gdual.gdual_exp(t)


print e1
print e2
