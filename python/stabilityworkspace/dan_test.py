import ngdual
import numpy as np
import scipy
import scipy.special
import gdual


def ls2logsign(l, s=1):
    l = np.array(l)
    z = np.zeros(l.shape, dtype=[('l', '<f8'), ('s', 'i1')])
    z['l'] = l
    z['s'] = s
    return z


def logsign(x):
    x = np.array(x)
    z = np.zeros(x.shape, dtype=[('l', '<f8'), ('s', 'i1')])
    z['l'] = np.log(np.abs(x))
    z['s'] = np.sign(x)
    return z


def invlogsign(z):
    return z['s'] * np.exp(z['l'])

def ls_mult(x, y):
    z = np.empty_like(x)
    z['l'] =  x['l'] + y['l']
    z['s'] =  x['s'] * y['s']
    return z

def ls_add(x, y):
    z = np.array([x, y])
    return ls_sum(z)


def ls_sum(x, axis=None):
    l, s = scipy.special.logsumexp(x['l'], b=x['s'], axis=axis, return_sign=True )
    return ls2logsign(l, s)


def ls_exp(x):
    z = np.empty_like(x)
    z['l'] = np.exp(x['l'])*x['s']
    z['s'] = 1
    return z


def ls_allclose(x, y, **kwargs):
    return np.allclose(x['l'], y['l']) and np.allclose(x['s'], y['s'], **kwargs)


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

    # Test ls_exp
    x = np.array([12.3, -141.1, 0.23])
    z1 = logsign(np.exp(x))
    z2 = ls_exp(logsign(x))
    assert( ls_allclose (z1, z2) )

    # Test ls2logsign
    z1 = ls2logsign(np.log(np.abs(x)), np.sign(x))
    z2 = logsign(x)
    assert( ls_allclose (z1, z2) )

    # Test ls2logsign with positive array
    x = np.array([1.0, 10.0, 400.0])
    z1 = ls2logsign(np.log(x))
    z2 = logsign(x)
    assert( ls_allclose (z1, z2) )

    print 'test_logsign: success'


def some_gdual():
    t1 = gdual.gdual_new(4, 7)
    t2 = gdual.gdual_reciprocal(t1)
    t3 = gdual.gdual_log(t2)
    t4 = gdual.gdual_compose(t3, t1)
    return t4


def dan_gdual_exp(F):
    # assume F is passed in as log-sign
    out = np.empty_like(F)
    q   = out.shape[0]
    Ftilde = F[1:].copy()

    out[0] = ls_exp(F[0])
    Ftilde = ls_mult(Ftilde, logsign(np.arange(1, q)))
    for i in xrange(1, q):
        tmp = ls_mult(out[:i][::-1], Ftilde[:i])
        out[i] = ls_mult( ls_sum( tmp, axis=0), logsign(1./i) )

    return out

test_logsign()


# Test gdual_exp
t = some_gdual()
e1 = gdual.gdual_exp(t)
e2 = invlogsign(dan_gdual_exp(logsign(t)))
print e1
print e2
assert(np.allclose(e1, e2))

