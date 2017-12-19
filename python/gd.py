import numpy as np
import logsign as ls
import gdual

def exp(x):
    return x.exp()

def log(x):
    return x.log()

def inv(x):
    return x.inv()

class PowerSeries:

    @classmethod
    def wrap_coefs(cls, coefs):
        return np.array(coefs, dtype.np.longdouble)

    @classmethod
    def unwrap_coefs(cls, coefs):
        return coefs

    def __init__(self, q=None, coefs=None):
        if q is None:
            if coefs is None:
                raise('Must supply coefs or q')
            else:
                q = len(coefs)
            
        self.coefs = np.zeros(q)

        if coefs is not None:
            p = min(q, len(coefs))
            self.coefs[:p] = coefs[:p]

    @classmethod
    def const(cls, q, c):
        """construct a new gdual object for <c, dx>_q"""
        assert np.isreal(c) and (not hasattr(c, "__len__") or len(c) == 1)
        assert q > 0

        F = cls(q, coefs=wrap_coefs([c]))
        return F

    @classmethod
    def one(cls, q):
        return cls.const(q, 1.0)

    @classmethod
    def x_dx(cls, q, x):
        F = cls(q, coefs=wrap_coefs([x, 1]))
        return F

    def __repr__(self):
        return self.coefs.__repr__()

    def __str__(self):
        return self.coefs.__str__()

    def as_real(self):
        return self.unwrap_coefs(self.coefs)
    
    def binary_op(self, other, f):
        if isinstance(other, PowerSeries):
            pass
        elif isinstance(other, (int, float)):
            other = PowerSeries.const(len(self.coefs), other)
        else:
            raise('Incompatible other type')
        
        return PowerSeries(
            coefs=f(self.coefs, other.coefs)
        )

    def __add__(self, other):
        return self.binary_op(other, lambda x,y: x + y)

    def __radd__(self, other):
        return self.binary_op(other, lambda x,y: x + y)
    
    def __mul__(self, other):
        return self.binary_op(other, gdual.gdual_mul)

    def __rmul__(self, other):
        return self.binary_op(other, gdual.gdual_mul)

    def __sub__(self, other):
        return self.binary_op(other, lambda x,y: x - y)

    def __rsub__(self, other):
        return self.binary_op(other, lambda x,y: y - x)
    
    def __truediv__(self, other):
        return self.binary_op(other, gdual.gdual_div)

    def __rtruediv__(self, other):
        return self.binary_op(other, lambda x,y: gdual.gdual_div(y, x))

    # Python 2.7
    __div__  = __truediv__
    __rdiv__ = __rtruediv__
    
    def __pow__(self, other):
        return PowerSeries(
            coefs=gdual.gdual_pow(self.coefs, other),
        )

    def __neg__(self):
        return PowerSeries(
            coefs = -self.coefs
        )
    
    def exp(self):
        return PowerSeries(
            coefs = gdual.gdual_exp(self.coefs)
        )
        
    def log(self):
        return PowerSeries(
            coefs=gdual.gdual_log(self.coefs)
        )

    def inv(self):
        return PowerSeries(
            coefs=cygdual.gdual_reciprocal(self.coefs)
        )
    

def poisson_pgf(s, theta):
    lmbda = theta[0]
    return PowerSeries.exp(lmbda*(s - 1))

def bernoulli_pgf(s, theta):
    p = theta[0]
    return 1-p + p*s


import scipy

if __name__ == "__main__":

    C = PowerSeries

    coefs = np.array([0.5, -1, 100, 0.8])
    q = 50
    x = C(q, coefs=coefs)

    print exp(x).coefs/scipy.special.gamma(np.arange(50)+1)
    
    # print (x*2).coefs
    # print (2*x).coefs
    
    # print (x/2.0).coefs
    # print (0.5*x).coefs

    print x
    print exp(log(x))
    print log(exp(x))
    
    # y = C.const(10, 1000)
    # print "y: ", y.coefs
    #
    # z = C.one(10)
    # print "z: ",  z.coefs
    #
    # w = C.x_dx(5, 2.4)
    # print "w: ",  w.coefs
    #
    # u = x + y
    # print "u: ", u.coefs
    #
    # v = x * y
    # print "v: ", v.coefs
    #
    # a = C.exp(v)
    # print "a: ", a.coefs
    #
    # b = C.log(a)
    # print "b: ", b.coefs
