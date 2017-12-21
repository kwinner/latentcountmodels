import numpy as np
import logsign as ls
import cygdual
import gdual_impl as gdual

def exp(x):
    return x.exp()

def log(x):
    return x.log()

def inv(x):
    return x.inv()


class GDualBase:

    @classmethod
    def zero_coefs(cls, q):
        raise NotImplementedError("Please Implement this method")

    @classmethod
    def wrap_coefs(cls, coefs):
        raise NotImplementedError("Please Implement this method")

    @classmethod
    def unwrap_coefs(cls, coefs):
        raise NotImplementedError("Please Implement this method")
    
    _div  = None
    _rdiv = None
    _mul  = None
    _add  = None
    _sub  = None
    _rsub = None
    _pow  = None
    _neg  = None
    _exp  = None
    _log  = None
    _inv  = None

    def __init__(self, q=None, coefs=None, wrap=False):

        if q is None:
            if coefs is None:
                raise('Must supply either coefficients or truncation order')
            else:
                q = len(coefs)
            
        self.set_coefs(q, coefs, wrap)

    def set_coefs(self, q, coefs, wrap=False):
        
        if q is None:
            raise('Must supply q')

        self.coefs = self.zero_coefs(q)

        if coefs is not None:
            p = min(q, len(coefs))
            if wrap:
                self.coefs[:p] = self.wrap_coefs(coefs[:p])
            else:
                self.coefs[:p] = coefs[:p]

    def set_truncation_order(self, q):
        self.set_coefs(q, self.coefs)
        
    @classmethod
    def const(cls, q, c):
        """construct a new gdual object for <c, dx>_q"""
        assert np.isreal(c) and (not hasattr(c, "__len__") or len(c) == 1)
        assert q > 0

        F = cls(q, coefs=cls.wrap_coefs([c]))
        return F

    @classmethod
    def one(cls, q):
        return cls.const(q, 1.0)

    @classmethod
    def x_dx(cls, q, x):
        F = cls(q, coefs=cls.wrap_coefs([x, 1]))
        return F

    def __repr__(self):
        return self.coefs.__repr__()

    def __str__(self):
        return self.coefs.__str__()

    def as_real(self):
        return self.unwrap_coefs(self.coefs)

    def binary_op(self, other, f):

        if isinstance(other, self.__class__): # same type

            # Extend to same truncation order if needed
            p = len(self.coefs)
            q = len(other.coefs)
            if p < q:
                self.set_truncation_order(q)
            elif q < p:
                other.set_truncation_order(p)
                
        elif isinstance(other, (int, float)):
            other = self.const(len(self.coefs), other)
        else:
            raise('Incompatible other type')
        
        return self.__class__(
            coefs=f(self.coefs, other.coefs)
        )

    def __add__(self, other):
        return self.binary_op(other, self._add)

    def __radd__(self, other):
        return self.binary_op(other, self._add)
    
    def __mul__(self, other):
        return self.binary_op(other, self._mul)

    def __rmul__(self, other):
        return self.binary_op(other, self._mul)

    def __sub__(self, other):
        return self.binary_op(other, self._sub)

    def __rsub__(self, other):
        return self.binary_op(other, self._rsub)

    def __truediv__(self, other):
        return self.binary_op(other, self._div)

    def __rtruediv__(self, other):
        return self.binary_op(other, self._rdiv)

    # Python 2.7
    __div__  = __truediv__
    __rdiv__ = __rtruediv__
    
    def __pow__(self, other):
        return self.__class__(
            coefs = self._pow(self.coefs, other),
        )

    def __neg__(self):
        return self.__class__(
            coefs = self._neg(self.coefs)
        )

    def exp(self):
        return self.__class__(
            coefs = self._exp(self.coefs)
        )
        
    def log(self):
        return self.__class__(
            coefs = self._log(self.coefs)
        )

    def inv(self):
        return self.__class__(
            coefs= self._inv(self.coefs)
        )

class LSGDual(GDualBase):

    @classmethod
    def zero_coefs(cls, q):
        return ls.zeros(q)

    @classmethod
    def wrap_coefs(cls, coefs):
        return ls.real2ls(coefs)

    @classmethod
    def unwrap_coefs(cls, coefs):
        return ls.ls2real(coefs)

    _div  = staticmethod( cygdual.div )
    _rdiv = staticmethod( cygdual.rdiv )
    _mul  = staticmethod( cygdual.mul )
    _add  = staticmethod( cygdual.add )
    _sub  = staticmethod( cygdual.sub )
    _rsub = staticmethod( cygdual.rsub )
    _pow  = staticmethod( cygdual.pow )
    _neg  = staticmethod( cygdual.neg )
    _exp  = staticmethod( cygdual.exp )
    _log  = staticmethod( cygdual.log )
    _inv  = staticmethod( cygdual.inv )

class GDual(GDualBase):

    DTYPE=np.double
    
    @classmethod
    def zero_coefs(cls, q):
        return np.zeros(q, dtype=cls.DTYPE)

    @classmethod
    def wrap_coefs(cls, coefs):
        return np.array(coefs, dtype=cls.DTYPE)

    @classmethod
    def unwrap_coefs(cls, coefs):
        return coefs

    _div  = staticmethod( gdual.gdual_div )
    _rdiv = staticmethod( lambda x,y: gdual.gdual_div(y, x) )
    _mul  = staticmethod( gdual.gdual_mul )
    _add  = staticmethod( np.ndarray.__add__ )
    _sub  = staticmethod( lambda x,y: x - y )
    _rsub = staticmethod( lambda x,y: y - x )
    _pow  = staticmethod( gdual.gdual_pow )
    _neg  = staticmethod( lambda x: -x )
    _exp  = staticmethod( lambda x: gdual.gdual_exp(x) )
    _log  = staticmethod( gdual.gdual_log )
    _inv  = staticmethod( gdual.gdual_reciprocal )


if __name__ == "__main__":


    for C in [GDual, LSGDual]:
        
        coefs = np.array([0.5, -1, 100, 0.8])
        x = C(30, coefs=coefs, wrap=True)
        
        print x.as_real()
        print exp(x).as_real()
        
        print (x*2).coefs
        print (2*x).coefs
        
        print ((x/2.0)*2.0).as_real()
        print (0.5*x).coefs
        
        print exp(log(x)).as_real()
        print log(exp(x)).as_real()
        
        y = C.const(10, 1000)
        print "y: ", y.coefs
        
        z = C.one(10)
        print "z: ",  z.coefs
        
        w = C.x_dx(5, 2.4)
        print "w: ",  w.coefs
        
        u = x + y
        print "u: ", u.coefs
        
        v = x * y
        print "v: ", v.coefs
        
        a = C.exp(v)
        print "a: ", a.coefs
        
        b = C.log(a)
        print "b: ", b.coefs
