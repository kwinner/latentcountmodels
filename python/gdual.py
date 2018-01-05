import numpy as np
import logsign as ls
import cygdual
import lsgdual_impl as lsgdual
import gdual_impl as gdual

def exp(x):
    return x.exp()

def log(x):
    return x.log()

def inv(x):
    return x.inv()


class GDualBase:
    """Abstract base class for GDual objects    
    """
    
    @classmethod
    def zero_coefs(cls, k):
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
    _deriv = None
    _get_derivatives = None
    _compose = None
    _compose_affine = None

    
    def __init__(self, val=None, q=None, as_log=False, coefs=None, wrap=False):
        """Create a new GDual object
        
        Examples:

        x = GDual(3.14, q=10)                         # Create new GDual 3.14 + 1*eps

        x = GDual(np.log(3.14), q=10, as_log=True)    # Same, but value provided in log space

        y = GDual(x, 10)                              # Creates GDual u + 1*eps where u is the 
                                                      # scalar components of GDual x. Derivatives
                                                      # of x are ignored

        z = GDual(coefs = coefs, q=10)                # Create a GDual with specified coefficients,
                                                      # (in data type native to underyling implementation)
                                                  
        
        z = GDual(coefs = [1, 2, 3],                  # Same as above, but coefficients are supplied
                  q=10, wrap=True)                    # as array of real numbers
                                                      
        
        """
        
        if coefs is not None:
            # If coefs are provided, advance to end
            pass
        
        elif isinstance(val, (int, float)):
            # If scalar is provided, set to val + 1*eps, and ensure wrapping into
            # native data type
            if as_log:
                coefs = [val, 0.0]
            else:
                coefs = [val, 1.0]
            wrap = True

        elif isinstance(val, self.__class__):
            # If another dual number is provided, set coefficients
            # in native data type and disable wrapping
            coefs = [
                val.coefs[0],
                self.wrap_coefs([1.0])
            ]
            wrap = False
            
        else:
            raise('Must supply either coefficients or value of compatible type')

        if wrap:
            coefs = self.wrap_coefs(coefs, as_log)
            
        self.set_coefs(coefs, q)
            

    def set_coefs(self, coefs, q=None):

        # If q is not provided, set equal to order, which is
        # one less than the number of coefficients
        if q is None:
            q = len(coefs)-1

        # Initialize to zeros, and then populate. This is a clean way to handle truncation
        # and it ensures the coefficients are copied and we are not retaining a view into
        # an existing array
        self.coefs = self.zero_coefs(q+1)
        p = min(q+1, len(coefs))
        self.coefs[:p] = coefs[:p]

    def set_order(self, q):
        self.set_coefs(self.coefs, q)

    def order(self):
        return len(self.coefs) - 1

    def get(self, k, as_log=False):
        return self.unwrap_coefs(self.coefs, as_log)[k]
    
    @classmethod
    def const(cls, c, q=0, as_log=False):
        """construct a new gdual object for <c, dx>_q"""
        assert np.isreal(c) and (not hasattr(c, "__len__") or len(c) == 1)
        assert q >= 0
        return cls(q=q, coefs=cls.wrap_coefs([c], as_log=as_log))

    def __repr__(self):
        return self.unwrap_coefs(self.coefs).__repr__()

    def __str__(self):
        return self.unwrap_coefs(self.coefs).__str__()

    def as_real(self):
        return self.unwrap_coefs(self.coefs)

    def binary_op(self, other, f):

        if isinstance(other, self.__class__): # same type

            # Extend to same truncation order if needed
            p = self.order()
            q = other.order()
            if p < q:
                self.set_order(q)
            elif q < p:
                other.set_order(p)
                
        elif isinstance(other, (int, float)):
            other = self.const(other, self.order())

        elif isinstance(other, np.ndarray) and other.size == 1:
            other = self.const(np.asscalar(other), self.order())
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

    def compose(self, other):
        return self.binary_op(other, self._compose)

    def compose_affine(self, other):
        return self.binary_op(other, self._compose_affine)
    
    
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
            coefs = self._inv(self.coefs)
        )

    def deriv(self, k):
        return self.__class__(
            coefs = self._deriv(self.coefs, k)
        )

    def get_derivatives(self):
        return self.__class__(
            coefs = self._get_derivatives(self.coefs)
        )

class LSGDual(GDualBase):

    @classmethod
    def zero_coefs(cls, k):
        return ls.zeros(k)

    @classmethod
    def wrap_coefs(cls, coefs, as_log=False):
        if not as_log:
            return ls.real2ls(coefs)
        else:
            out = ls.ls(shape=len(coefs))
            out['mag'] = coefs
            out['sgn'] = 1
            return out            

    @classmethod
    def unwrap_coefs(cls, coefs, as_log=False):
        if as_log:
            assert(all(coefs['sgn'] > 0))
            return coefs['mag']
        else:
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
    _deriv = staticmethod( lsgdual.deriv )
    _get_derivatives = staticmethod( lsgdual.get_derivatives )
    _compose = staticmethod( lsgdual.compose )
    _compose_affine = staticmethod( lsgdual.compose_affine )
    
class GDual(GDualBase):

    DTYPE=np.double
    
    @classmethod
    def zero_coefs(cls, k):
        return np.zeros(k, dtype=cls.DTYPE)

    @classmethod
    def wrap_coefs(cls, coefs, as_log=False):
        if not as_log:
            return np.array(coefs, dtype=cls.DTYPE)
        else:
            return np.array(np.exp(coefs), dtype=cls.DTYPE)

    @classmethod
    def unwrap_coefs(cls, coefs, as_log=False):
        if as_log:
            return np.log(coefs)
        else:
            return coefs

    _div  = staticmethod( gdual.div )
    _rdiv = staticmethod( lambda x,y: gdual.div(y, x) )
    _mul  = staticmethod( gdual.mul )
    _add  = staticmethod( lambda x,y: x + y )
    _sub  = staticmethod( lambda x,y: x - y )
    _rsub = staticmethod( lambda x,y: y - x )
    _pow  = staticmethod( gdual.pow )
    _neg  = staticmethod( lambda x: -x )
    _exp  = staticmethod( gdual.exp )
    _log  = staticmethod( gdual.log )
    _inv  = staticmethod( gdual.inv )
    _deriv = staticmethod( gdual.deriv )
    _get_derivatives = None  # TODO
    _compose = staticmethod( gdual.compose )
    _compose_affine = staticmethod( gdual.compose_affine )


def old_diff(f, x, k, GDualType=LSGDual):
    """Compute kth derivative of f evaluated at x

    f : function
    x : input
    k : number of times to differentiate
    """
    if isinstance(x, (int, float)):
        y = f( GDualType(x, k) )
        z = y.deriv(k)
        return z
    
    elif isinstance(x, GDualBase):
        GDualType = x.__class__
        q = x.order()
        return f( GDualType(x, q + k)).deriv(k).compose(x)

def diff(f, x, k, GDualType=LSGDual):
    """Compute kth derivative of f evaluated at x

    f : function
    x : input
    k : number of times to differentiate
    """
    if isinstance(x, (int, float)):
        y = f( GDualType(x, k) )
        if isinstance(y, (tuple)):
            return tuple(yi.deriv(k) for yi in y)
        else:
            return y.deriv(k)
    
    elif isinstance(x, GDualBase):
        GDualType = x.__class__
        q = x.order()
        y = f( GDualType(x, q + k))
        if isinstance(y, (tuple)):
            return tuple(yi.deriv(k).compose(x) for yi in y)
        else:
            return y.deriv(k).compose(x)

if __name__ == "__main__":


    for C in [GDual, LSGDual]:
        
        coefs = np.array([0.5, -1, 100, 0.8])
        x = C(q=30, coefs=coefs, wrap=True)
        
        print x
        print exp(x)
        
        print (x*2)
        print (2*x)
        
        print ((x/2.0)*2.0)
        print (0.5*x)
        
        print exp(log(x))
        print log(exp(x))
        
        y = C.const(10, 100)
        print "y: ", y
        
        z = C.const(1.0, 10)
        print "z: ",  z
        
        w = C(2.4, 5)
        print "w: ",  w
        
        u = x + y
        print "u: ", u
        
        v = x * y
        print "v: ", v
        
        a = C.exp(v)
        print "a: ", a
        
        b = C.log(a)
        print "b: ", b
