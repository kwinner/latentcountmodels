import scipy.signal
import numpy as np
import gdual as gd
from gdual import GDual, LSGDual, diff


def run_compose(n):

    x = np.exp(-np.random.rand(n))
    y = np.exp(-np.random.rand(n))
    
    X_LS = LSGDual(coefs=x, wrap=True)
    Y_LS = LSGDual(coefs=y, wrap=True)

    print("compose")
    Z = X_LS.compose(Y_LS)
    
    return
