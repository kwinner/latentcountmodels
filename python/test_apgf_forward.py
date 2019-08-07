from apgf_forward import APGF, APGF_NB, APGF_Poiss
import forward as fwd
import gdual as gd

import numpy as np
import scipy as sp

GDualType = gd.LSGDual

y = np.array([3, 2, 4])
lmbda = np.array([5, 5, 5]).reshape(-1, 1)
delta = np.array([0.4, 0.4, 0.4]).reshape(-1, 1)
rho = np.array([0.25, 0.25, 0.25])

immigration_pgf = fwd.poisson_pgf
offspring_pgf   = fwd.bernoulli_pgf

K = len(y)

# precompute all values of s_k, u_k (the backwards pass)
s = np.zeros(K+1)
u = np.zeros(K)

s[K] = 1.0
for i in range(K)[::-1]:
    u[i] = s[i+1] * (1 - rho[i])
    s[i] = offspring_pgf(u[i], theta=delta[i])

# print(s)
# print(u)

# does this need to type check s and return a dual number if s is a dual?
Alpha_0 = lambda s_0: s_0.__class__.const(1.0, q = s_0.order()) if isinstance(s_0, gd.GDualBase) else 1.0

Gamma_1 = lambda u_1: Alpha_0(offspring_pgf(u_1, delta[0])) * immigration_pgf(u_1, lmbda[0])
Gamma_1_realized = Gamma_1(u[0])
print(Gamma_1_realized)

hatGamma_1 = APGF(Gamma_1)
hatGamma_1_realized = hatGamma_1(u[0])
print(hatGamma_1_realized)

def Alpha_1(s_1):
    const = (s_1 * rho[0]) ** y[0] / sp.misc.factorial(y[0])
    d     = gd.diff(Gamma_1, u[0], y[0], GDualType = GDualType).get(0, as_log = False)
    return const * d

Alpha_1_realized = Alpha_1(s[0])
print(Alpha_1_realized)

def hatAlpha_1(s_1):
    const = (s_1 * rho[0]) ** y[0] / sp.misc.factorial(y[0])
    d     = gd.diff(hatGamma_1, u[0], y[0], GDualType = GDualType).get(0, as_log = False)
    return const * d

hatAlpha_1_realized = hatAlpha_1(s[0])
print(hatAlpha_1_realized)

Gamma_2 = lambda u_2: Alpha_1(offspring_pgf(u_2, delta[1])) * immigration_pgf(u_2, lmbda[1])
Gamma_2_realized = Gamma_2(u[1])
print(Gamma_2_realized)

hatGamma_2 = lambda u_2: hatAlpha_1(offspring_pgf(u_2, delta[1])) * immigration_pgf(u_2, lmbda[1])
hatGamma_2_realized = hatGamma_2(u[1])
print(hatGamma_2_realized)

hat2Gamma_2 = APGF(hatGamma_2)
hat2Gamma_2_realized = hat2Gamma_2(u[1])
print(hat2Gamma_2_realized)

# Alpha_1 = lambda s_
