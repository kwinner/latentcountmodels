import numpy as np
import logsign as ls
import cygdual
import gdual


def some_gdual():
    t1 = gdual.gdual_new(4, 7)
    t2 = gdual.gdual_reciprocal(t1)
    t3 = gdual.gdual_log(t2)
    t4 = gdual.gdual_compose(t3, t1)
    return t4


u = np.array([-1.38629436e+00,
              -2.50000000e-01,
              3.12500000e-02,
              -5.20833333e-03,
              9.76562500e-04,
              -1.95312500e-04,
              4.06901042e-05])

u_ls = ls.real2ls(u)

print('EXP')
print(gdual.gdual_exp(u))
print(ls.ls2real(cygdual.exp(u_ls)))
print('')

print('INV')
print(gdual.gdual_reciprocal(u))
print(ls.ls2real(cygdual.inv(u_ls)))
print('')

print('MUL')
print(gdual.gdual_mul(u, u))
print(ls.ls2real(cygdual.mul(u_ls, u_ls)))
print('')

print('POW')
#print(gdual.gdual_pow(u, 2.0)
v_ls = cygdual.pow(u_ls, 2.0)
print(ls.ls2real(v_ls))
print('')

u[0] = 2.5
u_ls = ls.real2ls(u)

print('LOG')
print(gdual.gdual_log(u))
print(ls.ls2real(cygdual.log(u_ls)))
print('')

