import gdual
import ngdual
import generatingfunctions

import numpy as np

x  = 3.0
y  = 1.4
q  = 4
c1 = 5.2    # scalar mul
c2 = -10.0  # scalar add (x)
c3 = 16.7   # scalar add (y)
k  = 2.5    # pow
lmbda = 1.9
rho   = 0.76
n     = 12.31
r     = 6.11
p     = 0.93
logp  = 0.051

###
print "\n\nTest new gduals"

gdual_utp    = gdual.gdual_new(x, q)
ngdual_tuple = ngdual.ngdual_new_x_dx(x, q)

print "gdual:  ", gdual_utp
print "ngdual: ", (np.exp(ngdual_tuple[0]) * ngdual_tuple[1])
###

###
print "\n\nTest scalar operations"

gdual_utp    = gdual.gdual_new(x, q)
gdual_utp    = gdual_utp * c1
ngdual_tuple = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple = ngdual.ngdual_scalar_mul(ngdual_tuple, c1)

print "gdual:  ", gdual_utp
print "ngdual: ", (np.exp(ngdual_tuple[0]) * ngdual_tuple[1])

gdual_utp    = gdual.gdual_new(x, q)
gdual_utp[0] = gdual_utp[0] + c2
ngdual_tuple = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple = ngdual.ngdual_scalar_add(ngdual_tuple, c2)

print "\ngdual:  ", gdual_utp
print "ngdual: ", (np.exp(ngdual_tuple[0]) * ngdual_tuple[1])

gdual_utp    = gdual.gdual_new(x, q)
gdual_utp    = gdual_utp * c1
gdual_utp[0] = gdual_utp[0] + c2
ngdual_tuple = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple = ngdual.ngdual_scalar_mul(ngdual_tuple, c1)
ngdual_tuple = ngdual.ngdual_scalar_add(ngdual_tuple, c2)

print "\ngdual:  ", gdual_utp
print "ngdual: ", (np.exp(ngdual_tuple[0]) * ngdual_tuple[1])

gdual_utp    = gdual.gdual_new(x, q)
gdual_utp[0] = gdual_utp[0] + c2
gdual_utp    = gdual_utp * c1
ngdual_tuple = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple = ngdual.ngdual_scalar_add(ngdual_tuple, c2)
ngdual_tuple = ngdual.ngdual_scalar_mul(ngdual_tuple, c1)

print "\ngdual:  ", gdual_utp
print "ngdual: ", (np.exp(ngdual_tuple[0]) * ngdual_tuple[1])
###

###
print "\n\nTest pow"

gdual_utp    = gdual.gdual_new(x, q)
gdual_utp    = gdual_utp * c1
gdual_utp    = gdual.gdual_pow(gdual_utp, k)
ngdual_tuple = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple = ngdual.ngdual_scalar_mul(ngdual_tuple, c1)
ngdual_tuple = ngdual.ngdual_pow(ngdual_tuple, k)

print "gdual:  ", gdual_utp
print "ngdual: ", (np.exp(ngdual_tuple[0]) * ngdual_tuple[1])
###

###
print "\n\nTest experimental pow operation"

ngdual_tuple = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple = ngdual.ngdual_scalar_mul(ngdual_tuple, c1)
ngdual_tuple = ngdual.ngdual_pow_safe(ngdual_tuple, k)
ngdual_tuple2 = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple2 = ngdual.ngdual_scalar_mul(ngdual_tuple2, c1)
ngdual_tuple2 = ngdual.ngdual_pow(ngdual_tuple2, k)

print "ngdual1: ", (np.exp(ngdual_tuple[0]) * ngdual_tuple[1])
print "ngdual2: ", (np.exp(ngdual_tuple2[0]) * ngdual_tuple2[1])
###

###
print "\n\nTest reciprocal"

gdual_utp    = gdual.gdual_new(x, q)
gdual_utp    = gdual_utp * c1
gdual_utp    = gdual.gdual_pow(gdual_utp, k)
gdual_utp    = gdual.gdual_reciprocal(gdual_utp)
ngdual_tuple = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple = ngdual.ngdual_scalar_mul(ngdual_tuple, c1)
ngdual_tuple = ngdual.ngdual_pow(ngdual_tuple, k)
ngdual_tuple = ngdual.ngdual_reciprocal(ngdual_tuple)

print "gdual:  ", gdual_utp
print "ngdual: ", (np.exp(ngdual_tuple[0]) * ngdual_tuple[1])
###

###
print "\n\nTest exp"

gdual_utp    = gdual.gdual_new(x, q)
gdual_utp    = gdual_utp * c1
gdual_utp    = gdual.gdual_exp(gdual_utp)
ngdual_tuple = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple = ngdual.ngdual_scalar_mul(ngdual_tuple, c1)
ngdual_tuple = ngdual.ngdual_exp(ngdual_tuple)

print "gdual:  ", gdual_utp
print "ngdual: ", (np.exp(ngdual_tuple[0]) * ngdual_tuple[1])
###

###
print "\n\nTest experimental exp operation"

ngdual_tuple = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple = ngdual.ngdual_scalar_mul(ngdual_tuple, c1)
ngdual_tuple = ngdual.ngdual_exp_safe(ngdual_tuple)
ngdual_tuple2 = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple2 = ngdual.ngdual_scalar_mul(ngdual_tuple2, c1)
ngdual_tuple2 = ngdual.ngdual_exp(ngdual_tuple2)

print "ngdual1: ", (np.exp(ngdual_tuple[0]) * ngdual_tuple[1])
print "ngdual2: ", (np.exp(ngdual_tuple2[0]) * ngdual_tuple2[1])
###

###
print "\n\nTest log"
gdual_utp    = gdual.gdual_new(x, q)
gdual_utp    = gdual_utp * c1
gdual_utp    = gdual.gdual_log(gdual_utp)
ngdual_tuple = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple = ngdual.ngdual_scalar_mul(ngdual_tuple, c1)
ngdual_tuple = ngdual.ngdual_log(ngdual_tuple)

print "gdual:  ", gdual_utp
print "ngdual: ", (np.exp(ngdual_tuple[0]) * ngdual_tuple[1])
###

###
print "\n\nTest experimental log operation"
ngdual_tuple = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple = ngdual.ngdual_scalar_mul(ngdual_tuple, c1)
ngdual_tuple = ngdual.ngdual_log_safe(ngdual_tuple)
ngdual_tuple2 = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple2 = ngdual.ngdual_scalar_mul(ngdual_tuple2, c1)
ngdual_tuple2 = ngdual.ngdual_log(ngdual_tuple2)

print "ngdual1: ", (np.exp(ngdual_tuple[0]) * ngdual_tuple[1])
print "ngdual2: ", (np.exp(ngdual_tuple2[0]) * ngdual_tuple2[1])
###

###
print "\n\nTest deriv"
gdual_utp    = gdual.gdual_new(x, q)
gdual_utp    = gdual_utp * c1
gdual_utp    = gdual.gdual_log(gdual_utp)
gdual_utp    = gdual.gdual_deriv(gdual_utp, 2)
ngdual_tuple = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple = ngdual.ngdual_scalar_mul(ngdual_tuple, c1)
ngdual_tuple = ngdual.ngdual_log(ngdual_tuple)
ngdual_tuple = ngdual.ngdual_deriv(ngdual_tuple, 2)

print "gdual:  ", gdual_utp
print "ngdual: ", (np.exp(ngdual_tuple[0]) * ngdual_tuple[1])
###

###
print "\n\nTest gdual mul"
F_gdual      = gdual.gdual_new(x, q)
G_gdual      = gdual.gdual_new(y, q)
F_gdual[0]   = F_gdual[0] + c2
G_gdual[0]   = G_gdual[0] + c3
F_ngdual     = ngdual.ngdual_new_x_dx(x, q)
G_ngdual     = ngdual.ngdual_new_x_dx(y, q)
F_ngdual     = ngdual.ngdual_scalar_add(F_ngdual, c2)
G_ngdual     = ngdual.ngdual_scalar_add(G_ngdual, c3)

print "F_gdual:  ", F_gdual
print "F_ngdual: ", (np.exp(F_ngdual[0]) * F_ngdual[1])
print "\nG_gdual:  ", G_gdual
print "G_ngdual: ", (np.exp(G_ngdual[0]) * G_ngdual[1])

H_gdual      = gdual.gdual_mul(F_gdual, G_gdual)
H_ngdual     = ngdual.ngdual_mul(F_ngdual, G_ngdual)

print "\nH_gdual:  ", H_gdual
print "H_ngdual: ", (np.exp(H_ngdual[0]) * H_ngdual[1])

##

F_gdual      = gdual.gdual_new(x, q)
G_gdual      = gdual.gdual_new(y, q)
F_gdual[0]   = F_gdual[0] + c3
G_gdual[0]   = G_gdual[0] + c3
F_gdual      = gdual.gdual_pow(F_gdual, k)
G_gdual      = gdual.gdual_log(G_gdual)
F_ngdual     = ngdual.ngdual_new_x_dx(x, q)
G_ngdual     = ngdual.ngdual_new_x_dx(y, q)
F_ngdual     = ngdual.ngdual_scalar_add(F_ngdual, c3)
G_ngdual     = ngdual.ngdual_scalar_add(G_ngdual, c3)
F_ngdual     = ngdual.ngdual_pow(F_ngdual, k)
G_ngdual     = ngdual.ngdual_log(G_ngdual)

print "\nF_gdual:  ", F_gdual
print "F_ngdual: ", (np.exp(F_ngdual[0]) * F_ngdual[1])
print "\nG_gdual:  ", G_gdual
print "G_ngdual: ", (np.exp(G_ngdual[0]) * G_ngdual[1])

H_gdual      = gdual.gdual_mul(F_gdual, G_gdual)
H_ngdual     = ngdual.ngdual_mul(F_ngdual, G_ngdual)

print "\nH_gdual:  ", H_gdual
print "H_ngdual: ", (np.exp(H_ngdual[0]) * H_ngdual[1])
###

###
print "\n\nTest gdual compose"
F_gdual      = gdual.gdual_new(x, q)
G_gdual      = gdual.gdual_new(y, q)
F_gdual[0]   = F_gdual[0] + c2
G_gdual[0]   = G_gdual[0] + c3
F_ngdual     = ngdual.ngdual_new_x_dx(x, q)
G_ngdual     = ngdual.ngdual_new_x_dx(y, q)
F_ngdual     = ngdual.ngdual_scalar_add(F_ngdual, c2)
G_ngdual     = ngdual.ngdual_scalar_add(G_ngdual, c3)

print "F_gdual:  ", F_gdual
print "F_ngdual: ", (np.exp(F_ngdual[0]) * F_ngdual[1])
print "\nG_gdual:  ", G_gdual
print "G_ngdual: ", (np.exp(G_ngdual[0]) * G_ngdual[1])

H_gdual      = gdual.gdual_compose(F_gdual, G_gdual)
H_ngdual     = ngdual.ngdual_compose(F_ngdual, G_ngdual)

print "\nH_gdual:  ", H_gdual
print "H_ngdual: ", (np.exp(H_ngdual[0]) * H_ngdual[1])

H_gdual      = gdual.gdual_compose(G_gdual, F_gdual)
H_ngdual     = ngdual.ngdual_compose(G_ngdual, F_ngdual)

print "\nH_gdual:  ", H_gdual
print "H_ngdual: ", (np.exp(H_ngdual[0]) * H_ngdual[1])

##

F_gdual      = gdual.gdual_new(x, q)
G_gdual      = gdual.gdual_new(y, q)
F_gdual[0]   = F_gdual[0] + c3
G_gdual[0]   = G_gdual[0] + c3
F_gdual      = gdual.gdual_pow(F_gdual, k)
G_gdual      = gdual.gdual_log(G_gdual)
F_ngdual     = ngdual.ngdual_new_x_dx(x, q)
G_ngdual     = ngdual.ngdual_new_x_dx(y, q)
F_ngdual     = ngdual.ngdual_scalar_add(F_ngdual, c3)
G_ngdual     = ngdual.ngdual_scalar_add(G_ngdual, c3)
F_ngdual     = ngdual.ngdual_pow(F_ngdual, k)
G_ngdual     = ngdual.ngdual_log(G_ngdual)

print "\nF_gdual:  ", F_gdual
print "F_ngdual: ", (np.exp(F_ngdual[0]) * F_ngdual[1])
print "\nG_gdual:  ", G_gdual
print "G_ngdual: ", (np.exp(G_ngdual[0]) * G_ngdual[1])

H_gdual      = gdual.gdual_compose(F_gdual, G_gdual)
H_ngdual     = ngdual.ngdual_compose(F_ngdual, G_ngdual)

print "\nH_gdual:  ", H_gdual
print "H_ngdual: ", (np.exp(H_ngdual[0]) * H_ngdual[1])

H_gdual      = gdual.gdual_compose(G_gdual, F_gdual)
H_ngdual     = ngdual.ngdual_compose(G_ngdual, F_ngdual)

print "\nH_gdual:  ", H_gdual
print "H_ngdual: ", (np.exp(H_ngdual[0]) * H_ngdual[1])

##

F_gdual      = gdual.gdual_new(x, q)
G_gdual      = gdual.gdual_new(y, q)
F_gdual[0]   = F_gdual[0] + c3
G_gdual[0]   = G_gdual[0] + c3
F_gdual      = gdual.gdual_pow(F_gdual, k)
F_ngdual     = ngdual.ngdual_new_x_dx(x, q)
G_ngdual     = ngdual.ngdual_new_x_dx(y, q)
F_ngdual     = ngdual.ngdual_scalar_add(F_ngdual, c3)
G_ngdual     = ngdual.ngdual_scalar_add(G_ngdual, c3)
F_ngdual     = ngdual.ngdual_pow(F_ngdual, k)

print "\nF_gdual:  ", F_gdual
print "F_ngdual: ", (np.exp(F_ngdual[0]) * F_ngdual[1])
print "\nG_gdual:  ", G_gdual
print "G_ngdual: ", (np.exp(G_ngdual[0]) * G_ngdual[1])

H_gdual      = gdual.gdual_compose_affine(F_gdual, G_gdual)
H_ngdual     = ngdual.ngdual_compose_affine(F_ngdual, G_ngdual)

print "\nH_gdual:  ", H_gdual
print "H_ngdual: ", (np.exp(H_ngdual[0]) * H_ngdual[1])
###

###
print "\n\nTest generating functions"
gdual_utp    = gdual.gdual_new(x, q)
gdual_utp    = generatingfunctions.poisson_gdual(gdual_utp, [lmbda])
ngdual_tuple = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple = generatingfunctions.poisson_ngdual(ngdual_tuple, [lmbda])

print "poisson"
print "gdual:  ", gdual_utp
print "ngdual: ", (np.exp(ngdual_tuple[0]) * ngdual_tuple[1])

##

gdual_utp    = gdual.gdual_new(x, q)
gdual_utp    = generatingfunctions.bernoulli_gdual(gdual_utp, [rho])
ngdual_tuple = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple = generatingfunctions.bernoulli_ngdual(ngdual_tuple, [rho])

print "bernoulli"
print "gdual:  ", gdual_utp
print "ngdual: ", (np.exp(ngdual_tuple[0]) * ngdual_tuple[1])

##

gdual_utp    = gdual.gdual_new(x, q)
gdual_utp    = generatingfunctions.binomial_gdual(gdual_utp, [n, p])
ngdual_tuple = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple = generatingfunctions.binomial_ngdual(ngdual_tuple, [n, p])

print "binomial"
print "gdual:  ", gdual_utp
print "ngdual: ", (np.exp(ngdual_tuple[0]) * ngdual_tuple[1])

##

gdual_utp    = gdual.gdual_new(x, q)
gdual_utp    = generatingfunctions.negbin_gdual(gdual_utp, [r, p])
ngdual_tuple = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple = generatingfunctions.negbin_ngdual(ngdual_tuple, [r, p])

print "negbin"
print "gdual:  ", gdual_utp
print "ngdual: ", (np.exp(ngdual_tuple[0]) * ngdual_tuple[1])

##

gdual_utp    = gdual.gdual_new(x, q)
gdual_utp    = generatingfunctions.logarithmic_gdual(gdual_utp, [logp])
ngdual_tuple = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple = generatingfunctions.logarithmic_ngdual(ngdual_tuple, [logp])

print "logarithmic"
print "gdual:  ", gdual_utp
print "ngdual: ", (np.exp(ngdual_tuple[0]) * ngdual_tuple[1])

##

gdual_utp    = gdual.gdual_new(x, q)
gdual_utp    = generatingfunctions.geometric_gdual(gdual_utp, [p])
ngdual_tuple = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple = generatingfunctions.geometric_ngdual(ngdual_tuple, [p])

print "geometric"
print "gdual:  ", gdual_utp
print "ngdual: ", (np.exp(ngdual_tuple[0]) * ngdual_tuple[1])

##

gdual_utp    = gdual.gdual_new(x, q)
gdual_utp    = generatingfunctions.geometric2_gdual(gdual_utp, [p])
ngdual_tuple = ngdual.ngdual_new_x_dx(x, q)
ngdual_tuple = generatingfunctions.geometric2_ngdual(ngdual_tuple, [p])

print "geometric2"
print "gdual:  ", gdual_utp
print "ngdual: ", (np.exp(ngdual_tuple[0]) * ngdual_tuple[1])
###