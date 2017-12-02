# First run
#   >> python setup.py build_ext --inplace
# The run this script

import numpy as np
import cygdual
import gdual.gdual as gdual
import logsign

u = np.array([0.5, -1.0, 3, -1.0], dtype="double")

print "gdual_exp(u): "
print gdual.gdual_exp(u)

print "cython version: "
u_ls = logsign.real2ls(u)
v_ls = cygdual.exp(u_ls)
v = logsign.ls2real(v_ls)
print v
