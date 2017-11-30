# First run
#   >> python setup.py build_ext --inplace
# The run this script

import numpy as np
import cygdual
import gdual.gdual as gdual
import lsgdual.logsign as ls

u = np.array([0.5, -1.0, 3, -1.0], dtype="double")

print "gdual_exp(u): "
print gdual.gdual_exp(u)

print "cython version: "
u_ls = ls.real2ls(u)
v_ls = cygdual.cygdual_exp(u_ls)
v = ls.ls2real(v_ls)
print v

