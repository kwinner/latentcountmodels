from default_data import *
from UTPPGFFA_phmm import *

import time

reps = 100
t_start = time.clock()
for i in range(0,reps):
    Alpha, Gamma, Psi = UTP_PGFFA_phmm(y, Lambda, Delta, Rho, d=1)
total_time = time.clock() - t_start

print total_time / reps

print "\n", "Compare with the following Matlab code:"
print "y =", y, ";"
print "lambda =", Lambda, ";"
print "delta =", Delta, ";"
print "rho =", Rho[0], ";"
print "nIter =", reps, ";"
print "tStart = tic;"
print "for i = 1:nIter"
print "ll = gf_forward(y, lambda, rho, delta);"
print "end"
print "tTotal = toc(tStart)/nIter;"
print "fprintf('elapsed time: %0.8f\\n', tTotal)"