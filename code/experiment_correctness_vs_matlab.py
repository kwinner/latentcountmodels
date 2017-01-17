from default_data import *
from UTPPGFFA_phmm import *
from UTPPGF_util import *

Alpha, Gamma, Psi = UTP_PGFFA_phmm(y, Lambda, Delta, Rho, d=3)

print "ll:\t%e"   % Alpha[-1].data[0]
print "Mean:\t%e" % UTPPGF_mean(Alpha[-1])
print "Var:\t%e"  % UTPPGF_var(Alpha[-1])

print "\n", "Compare with the following Matlab code:"
print "y =", y, ";"
print "lambda =", Lambda, ";"
print "delta =", Delta, ";"
print "rho =", Rho[0], ";"
print "[ll, ~, ~, ~, messages] = gf_forward(y, lambda, rho, delta);"
print "[mean, var] = moments_pgf(messages(end).f, messages(end).a, messages(end).b);"
print "fprintf('ll:\\t %.06e\\n', exp(ll));"
print "fprintf('mean:\\t %.06e\\n', mean);"
print "fprintf('mean:\\t %.06e\\n', var);"