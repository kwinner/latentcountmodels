import UTPPGFFA_phmm
import UTPPGFFA
from distributions import *
import truncatedfa
import numpy as np
from scipy import stats

y = np.array([6,8,10,6,8,10,6,8,10])
Lambda = np.array([16, 20, 24, 16, 20, 24, 16, 20, 24])
Delta = np.array([0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4])
Rho = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])

N_LIMIT = 1000

#poisson arrival, binom branching correctness
arrival_pmf = stats.poisson
arrival_pgf = lambda s, theta: poisson_pgf(s, theta)
branch_fun  = truncatedfa.binomial_branching
branch_pgf  = lambda s, theta: bernoulli_pgf(s, theta)
observ_pgf  = None

Theta = {'arrival': Lambda,
         'branch':  Delta,
         'observ':  Rho}

# Alpha_utppgffa, Gamma_utppgffa, Psi_utppgffa = UTPPGFFA.UTP_PGFFA(y,
#                                                                   Theta,
#                                                                   arrival_pgf,
#                                                                   branch_pgf,
#                                                                   observ_pgf,
#                                                                   d=3)
# Alpha_utppgffa = UTPPGFFA.utppgffa(y,
#                                    Theta,
#                                    arrival_pgf,
#                                    branch_pgf,
#                                    observ_pgf,
#                                    d=1)
Alpha_utppgffa, logZ_utppgffa = UTPPGFFA.utppgffa(y,
                                                  Theta,
                                                  arrival_pgf,
                                                  branch_pgf,
                                                  observ_pgf,
                                                  d=1,
                                                  normalized=True)
Alpha_pgffa, Gamma_pgffa, Psi_pgffa = UTPPGFFA_phmm.UTP_PGFFA_phmm(y,
                                                                   Lambda,
                                                                   Delta,
                                                                   Rho,
                                                                   d=3)
Alpha_tfa, z_tfa = truncatedfa.truncated_forward(arrival_pmf,
                                                 Lambda.reshape((-1, 1)),
                                                 branch_fun,
                                                 Delta.reshape((-1, 1)),
                                                 Rho,
                                                 y,
                                                 n_max=N_LIMIT)

# likelihood_utppgffa = Alpha_utppgffa[-1].data[0,0]   # original AlgoPy impl
# likelihood_utppgffa = Alpha_utppgffa[-1][0]          # vector impl
# likelihood_utppgffa = np.exp(Alpha_utppgffa[-1][0])  # failed log-space impl
# likelihood_utppgffa = Alpha_utppgffa[-1][0] * np.prod(Z_utppgffa)
# loglikelihood_utppgffa = np.log(Alpha_utppgffa[-1][0]) + np.sum(np.log(Z_utppgffa)) # linear Z
loglikelihood_utppgffa = np.log(Alpha_utppgffa[-1][0]) + np.sum(logZ_utppgffa)
loglikelihood_pgffa    = np.log(Alpha_pgffa[-1].data[0,0])
loglikelihood_tfa      = truncatedfa.likelihood(z_tfa, log=True)

print "--Poisson arrivals, Binom branching--"
print "UTPPGFFA loglikelihood: {0}".format(loglikelihood_utppgffa)
print "PGFFA loglikelihood:    {0}".format(loglikelihood_pgffa)
print "Trunc loglikelihood:    {0}".format(loglikelihood_tfa)

#poisson arrival, poisson branching correctness
arrival_pmf = stats.poisson
arrival_pgf = lambda s, theta: poisson_pgf(s, theta)
branch_fun  = truncatedfa.poisson_branching
branch_pgf  = lambda s, theta: poisson_pgf(s, theta)
observ_pgf  = None

Theta = {'arrival': Lambda,
         'branch':  Delta,
         'observ':  Rho}

# Alpha_utppgffa, Gamma_utppgffa, Psi_utppgffa = UTPPGFFA.UTP_PGFFA(y,
#                                                                   Theta,
#                                                                   arrival_pgf,
#                                                                   branch_pgf,
#                                                                   observ_pgf,
#                                                                   d=3)
# Alpha_utppgffa = UTPPGFFA.utppgffa(y,
#                                    Theta,
#                                    arrival_pgf,
#                                    branch_pgf,
#                                    observ_pgf,
#                                    d=1)
Alpha_utppgffa, logZ_utppgffa = UTPPGFFA.utppgffa(y,
                                                  Theta,
                                                  arrival_pgf,
                                                  branch_pgf,
                                                  observ_pgf,
                                                  d=1,
                                                  normalized=True)
Alpha_tfa, z_tfa = truncatedfa.truncated_forward(arrival_pmf,
                                                 Lambda.reshape((-1, 1)),
                                                 branch_fun,
                                                 Delta.reshape((-1, 1)),
                                                 Rho,
                                                 y,
                                                 n_max=500)

# likelihood_utppgffa = Alpha_utppgffa[-1].data[0,0]   # original AlgoPy impl
# likelihood_utppgffa = Alpha_utppgffa[-1][0]          # vector impl
# likelihood_utppgffa = np.exp(Alpha_utppgffa[-1][0])  # failed log-space impl
# likelihood_utppgffa = Alpha_utppgffa[-1][0] * np.prod(Z_utppgffa)
# loglikelihood_utppgffa = np.log(Alpha_utppgffa[-1][0]) + np.sum(np.log(Z_utppgffa)) # linear Z
loglikelihood_utppgffa = np.log(Alpha_utppgffa[-1][0]) + np.sum(logZ_utppgffa)
loglikelihood_tfa      = truncatedfa.likelihood(z_tfa, log=True)

print "--Poisson arrivals, Poisson branching--"
print "UTPPGFFA loglikelihood: {0}".format(loglikelihood_utppgffa)
print "Trunc loglikelihood:    {0}".format(loglikelihood_tfa)