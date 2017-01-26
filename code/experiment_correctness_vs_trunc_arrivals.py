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

#poisson arrival correctness
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
Alpha_utppgffa = UTPPGFFA.utppgffa(y,
                                   Theta,
                                   arrival_pgf,
                                   branch_pgf,
                                   observ_pgf,
                                   d=3)
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
                                                 n_max=100)

likelihood_utppgffa = Alpha_utppgffa[-1].data[0,0]
likelihood_pgffa    = Alpha_pgffa[-1].data[0,0]
likelihood_tfa      = truncatedfa.likelihood(z_tfa, log=False)

print "--Poisson arrivals--"
print "UTPPGFFA likelihood: {0}".format(likelihood_utppgffa)
print "PGFFA likelihood:    {0}".format(likelihood_pgffa)
print "Trunc likelihood:    {0}".format(likelihood_tfa)

#negbin arrival correctness
# r = np.array([6,   8,   10,  6,   8,   10,  6,   8,   10])
# p = np.array([0.4, 0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.5, 0.4])
r = np.array([16,  20,  24,  16,  20,  24,  16,  20,  24])
p = np.array([0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.8])
arrival_params = np.stack((r,p), axis=1)

# Delta = Delta
# Rho = Rho

arrival_pmf = stats.nbinom
arrival_pgf = lambda s, theta: negbin_pgf(s, theta)
branch_fun  = truncatedfa.binomial_branching
branch_pgf  = lambda s, theta: bernoulli_pgf(s, theta)
observ_pgf  = None

Theta = {'arrival': arrival_params,
         'branch':  Delta,
         'observ':  Rho}

# Alpha_utppgffa, Gamma_utppgffa, Psi_utppgffa = UTPPGFFA.UTP_PGFFA(y,
#                                                                   Theta,
#                                                                   arrival_pgf,
#                                                                   branch_pgf,
#                                                                   observ_pgf,
#                                                                   d=3)
Alpha_utppgffa = UTPPGFFA.utppgffa(y,
                                   Theta,
                                   arrival_pgf,
                                   branch_pgf,
                                   observ_pgf,
                                   d=3)
Alpha_tfa, z_tfa = truncatedfa.truncated_forward(arrival_pmf,
                                                 arrival_params,
                                                 branch_fun,
                                                 Delta.reshape((-1, 1)),
                                                 Rho,
                                                 y,
                                                 n_max=100)

likelihood_utppgffa = Alpha_utppgffa[-1].data[0,0]
likelihood_tfa      = truncatedfa.likelihood(z_tfa, log=False)

print "--Negbin arrivals--"
print "UTPPGFFA likelihood: {0}".format(likelihood_utppgffa)
print "Trunc likelihood:    {0}".format(likelihood_tfa)


#logarithmic arrival correctness
arrival_pmf = stats.logser
arrival_pgf = lambda s, theta: logarithmic_pgf(s, theta)
branch_fun  = truncatedfa.binomial_branching
branch_pgf  = lambda s, theta: bernoulli_pgf(s, theta)
observ_pgf  = None

invLambda = 1.0 / Lambda

Theta = {'arrival':invLambda,
         'branch':  Delta,
         'observ':  Rho}

# Alpha_utppgffa, Gamma_utppgffa, Psi_utppgffa = UTPPGFFA.UTP_PGFFA(y,
#                                                                   Theta,
#                                                                   arrival_pgf,
#                                                                   branch_pgf,
#                                                                   observ_pgf,
#                                                                   d=3)
Alpha_utppgffa = UTPPGFFA.utppgffa(y,
                                   Theta,
                                   arrival_pgf,
                                   branch_pgf,
                                   observ_pgf,
                                   d=3)
Alpha_tfa, z_tfa = truncatedfa.truncated_forward(arrival_pmf,
                                                 invLambda.reshape((-1, 1)),
                                                 branch_fun,
                                                 Delta.reshape((-1, 1)),
                                                 Rho,
                                                 y,
                                                 n_max=100)

likelihood_utppgffa = Alpha_utppgffa[-1].data[0,0]
likelihood_tfa      = truncatedfa.likelihood(z_tfa, log=False)

print "--Logarithmic arrivals--"
print "UTPPGFFA likelihood: {0}".format(likelihood_utppgffa)
print "Trunc likelihood:    {0}".format(likelihood_tfa)


#geometric arrival correctness
arrival_pmf = stats.geom
arrival_pgf = lambda s, theta: geometric_pgf(s, theta)
branch_fun  = truncatedfa.binomial_branching
branch_pgf  = lambda s, theta: bernoulli_pgf(s, theta)
observ_pgf  = None

invLambda = 1.0 / Lambda

Theta = {'arrival':invLambda,
         'branch':  Delta,
         'observ':  Rho}

# Alpha_utppgffa, Gamma_utppgffa, Psi_utppgffa = UTPPGFFA.UTP_PGFFA(y,
#                                                                   Theta,
#                                                                   arrival_pgf,
#                                                                   branch_pgf,
#                                                                   observ_pgf,
#                                                                   d=3)
Alpha_utppgffa = UTPPGFFA.utppgffa(y,
                                   Theta,
                                   arrival_pgf,
                                   branch_pgf,
                                   observ_pgf,
                                   d=3)
Alpha_tfa, z_tfa = truncatedfa.truncated_forward(arrival_pmf,
                                                 invLambda.reshape((-1, 1)),
                                                 branch_fun,
                                                 Delta.reshape((-1, 1)),
                                                 Rho,
                                                 y,
                                                 n_max=100)

likelihood_utppgffa = Alpha_utppgffa[-1].data[0,0]
likelihood_tfa      = truncatedfa.likelihood(z_tfa, log=False)

print "--Geometric arrivals--"
print "UTPPGFFA likelihood: {0}".format(likelihood_utppgffa)
print "Trunc likelihood:    {0}".format(likelihood_tfa)