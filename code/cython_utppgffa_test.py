import UTPPGFFA_phmm
import UTPPGFFA
import UTPPGFFA_cython
# from distributions import *
from distributions_cython import *
import truncatedfa
import numpy as np
from scipy import stats

y = np.array([6,8,10,6,8,10,6,8,10], dtype=np.int32) #* 10
Lambda = np.array([16, 20, 24, 16, 20, 24, 16, 20, 24], dtype=np.float64).reshape((-1,1)) #* 10
Delta = np.array([0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4], dtype=np.float64).reshape((-1,1))
Rho = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], dtype=np.float64)
# Lambda = np.array([16, 20, 24, 16, 20, 24, 16, 20, 24], dtype=np.float64) #* 10
# Delta = np.array([0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4], dtype=np.float64)
# Rho = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], dtype=np.float64)

N_LIMIT = 1000

#poisson arrival correctness
arrival_pmf = stats.poisson
# arrival_pgf = lambda s, theta: poisson_pgf(s, theta)
arrival_pgf = poisson_utppgf_cython
arrival_pgf_name = 'poisson'
branch_fun  = truncatedfa.binomial_branching
# branch_pgf  = lambda s, theta: bernoulli_pgf(s, theta)
branch_pgf = bernoulli_utppgf_cython
branch_pgf_name = 'bernoulli'
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
# Alpha_utppgffa, logZ_utppgffa = UTPPGFFA.utppgffa(y,
#                                                   Theta,
#                                                   arrival_pgf,
#                                                   branch_pgf,
#                                                   observ_pgf,
#                                                   d=1,
#                                                   normalized=True)
alpha_utppgffa, logZ_utppgffa = UTPPGFFA_cython.utppgffa_cython(y,
                                                                arrival_pgf_name,
                                                                Lambda,
                                                                branch_pgf_name,
                                                                Delta,
                                                                Rho,
                                                                d=1)

Alpha_utppgffa2, logZ_utppgffa2 = UTPPGFFA.utppgffa(y,
                                                    Theta,
                                                    arrival_pgf,
                                                    branch_pgf,
                                                    observ_pgf,
                                                    d=1,
                                                    normalized=True)
Alpha_pgffa, Gamma_pgffa, Psi_pgffa = UTPPGFFA_phmm.UTP_PGFFA_phmm(y,
                                                                   Lambda.reshape((-1)),
                                                                   Delta.reshape((-1)),
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
loglikelihood_utppgffa  = np.log(alpha_utppgffa[0]) + np.sum(logZ_utppgffa)
loglikelihood_utppgffa2 = np.log(Alpha_utppgffa2[-1][0]) + np.sum(logZ_utppgffa2)
loglikelihood_pgffa     = np.log(Alpha_pgffa[-1].data[0,0])
loglikelihood_tfa       = truncatedfa.likelihood(z_tfa, log=True)

print "--Poisson arrivals--"
print "UTPPGFFA loglikelihood:  {0}".format(loglikelihood_utppgffa)
print "UTPPGFFA2 loglikelihood: {0}".format(loglikelihood_utppgffa2)
print "PGFFA loglikelihood:     {0}".format(loglikelihood_pgffa)
print "Trunc loglikelihood:     {0}".format(loglikelihood_tfa)

#negbin arrival correctness
# r = np.array([6,   8,   10,  6,   8,   10,  6,   8,   10])
# p = np.array([0.4, 0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.5, 0.4])
r = np.array([16,  20,  24,  16,  20,  24,  16,  20,  24])
p = np.array([0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.8])
arrival_params = np.stack((r,p), axis=1)

# Delta = Delta
# Rho = Rho

arrival_pmf = stats.nbinom
# arrival_pgf = lambda s, theta: negbin_pgf(s, theta)
arrival_pgf = negbin_utppgf_cython
arrival_pgf_name = 'negbin'
branch_fun  = truncatedfa.binomial_branching
# branch_pgf  = lambda s, theta: bernoulli_pgf(s, theta)
branch_pgf  = bernoulli_utppgf_cython
branch_pgf_name = 'bernoulli'
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
# Alpha_utppgffa = UTPPGFFA.utppgffa(y,
#                                    Theta,
#                                    arrival_pgf,
#                                    branch_pgf,
#                                    observ_pgf,
#                                    d=1)
alpha_utppgffa, logZ_utppgffa = UTPPGFFA_cython.utppgffa_cython(y,
                                                                arrival_pgf_name,
                                                                arrival_params,
                                                                branch_pgf_name,
                                                                Delta,
                                                                Rho,
                                                                d=1)

Alpha_utppgffa2, logZ_utppgffa2 = UTPPGFFA.utppgffa(y,
                                                    Theta,
                                                    arrival_pgf,
                                                    branch_pgf,
                                                    observ_pgf,
                                                    d=1,
                                                    normalized=True)
Alpha_tfa, z_tfa = truncatedfa.truncated_forward(arrival_pmf,
                                                 arrival_params,
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
loglikelihood_utppgffa  = np.log(alpha_utppgffa[0]) + np.sum(logZ_utppgffa)
loglikelihood_utppgffa2 = np.log(Alpha_utppgffa2[-1][0]) + np.sum(logZ_utppgffa2)
loglikelihood_tfa       = truncatedfa.likelihood(z_tfa, log=True)

print "--NegBin arrivals--"
print "UTPPGFFA loglikelihood:  {0}".format(loglikelihood_utppgffa)
print "UTPPGFFA2 loglikelihood: {0}".format(loglikelihood_utppgffa2)
print "Trunc loglikelihood:     {0}".format(loglikelihood_tfa)


# #logarithmic arrival correctness
# arrival_pmf = stats.logser
# # arrival_pgf = lambda s, theta: logarithmic_pgf(s, theta)
# arrival_pgf = logarithmic_utppgf_cython
# branch_fun  = truncatedfa.binomial_branching
# # branch_pgf  = lambda s, theta: bernoulli_pgf(s, theta)
# branch_pgf  = bernoulli_utppgf_cython
# observ_pgf  = None
#
# invLambda = 1.0 / Lambda
#
# Theta = {'arrival':invLambda,
#          'branch':  Delta,
#          'observ':  Rho}
#
# # Alpha_utppgffa, Gamma_utppgffa, Psi_utppgffa = UTPPGFFA.UTP_PGFFA(y,
# #                                                                   Theta,
# #                                                                   arrival_pgf,
# #                                                                   branch_pgf,
# #                                                                   observ_pgf,
# #                                                                   d=3)
# # Alpha_utppgffa = UTPPGFFA.utppgffa(y,
# #                                    Theta,
# #                                    arrival_pgf,
# #                                    branch_pgf,
# #                                    observ_pgf,
# #                                    d=1)
# Alpha_utppgffa, logZ_utppgffa = UTPPGFFA.utppgffa(y,
#                                                   Theta,
#                                                   arrival_pgf,
#                                                   branch_pgf,
#                                                   observ_pgf,
#                                                   d=1,
#                                                   normalized=True)
# Alpha_tfa, z_tfa = truncatedfa.truncated_forward(arrival_pmf,
#                                                  invLambda.reshape((-1, 1)),
#                                                  branch_fun,
#                                                  Delta.reshape((-1, 1)),
#                                                  Rho,
#                                                  y,
#                                                  n_max=N_LIMIT)
# # likelihood_utppgffa = Alpha_utppgffa[-1].data[0,0]   # original AlgoPy impl
# # likelihood_utppgffa = Alpha_utppgffa[-1][0]          # vector impl
# # likelihood_utppgffa = np.exp(Alpha_utppgffa[-1][0])  # failed log-space impl
# # likelihood_utppgffa = Alpha_utppgffa[-1][0] * np.prod(Z_utppgffa)
# # loglikelihood_utppgffa = np.log(Alpha_utppgffa[-1][0]) + np.sum(np.log(Z_utppgffa)) # linear Z
# loglikelihood_utppgffa = np.log(Alpha_utppgffa[-1][0]) + np.sum(logZ_utppgffa)
# loglikelihood_tfa      = truncatedfa.likelihood(z_tfa, log=True)
#
# print "--Logarithmic arrivals--"
# print "UTPPGFFA loglikelihood: {0}".format(loglikelihood_utppgffa)
# print "Trunc loglikelihood:    {0}".format(loglikelihood_tfa)
#
#
# #geometric arrival correctness
# arrival_pmf = stats.geom
# # arrival_pgf = lambda s, theta: geometric_pgf(s, theta)
# arrival_pgf = geometric_utppgf_cython
# branch_fun  = truncatedfa.binomial_branching
# # branch_pgf  = lambda s, theta: bernoulli_pgf(s, theta)
# branch_pgf  = bernoulli_utppgf_cython
# observ_pgf  = None
#
# invLambda = 1.0 / Lambda
#
# Theta = {'arrival':invLambda,
#          'branch':  Delta,
#          'observ':  Rho}
#
# # Alpha_utppgffa, Gamma_utppgffa, Psi_utppgffa = UTPPGFFA.UTP_PGFFA(y,
# #                                                                   Theta,
# #                                                                   arrival_pgf,
# #                                                                   branch_pgf,
# #                                                                   observ_pgf,
# #                                                                   d=3)
# # Alpha_utppgffa = UTPPGFFA.utppgffa(y,
# #                                    Theta,
# #                                    arrival_pgf,
# #                                    branch_pgf,
# #                                    observ_pgf,
# #                                    d=1)
# Alpha_utppgffa, logZ_utppgffa = UTPPGFFA.utppgffa(y,
#                                                   Theta,
#                                                   arrival_pgf,
#                                                   branch_pgf,
#                                                   observ_pgf,
#                                                   d=1,
#                                                   normalized=True)
# Alpha_tfa, z_tfa = truncatedfa.truncated_forward(arrival_pmf,
#                                                  invLambda.reshape((-1, 1)),
#                                                  branch_fun,
#                                                  Delta.reshape((-1, 1)),
#                                                  Rho,
#                                                  y,
#                                                  n_max=N_LIMIT)
# # likelihood_utppgffa = Alpha_utppgffa[-1].data[0,0]   # original AlgoPy impl
# # likelihood_utppgffa = Alpha_utppgffa[-1][0]          # vector impl
# # likelihood_utppgffa = np.exp(Alpha_utppgffa[-1][0])  # failed log-space impl
# # likelihood_utppgffa = Alpha_utppgffa[-1][0] * np.prod(Z_utppgffa)
# # loglikelihood_utppgffa = np.log(Alpha_utppgffa[-1][0]) + np.sum(np.log(Z_utppgffa)) # linear Z
# loglikelihood_utppgffa = np.log(Alpha_utppgffa[-1][0]) + np.sum(logZ_utppgffa)
# loglikelihood_tfa      = truncatedfa.likelihood(z_tfa, log=True)
#
# print "--Geometric arrivals--"
# print "UTPPGFFA loglikelihood: {0}".format(loglikelihood_utppgffa)
# print "Trunc loglikelihood:    {0}".format(loglikelihood_tfa)