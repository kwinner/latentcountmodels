import UTPPGFFA
# import UTPPGFFA_cython
from distributions import *
# from distributions_cython import *
import numpy as np

# lmbda, r, rho = np.array([  9.11688589e+02,   1.06281698e+00,   5.66221059e-02])
# theta = {'arrival': np.array([lmbda] + [0] * (K - 1)),
#          'branch': np.array([r] * (K - 1)),
#          'observ': np.array([rho] * K)}

# y = np.array([23, 81, 86, 40, 102, 51, 72], dtype=np.int32)
# y = np.array([0, 1, 0, 3, 3], dtype=np.int32)
# K = y.shape[0]

# lmbda = np.zeros((K, 1), dtype=np.float64)
# lmbda[0] = 50
# lmbda = np.array([ 8.34363861, 37.79269177, 68.38119097, 48.89461165, 13.91515157], dtype=np.float64).reshape(-1,1)
# lmbda = np.array([[  3.20909177],
#        [ 14.53565068],
#        [ 26.30045807],
#        [ 18.80561987],
#        [  5.35198137]], dtype=np.float64).reshape(-1)

# delta = 1.50000001 * np.ones((K-1,1), np.float64)
# delta = np.array([[ 0.26359714],
#        [ 0.26359714],
#        [ 0.26359714],
#        [ 0.26359714]], dtype=np.float64)
# delta = np.array([[ 0.26359714],
#        [ 0.26359714],
#        [ 0.26359714],
#        [ 0.26359714]], dtype=np.float64).reshape(-1)


# rho = 0.5 * np.ones(K, np.float64)
# rho = 0.05 * np.ones(K, np.float64)
# rho = 0.55 * np.ones(K, np.float64)
# Theta = Theta = {'arrival': lmbda,
#         		 'branch':  delta,
#         		 'observ':  rho}

y = np.array([23, 81])#, 86, 40, 102, 51, 72])
K = y.shape[0]
lmbda, r, rho = np.array([  9.11688589e+02,   1.06281698e+00,   5.66221059e-02])
Theta = {'arrival': np.array([lmbda] + [0] * (K - 1)),
         'branch': np.array([r] * (K - 1)),
         'observ': np.array([rho] * K)}

arrival_name = 'poisson'
# arrival_pgf  = poisson_utppgf_cython
arrival_pgf = poisson_pgf
# branch_name = 'poisson'
# branch_pgf  = poisson_utppgf_cython
# branch_name = 'bernoulli'
# branch_pgf  = bernoulli_utppgf_cython
branch_pgf = geometric_pgf

# alpha_cython, logZ_cython = UTPPGFFA_cython.utppgffa_cython(y,
# 														  	arrival_name,
# 															lmbda,
# 											  				branch_name,
# 															delta,
# 											  				rho,
# 											  				d=4)
# ll_cython = np.log(alpha_cython[0]) + np.sum(logZ_cython)

Alpha_reg = UTPPGFFA.utppgffa(y,
									  	Theta,
									  	arrival_pgf,
									  	branch_pgf,
									  	None,
									  	d=100)
									  	# normalized=True)
# ll_reg = np.log(Alpha_reg[-1][0]) + np.sum(logZ_reg)

True

#97.3321797998 [ 1.00158457  0.50000335]
#1e+12 [ 1.00158456  0.50000336]