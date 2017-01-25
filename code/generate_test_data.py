import numpy as np
from lib import arrival, branching
from lib.model import NMixture, Zonneveld

"""
Script to generate data for parameter estimation
"""

def print_data(y):
    print np.array2string(y, separator=', ')

# N-mixture, Poisson arrival, binomial branching
model = NMixture(arrival.poisson)
print 'N-mixture, Poisson arrival'
lmbda, rho = 76, 0.3
print_data(model.generate_data([lmbda], rho, 10))
print model.theta2dict([lmbda, rho])
print

# N-mixture, NB arrival, binomial branching
model = NMixture(arrival.nbinom)
print 'N-mixture, NB arrival'
r, p, rho = 15, 0.15, 0.3
print_data(model.generate_data([r, p], rho, 10))
print model.theta2dict([r, p, rho])
print

# N-mixture, geometric arrival, binomial branching
model = NMixture(arrival.geom)
print 'N-mixture, geometric arrival'
print_data(model.generate_data([0.01], 0.7, 10))
print model.theta2dict([p, rho])
print

# Zonneveld, Poisson arrival, binomial branching
print 'Zonneveld, Poisson arrival, binomial branching'
model = Zonneveld(arrival.poisson, branching.binom)
mu, sigma, c, lmbda, rho = 2, 1, 300, 1.5, 0.6
T = range(0, 10, 2) + range(9, 14)
print_data(model.generate_data([mu, sigma, c], [lmbda], rho, T))
print model.theta2dict([mu, sigma, c, lmbda, rho])
print

# Zonneveld, Poisson arrival, Poisson branching
print 'Zonneveld, Poisson arrival, Poisson branching'
mu, sigma, c, lmbda, rho = 2, 1, 100, 1.2, 0.6 # if too big, try lmbda = 1.2
T = range(0, 10, 2) + range(9, 14)
model = Zonneveld(arrival.poisson, branching.poisson)
print_data(model.generate_data([mu, sigma, c], [lmbda], rho, T))
print model.theta2dict([mu, sigma, c, lmbda, rho])
print

# Zonneveld, NB arrival, binomial branching
print 'Zonneveld, NB arrival, binomial branching'
mu, sigma, c, r, lmbda, rho = 2, 1, 300, 3, 1.5, 0.8
T = range(0, 10, 2) + range(9, 14)
model = Zonneveld(arrival.nbinom, branching.binom)
print_data(model.generate_data([mu, sigma, c, r], [lmbda], rho, T))
print model.theta2dict([mu, sigma, c, r, lmbda, rho])
print

# Zonneveld, NB arrival, Poisson branching
print 'Zonneveld, NB arrival, Poisson branching'
mu, sigma, c, r, lmbda, rho = 2, 1, 200, 3, 1.1, 0.8
T = range(0, 10, 2) + range(9, 14)
model = Zonneveld(arrival.nbinom, branching.poisson)
print_data(model.generate_data([mu, sigma, c, r], [lmbda], rho, T))
print model.theta2dict([mu, sigma, c, r, lmbda, rho])
print

# Zonneveld, geom arrival, binom branching
print 'Zonneveld, geom arrival, binom branching'
mu, sigma, c, lmbda, rho = 2, 1, 300, 1.5, 0.8
T = range(0, 10, 2) + range(9, 14)
model = Zonneveld(arrival.geom, branching.binom)
print_data(model.generate_data([mu, sigma, c], [lmbda], rho, T))
print model.theta2dict([mu, sigma, c, lmbda, rho])
print

# Zonneveld, geom arrival, Poisson branching
print 'Zonneveld, geom arrival, Poisson branching'
mu, sigma, c, lmbda, rho = 2, 2, 100, 1.4, 0.7
T = range(0, 10, 2) + range(9, 14)
model = Zonneveld(arrival.geom, branching.poisson)
print_data(model.generate_data([mu, sigma, c], [lmbda], rho, T))
print model.theta2dict([mu, sigma, c, lmbda, rho])
print
