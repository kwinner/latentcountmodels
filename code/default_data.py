import numpy

y = numpy.array([3,4,5])
Lambda = numpy.array([8, 10, 12])
Delta = numpy.array([0.6, 0.4])
Rho = numpy.array([0.8, 0.8, 0.8])


# y = numpy.array([6,8,10,6,8,10,6,8,10])
# Lambda = numpy.array([16, 20, 24, 16, 20, 24, 16, 20, 24])
# Delta = numpy.array([0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4])
# Rho = numpy.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])


K = len(y)
Y = sum(y)