import numpy as np

y = np.array([3,4,5])
Lambda = np.array([8, 10, 12])
Delta = np.array([0.6, 0.4])
Rho = np.array([0.8, 0.8, 0.8])


# y = np.array([6,8,10,6,8,10,6,8,10])
# Lambda = np.array([16, 20, 24, 16, 20, 24, 16, 20, 24])
# Delta = np.array([0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4])
# Rho = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])


K = len(y)
Y = sum(y)