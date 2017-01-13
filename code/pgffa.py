import numpy as np
from scipy.misc import factorial

def pgf_forward(lmbda, rho, delta, y):
	"""
	lmbda : new arrival rates, len(lmbda) = K
	rho   : detection probabilities, len(rho) = K
	delta : survival probabilities, len(delta) = K - 1
	y     : observations, len(y) = K
	"""
	K = len(y)
	a = 0
	b = 0
	f = np.poly1d(1)

	for k in range(K):
		a, b = arrivals(a, b, lmbda[k])
		a, f = evidence(a, f, y[k], rho[k])
		if k < K - 1:
			a, b, f = survivors(a, b, f, delta[k])

	return np.polyval(f, 1) * np.exp(a + b)

def arrivals(a, b, lmbda):
	a_prime = a + lmbda
	b_prime = b - lmbda
	return a_prime, b_prime

def evidence(a, f, y, rho):
	a_prime = a * (1 - rho)
	g = np.poly1d(0)
	df = f

	for l in range(y + 1):
		g = g + df / (np.power(a, l) * factorial(l) * factorial(y - l))
		df = np.polyder(df)

	g = np.polyval(g, np.poly1d([1-rho, 0])) # g(s(1 - rho))
	g = g * np.power(a*rho, y) * np.poly1d([1] + [0]*y)

	return a_prime, g

def survivors(a, b, f, delta):
	a_prime = a * delta
	b_prime = b + a*(1 - delta)
	g = np.polyval(f, np.poly1d([delta, 1 - delta])) # f(delta * s + 1 - delta)

	return a_prime, b_prime, g


y = np.array([3,4,5])
Lambda = np.array([8, 10, 12])
Delta = np.array([0.6, 0.4])
Rho = np.array([0.8, 0.8, 0.8])
print pgf_forward(Lambda, Rho, Delta, y)
