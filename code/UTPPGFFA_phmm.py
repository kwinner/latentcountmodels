# a collection of UTP-PGFFA methods specific to PHMM models

import numpy
import scipy
import scipy.misc
import scipy.io
from algopy import UTPM

from UTPPGF_util import *


def UTP_Reverse_phmm(s_K, K, Rho, Delta):
    S = numpy.zeros(K);
    S[K - 1] = s_K
    U = numpy.zeros(K)
    V = numpy.zeros(K)
    for k in range(K - 1, -1, -1):
        V[k] = S[k] * (1 - Rho[k])
        U[k] = V[k]
        if k > 0:
            S[k - 1] = Delta[k - 1] * U[k] + 1 - Delta[k - 1]

    return S, U, V


def Gamma_UTP_phmm(Psi_k, lambda_k, V_k):
    D = len(Psi_k.data)

    # setup the indeterminate variable
    v_k = UTPM(numpy.zeros((D, 1)))
    v_k.data[0, 0] = V_k

    if D > 1:
        v_k.data[1, 0] = 1

    Gamma_k = Psi_k * numpy.exp(lambda_k * (v_k - 1))

    return Gamma_k


def Alpha_UTP_phmm(Gamma_k, rho_k, y_k, S_k):
    D = len(Gamma_k.data)
    D_prime = D - y_k

    # setup the indeterminate variable
    s_k = UTPM(numpy.zeros((D_prime, 1)))
    s_k.data[0, 0] = S_k

    if D_prime > 1:
        s_k.data[1, 0] = 1

    # initialize a dummy UTP for Alpha
    Alpha_k = UTPM(numpy.zeros((D_prime, 1)))

    Alpha_k.data[:, 0] = UTP_deriv(Gamma_k.data[:, 0], y_k)

    Alpha_k.data[:, 0] = Alpha_k.data[:, 0] * numpy.power(1 - rho_k, numpy.arange(D_prime))

    Alpha_k = Alpha_k * 1. / scipy.misc.factorial(y_k) * numpy.power(s_k * rho_k, y_k)

    return Alpha_k


# j subscript here is equivalent to k-1
def Psi_UTP_phmm(Alpha_j, delta_j, U_k):
    D = len(Alpha_j.data)

    # setup the indeterminate variable
    u_k = UTPM(numpy.zeros((D, 1)))
    u_k.data[0, 0] = U_k

    if D > 1:
        u_k.data[1, 0] = 1

    # chain rule
    Psi_k = Alpha_j.copy()
    Psi_k.data[:, 0] = Psi_k.data[:, 0] * numpy.power(delta_j, numpy.arange(D))

    return Psi_k


def UTP_PGFFA_phmm(y, Lambda, Delta, Rho, d=1):
    assert d >= 1

    K = len(y)
    Y = sum(y)

    S, U, V = UTP_Reverse_phmm(1, K, Rho, Delta)

    # allocate/wipe message storage
    Gamma = [None] * K
    Alpha = [None] * K
    Psi = [None] * K

    # initialize first Psi (survivor) message to 1
    Psi[0] = UTPM(numpy.zeros((Y + d, 1)))
    Psi[0].data[0, 0] = 1

    for k in range(0, K):
        Gamma[k] = Gamma_UTP_phmm(Psi[k], Lambda[k], V[k])

        Alpha[k] = Alpha_UTP_phmm(Gamma[k], Rho[k], y[k], S[k])

        if k < K - 1:
            Psi[k + 1] = Psi_UTP_phmm(Alpha[k], Delta[k], U[k])

    return Alpha, Gamma, Psi
