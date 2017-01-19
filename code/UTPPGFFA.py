# a collection of UTP-PGFFA methods specific to PHMM models

import numpy
import scipy
import scipy.misc
import scipy.io
from algopy import UTPM

from UTPPGF_util import *

def UTP_Reverse(s_K, K, branch_pgf, observ_pgf, Theta):
    S = numpy.zeros(K);
    S[K - 1] = s_K
    U = numpy.zeros(K)
    V = numpy.zeros(K)
    for k in range(K - 1, -1, -1):
        V[k] = S[k] * (1 - Theta['observ'][k]) #hardcoded for now, needs to be tied to observ_pgf still
        U[k] = V[k]
        if k > 0:
            S[k - 1] = branch_pgf(U[k], Theta['branch'][k-1])

    return S, U, V


def Gamma_UTP(V_k, Psi_k, arrival_pgf, theta):
    D = len(Psi_k.data)

    # setup the indeterminate variable
    v_k = UTPM(numpy.zeros((D, 1)))
    v_k.data[0, 0] = V_k

    if D > 1:
        v_k.data[1, 0] = 1

    Gamma_k = Psi_k * arrival_pgf(v_k, theta)

    return Gamma_k


#not really changed yet, still need to work out this one
def Alpha_UTP(S_k, Gamma_k, y_k, observ_pgf, theta):
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

    Alpha_k.data[:, 0] = Alpha_k.data[:, 0] * numpy.power(1 - theta, numpy.arange(D_prime))

    Alpha_k = Alpha_k * 1. / scipy.misc.factorial(y_k) * numpy.power(s_k * theta, y_k)

    return Alpha_k


# j subscript here is equivalent to k-1
def Psi_UTP(U_k, Alpha_j, branch_pgf, theta):
    D = len(Alpha_j.data)

    # setup the indeterminate variable
    u_k = UTPM(numpy.zeros((D, 1)))
    u_k.data[0, 0] = U_k

    if D > 1:
        u_k.data[1, 0] = 1

    # compose Alpha_j, branch_pgf in UTP form
    Psi_k = UTPPGF_compose(Alpha_j, branch_pgf(u_k, theta))

    return Psi_k


def UTP_PGFFA(y, Theta, arrival_pgf, branch_pgf, observ_pgf, d=1):
    assert d >= 1

    K = len(y)
    Y = sum(y)

    # S, U, V = UTP_Reverse(1, K, Theta, branch_pgf, observ_pgf)
    S, U, V = UTP_Reverse(1, K, branch_pgf, observ_pgf, Theta)

    # allocate/wipe message storage
    Gamma = [None] * K
    Alpha = [None] * K
    Psi = [None] * K

    # initialize first Psi (survivor) message to 1
    Psi[0] = UTPM(numpy.zeros((Y + d, 1)))
    Psi[0].data[0, 0] = 1

    for k in range(0, K):
        Gamma[k] = Gamma_UTP(V[k], Psi[k], arrival_pgf, Theta['arrival'][k])

        Alpha[k] = Alpha_UTP(S[k], Gamma[k], y[k], observ_pgf, Theta['observ'][k])

        if k < K - 1:
            Psi[k + 1] = Psi_UTP(U[k], Alpha[k], branch_pgf, Theta['branch'][k])

    return Alpha, Gamma, Psi