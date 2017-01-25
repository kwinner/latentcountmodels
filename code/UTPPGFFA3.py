# a collection of UTP-PGFFA methods specific to PHMM models

import numpy
import scipy
import scipy.misc
import scipy.io
from algopy import UTPM

from UTPPGF_util import *


def UTP_Reverse(s_K, K, D, branch_pgf, observ_pgf, Theta):
    # initialize storage for return values
    # S_dU and U_dS are UTPs wrt the previous variable
    S_dU = [None] * K
    U_dS = [None] * K
    # S_dS and U_dU are UTPs wrt the same variable
    # importantly, S_dS and S_dU have the same first value, but the derivatives of S_dS are fixed
    # and analogously U_dU and U_dS are closely related
    S_dS = [None] * K
    U_dU = [None] * K

    # initialize the first indeterminate to the initial value provided
    S_dU[K - 1] = UTPM(numpy.zeros((D, 1)))
    S_dU[K - 1].data[0,0] = s_K
    S_dU[K - 1].data[1,0] = 1
    # create copy of S_dU[K - 1] whose derivatives are taken wrt itself
    S_dS[K - 1] = UTPM(numpy.zeros((D,1)))
    S_dS[K - 1].data[0,0] = S_dU[K - 1].data[0,0]
    S_dS[K - 1].data[1,0] = 1

    for k in range(K - 1, -1, -1):
        # compute U_dS[k]
        U_dS[k] = S_dS[k] * (1 - Theta['observ'][k]) #hardcoded for now, needs to be tied to observ_pgf still

        # create copy of U_dS[k] whose derivatives are taken wrt itself
        U_dU[k] = UTPM(numpy.zeros((D,1)))
        U_dU[k].data[0,0] = U_dS[k].data[0,0]
        U_dU[k].data[1,0] = 1

        if k > 0:
            # compute S_dU[k - 1]
            S_dU[k - 1] = branch_pgf(U_dU[k], Theta['branch'][k-1])

            # create copy of S_dU[k] whose derivatives are taken wrt itself
            S_dS[k - 1] = UTPM(numpy.zeros((D,1)))
            S_dS[k - 1].data[0,0] = S_dU[k - 1].data[0,0]
            S_dS[k - 1].data[1,0] = 1

    return S_dU, U_dS, S_dS, U_dU


def alpha_k_ds_k(alpha_j_ds_j, y_k, s_j_du_k, s_k_ds_k, u_k_ds_k, u_k_du_k, arrival_pgf, theta_arrival_k, rho_k):
    # compose the input message with <s_j, du_k>
    alpha_j_du_k = UTPPGF_compose(alpha_j_ds_j, s_j_du_k)

    D = alpha_j_du_k.data.shape[0]

    # truncate u_k_du_k to match the length of alpha_j_du_k
    u_k_du_k.data = u_k_du_k.data[:D]

    # compute beta
    beta_k_du_k = alpha_j_du_k * arrival_pgf(u_k_du_k, theta_arrival_k)

    # take the derivative
    gamma_k_du_k = UTPM(numpy.zeros((D - y_k, 1)))
    gamma_k_du_k.data[:,0] = UTP_deriv(beta_k_du_k.data[:,0], y_k) # UTP length is now D - y_k

    # correct for derivative wrt s_k
    gamma_k_ds_k = UTPPGF_compose(gamma_k_du_k, u_k_ds_k)

    # truncate s_k_ds_k to match length of gamma_k_ds_k (D - y_k, in this case)
    s_k_ds_k.data = s_k_ds_k.data[:D - y_k]

    # compute the return value alpha_k_ds_k
    return gamma_k_ds_k / scipy.misc.factorial(y_k) * numpy.power(s_k_ds_k * rho_k, y_k)


def alpha_0_ds_0(y_0, Y, s_0_ds_0, u_0_ds_0, u_0_du_0, arrival_pgf, theta_arrival_0, rho_0):
    # compute beta
    beta_0_du_0 = arrival_pgf(u_0_du_0, theta_arrival_0)

    # take the derivative
    gamma_0_du_0 = UTPM(numpy.zeros((Y - y_0, 1)))
    gamma_0_du_0.data[:,0] = UTP_deriv(beta_0_du_0.data[:,0], y_0)  # UTP length is now D - y_k

    # correct for derivative wrt s_k
    gamma_0_ds_0 = UTPPGF_compose(gamma_0_du_0, u_0_ds_0)

    # truncate s_0_ds_0 to match length of gamma_0_ds_0 (Y - y_0, in this case)
    s_0_ds_0.data = s_0_ds_0.data[:Y - y_0]

    # compute the return value alpha_k_ds_k
    return gamma_0_ds_0 / scipy.misc.factorial(y_0) * numpy.power(s_0_ds_0 * rho_0, y_0)


def UTP_PGFFA(y, Theta, arrival_pgf, branch_pgf, observ_pgf, d=1):
    assert d >= 1

    K = len(y)
    Y = sum(y)

    # S, U, V = UTP_Reverse(1, K, Theta, branch_pgf, observ_pgf)
    S_dU, U_dS, S_dS, U_dU = UTP_Reverse(1, K, Y + d, branch_pgf, observ_pgf, Theta)

    # allocate message storage
    Alpha = [None] * K

    # compute first Alpha message w/ special case no-survival method
    Alpha[0] = alpha_0_ds_0(y[0],
                            Y + d,
                            S_dS[0],
                            U_dS[0],
                            U_dU[0],
                            arrival_pgf,
                            Theta['arrival'][0],
                            Theta['observ'][0])

    for k in range(1, K):
        Alpha[k] = alpha_k_ds_k(Alpha[k-1],
                                y[k],
                                S_dU[k-1],
                                S_dS[k],
                                U_dS[k],
                                U_dU[k],
                                arrival_pgf,
                                Theta['arrival'][k],
                                Theta['observ'][k])

    return Alpha, None, None