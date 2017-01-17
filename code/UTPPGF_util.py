import numpy
from algopy import UTPM


def UTPPGF_mean(F):
    return F.data[1, 0]


def UTPPGF_var(F):
    return (2 * F.data[2, 0]) - numpy.power(F.data[1, 0], 2) + F.data[1, 0]


# compose two PGFs represented as UTPs
# Function returns H = G(F(s)), truncated to be a UTP of degree d
#  if d = 'F' or 'G', then the degree will match the size of the corresponding input
#  if d = 'FG', then the max of the two will be used
#  if d = None or d < 0, then the UTP will not be truncated
# note: the actual value of G(F(-)) cannot be computed this way, and the actual value of G is used instead
def UTPPGF_compose(G, F, d='G'):
    # representation note: numpy.poly1d objects store coefficients in decreasing order
    #                      algopy.UTPM  objects store coefficients in increasing order
    #                      UTP.data[::-1] or np.array(poly1d)[::-1] reverses them using a view
    if d == 'G':
        d = G.data.shape[0]
    elif d == 'F':
        d = F.data.shape[0]
    elif d == 'FG':
        d = max(G.data.shape[0], F.data.shape[0])

    # extract the nonconstant terms as vectors
    Gbar = numpy.squeeze(G.data.copy())
    Gbar[0] = 0
    Fbar = numpy.squeeze(F.data.copy())
    Fbar[0] = 0

    H = numpy.polyval(Gbar[::-1],
                      numpy.poly1d(Fbar[::-1])) #F must be passed as a poly1d object or it is treated as a set of points to eval at

    # convert H back to a UTP
    H = numpy.array(H)[::-1]

    # reinsert the actual value of G (hopefully storing G(F(-))...)
    # H = numpy.insert(H, 0, G.data[0])
    H[0] = G.data[0]

    print G.data[0]
    print H

    if d == None or d < 0:
        H = UTPM(H.reshape(-1, 1))
    else:
        H = UTPM(H.reshape(-1, 1)[0:d])

    return H
