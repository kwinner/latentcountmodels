from UTPPGFFA_phmm import *

import time

def utppgffa_vs_matlab(resultdir):
    # read in the data saved out from a matlab experiment
    data = scipy.io.loadmat(os.path.join(resultdir, "data.mat"))
    y = data.get('y')
    Lambda = data.get('lambda')
    Rho = numpy.squeeze(data.get('rho'))
    Delta = numpy.squeeze(data.get('delta'))

    print y.shape
    print Lambda.shape
    print Rho.shape
    print Delta.shape

    # dimensions
    # nN and nRho were the grid search dimensions
    # nIter is how many times each experiment was repeated
    # K is the number of observations in each sample
    [nN, nRho, nIter, K] = y.shape

    # repeat each experiment using UTPPGFFA
    runtime = numpy.zeros((nN, nRho, nIter))
    for iN in range(0, nN):  # indexing through rows of lambda
        for iRho in range(0, nRho):
            for iIter in range(0, nIter):
                print "Experiment %d of %d, iter %d of %d" % ((iRho + 1) + iN * nRho,
                                                              nN * nRho,
                                                              iIter,
                                                              nIter)
                t_start = time.clock()
                [Alpha, Gamma, Psi] = UTP_PGFFA_phmm(y[iN, iRho, iIter,],
                                                     Lambda[iN,],
                                                     Delta,
                                                     Rho[iRho] * numpy.ones(K),
                                                     d=1)
                runtime[iN, iRho, iIter] = time.clock() - t_start
    return runtime


resultdir = "/Users/kwinner/Work/Data/Results/20170111T134734854"
runtime = utppgffa_vs_matlab(resultdir)
scipy.io.savemat(os.path.join(resultdir, "utppgffaruntime.mat"),
                 {'runtime': runtime})