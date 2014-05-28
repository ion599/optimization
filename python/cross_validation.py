from sklearn.cross_validation import KFold
from main import parser
import solvers
import scipy.io as sio
from util import PROB_SIMPLEX, EQ_CONSTR_ELIM, \
        L_BFGS, ADMM, SPG, load_data
import util
import numpy as np
import numpy.linalg as la
from numpy import ones, array
from proj_PAV import simplex_projection
import matplotlib.pyplot as plt
import argparse
import logging
import operator
import BB, LBFGS

def cross_validation(k=3):
    p = parser()
    args = p.parse_args()

    # load data
    A, b, N, block_sizes, x_true = load_data(args.file)

    n = np.size(b)
    x_true = np.squeeze(np.asarray(x_true))

    logging.debug("Blocks: %s" % block_sizes.shape)
    x0 = util.block_e(block_sizes - 1, block_sizes)
    target = b-np.squeeze(A.dot(x0))

    options = { 'max_iter': 500,
                'verbose': 1,
                'suff_dec': 0.003, # FIXME unused
                'corrections': 500 } # FIXME unused

    proj = lambda x: simplex_projection(block_sizes - 1,x)
    z0 = np.zeros(N.shape[1])

    kf = KFold(n,n_folds=k, indices=True)
    cv_errors = []
    for train,test in kf:
        # Setup
        b_train = b[train]
        A_train = A[train,:]
        b_test = b[test]
        A_test = A[test,:]

        AN = A_train.dot(N)
        f = lambda z: 0.5 * la.norm(AN.dot(z) + A_train.dot(x0) - b_train)**2
        nabla_f = lambda z: AN.T.dot(A_train.dot(x0)+AN.dot(z)-b_train)

        # Solve
        if args.solver == 'LBFGS':
            logging.debug('Starting LBFGS solver...')
            iters,times,state = LBFGS.solve(z0 + 1, f, nabla_f, solvers.stopping,
                    proj=proj, options=options)
            logging.debug('Stopping LBFGS solver...')
        elif args.solver == 'BB':
            logging.debug('Starting BB solver...')
            iters,times,state  = BB.solve(z0, f, nabla_f, solvers.stopping,
                    proj=proj, options=options)
            logging.debug('Stopping BB solver...')

        # Post processing
        x_hat = np.squeeze(np.asarray(N.dot(state[-1]) + x0))
        x_init = np.squeeze(np.asarray(x0))

        logging.debug("Shape of x_init: %s" % repr(x_init.shape))
        logging.debug("Shape of x_hat: %s" % repr(x_hat.shape))

        starting_error = la.norm(A_train.dot(x_init)-b_train)
        training_error = la.norm(A_train.dot(x_hat)-b_train)
        test_error = la.norm(A_test.dot(x_hat)-b_test)
        dist_from_true = np.max(np.abs(x_hat-x_true))
        start_dist_from_true = np.max(np.abs(x_true-x_init))
        cv_errors.append(test_error)

        print 'norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\nmax|x_init-x_true|: %.2f\nCV error: %8.5e\n\n\n' % \
            (training_error, starting_error, dist_from_true,
            start_dist_from_true, test_error)
    print 'cv error: %8.5e' % np.mean(cv_errors)

if __name__ == "__main__":
    cross_validation()
