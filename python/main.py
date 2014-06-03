import solvers
import scipy.io as sio
from util import PROB_SIMPLEX, EQ_CONSTR_ELIM, \
        L_BFGS, ADMM, SPG, load_data
import util
import numpy as np
import numpy.linalg as la
from numpy import ones, array
from c_extensions import simplex_projection
from projection import pysimplex_projection, proj_PAV, proj_l1ball
import matplotlib.pyplot as plt
import argparse
import logging
import operator
import BB, LBFGS

ACCEPTED_LOG_LEVELS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'WARN']

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='Data file (*.mat)',
                        default='data/stevesSmallData.mat')
    parser.add_argument('--log', dest='log', nargs='?', const='INFO',
            default='WARN', help='Set log level (default: WARN)')
    parser.add_argument('--solver',dest='solver',type=str,default='LBFGS',
            help='Solver name')
    return parser

def main():
    p = parser()
    args = p.parse_args()
    if args.log in ACCEPTED_LOG_LEVELS:
        logging.basicConfig(level=eval('logging.'+args.log))

    # load data
    A, b, N, block_sizes, x_true = load_data(args.file)
    sio.savemat('fullData.mat', {'A':A,'b':b,'N':block_sizes,'N2':N,
        'x_true':x_true})

    x_true = np.squeeze(np.asarray(x_true))

    # Sample usage
    #P = A.T.dot(A)
    #q = A.T.dot(b)
    #solvers.qp2(P, q, block_sizes=block_sizes, constraints={PROB_SIMPLEX}, \
    #        reduction=EQ_CONSTR_ELIM, method=L_BFGS)

    logging.debug("Blocks: %s" % block_sizes.shape)
    x0 = util.block_e(block_sizes - 1, block_sizes)
    target = b-np.squeeze(A.dot(x0))

    options = { 'max_iter': 200,
                'verbose': 1,
                'suff_dec': 0.003, # FIXME unused
                'corrections': 500 } # FIXME unused
    AT = A.T.tocsr()
    NT = N.T.tocsr()

    target = A.dot(x0) - b

    f = lambda z: 0.5 * la.norm(A.dot(N.dot(z)) + target)**2
    nabla_f = lambda z: NT.dot(AT.dot(A.dot(N.dot(z)) + target))

    def proj(x):
        projected_value = simplex_projection.simplex_projection(block_sizes - 1,x)
        #projected_value = projection.pysimplex_projection(block_sizes - 1,x)
        return projected_value

    z0 = np.zeros(N.shape[1])
    if args.solver == 'LBFGS':
        logging.debug('Starting LBFGS solver...')
        iters,times,state = LBFGS.solve(z0 + 1, f, nabla_f, solvers.stopping,
                proj=proj, options=options)
        logging.debug("Took %s time" % str(np.sum(times)))
        logging.debug('Stopping LBFGS solver...')
    elif args.solver == 'BB':
        logging.debug('Starting BB solver...')
        iters,times,state  = BB.solve(z0, f, nabla_f, solvers.stopping,
                proj=proj, options=options)
        logging.debug('Stopping BB solver...')
    elif args.solver == 'DORE':
        progress = {}

        def diagnostics(value, iter_):
            progress[iter_] = la.norm(A.dot(N.dot(value)) - target, 2)


        lsv = util.lsv_operator(A, N)
        logging.info("Largest singular value: %s" % lsv)
        A_dore = A*0.99/lsv
        target_dore = target*0.99/lsv

        logging.debug('Starting DORE solver...')
        z, dore_time = util.timer(lambda: solvers.least_squares(lambda z: \
                A_dore.dot(N.dot(z)), lambda b: N.T.dot(A_dore.T.dot(b)), \
                target_dore, proj, z0, diagnostics=diagnostics,options=options))
        logging.debug('Stopping DORE solver...')
        print 'Time (DORE):', float(dore_time)
        xDORE = np.squeeze(np.asarray(N.dot(z) + x0))
        x_init = np.squeeze(np.asarray(x0))

        logging.debug("Shape of x_init: %s" % repr(x_init.shape))
        logging.debug("Shape of xDORE: %s" % repr(xDORE.shape))

        starting_error = la.norm(A.dot(x_init)-b)
        training_error = la.norm(A.dot(xDORE)-b)
        dist_from_true = np.max(np.abs(xDORE-x_true))
        start_dist_from_true = np.max(np.abs(x_true-x_init))

        A_dore = None

        print 'norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\nmax|x_init-x_true|: %.2f\nCV error: %8.5e\n\n\n' % \
            (training_error, starting_error, dist_from_true,
            start_dist_from_true, 0)
        plt.figure()
        plt.hist(xDORE)

        progress = sorted(progress.iteritems(), key=operator.itemgetter(0))
        plt.figure()
        plt.plot([p[0] for p in progress], [p[1] for p in progress])
        plt.show()

if __name__ == "__main__":
    main()
