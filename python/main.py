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

ACCEPTED_LOG_LEVELS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'WARN']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Data file (*.mat)',
                        default='../data/4_6_3_2_20140311T183121_1_small_graph_OD_dense.mat')
    parser.add_argument('--log', dest='log', nargs='?', const='INFO',
            default='WARN', help='Set log level (default: WARN)')
    args = parser.parse_args()
    if args.log in ACCEPTED_LOG_LEVELS:
        logging.basicConfig(level=eval('logging.'+args.log))
    # load data

    A, b, N, block_sizes, x_true = load_data(args.file)
    sio.savemat('fullData.mat', {'A':A,'b':b,'N':block_sizes,'N2':N,'x_true':x_true})

    x_true = np.squeeze(np.asarray(x_true))

    # Sample usage
    #P = A.T.dot(A)
    #q = A.T.dot(b)
    #solvers.qp2(P, q, block_sizes=block_sizes, constraints={PROB_SIMPLEX}, \
    #        reduction=EQ_CONSTR_ELIM, method=L_BFGS)

    logging.debug("Blocks: %s" % block_sizes.shape)
    x0 = util.block_e(block_sizes - 1, block_sizes)
    target = b-np.squeeze(A.dot(x0))

    lsv = util.lsv_operator(A, N)
    logging.info("Largest singular value: %s" % lsv)
    A_dore = A*0.99/lsv
    target_dore = target*0.99/lsv

    progress = {}

    def diagnostics(value, iter_):
        progress[iter_] = la.norm(A.dot(N.dot(value)) - target, 2)

    z, dore_time = util.timer(lambda: solvers.least_squares(lambda z: A_dore.dot(N.dot(z)), lambda b: N.T.dot(A_dore.T.dot(b)), \
            target_dore, lambda x: simplex_projection(block_sizes - 1, x), \
            np.zeros(N.shape[1]), diagnostics=diagnostics))
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
        (training_error, starting_error, dist_from_true, start_dist_from_true, 0)
    plt.figure()
    plt.hist(xDORE)

    progress = sorted(progress.iteritems(), key=operator.itemgetter(0))
    plt.figure()
    plt.plot([p[0] for p in progress], [p[1] for p in progress])
    plt.show()

if __name__ == "__main__":
    main()
