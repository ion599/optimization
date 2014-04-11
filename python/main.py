import solvers
from util import PROB_SIMPLEX, EQ_CONSTR_ELIM, \
        L_BFGS, ADMM, SPG, load_data
import util
import numpy as np
from numpy import ones, array
import argparse
import logging

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

    # Sample usage
    #P = A.T.dot(A)
    #q = A.T.dot(b)
    #solvers.qp2(P, q, block_sizes=block_sizes, constraints={PROB_SIMPLEX}, \
    #        reduction=EQ_CONSTR_ELIM, method=L_BFGS)

    print block_sizes.shape
    print block_sizes
    x0 = util.block_e(block_sizes - 1, block_sizes)
    logging.info("Computing AN")
    AN = util.AN(A, N)
    logging.info("Computation complete")
    q = -(AN.T.dot(b-np.squeeze(A.dot(x0))))
    logging.debug('q has %d dimensions' % q.ndim)
    logging.info("Computing AN'AN")
    #P = AN.T.dot(AN)
    logging.info("Computation complete")
    solvers.qp2(AN, q, block_sizes=block_sizes, constraints={PROB_SIMPLEX}, \
            method=SPG, reduction=EQ_CONSTR_ELIM)

if __name__ == "__main__":
    main()
