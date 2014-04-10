import solvers
from util import PROB_SIMPLEX, EQ_CONSTR_ELIM, \
        L_BFGS, ADMM, SPG, load_data
import util
import numpy as np
import scipy.io as sio
from numpy import ones, array

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Data file (*.mat)',
                        default='../data/4_6_3_2_20140311T183121_1_small_graph_OD_dense.mat')
    args = parser.parse_args()
    # load data

    A, b, N, block_sizes, x_true = load_data(args.file)

    # Sample usage
    P = A.T.dot(A)
    q = A.T.dot(b)
    solvers.qp2(P, q, block_sizes=block_sizes, constraints={PROB_SIMPLEX}, \
            reduction=EQ_CONSTR_ELIM, method=L_BFGS)

    # TODO compute AN, AN^TAN
    AN = util.AN(A, N)
    P = AN.T.dot(AN)
    q = AN.T.dot(b)
    solvers.qp2(P, q, block_sizes=block_sizes, constraints={PROB_SIMPLEX}, \
            method=L_BFGS)

if __name__ == "__main__":
    main()
