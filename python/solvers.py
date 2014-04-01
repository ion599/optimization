from util import is_ones, all_equal, block_e, PROB_SIMPLEX, EQ_CONSTR_ELIM, \
        L_BFGS, ADMM, SPG
import numpy as np
from numpy import ones, array

class solvers:
    # Reference: http://cvxopt.org/userguide/coneprog.html#quadratic-programming

    @staticmethod
    def qp(P, q, G=None, h=None, A=None, b=None, solver=None, initvals=None):
        from cvxopt.solvers import qp
        return qp(P, q, G, h, A, b, solver, initvals)

    @staticmethod
    def qp2(P, q, G=None, h=None, A=None, b=None, solver=None, initvals=None, \
            block_sizes=None, reduction=None, constraints=None, method=None):
        
        if block_sizes != None and PROB_SIMPLEX in constraints:
            if reduction == EQ_CONSTR_ELIM: # work in z's
                if method == L_BFGS:
                    pass
                elif method == SPG:
                    pass
                elif method == ADMM:
                    pass
            else: # work in x's
                x_0 = block_e(block_sizes-1,block_sizes)
                print x_0
                if method == L_BFGS:
                    pass
                elif method == SPG:
                    pass
                elif method == ADMM:
                    pass
        else:
            print "'block_sizes' parameter required"

if __name__ == "__main__":
    from scipy.io import loadmat
    # load data
    data = loadmat('../data/4_6_3_2_20140311T183121_1_small_graph_OD_dense.mat')
    A = data['phi']
    b = data['f']
    block_sizes = data['block_sizes']
    x_true = data['real_a']

    # Sample usage
    P = A.T.dot(A)
    q = A.T.dot(b)
    solvers.qp2(P, q, block_sizes=block_sizes, constraints={PROB_SIMPLEX}, \
            reduction=EQ_CONSTR_ELIM, method=L_BFGS)

    # TODO compute AN, AN^TAN
    # P = ???
    # q = ???
    solvers.qp2(P, q, block_sizes=block_sizes, constraints={PROB_SIMPLEX}, \
            method=L_BFGS)
