from util import is_ones, all_equal, block_e, PROB_SIMPLEX, EQ_CONSTR_ELIM, \
        L_BFGS, ADMM, SPG
import numpy as np
import scipy.io as sio
from numpy import ones, array
import spg

def qp(P, q, G=None, h=None, A=None, b=None, solver=None, initvals=None):
    from cvxopt.solvers import qp
    return qp(P, q, G, h, A, b, solver, initvals)

def qp2(P, q, G=None, h=None, A=None, b=None, solver=None, initvals=None, \
        N=None, block_sizes=None, reduction=None, constraints=None, method=None):

    if block_sizes != None and PROB_SIMPLEX in constraints:
        if reduction == EQ_CONSTR_ELIM: # work in z's
            if method == L_BFGS:
                pass
            elif method == SPG:
                return spg.solver(P, q, G, h, A, b, solver, initvals, N, block_sizes, reduction, constraints)
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
