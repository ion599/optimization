import scipy.sparse
import scipy.sparse.linalg
import scipy.sparse.linalg as sla
import numpy as np
import scipy.io as sio
from scipy.linalg import block_diag
import scipy.sparse as sps
import sys
import time
import scipy.io as sio
import logging

# Constraints
PROB_SIMPLEX = 'probability simplex'
# Reductions
EQ_CONSTR_ELIM = 'equality constraint elimination'
# Methods
L_BFGS = 'L-BFGS'
SPG = 'SPG'
ADMM = 'ADMM'

# Check if all entries equal
def all_equal(x,y):
    return np.all(np.equal(x,y))

# Check if ndarray consists of all ones
def is_ones(x):
    return np.all(x == np.ones(x.shape))

# Return a vector of size n with a 1 in the ith position
def e(i,n, val=1):
    x = np.zeros((n,1))
    x[i] = val
    return x

# Returns a vector with blocks of e, given a vector of sizes N and positions I
def block_e(I,N):
    return np.squeeze(np.vstack([e(i,n) for (i,n) in zip(I,N)]))

def J():
    # TODO
    pass

def block_J(Js):
    return block_diag(Js)

def block_sizes_from_U(U):
    # Sum along rows
    return np.squeeze(np.asarray(U.sum(axis=1)))

def block_sizes_to_N(block_sizes):
    """Converts a list of the block sizes to a scipy.sparse matrix.

    The matrix will start in lil format, as this is the best way to generate it,
    but can easily be converted to another format such as csr for efficient multiplication.
    I will return it in csr so that each function doesn't need to convert it itself.
    """
    block_sizes = np.squeeze(np.asarray(block_sizes))
    m = np.sum(block_sizes)
    n = m - block_sizes.shape[0]
    N = sps.lil_matrix((m, n))
    start_row = 0
    start_col = 0
    for i, block_size in enumerate(block_sizes):
        if block_size < 2:
            start_row += block_size
            start_col += block_size - 1
            continue
        for j in xrange(block_size-1):
            N[start_row+j, start_col+j] = 1
            N[start_row+j+1, start_col+j] = -1
        start_row += block_size
        start_col += block_size - 1
    return N.tocsr()

# Convenience functions
# -----------------------------------------------------------------------------

def is_sparse_matrix(A):
    return not sps.sputils.isdense(A)

def load_data(filename):
    logging.debug('Loading %s...' % filename)
    data = sio.loadmat(filename)

    logging.debug('Unpacking...')
    if data.has_key('phi'):
        A = data['phi']
    else:
        A = data['A']

    if data.has_key('b'):
        b = data['b']
    else:
        b = data['f']
    b = np.squeeze(np.asarray(b))

    if data.has_key('block_sizes'):
        block_sizes = data['block_sizes']
    elif data.has_key('U'):
        block_sizes = block_sizes_from_U(data['U']).astype(int)

    if data.has_key('x'):
        x_true = data['x']
    else:
        x_true = data['real_a']

    logging.debug('Creating sparse N matrix')
    N = block_sizes_to_N(block_sizes)

    logging.debug('File loaded successfully')

    return (A, b, N, block_sizes, x_true)

def AN(A,N):
    # TODO port from preADMM.m (lines 3-21)
    return A.dot(N)

# Do we ever use U?
def U(block_sizes):
    pass

def lsv_operator(A, N):
    """Computes largest singular value of AN
    
    Computation is done without computing AN or (AN)^T(AN)
    by using functions that act as these linear operators on a vector
    """

    # Build linear operator for AN
    def matmuldyad(v):
        return A.dot(N.dot(v))

    def rmatmuldyad(v):
        return N.T.dot(A.T.dot(v))
    normalized_lin_op = scipy.sparse.linalg.LinearOperator((A.shape[0], N.shape[1]), matmuldyad, rmatmuldyad)

    # Given v, computes (N^TA^TAN)v
    def matvec_XH_X(v):
        return normalized_lin_op.rmatvec(normalized_lin_op.matvec(v))

    which='LM'
    v0=None
    maxiter=None
    return_singular_vectors=False

    # Builds linear operator object
    XH_X = scipy.sparse.linalg.LinearOperator(matvec=matvec_XH_X, dtype=A.dtype, shape=(N.shape[1], N.shape[1]))
    # Computes eigenvalues of (N^TA^TAN), the largest of which is the LSV of AN
    eigvals = sla.eigs(XH_X, k=1, tol=0, maxiter=None, ncv=10, which=which, v0=v0, return_eigenvectors=False)
    lsv = np.sqrt(eigvals)
    # Take largest one
    return lsv[0].real

def timer(func, number= 1):
    '''
    Output the average time
    '''
    total = 0
    for _ in xrange(number):
        if sys.platform == "win32":
            t = time.clock
        else:
            t = time.time
        start = t()
        output = func()
        end = t()
        total += end - start

    return output, total / number
