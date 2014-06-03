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

def block_sizes_to_x0(block_sizes):
    """Converts a list of the block sizes to a scipy.sparse vector x0
    """
    x0 = sps.dok_matrix((np.sum(block_sizes),1))
    for i in np.cumsum(block_sizes)-1: x0[(i,0)] = 1
    return x0.transpose()

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
    A = A.tocsr()

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
    N = N.tocsr()

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


def x2z(x, block_sizes):
    p = len(block_sizes)
    ind_end = np.cumsum(block_sizes)
    ind_start = np.hstack(([0],ind_end[:-1]))
    z = np.concatenate([np.cumsum(x[i:j-1]) for i,j \
                in zip(ind_start,ind_end) if i<j-1])
    return z


def init_xz(block_sizes, x_true):
    """Generate initial points
    1: random
    2: by importance (cheating-ish)
    3: 10^importance (cheating-ish)
    4: uniform
    """
    n = np.sum(block_sizes)
    x1 = np.random.random_sample((n, 1))
    ind_end = np.cumsum(block_sizes)
    ind_start = np.hstack(([0],ind_end[:-1]))
    x1 = np.divide(x1, \
                   np.concatenate([np.sum(x1[i:j])*np.ones((k,1)) for i,j,k in zip(ind_start,ind_end,block_sizes)]))
    
    tmp = np.concatenate([np.argsort(np.argsort(x_true[i:j])) for i,j in zip(ind_start,ind_end)]) + 1
    x2 = np.divide(tmp, \
                   np.squeeze(np.concatenate([np.sum(tmp[i:j])*np.ones((k,1)) for i,j,k in zip(ind_start,ind_end,block_sizes)])))
    tmp = np.power(10, tmp)
    x3 = np.divide(tmp, \
                   np.squeeze(np.concatenate([np.sum(tmp[i:j])*np.ones((k,1)) for i,j,k in zip(ind_start,ind_end,block_sizes)])))
    x4 = np.concatenate([(1./k)*np.ones((k,1)) for k in block_sizes])
    
    z1 = x2z(x1, block_sizes)
    z2 = x2z(x2, block_sizes)
    z3 = x2z(x3, block_sizes)
    z4 = x2z(x4, block_sizes)
    
    return x1,x2,x3,x4,z1,z2,z3,z4


if __name__ == "__main__":
    x = np.array([1/6.,2/6.,3/6.,1,.5,.1,.4])
    
    print "Demonstration of convenience functions (x2z, x2z)"
    block_sizes = np.array([3,1,3])
    z = x2z(x, block_sizes)
    x0 = block_sizes_to_x0(block_sizes)
    N = block_sizes_to_N(block_sizes)
    
    #print x
    #print z
    #print N.dot(z) +x0
    print init_xz(block_sizes, x)
