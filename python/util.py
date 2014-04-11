import numpy as np
from scipy.linalg import block_diag
import scipy.sparse as sps
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
