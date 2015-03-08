import ipdb

import scipy.sparse
import scipy.sparse.linalg
import scipy.sparse.linalg as sla
import numpy as np
import numpy.linalg as la
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

# Clean array wrapper
def array(x):
    return np.squeeze(np.array(x))

# Clean sparse matrix wrapper
def sparse(A):
    if type(A) == np.ndarray:
        return sps.csr_matrix(A)
    return A.tocsr()

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

def get_block_sizes(U):
    # Sum along rows
    return array(U.sum(axis=1)).astype(int)

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
    return sparse(N)

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

def load_weights(filename,block_sizes,weight=1):
    import pickle
    with open(filename) as f:
        data = pickle.load(f)
    D = np.array([v for (i,v) in data])
    # normalize weights
    blocks_end = np.cumsum(block_sizes)
    blocks_start = np.hstack((0,blocks_end[:-1]))
    blocks = [D[s:e] for s,e in np.vstack((blocks_start,blocks_end)).T]
    blocks = [b/sum(b) for b in blocks]
    return weight*np.array([e for b in blocks for e in b])

def assert_simplex_incidence(M,n):
    """
    1. Check that the width of the matrix is correct.
    2. Check that each column sums to 1
    3. Check that there are exactly n nonzero values
    :param M:
    :param n:
    :return:
    """
    assert M.shape[1] == n, 'Incidence matrix: wrong size'
    assert (M.sum(axis=0)-1).any() == False,\
        'Incidence matrix: columns should sum to 1'
    assert M.nnz == n, 'Incidence matrix: should be n nonzero values'

def assert_scaled_incidence(M):
    """
    Check that all column entries are either 0 or the same entry value

    :param M:
    :return:
    """
    m,n = M.shape
    col_sum = M.sum(axis=0)
    col_nz = (M > 0).sum(axis=0)
    entry_val = np.array([0 if M[:,i].nonzero()[0].size == 0 else \
                              M[M[:,i].nonzero()[0][0],i] for i in range(n)])
    assert (np.abs(array(col_sum) - array(col_nz) * entry_val) < 1e-10).all(), \
        'Not a proper scaled incidence matrix, check column entries'

def load_x_true(data):
    if 'x_true' in data:
        return array(data['x_true'])
    if 'x' in data:
        return array(data['x'])

def load_data(filename,full=False,OD=False,CP=False,eq=None):
    """
    Load data from file about network state

    Notation:
    x_true = route flow
    x_split = route split

    :param filename:
    :param full: Use A_full, b_full instead of A,b
    :param OD: Extract information from T
    :param CP: Extract information from U
    :param eq: None uses block_sizes to generate equality constraint; OD uses
                T to generate equality constraint; CP uses U
    :return:
    """
    print filename
    logging.debug('Loading %s...' % filename)
    data = sio.loadmat(filename)
    logging.debug('Unpacking...')

    # Link-route and route
    # FIXME deprecate use of key 'x'
    if full and 'A_full' in data and 'b_full' in data and 'x_true' in data:
        x_true = array(data['x_true'])
        A = sparse(data['A_full'])
        b = array(data['b_full'])
    elif 'A' in data and 'b' in data:
        x_true = load_x_true(data)
        A = sparse(data['A'])
        b = array(data['b'])
    elif 'phi' in data and 'b' in data and 'real_a' in data:
        x_true = array(data['real_a'])
        A = sparse(data['phi'])
        b = array(data['b'])
    #assert_scaled_incidence(A)

    # Remove rows of zeros (unused sensors)
    nz = [i for i in xrange(A.shape[0]) if A[i,:].nnz == 0]
    nnz = [i for i in xrange(A.shape[0]) if A[i,:].nnz > 0]
    A, b = A[nnz,:], b[nnz]
    assert la.norm(A.dot(x_true) - b) < 1e-3, 'Check data input: Ax != b'

    n = x_true.shape[0]
    # OD-route
    if OD and 'T' in data and 'd' in data:
        T,d = sparse(data['T']), array(data['d'])
        assert_simplex_incidence(T, n) # ASSERT
    # Cellpath-route
    if CP and 'U' in data and 'f' in data:
        U,f = sparse(data['U']), array(data['f'])
        assert_simplex_incidence(U, n) # ASSERT

    # Reorder routes by blocks of flow, e.g. OD flow or waypoint flow given by U
    if data.has_key('block_sizes'):
        eq = None
        block_sizes = array(data['block_sizes'])
        rsort_index = None
    else:
        W = T if eq == 'OD' else U
        block_sizes = get_block_sizes(W)
        rank = W.nonzero()[0]
        sort_index = np.argsort(rank)

        if CP and 'U' in data:
            U = U[:,sort_index] # reorder
        if OD and 'T' in data:
            T = T[:,sort_index] # reorder
        A = A[:,sort_index] # reorder
        x_true = x_true[sort_index] # reorder
        rsort_index = np.argsort(sort_index) # revert sort

    logging.debug('Creating sparse N matrix')
    N = block_sizes_to_N(block_sizes)

    logging.debug('File loaded successfully')

    # Scale matrices by block
    print la.norm(A.dot(x_true) - b)
    if eq == 'OD' and 'T' in data:
        scaling =  T.T.dot(T.dot(x_true))
        x_split = x_true / scaling
        DT = sps.diags([scaling],[0])
        A = A.dot(DT)
        if CP and 'U' in data:
            U = U.dot(DT)
            AA,bb = sps.vstack([A,U]), np.concatenate((b,f))
        else:
            AA,bb = A,b
    elif eq == 'CP' and 'U' in data:
        scaling =  U.T.dot(U.dot(x_true))
        x_split = x_true / scaling
        DU = sps.diags([scaling],[0])
        A = A.dot(DU)
        if OD and 'T' in data:
            T = T.dot(DU)
            AA,bb = sps.vstack([A,T]), np.concatenate((b,d))
        else:
            AA,bb = A,b
    else:
        x_split = x_true
        # TODO what is going on here????
        #scaling = array(A.sum(axis=0)/(A > 0).sum(axis=0))
        #scaling[np.isnan(scaling)]=0 # FIXME this is not accurate
        scaling = f
        AA,bb = A,b
    assert la.norm(A.dot(x_split) - b) < 1e-3, 'Improper scaling: Ax != b'

    return (AA, bb, N, block_sizes, x_split, nz, scaling, rsort_index)

def AN(A,N):
    # TODO port from preADMM.m (lines 3-21)
    return A.dot(N)

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
    normalized_lin_op = scipy.sparse.linalg.LinearOperator((A.shape[0],
                                                            N.shape[1]),
                                                           matmuldyad,
                                                           rmatmuldyad)

    # Given v, computes (N^TA^TAN)v
    def matvec_XH_X(v):
        return normalized_lin_op.rmatvec(normalized_lin_op.matvec(v))

    which='LM'
    v0=None
    maxiter=None
    return_singular_vectors=False

    # Builds linear operator object
    XH_X = scipy.sparse.linalg.LinearOperator(matvec=matvec_XH_X, dtype=A.dtype,
                                              shape=(N.shape[1], N.shape[1]))
    # Computes eigenvalues of (N^TA^TAN), the largest of which is the LSV of AN
    eigvals = sla.eigs(XH_X, k=1, tol=0, maxiter=None, ncv=10, which=which,
                       v0=v0, return_eigenvectors=False)
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
                   np.concatenate([np.sum(x1[i:j])*np.ones((k,1)) for i, j, k \
                                   in zip(ind_start,ind_end,block_sizes)]))
    
    tmp = np.concatenate([np.argsort(np.argsort(x_true[i:j])) for i,j in \
                          zip(ind_start,ind_end)]) + 1
    x2 = np.divide(tmp, \
                   np.squeeze(np.concatenate([np.sum(tmp[i:j])*np.ones((k,1)) \
                                              for i,j,k in zip(ind_start,
                                                               ind_end,
                                                               block_sizes)])))
    tmp = np.power(10, tmp)
    x3 = np.divide(tmp, \
                   np.squeeze(np.concatenate([np.sum(tmp[i:j])*np.ones((k,1)) \
                                              for i,j,k in zip(ind_start,
                                                               ind_end,
                                                               block_sizes)])))
    x4 = np.concatenate([(1./k)*np.ones((k,1)) for k in block_sizes])
    
    z1 = x2z(x1, block_sizes)
    z2 = x2z(x2, block_sizes)
    z3 = x2z(x3, block_sizes)
    z4 = x2z(x4, block_sizes)
    
    return x1,x2,x3,x4,z1,z2,z3,z4

def mask(arr):
    k = len(arr)
    size = np.max([len(arr[i]) for i in range(k)])
    masked = np.ma.empty((size,k))
    masked.mask = True
    for i in range(k):
        masked[:len(arr[i]),i] = np.array(arr[i])
    return masked

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
