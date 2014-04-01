from numpy import ones,all,zeros,equal,vstack
from scipy.linalg import block_diag

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
    return all(equal(x,y))

# Check if ndarray consists of all ones
def is_ones(x):
    return all(x == ones(x.shape))

# Return a vector of size n with a 1 in the ith position
def e(i,n, val=1):
    x = zeros((n,1))
    x[i] = val
    return x

# Returns a vector with blocks of e, given a vector of sizes N and positions I
def block_e(I,N):
    return vstack([e(i,n) for (i,n) in zip(I,N)])

def J():
    # TODO
    pass

def block_J(Js):
    return block_diag(Js)

# Convenience functions
# -----------------------------------------------------------------------------
def AN(A,block_sizes):
    # TODO port from preADMM.m (lines 3-21)
    pass

def U(block_sizes):
    pass

