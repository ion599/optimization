import util
import numpy as np
from numpy.random import randint, choice
import numpy.linalg as la

epsilon = 0.001
(A, b, N, block_sizes, x_true) = util.load_data('../data/2_3_3_1_20140421T151732_1_small_graph_OD_dense.mat')

def trials(n,fn):
    dx_sum, res_sum = 0, 0
    for i in range(n):
        (dx,res) = fn()
        res_sum += res
        dx_sum += la.norm(dx,2)
    return (dx_sum/n,res_sum/n)

def mask_block(block):
    cum_block_sizes = np.cumsum(block_sizes)
    block_len = block_sizes[block][0]
    block_end = cum_block_sizes[block]
    mask = np.zeros(A.shape)
    mask[:,block_end-block_len:block_end] = np.ones((A.shape[0],block_len))
    return mask

def perturb_one_entry():
    dA = np.zeros(A.shape)
    dA[randint(0,A.shape[0]),randint(0,A.shape[1])] = epsilon
    (dx,res,rank,s) = la.lstsq(A + dA,-dA.dot(x_true))
    return (dx,res)

def perturb_one_block():
    # generate mask
    block = randint(0,len(block_sizes))
    mask = mask_block(block)
    mask = mask * A > 0
    
    # generate perturbation
    dA = mask * epsilon
    (dx,res,rank,s) = la.lstsq(A + dA,-dA.dot(x_true))
    return (dx,res)

def perturb_one_block_conserve():
    # TODO caution that the real problem will have more structure than this
    # that is, 'unblock' will be determined by neighboring regions / routes, and
    # not entirely random. This will become relevant for larger networks.

    # generate masks
    (block,unblock) = choice(range(len(block_sizes)),size=2,replace=False)
    mask = mask_block(block)
    mask = mask * A > 0
    unmask = mask_block(unblock)
    unmask = unmask * A > 0
    
    # generate perturbation
    dA = mask * epsilon - unmask * epsilon
    (dx,res,rank,s) = la.lstsq(A + dA,-dA.dot(x_true))
    return (dx,res)

epsilons = [0.05,0.01,0.005,0.001]
for epsilon in epsilons:
    print 'Epsilon = \t\t\t%f' % (epsilon)
    (dx,res) = trials(1000,perturb_one_entry)
    print "Perturbing 1 entry: \t\t||dx|| = %f, res = %f, rel = %f" % \
            (dx, res, dx/epsilon)
    (dx,res) = trials(1000,perturb_one_block)
    print "Perturbing 1 block: \t\t||dx|| = %f, res = %f, rel = %f" % \
            (dx, res, dx/epsilon)
    (dx,res) = trials(1000,perturb_one_block_conserve)
    print "Perturbing 1 block conserve: \t||dx|| = %f, res = %f, rel = %f" % \
            (dx, res, dx/epsilon)
