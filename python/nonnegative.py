__author__ = 'lei'
from cvxopt import solvers, spmatrix, matrix
import scipy.sparse as sparse
import scipy.optimize as optimize
import numpy
import config

def isnumber(x):
    return isinstance(x, (int, long, float, complex))

def all_numbers(xs):
    for x in xs:
        if not isnumber(x):
            print x, type(x)

def func(x, A, b, l):
    p1 = numpy.linalg.norm(A.dot(x)-b)**2
    p2 = l* numpy.linalg.norm(x)**2
    return .5*(p1 + p2)

def funcprime(x,A,b,l):
    return A.T.dot(A.dot(x) - b) + numpy.multiply(l, x)

def spnnls(A, b, l, bounds):
    f = lambda (x): func(x, A, b, l)

    fprime = lambda (x): funcprime(x, A, b, l)

    x0 = .5*numpy.ones((A.shape[1],))

    x, objective, d = optimize.fmin_l_bfgs_b(f, x0, fprime=fprime, bounds=bounds, factr=1e7, m=20)
    #x, objective, d = optimize.fmin_l_bfgs_b(f, x0, fprime=fprime, bounds=bounds,approx_grad=True)
    return x, objective, d

import scipy.io as sio
call = 0

def solve(A, b, l):
    bounds = [(0, 1) for i in range(A.shape[1])]
    x, obj, d = spnnls(A, b, l, bounds)
    return x, obj

def load_matries(probablity, routes):
    filepath = config.DATA_DIR + "/experiment_matrices/{0}/sampled_links_routes_{1}.mat".format(probablity, routes)

    matrices = sio.loadmat(filepath)
    return matrices['A'], numpy.squeeze(matrices['b']), numpy.squeeze(matrices['x_true'])

def run():
    for p in [.01, .1, .2, .3, .4, .5, .6, .7, .8, .9]:
        for r in [3,10,20,30,40,50]:
            A, b, xtrue = load_matries(p, r)
            print solve(A, b, 0)

if __name__ == '__main__':
    run()