import scipy.io as sio
import matplotlib.pyplot as plt
import argparse
import logging
import numpy as np
import numpy.linalg as la

import solvers
import util
from c_extensions.simplex_projection import simplex_projection
# from projection import pysimplex_projection
import BB, LBFGS, DORE
import config as c

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='Data file (*.mat)',
                        default='route_assignment_matrices_ntt.mat')
    parser.add_argument('--log', dest='log', nargs='?', const='INFO',
            default='WARN', help='Set log level (default: WARN)')
    parser.add_argument('--solver',dest='solver',type=str,default='LBFGS',
            help='Solver name')
    parser.add_argument('--noise',dest='noise',type=float,default=None,
            help='Noise level')
    return parser

def main():
    p = parser()
    args = p.parse_args()
    if args.log in c.ACCEPTED_LOG_LEVELS:
        logging.basicConfig(level=eval('logging.'+args.log))

    # load data
    filepath = '%s/%s/%s' % (c.DATA_DIR, c.EXPERIMENT_MATRICES_DIR, args.file)
    A, b, N, block_sizes, x_true, nz, flow = util.load_data(filepath)
    sio.savemat('fullData.mat', {'A':A,'b':b,'N':block_sizes,'N2':N,
        'x_true':x_true})

    if args.noise:
        b_true = b
        delta = np.random.normal(scale=b*args.noise)
        b = b + delta

    # Sample usage
    #P = A.T.dot(A)
    #q = A.T.dot(b)
    #solvers.qp2(P, q, block_sizes=block_sizes, constraints={PROB_SIMPLEX}, \
    #        reduction=EQ_CONSTR_ELIM, method=L_BFGS)

    logging.debug("Blocks: %s" % block_sizes.shape)
    x0 = np.array(util.block_e(block_sizes - 1, block_sizes))
    target = A.dot(x0)-b

    options = { 'max_iter': 5000,
                'verbose': 1,
                'suff_dec': 0.003, # FIXME unused
                'corrections': 500 } # FIXME unused
    AT = A.T.tocsr()
    NT = N.T.tocsr()

    f = lambda z: 0.5 * la.norm(A.dot(N.dot(z)) + target)**2
    nabla_f = lambda z: NT.dot(AT.dot(A.dot(N.dot(z)) + target))

    def proj(x):
        projected_value = simplex_projection(block_sizes - 1,x)
        # projected_value = pysimplex_projection(block_sizes - 1,x)
        return projected_value

    z0 = np.zeros(N.shape[1])

    import time
    iters, times, states = [], [], []
    def log(iter_,state,duration):
        iters.append(iter_)
        times.append(duration)
        states.append(state)
        start = time.time()
        return start

    logging.debug('Starting %s solver...' % args.solver)
    if args.solver == 'LBFGS':
        LBFGS.solve(z0+1, f, nabla_f, solvers.stopping, log=log,proj=proj,
                options=options)
        logging.debug("Took %s time" % str(np.sum(times)))
    elif args.solver == 'BB':
        BB.solve(z0,f,nabla_f,solvers.stopping,log=log,proj=proj,
                options=options)
    elif args.solver == 'DORE':
        # setup for DORE
        alpha = 0.99
        lsv = util.lsv_operator(A, N)
        logging.info("Largest singular value: %s" % lsv)
        A_dore = A*alpha/lsv
        target_dore = target*alpha/lsv

        DORE.solve(z0, lambda z: A_dore.dot(N.dot(z)),
                lambda b: N.T.dot(A_dore.T.dot(b)), target_dore, proj=proj,
                log=log,options=options)
        A_dore = None
    logging.debug('Stopping %s solver...' % args.solver)

    # Plot some stuff
    d = len(states)
    x_hat = N.dot(np.array(states).T) + np.tile(x0,(d,1)).T
    x_last = x_hat[:,-1]

    logging.debug("Shape of x0: %s" % repr(x0.shape))
    logging.debug("Shape of x_hat: %s" % repr(x_hat.shape))

    starting_error = 0.5 * la.norm(A.dot(x0)-b)**2
    opt_error = 0.5 * la.norm(A.dot(x_true)-b)**2
    diff = A.dot(x_hat) - np.tile(b,(d,1)).T
    error = 0.5 * np.diag(diff.T.dot(diff))

    dist_from_true = np.max(np.abs(x_last-x_true))
    start_dist_from_true = np.max(np.abs(x_last-x0))

    x_diff = x_true - x_last
    print 'incorrect x entries: %s' % x_diff[np.abs(x_diff) > 1e-3].shape[0]
    per_flow = np.sum(np.abs(flow * (x_last-x_true))) / np.sum(flow * x_true)
    print 'percent flow allocated incorrectly: %f' % per_flow
    print '0.5norm(A*x-b)^2: %8.5e\n0.5norm(A*x_init-b)^2: %8.5e\n0.5norm(A*x*-b)^2: %8.5e\nmax|x-x_true|: %.2f\nmax|x_init-x_true|: %.2f\n\n\n' % \
        (error[-1], starting_error, opt_error, dist_from_true,start_dist_from_true)
    import ipdb
    ipdb.set_trace()

    plt.figure()
    plt.hist(x_last)

    plt.figure()
    plt.loglog(np.cumsum(times),error)
    plt.show()

    return iters, times, states

if __name__ == "__main__":
    iters, times, states = main()
