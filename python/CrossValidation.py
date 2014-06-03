from sklearn.cross_validation import KFold
from main import parser
import solvers
import scipy.io as sio
from util import PROB_SIMPLEX, EQ_CONSTR_ELIM, \
        L_BFGS, ADMM, SPG, load_data
import util
import numpy as np
import numpy.linalg as la
from numpy import ones, array
from proj_PAV import simplex_projection
import matplotlib.pyplot as plt
import argparse
import logging
import operator
import BB, LBFGS
import config as c

class CrossValidation:

    def __init__(self,k=3,f=None,solver=None):
        self.f=f
        self.solver=solver
        self.k=k
        self.setup()
        self.kf = KFold(self.n,n_folds=k, indices=True)
        self.iters = [None]*k
        self.times = [None]*k
        self.state = [None]*k

    def save(self):
        pass

    def load(self):
        pass

    def setup(self):
        # load data
        self.A, self.b, self.N, self.block_sizes, x_true = load_data(self.f)

        self.n = np.size(self.b)
        self.x_true = np.squeeze(np.array(x_true))

        self.x0 = util.block_e(self.block_sizes - 1, 
            self.block_sizes)
        logging.debug("Blocks: %s" % self.block_sizes.shape)

        self.options = { 'max_iter': 10,
                    'verbose': 1,
                    'suff_dec': 0.003, # FIXME unused
                    'corrections': 500 } # FIXME unused

        self.proj = lambda x: simplex_projection(self.block_sizes - 1,x)
        self.z0 = np.zeros(self.N.shape[1])

    def run(self):
        for i,(train,test) in enumerate(self.kf):
            # Setup
            b_train,A_train = self.b[train],self.A[train,:]
            b_test,A_test = self.b[test],self.A[test,:]

            AN = A_train.dot(self.N)
            f = lambda z: 0.5*la.norm(AN.dot(z)+A_train.dot(self.x0)-b_train)**2
            nabla_f = lambda z: AN.T.dot(A_train.dot(self.x0)+AN.dot(z)-b_train)

            # Solve
            if self.solver == 'LBFGS':
                logging.debug('Starting LBFGS solver...')
                iters,times,state = LBFGS.solve(self.z0 + 1, f, nabla_f, 
                        solvers.stopping,proj=self.proj, options=self.options)
                logging.debug('Stopping LBFGS solver...')
            elif self.solver == 'BB':
                logging.debug('Starting BB solver...')
                iters,times,state = BB.solve(self.z0, f, nabla_f,
                        solvers.stopping,proj=self.proj, options=self.options)
                logging.debug('Stopping BB solver...')

            self.iters[i] = iters
            self.times[i] = times
            self.state[i] = state

    def post_process(self):
        self.cv_errors = []
        self.train_error = []
        self.test_error = []
        logging.debug("Shape of x0: %s" % repr(self.x0.shape))
        for i,(train,test) in enumerate(self.kf):
            d = len(self.state[i])
            b_train,A_train = self.b[train],self.A[train,:]
            b_test,A_test = self.b[test],self.A[test,:]
            self.x_hat = self.N.dot(np.array(self.state[i]).T) - np.tile(self.x0,(d,1)).T
            logging.debug("Shape of x_hat: %s" % repr(self.x_hat.shape))


            starting_error = la.norm(A_train.dot(self.x0)-b_train)
            train_diff = A_train.dot(self.x_hat) - np.tile(b_train,(d,1)).T
            train_error = 0.5 * np.diag(train_diff.T.dot(train_diff))
            self.train_error.append(train_error)

            test_diff = A_test.dot(self.x_hat) - np.tile(b_test,(d,1)).T
            test_error = 0.5 * np.diag(test_diff.T.dot(test_diff))
            self.test_error.append(test_error)

            x_last = self.x_hat[:,-1]
            dist_from_true = np.max(np.abs(x_last-self.x_true))
            start_dist_from_true = np.max(np.abs(self.x_true-self.x0))
            self.cv_errors.append(test_error)

            logging.debug('Train error')
            logging.debug(train_error)
            logging.debug('Test error')
            logging.debug(test_error)

            print 'norm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\nmax|x_init-x_true|: %.2f\n\n\n' % (starting_error, dist_from_true, start_dist_from_true)
        print 'cv error: %8.5e' % np.mean(self.cv_errors)

        self.mean_time = np.mean([np.cumsum(self.times[i])[-1] for i in range(self.k)])
        self.mean_error = np.mean([self.test_error[i][-1] for i in range(self.k)])

    def plot_all(self,subplot=None):
        if subplot:
            plt.subplot(subplot)
        for i in range(self.k):
            times = np.cumsum(self.times[i])
            plt.plot(times,self.test_error[i])
            plt.hold(True)
        plt.xlabel('CPU time (minutes)')
        plt.ylabel('%d-fold CV holdout error (L2)' % self.k)
        plt.title('CV error')

    def plot(self,subplot=None):
        if subplot:
            plt.subplot(subplot)
        fig, ax = plt.subplots()
        ax.plot(self.mean_time,self.mean_error,marker='.',label=self.solver)
        ax.legend(shadow=True)
        plt.xlabel('CPU time (minutes)')
        plt.ylabel('%d-fold CV holdout error (L2)' % self.k)
        plt.title('Average CV error')

if __name__ == "__main__":
    p = parser()
    args = p.parse_args()
    if args.log in c.ACCEPTED_LOG_LEVELS:
        logging.basicConfig(level=eval('logging.'+args.log))

    cv = CrossValidation(k=3,f=args.file,solver=args.solver)
    cv.run()
    cv.post_process()
    cv.plot(subplot=211)
    cv.plot_all(subplot=212)
    plt.show()
        
