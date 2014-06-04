import matplotlib.pyplot as plt
import logging
import numpy as np
import numpy.linalg as la
import time
from sklearn.cross_validation import KFold

import solvers
import util
from simplex_projection import simplex_projection
# from projection import pysimplex_projection
import BB, LBFGS, DORE
import config as c
from main import parser

class CrossValidation:

    def __init__(self,k=3,f=None,solver=None,var=None,iter=200):
        self.f=f
        self.solver=solver
        self.var = var
        self.iter = iter
        self.k=k
        self.setup()
        self.kf = KFold(self.n,n_folds=k, indices=True)
        self.iters = [None]*k
        self.times = [None]*k
        self.states = [None]*k

    def save(self):
        pass

    def load(self):
        pass

    def setup(self):
        # load data
        self.A, self.b, self.N, self.block_sizes, x_true=util.load_data(self.f)
        self.NT = self.N.T.tocsr()

        self.n = np.size(self.b)
        self.x_true = np.squeeze(np.array(x_true))

        self.x0 = np.array(util.block_e(self.block_sizes - 1, self.block_sizes))
        logging.debug("Blocks: %s" % self.block_sizes.shape)

        self.options = { 'max_iter': self.iter,
                    'verbose': 0,
                    'suff_dec': 0.003, # FIXME unused
                    'corrections': 500 } # FIXME unused

        self.proj = lambda x: simplex_projection(self.block_sizes - 1,x)
        self.z0 = np.zeros(self.N.shape[1])

    # Run cross-validation and store intermediate states of each run
    def run(self):
        for i,(train,test) in enumerate(self.kf):
            # Setup
            b_train,A_train = self.b[train],self.A[train,:]
            b_test,A_test = self.b[test],self.A[test,:]

            AT = A_train.T.tocsr()

            target = A_train.dot(self.x0) - b_train

            f = lambda z: 0.5 * la.norm(A_train.dot(self.N.dot(z)) + target)**2
            nabla_f = lambda z: self.NT.dot(AT.dot(A_train.dot(self.N.dot(z)) \
                    + target))

            iters, times, states = [], [], []
            def log(iter_,state,duration):
                iters.append(iter_)
                times.append(duration)
                states.append(state)
                start = time.time()
                return start

            # Solve
            logging.debug('[%d] Starting %s solver...' % (i,self.solver))
            if self.solver == 'LBFGS':
                LBFGS.solve(self.z0+1, f, nabla_f, solvers.stopping, log=log,
                        proj=self.proj,options=self.options)
                logging.debug("Took %s time" % str(np.sum(times)))
            elif self.solver == 'BB':
                BB.solve(self.z0,f,nabla_f,solvers.stopping,log=log,
                        proj=self.proj,options=self.options)
            elif self.solver == 'DORE':
                # setup for DORE
                alpha = 0.99
                lsv = util.lsv_operator(A_train, self.N)
                logging.info("Largest singular value: %s" % lsv)
                A_dore = A_train*alpha/lsv
                target_dore = target*alpha/lsv
                DORE.solve(self.z0, lambda z: A_dore.dot(self.N.dot(z)),
                        lambda b: self.N.T.dot(A_dore.T.dot(b)), 
                        target_dore,proj=self.proj,log=log,options=self.options)
                A_dore = None
            logging.debug('[%d] Stopping %s solver...' % (i,self.solver))

            self.iters[i] = iters
            self.times[i] = times
            self.states[i] = states
            AT,A_train,A_test = None,None,None

    # Post process intermediate states of runs
    def post_process(self):
        self.train_error = []
        self.test_error = []
        logging.debug("Shape of x0: %s" % repr(self.x0.shape))
        for i,(train,test) in enumerate(self.kf):
            d = len(self.states[i])
            b_train,A_train = self.b[train],self.A[train,:]
            b_test,A_test = self.b[test],self.A[test,:]
            self.x_hat = self.N.dot(np.array(self.states[i]).T) + \
                    np.tile(self.x0,(d,1)).T
            logging.debug("Shape of x_hat: %s" % repr(self.x_hat.shape))

            starting_error = 0.5 * la.norm(A_train.dot(self.x0)-b_train) ** 2
            train_diff = A_train.dot(self.x_hat) - np.tile(b_train,(d,1)).T
            train_error = 0.5 * np.diag(train_diff.T.dot(train_diff))
            self.train_error.append(train_error)

            test_diff = A_test.dot(self.x_hat) - np.tile(b_test,(d,1)).T
            test_error = 0.5 * np.diag(test_diff.T.dot(test_diff))
            self.test_error.append(test_error)

            x_last = self.x_hat[:,-1]
            dist_from_true = np.max(np.abs(x_last-self.x_true))
            start_dist_from_true = np.max(np.abs(self.x_true-self.x0))

            logging.debug('Train: %8.5e to %8.5e' % (train_error[0],
                train_error[-1]))
            logging.debug('Test:  %8.5e to %8.5e' % (test_error[0],
                test_error[-1]))

            print '0.5norm(A*x_init-b)^2: %8.5e\nmax|x-x_true|: %.2f\nmax|x_init-x_true|: %.2f\n\n\n' \
                    % (starting_error,dist_from_true, start_dist_from_true)

        self.mean_time = np.mean([np.cumsum(self.times[i])[-1] for i in \
                range(self.k)])
        self.mean_error = np.mean([self.test_error[i][-1] for i in \
                range(self.k)])
        print 'mean time: %8.5e, mean error: %8.5e' % (self.mean_time,
                self.mean_error)

    def cleanup(self):
        from sys import getsizeof
        self.A = None
        self.N = None
        self.NT = None
        self.states = None

    # Plot each of the k tests separately
    def plot_all(self,subplot=None,color='k'):
        if subplot:
            plt.subplot(subplot)

        for i in range(self.k):
            times = np.cumsum(self.times[i])
            if i == 0:
                plt.loglog(times,self.test_error[i],color=color,
                        label='%s-%s' % (self.solver,self.var))
            else:
                plt.loglog(times,self.test_error[i],color=color)

            plt.loglog(times,self.train_error[i],color=color,alpha=0.25)
            plt.hold(True)
        plt.xlabel('CPU time (seconds)')
        plt.ylabel('%d-fold CV holdout error (L2)' % self.k)
        plt.title('CV error (%d iterations)' % self.iter)
        plt.legend(shadow=True)

    # Plot summary dot for this solver
    def plot(self,subplot=None,color='k'):
        if subplot:
            plt.subplot(subplot)

        plt.plot(self.mean_time,self.mean_error,marker='.',color=color,
                label='%s-%s' % (self.solver,self.var))
        plt.xlabel('CPU time (seconds)')
        plt.ylabel('%d-fold CV holdout error (L2)' % self.k)
        plt.title('Average CV error')
        plt.legend(shadow=True)

if __name__ == "__main__":
    p = parser()
    args = p.parse_args()
    if args.log in c.ACCEPTED_LOG_LEVELS:
        logging.basicConfig(level=eval('logging.'+args.log))

    cv3 = CrossValidation(k=3,f=args.file,solver='DORE',var='z',iter=380)
    cv3.run()
    cv3.post_process()
    cv3.cleanup()

    cv = CrossValidation(k=3,f=args.file,solver='BB',var='z',iter=650)
    cv.run()
    cv.post_process()
    cv.cleanup()
        
    cv2 = CrossValidation(k=3,f=args.file,solver='LBFGS',var='z',iter=195)
    cv2.run()
    cv2.post_process()
    cv2.cleanup()

    cv.plot(subplot=121,color='b')
    cv.plot_all(subplot=122,color='b')
    cv2.plot(subplot=121,color='m')
    cv2.plot_all(subplot=122,color='m')
    cv3.plot(subplot=121,color='g')
    cv3.plot_all(subplot=122,color='g')
    plt.show()
