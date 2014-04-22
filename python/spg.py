import numpy as np
from math import isnan
import scipy.sparse as sps
import proj_PAV
import numpy.linalg as la
import scipy.sparse.linalg as sla
import logging

def spg_line(f, x, d, gtd, f_max, objective):
    max_iterations = 10
    step = 1
    n_obj = 0
    gamma = 10**-4

    for iter_index in xrange(max_iterations):
        x_new = x + step*d
        f_new, _ = objective(x)
        n_obj += 1

        if f_new < f_max + gamma*step*gtd:
            break

        if step <= 0.1:
            step = step / 2.
        else:
            tmp = (-gtd*step**2) / float(2*(f_new - f - step*gtd))
            if tmp < 0.1 or tmp > 0.9*step or isnan(tmp):
                tmp = step / 2.
            step = tmp
    return f_new, x_new, step, n_obj

def solver(AN, q, G=None, h=None, A=None, b=None, solver=None, initvals=None,
           N=None, block_sizes=None, reduction=None, constraints=None):
    x = np.zeros(q.shape[0])
    logging.debug('|q| = %s' % la.norm(q))

    def projection(x):
        return proj_PAV.simplex_projection(block_sizes - 1, x)

    def objective(x):
        """Computes objective and gradient of objective at x

        returns tuple (f, g) where f=obj(x) and g = grad_x obj(x)
        """
        Ax = AN.dot(x)
        f = x.dot(q) + (Ax.dot(Ax)/2.)
        g = AN.T.dot(AN.dot(x)) + q
        return (f, g)

    m = 1
    tolerance = 10**-7
    max_iterations = 1000
    max_fn_evals = 10*max_iterations
    min_step_length = 10**-5
    max_step_length = 10**5

    n_obj = 0
    n_grd = 0
    last_m_function_values = np.empty(m)
    last_m_function_values[:] = -np.inf
    line_search_step = 0
    function_history = np.zeros(max_iterations + 1)
    norm_history = np.zeros(max_iterations + 1)
    eval_history = np.zeros(max_iterations + 1)

    # Compute PG direction and starting steplength
    x = projection(x)
    f, g = objective(x)
    n_obj += 1
    n_grd += 1
    last_m_function_values[0] = f;
    last_m_function_values = np.roll(last_m_function_values, -1)
    best_f = f
    best_x = x
    function_history[0] = f
    eval_history[0] = n_obj
    d = projection(x - g) - x
    d_norm = la.norm(d, np.inf)
    g_step = min(max_step_length, max(min_step_length, 1./d_norm))
    for iter_index in xrange(max_iterations):
        logging.info('Running iteration %d of SPG' % iter_index)
        if d_norm < tolerance or n_obj >= max_fn_evals:
            break

        # Right now, I am ignoring curvyFlag (i.e. setting it to false)
        d = projection(x - g_step*g) - x
        gtd = g.dot(d)

        f_new, x_new, l_step, ln_obj = spg_line(f, x, d, gtd, last_m_function_values[0], objective)

        if best_f < f_new:
            best_f = f_new
            best_x = x_new

        f_new, g_new = objective(x_new)
        n_obj += 1
        n_grd += 1
        eval_history[iter_index] = n_obj

        s = x_new - x
        y = g_new - g
        x = x_new
        g = g_new
        f = f_new
        logging.info('Function value on iteration %d: %s' % (iter_index, f))
        last_m_function_values[0] = f;
        last_m_function_values = np.roll(last_m_function_values, -1)

        d = projection(x - g) - x
        d_norm = la.norm(d, np.inf)
        logging.debug('d_norm = %s' % d_norm)

        sts = s.dot(s)
        sty = s.dot(y)
        if sty <= 0:
            g_step = max_step_length
        else:
            g_step = min(max_step_length, max(min_step_length, sts/float(sty)))

    if f > best_f:
        x = best_x
        f, g = objective(x)
        n_obj += 1
        n_grd += 1

    return x
