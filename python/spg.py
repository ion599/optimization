import numpy as np
from math import isnan
import scipy.sparse as sps

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

def solver(P, q, G=None, h=None, A=None, b=None, solver=None, initvals=None,
           N=None, block_sizes=None, reduction=None, constraints=None):
    # TODO: set projection, objective based on QP
    def projection(x):
        pass

    def objective(x):
        """Computes objective and gradient of objectvie at x

        returns tuple (f, g) where f=obj(x) and g = grad_x obj(x)
        """
        pass

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
    last_function_value = f;
    best_f = f
    best_x = x
    function_hist[0] = f
    eval_history[0] = b_obj
    d = projection(x - g) - x
    d_norm = la.norm(d, np.inf)
    g_step = min(max_step_length, max(min_step_length, 1./d_norm))
    for iter_index in xrange(max_iterations):
        if d_norm < tolerance or n_obj >= max_fn_evals:
            break

        # Right now, I am ignoring curvyFlag (i.e. setting it to false)
        d = projection(x - g_step*g) - x
        gtd = g.dot(g)

        f_new, x_new, l_step, ln_obj = spg_line(f, x, d, gtd, max(last_m_function_values), objective)

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

        d = projection(x - g) - x
        d_norm = la.norm(d, np.inf)

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
