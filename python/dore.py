import numpy as np
import scipy.sparse
from numpy import linalg as la
import random

def solve(linop, linop_T, target, projection, initial, diagnostics = None, options=None, i = 10000, eps = 10**-16):
    """Solves DORE accelerated least squares via projection
    @param diagnostics is a closure that accepts the current solution and iteration number
    """
    # override defaults
    if options and 'max_iter' in options:
        i = options['max_iter']
    if options and 'opt_tol' in options:
        eps = options['opt_tol']

    x = np.squeeze(np.asarray(target))
    y = np.squeeze(np.asarray(initial))
    n = initial.shape[0]

    y_start = y
    y_prev = y
    phi_y = 0
    old_phi_y = 0
    very_old_phi_y = 0

    for iter_ in xrange(i):
        very_old_phi_y = old_phi_y
        old_phi_y = phi_y
        phi_y = linop(y)
        phi_y = np.squeeze(np.asarray(phi_y))
        err = x - phi_y
        norm_change = ((la.norm(y - y_prev)**2))
        print err.dot(err), norm_change, la.norm(y)

        if (iter_ % 10 == 0) and diagnostics is not None:
            diagnostics(y, iter_)

        if iter_ > 0 and (norm_change <= eps):
            break
        y_new = y + np.squeeze(np.asarray(linop_T(err)))

        y_new = projection(y_new)
        phi_y = linop(y_new)
        phi_y = np.squeeze(phi_y)
        err = x - phi_y

        if iter_ > 2:
            delta_phi_y = phi_y - old_phi_y
            dp = delta_phi_y.dot(delta_phi_y)
            if dp > 0:
                a1 = delta_phi_y.dot(err)/dp
                phi_y_1 = (1+a1)*phi_y - a1*old_phi_y
                y_1 = y_new + a1*(y_new - y)
                err_1 = x - phi_y_1

                delta_phi_y = phi_y_1 - very_old_phi_y
                dp = delta_phi_y.dot(delta_phi_y)
                if dp > 0:
                    a2 = delta_phi_y.dot(err_1)/dp
                    y_2 = y_1 + a2*(y_1 - y_prev)
                    y_2 = projection(y_2)

                    phi_y_2 = linop(y_2)
                    phi_y_2 = np.squeeze(np.asarray(phi_y_2))
                    err_2 = x - phi_y_2

                    if err_2.dot(err_2) / err.dot(err) < 1:
                        y_select = y_2
                        phi_y = phi_y_2
                    else:
                        y_select = y_new
                else:
                    y_select = y_new
            else:
                y_select = y_new
        else:
            y_select = y_new

        y_prev = y
        y = y_select

    return y
