import numpy as np
import numpy.linalg as la

# Weak Wolfe line search
# t: step size
# d: search direction
# x: current position
def weak_wolfe_ls(x,d,f,nabla_f,c1=1e-3,c2=0.9):
    alpha, beta = 0, float('inf')
    t = 1
    stop = False
    # bisection method
    # https://www.math.washington.edu/~burke/crs/408/notes/nlp/line.pdf [Page 8]
    while not stop:
        # armijo condition violated
        nabla_fx = nabla_f(x)
        if f(x + t*d) > f(x) + c1 * t * d.dot(nabla_fx):
            # print 'ar', alpha, beta, t
            beta = t
            t = 0.5 * (alpha + beta)
        # curvature condition violated
        elif d.dot(nabla_f(x + t*d)) < c2 * d.dot(nabla_fx):
            # print 'cc', alpha,beta, t
            alpha = t
            t = 2 * alpha if beta == float('inf') else 0.5*(alpha+beta)
        # both conditions pass
        else:
            stop = True
    return t

# Limited memory BFGS
def solve(x, f, nabla_f, stopping, m=10, proj=None, options=None):
    # FIXME has issues when x == 0 to start

    def search_dir(g_new,y_new,s_new,rho,y,s,m=10):
        q = g_new
        alpha = [0]*m
        for i in range(m-1,-1,-1):
            alpha[i] = rho[i] * (s[i].dot(q))
            q = q - alpha[i] * y[i]
        H = y_new.dot(s_new) / (y_new.dot(y_new))
        # H = y[-1].T*s[-1] / (y[-1].T*y[-1])
        r = H * q
        for i in range(0,m):
            beta = rho[i] * y[i].dot(r)
            r = r + s[i] * (alpha[i] - beta)
        return -r

    # Initializations
    it,stop = 0,False
    n = x.shape[0]
    y, s = [np.zeros((n))]*m, [np.zeros((n))]*m
    g_new = nabla_f(x)
    y_new, s_new = g_new, np.ones((n))
    rho, rho_new = [0]*m, 1 / (y_new.dot(s_new))
    while not stop:
        it+=1
        d = search_dir(g_new,y_new,s_new,rho,y,s,m=m)
        # update history
        y.pop(0)
        y.append(y_new)
        s.pop(0)
        s.append(s_new)
        rho.pop(0)
        rho.append(rho_new)

        # update position
        t = weak_wolfe_ls(x,d,f,nabla_f)
        s_new = t * d
        x_next = x + s_new
        g = nabla_f(x)
        g_new = nabla_f(x_next)
        y_new = g_new - g 
        rho_new = 1 / (y_new.dot(s_new))
        x = x_next

        fx = f(x)
        stop = stopping(g,fx,it,t,options)
    return x
