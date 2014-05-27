import numpy as np
import numpy.linalg as la

# Barzilai-Borwein (BB)
def solve(x, f, nabla_f,proj=None,options=None):
    i,stop = 0,False
    x_prev = x + 1

    g_prev = nabla_f(x_prev)
    while not stop:
        i+=1
        delta_x = x - x_prev

        g = nabla_f(x)
        delta_g = g - g_prev

        t = delta_x.dot(delta_g) / delta_g.dot(delta_g)
        x_next = x - t * g # next position
        x_prev, x = x, x_next # update
        g_prev = g

        x = proj(x)
        fx = f(x)
        stop = stopping(g,fx,i,t,options)
    return x

# Stopping condition
def stopping(g,fx,i,t,options=None):
    if options and 'max_iter' in options:
        if i >= options['max_iter']:
            return True
    if options and 'opt_tol' in options:
        TOLER = options['opt_tol']
    else:
        TOLER = 1e-6

    norm2_nabla_f = np.square(la.norm(g))
    thresh = TOLER * (1 + abs(fx))
    if options and 'verbose' in options and options['verbose'] >= 1:
        print "iter=%d: %e %e %e %f" % (i,t,norm2_nabla_f,thresh,fx) 
    if norm2_nabla_f <= thresh:
        print "iter=%d: %e %e %e %f" % (i,t,norm2_nabla_f,thresh,fx) 
        return True
    return False
