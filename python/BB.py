import numpy as np
import numpy.linalg as la

# Barzilai-Borwein (BB)
def solve(x, f, nabla_f, stopping, proj=None, options=None):
    i,stop = 0,False
    x_prev = x + 1
    g_prev = nabla_f(x_prev)

    while not stop:
        i+=1
        g = nabla_f(x)

        delta_g = g - g_prev
        delta_x = x - x_prev
        t = delta_x.dot(delta_g) / delta_g.dot(delta_g) # BB step
        x_next = x - t * g # next position

        x_prev, x = x, x_next # update
        g_prev = g

        if proj:
            x = proj(x)
        fx = f(x)
        stop = stopping(g,fx,i,t,options)
    return x
