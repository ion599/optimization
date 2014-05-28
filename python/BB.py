import numpy as np
import numpy.linalg as la
import time

# Barzilai-Borwein (BB)
def solve(x, f, nabla_f, stopping, record_every=5, proj=None, options=None):
    # Save initial state
    iters,times,state = [0],[0],[x]
    start = time.time()

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

        # Save intermediate state
        if i % record_every == 0:
            iters.append(i)
            times.append(time.time() - start)
            start = time.time()
            state.append(x)

    # Save final state
    iters.append(i)
    times.append(time.time() - start)
    state.append(x)
    return (iters,times,state)
