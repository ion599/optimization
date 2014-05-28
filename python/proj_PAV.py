from __future__ import division
from numpy import array, inf, dot, ones, float
import numpy as np
import time

def proj_PAV(y, w, l=-inf, u=inf):
    """PAV algorithm with box constraints
    """

    if y.size != w.size:
        print y
        print w
        raise Exception("Shape of y (%s) != shape of w (%d)" % (y.size, w.size))

    n = len(y)
    y = y.astype(float)
    x=y.copy()

    if n==2:
        if y[0]>y[1]:
            x = (w.dot(y)/w.sum())*np.ones(2)
    elif n>2:
        j=range(n+1) # j contains the first index of each block
        ind = 0

        while ind < len(j)-2:
            if weighted_block_avg(y,w,j,ind+1) < weighted_block_avg(y,w,j,ind):
                j.pop(ind+1)
                while ind > 0 and weighted_block_avg(y,w,j,ind-1) > weighted_block_avg(y,w,j,ind):
                    if weighted_block_avg(y,w,j,ind) <= weighted_block_avg(y,w,j,ind-1):
                        j.pop(ind)
                        ind -= 1
            else:
                ind += 1

        for i in xrange(len(j)-1):
            x[j[i]:j[i+1]] = weighted_block_avg(y,w,j,i)*ones(j[i+1]-j[i])

    x[np.where(x < l)] = l
    x[np.where(x > u)] = u

    return x

def simplex_projection(block_sizes, x):
    k = 0
    for i, block_size in enumerate(block_sizes):
        x[k:k+block_size] = proj_PAV(x[k:k+block_size],np.ones(block_size),0,1)
        k += block_size
    return x

# weighted average
def weighted_block_avg(y,w,j,ind):
    block = range(j[ind],j[ind+1])
    #print block
    wB = w[block]
    return dot(wB,y[block])/wB.sum()

# DEMO starts here
if __name__ == "__main__":
    print """
Demonstration of the PAV algorithm on a small example."""
    print
    y = array([4,5,1,6,8,7])
    w = array([1,1,1,1,1,1])
    print "y vector", y
    print "weights", w
    print "solution", proj_PAV(y,w)
    print "solution with bounds", proj_PAV(y,w,5,7)

    N = 3*ones(20000)
    w = array([i%5 for i in range(60000)])
    print w[range(20)]
    start = time.clock()
    w = simplex_projection(N, w)
    print (time.clock() - start)
    print w[range(20)]

