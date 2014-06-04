from __future__ import division
from numpy import array, inf, dot, ones, float
import numpy as np
import time
import simplex_projection
import sys
from multiprocessing import Pool

def proj_PAV(y, w=None, l=0, u=1):
#def proj_PAV(s):
    """PAV algorithm with box constraints
    """
    #y, w, l, u = s

    # if y.size != w.size:
    #     print y
    #     print w
    #     raise Exception("Shape of y (%s) != shape of w (%d)" % (y.size, w.size))

    n = len(y)
    if w is None: w = np.ones(n)
    y = y.astype(float)
    # x=y.copy()
    x=y

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

    return np.maximum(l,np.minimum(u,x))


def proj_l1ball(y):
    """Projection on l1-ball
    """
    n, x = len(y), y.copy()
    x.sort()
    x = x[::-1]
    tmp = np.multiply(np.cumsum(x) - 1, [1/n for n in range(1,n+1)])
    return np.maximum(y - tmp[np.sum(x > tmp)-1],0)


def pysimplex_projection(block_sizes, x, processes=1):
    ind_end = np.cumsum(block_sizes)
    ind_start = np.hstack(([0],ind_end[:-1]))
    if processes == 1:
        x = np.concatenate([proj_PAV(x[i:j], l=0, u=1) for i,j \
                in zip(ind_start,ind_end)])
    else:
        pool = Pool(processes)
        everything = [x[i:j] for i,j in \
                zip(ind_start,ind_end)]
        results = pool.map(projection, everything, chunksize=25)
        pool.close()
        x = np.concatenate(results)
    return x


# weighted average
def weighted_block_avg(y,w,j,ind):
    wB = w[j[ind]:j[ind+1]]
    return dot(wB,y[j[ind]:j[ind+1]])/wB.sum()

# DEMO starts here
if __name__ == "__main__":
    print >> sys.stderr, """Demonstration of the PAV algorithm on a small example."""
    print >> sys.stderr
    y = array([4,5,1,6,8,7])
    w = array([1,1,1,1,1,1])
    print >> sys.stderr, "y vector", y
    print >> sys.stderr, "weights", w
    print >> sys.stderr, "solution", proj_PAV(y,w,-inf,inf)
    tic = time.time()
    for idx in xrange(1000):
        proj_PAV(y,w,5,7)
    toc = time.time()
    print toc - tic
    print >> sys.stderr, "solution with bounds", proj_PAV(y,w,5,7)
    tic = time.time()
    for idx in xrange(1000):
        simplex_projection.pav_projection(y,5,7)
    toc = time.time()
    print toc - tic
    print >> sys.stderr, "solution with bounds", simplex_projection.pav_projection(y,5,7)

    print "Demonstration of projection on l1-ball"
    print "solution with bounds", proj_l1ball(array([.5,.2,.9,.5,.2]))
    
    print "Demonstration of simplex projection"
    
    N = 3*ones(20000)
    w = array([(i%5)/10. for i in range(60000)])
    print >> sys.stderr, w[range(20)]
    start = time.clock()
    w = pysimplex_projection(N, w)
    print >> sys.stderr, (time.clock() - start)
    print >> sys.stderr, w[range(20)]
    start = time.clock()
    w = simplex_projection.simplex_projection(N, w)
    print >> sys.stderr, (time.clock() - start)
    print >> sys.stderr, w[range(20)]
