from __future__ import division
from numpy import array, inf, dot, ones, float
import numpy as np
import time
from multiprocessing import Pool

# def proj_PAV(y, w, l=-inf, u=inf):
def proj_PAV(s):
    """PAV algorithm with box constraints
    """
    y, w, l, u = s

    # if y.size != w.size:
    #     print y
    #     print w
    #     raise Exception("Shape of y (%s) != shape of w (%d)" % (y.size, w.size))

    n = len(y)
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

"""
function X = SimplexProj(Y)
[N,D] = size(Y);
X = sort(Y,2,'descend');
Xtmp = (cumsum(X,2)-1)*diag(sparse(1./(1:D)));
X = max(bsxfun(@minus,Y,Xtmp(sub2ind([N,D],(1:N)',sum(X>Xtmp,2)))),0);
"""

def simplex_projection(block_sizes, x, processes=1):
    ind_end = np.cumsum(block_sizes)
    ind_start = np.hstack(([0],ind_end[:-1]))
    if processes == 1:
        x = np.concatenate([proj_PAV((x[i:j],np.ones(k),0,1)) for i,j,k \
                in zip(ind_start,ind_end,block_sizes)])
    else:
        pool = Pool(processes)
        everything = [(x[i:j],np.ones(k),0,1) for i,j,k in \
                zip(ind_start,ind_end,block_sizes)]
        results = pool.map(proj_PAV, everything, chunksize=25)
        pool.close()
        x = np.concatenate(results)
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
    print "solution", proj_PAV((y,w,-inf,inf))
    print "solution with bounds", proj_PAV((y,w,5,7))
    
    print proj_l1ball(array([.5,.2,.9,.5,.2]))
    
    """
    N = 3*ones(20000)
    w = array([i%5 for i in range(60000)])
    print w[range(20)]
    start = time.clock()
    w = simplex_projection(N, w)
    print (time.clock() - start)
    print w[range(20)]
    """

