from __future__ import division
from numpy import array, inf, dot, ones, float
import time
# PAV algorithm with box constraints
def proj_PAV(y, w, l=-inf, u=inf):
	
	assert(y.size == w.size)
	n = len(y)
	y = y.astype(float)
	x=y.copy()
	
	if n==2:
		if y[0]>y[1]:
			x = (dot(w,y)/w.sum())*ones(2)
	elif n>2:
		j=range(n+1) # j contains the first index of each block
		ind = 0
		
		while ind < len(j)-2:
			if avg(y,w,j,ind+1) < avg(y,w,j,ind):
				j.pop(ind+1)
				while ind > 0 and avg(y,w,j,ind-1) > avg(y,w,j,ind):
					if avg(y,w,j,ind) <= avg(y,w,j,ind-1):
						j.pop(ind)
						ind -= 1
			else:
				ind += 1	
		
		for i in range(len(j)-1):
			x[j[i]:j[i+1]] = avg(y,w,j,i)*ones(j[i+1]-j[i])
	
	x[x<l] = l
	x[x>u] = u	
		
	return x

# weighted average
def avg(y,w,j,ind):
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
	
	N = 3*ones(30000)
	w = array([i%5 for i in range(60000)])
	print w[range(20)]
	start = time.clock()
	k = 0
	for i in range(30000):
		w[k:k+N[i]-1] = proj_PAV(w[k:k+N[i]-1],ones(N[i]-1),1,4)
		k = k+N[i]-1
	print (time.clock() - start)
	print w[range(20)]
		