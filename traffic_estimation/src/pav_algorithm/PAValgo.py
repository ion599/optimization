from numpy import array, inf, dot, ones, float

# PAV algorithm with box constraints
def pav_algo(y, w, l=-inf, u=inf):
	assert(y.size == w.size)
	n = len(y)
	x = y #x=y.copy()
	
	if n>1:
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
		
		#x = array([avg(y,w,j,i)*ones(j[i+1]-j[i]) for i in range(len(j)-1)])
		for i in range(len(j)-1): x[j[i]:j[i+1]] = avg(y,w,j,i)*ones(j[i+1]-j[i])
		
	return x

# weighted average
def avg(y,w,j,ind):
	block = range(j[ind],j[ind+1])
	#print block
	wB = w[block]
	return dot(wB,y[block])/float(wB.sum())

# DEMO starts here
if __name__ == "__main__":
	print """
Demonstration of the PAV algorithm on a small example."""
	print
	y = [4.,5.,1.,6.,8.,7.]
	w = array([1,1,1,1,1,1])
	y=array(map(float, y))
	print "y vector", y
	print "weights", w
	print "solution", pav_algo(y,w)
	