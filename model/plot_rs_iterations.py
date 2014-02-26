from plot_utilities import text_to_np

from matplotlib import pyplot
import numpy as np
import os

L1_errs = []

data_dir = 'plot_rs_iterations_175'
# data_dir = 'plot_rs_iterations_155'
data_files = os.listdir(data_dir)
for data_file in data_files:
    if data_file[0] == '.':
        continue
    with open("%s/%s" % (data_dir, data_file)) as f:
        L1_errs.append([float(x) for x in f.readlines()])
L1_err = np.array(L1_errs)
L1_err_mean = np.sum(L1_err,axis=0)/np.shape(L1_err)[0]
for i in range(1, np.shape(L1_err)[0]):
    pyplot.plot(np.array(range(1,np.shape(L1_err)[1]+1)), L1_err[i], '-', hold=True)
pyplot.plot(np.array(range(1,np.shape(L1_err)[1]+1)), L1_err_mean, 'r--', hold=True)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('L1 error')
pyplot.title('Progression of L1 error by iteration for random sampling')
pyplot.show()
