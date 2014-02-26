from __future__ import division
from plot_utilities import text_to_np, dir_to_np

from matplotlib import pyplot
import numpy as np
from glob import glob

def reconstructed(errs):
    return np.sum(errs < 1e-4, axis=0)/np.shape(errs)[0]

from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')
linewidth = 1.5

# pyplot.subplot(2,2,1)
# L1_err_0 = dir_to_np('plot_rs_convergence_0_0')
# L1_err_001 = dir_to_np('plot_rs_convergence_0_001')
# L1_err_01 = dir_to_np('plot_rs_convergence_0_01')
# L1_err_1 = dir_to_np('plot_rs_convergence_0_1')
# reconstructed_0 = reconstructed(L1_err_0)
# reconstructed_001 = reconstructed(L1_err_001)
# reconstructed_01 = reconstructed(L1_err_01)
# reconstructed_1 = reconstructed(L1_err_1)
# 
# # L1_err_mean = np.sum(L1_err,axis=0)/np.shape(L1_err)[0]
# # for i in range(1, np.shape(L1_err)[0]):
# #     pyplot.plot(np.array(range(1,np.shape(L1_err)[1]+1)), L1_err[i], '-', hold=True)
# # pyplot.plot(np.array(range(1,np.shape(L1_err)[1]+1)), L1_err_mean, 'r--', hold=True)
# pyplot.plot(np.array(range(1,np.shape(L1_err_0)[1]+1)), reconstructed_0, '-', hold=True,linewidth=linewidth)
# pyplot.plot(np.array(range(1,np.shape(L1_err_001)[1]+1)), reconstructed_001, '-', hold=True,linewidth=linewidth)
# pyplot.plot(np.array(range(1,np.shape(L1_err_01)[1]+1)), reconstructed_01, '-', hold=True,linewidth=linewidth)
# pyplot.plot(np.array(range(1,np.shape(L1_err_1)[1]+1)), reconstructed_1, '-', hold=True,linewidth=linewidth)
# pyplot.legend(['mu=0.0','mu=0.0019','mu=0.01','mu=0.1'], prop = fontP)
# # pyplot.xlabel('Number of iterations')
# pyplot.ylabel('Percentage reconstructed')
# pyplot.ylim([0, 1])
# pyplot.title('Update: none (old)')
# # pyplot.title('Progression of reconstruction by iteration for random sampling (no update)')
# # pyplot.show()

# Update function: a0 <- a0
pyplot.subplot(2,2,1)
dirs = glob('plot_rs_convergence_old_*')
dirs.sort(key=lambda x: float(x.split('_')[-1]))
legend = []
for d in dirs:
    err = dir_to_np(d)
    r = reconstructed(err)
    pyplot.plot(np.array(range(1,np.shape(err)[1]+1)), r, '-', hold=True,linewidth=linewidth)
    mu = d.split('_')[-1]
    legend.append('mu=%.2e' % float(mu))
# pyplot.legend(legend,ncol=2, prop = fontP, loc='center left', bbox_to_anchor=(0.2, 0.35))
# pyplot.xlabel('Number of iterations')
pyplot.ylabel('Percentage reconstructed')
pyplot.ylim([0, 1])
pyplot.title("Update: none (old)")
# pyplot.title('Progression of reconstruction by iteration for random sampling (update=new)')

# pyplot.show()
# Update function: a0 <- a
pyplot.subplot(2,2,2)
dirs = glob('plot_rs_convergence_new_*')
dirs.sort(key=lambda x: float(x.split('_')[-1]))
legend = []
for d in dirs:
    err = dir_to_np(d)
    r = reconstructed(err)
    pyplot.plot(np.array(range(1,np.shape(err)[1]+1)), r, '-', hold=True,linewidth=linewidth)
    mu = d.split('_')[-1]
    legend.append(r'$\mu$ = %.2e' % float(mu))
pyplot.legend(legend,ncol=2, prop = fontP, loc='center left', bbox_to_anchor=(0.2, 0.35))
# pyplot.xlabel('Number of iterations')
# pyplot.ylabel('Percentage reconstructed')
pyplot.ylim([0, 1])
pyplot.title("Update: replace (new)")
# pyplot.title('Progression of reconstruction by iteration for random sampling (update=new)')
# pyplot.show()

# Update function: a0 <- 0.75 * a0 + 0.25 * a
pyplot.subplot(2,2,3)
dirs = glob('plot_rs_convergence_backoff75_*')
dirs.sort(key=lambda x: float(x.split('_')[-1]))
legend = []
for d in dirs:
    err = dir_to_np(d)
    r = reconstructed(err)
    pyplot.plot(np.array(range(1,np.shape(err)[1]+1)), r, '-', hold=True,linewidth=linewidth)
    mu = d.split('_')[-1]
    legend.append('mu=%.2e' % float(mu))
# pyplot.legend(legend,loc='best',ncol=3, mode="expand", prop = fontP)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Percentage reconstructed')
pyplot.ylim([0, 1])
pyplot.title('Update: 0.75 old + 0.25 new')
# pyplot.title('Progression of reconstruction by iteration for random sampling (update=backoff, 0.75)')

# Update function: a0 <- 0.25 * a0 + 0.75 * a
pyplot.subplot(2,2,4)
dirs = glob('plot_rs_convergence_backoff25_*')
dirs.sort(key=lambda x: float(x.split('_')[-1]))
legend = []
for d in dirs:
    err = dir_to_np(d)
    r = reconstructed(err)
    pyplot.plot(np.array(range(1,np.shape(err)[1]+1)), r, '-', hold=True,linewidth=linewidth)
    mu = d.split('_')[-1]
    legend.append('mu=%.2e' % float(mu))
# pyplot.legend(legend,loc='best',ncol=3, mode="expand", prop = fontP)
pyplot.xlabel('Number of iterations')
# pyplot.ylabel('Percentage reconstructed')
pyplot.ylim([0, 1])
pyplot.title('Update: 0.25 old + 0.75 new')
# pyplot.title('Progression of reconstruction by iteration for random sampling (update=backoff, 0.25)')
# pyplot.show()

pyplot.suptitle('Progression of reconstruction by iteration for random sampling')
pyplot.show()

