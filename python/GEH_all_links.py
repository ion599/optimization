from scipy import io as sio
import util
import config
import numpy as np
from matplotlib import pyplot

def allLinksB():
    dir = '{0}/{1}/{2}/experiment2_total_link_matrices_routes_2000.mat'.format(config.PLOT_DIR, config.EXPERIMENT_MATRICES_DIR, config.ALL_LINK_DIR)
    matrix = sio.loadmat(dir)['b']
    return matrix

def allLinksA(density, routes):
    dir1 = '{0}/{1}/{2}'.format(config.PLOT_DIR, config.EXPERIMENT_MATRICES_DIR, config.ALL_LINK_DIR)
    dir2 = '{0}/experiment2_total_link_matrices_routes_{1}.mat'.format(density,routes)
    matrix = sio.loadmat('{0}/{1}'.format(dir1, dir2))['A']
    return matrix

def readZ(density, routes):
    dir1 = '{0}/{1}'.format(config.PLOT_DIR, config.EXPERIMENT_MATRICES_DIR)
    dir2 = '{0}/output_waypoints{1}.mat'.format(density,routes)
    matrix = sio.loadmat('{0}/{1}'.format(dir1, dir2))['x']
    return matrix

def readX(density,routes):
    dir1 = '{0}/{1}'.format(config.PLOT_DIR, config.EXPERIMENT_MATRICES_DIR)
    dir2 = '{0}/experiment2_waypoints_matrices_routes_{1}.mat'.format(density,routes)
    A, b, N, block_sizes, x_true, nz, f = util.load_data('{0}/{1}'.format(dir1, dir2))
    x0 = np.array([util.block_e(block_sizes - 1, block_sizes)])
    return x0.T + N*readZ(density,routes).T

def GEH(b_estimate, b_true):
    diff = b_estimate - b_true
    plus = b_estimate + b_true
    GEH = np.sqrt(2 * diff**2 / np.abs(plus))
    return GEH

def bin(f, b_estimate, b_true):
    bin_b_estimate = [b for b, x in zip(b_estimate,b_true) if f(x)]
    bin_b_true = [b for b in b_true if f(b)]
    return bin_b_estimate, bin_b_true

def percent_above_5(geh):
    return len(list(filter(lambda x:x>5, geh)))/float(len(geh))

def GEH_bin2700(b_est, b_true):
    b_est, b_true = bin(lambda x: x > 2700, b_est, b_true)
    return GEH(np.array(b_est), np.array(b_true))

def GEH_bin700_2700(b_est, b_true):
    b_est, b_true = bin(lambda x:700 < x < 2700, b_est, b_true)
    return GEH(np.array(b_est), np.array(b_true))

def GEH_bin700(b_est, b_true):
    b_est, b_true = bin(lambda x:700 > x, b_est, b_true)
    return GEH(np.array(b_est), np.array(b_true))

def percent_under5(xs):
    return sum(1.0 for x in xs if x < 5)/len(xs)

def geh_by_bin(b_est, b_true):
    geh = percent_under5(GEH_bin2700(b_est, b_true))
    geh2 = percent_under5(GEH_bin700_2700(b_est, b_true))
    geh3 = percent_under5(GEH_bin700(b_est, b_true))
    return geh, geh2, geh3

geh_values = []

for d in config.WAYPOINT_DENSITIES:
    b_est = allLinksA(d,50) * readX(d,50)
    b_true = allLinksB().T
    g1,g2,g3 = geh_by_bin(b_est, b_true)
    geh_values.append([g1,g2,g3])

pyplot.plot([0,4000],[.85,.85],'--kb')
geh_values =np.array(geh_values)
pyplot.plot(config.WAYPOINT_DENSITIES, geh_values,'-o')
pyplot.ylim([0,1.1])
pyplot.xlim([0,4000])
font = {'size': 22}

pyplot.title('MATSim link flow error', fontsize=22, weight='bold')
pyplot.xlabel('Number of cells', fontsize=22)
pyplot.ylabel('% (GEH < 5)', fontsize=22)

pyplot.show()