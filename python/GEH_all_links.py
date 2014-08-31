from scipy import io as sio
import config
# read big b
# read big A
# read z, fz
# read support matrices to calculate real x

def allLinksB():
    dir = '{0}/{1}/{2}/experiment2_total_link_matrices_routes_2000.mat'.format(config.PLOT_DIR, config.EXPERIMENT_MATRICES_DIR, config.ALL_LINK_DIR)
    matrix = sio.loadmat(dir)['b']
    return matrix
def allLinksA(density, routes):
    dir1 = '{0}/{1}/{2}'.format(config.PLOT_DIR, config.EXPERIMENT_MATRICES_DIR, config.ALL_LINK_DIR)
    dir2 = '{0}/experiment2_total_link_matrices_routes_{1}.mat'.format(density,routes)
    matrix = sio.loadmat('{0}/{1}'.format(dir1, dir2))['A']
    return matrix
def
print allLinksA(3800,50)