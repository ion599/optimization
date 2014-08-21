import config
import matplotlib.pyplot as pyplot
import scipy.io as sio
import numpy as np
import util
BASE_DIR = '%s/%s'%(config.DATA_DIR, config.EXPERIMENT_MATRICES_DIR)
b_50 = None
def b_estimate():
    global b_50
    if b_50 == None:
         A, b_50, N, block_sizes, x_true, nz, flow = util.load_data('%s/experiment2_waypoints_matrices_routes_%s.mat'% (BASE_DIR, 50))
    return b_50
def read_problem_matrices(filepath):
    A, b, N, block_sizes, x_true, nz, flow = util.load_data(filepath)
    return A, x_true, b_estimate(), N, block_sizes, flow

def read_x_computed(filepath, block_sizes, N):
    matrices = sio.loadmat(filepath)
    x0 = np.array([util.block_e(block_sizes - 1, block_sizes)])
    x = matrices['x']
    fx = matrices['fx']
    x = x0.T + N*x.T
    return x, fx

def metrics(A,b,X):
    d = X.shape[1]
    diff = A.dot(X) - np.tile(b,(d,1)).T
    error = 0.5 * np.diag(diff.T.dot(diff))
    RMSE = np.sqrt(error/b.size)
    den = np.sum(b)/np.sqrt(b.size)
    pRMSE = RMSE / den
    plus = A.dot(X) + np.tile(b,(d,1)).T

    # GEH metric [See https://en.wikipedia.org/wiki/GEH_statistic]
    GEH = np.sqrt(2 * diff**2 / np.abs(plus))
    meanGEH = np.mean(GEH,axis=0)
    maxGEH = np.max(GEH,axis=0)
    GEHunder5 = np.mean(GEH < 5,axis=0)
    GEHunder1 = np.mean(GEH < 1,axis=0)
    GEHunder05 = np.mean(GEH < 0.5,axis=0)
    GEHunder005 = np.mean(GEH < 0.05,axis=0)
    return { 'error': error, 'RMSE': RMSE, 'pRMSE': pRMSE,
        'mean_GEH': meanGEH, 'max_GEH': maxGEH,
        'GEH_under_5': GEHunder5,'GEH_under_1': GEHunder1,
        'GEH_under_0.5': GEHunder05,'GEH_under_0.05': GEHunder005
        }

def flowerror(x_sol, x_true, flow):
     return np.sum(flow * np.abs((x_sol-np.matrix(x_true).T))) / np.sum(flow * x_true)

def get_GEH_from(solution_file, problem_file):
    print problem_file, solution_file
    A, x, b, N, block_size, flow= read_problem_matrices(problem_file)
    x_hat, fx = read_x_computed(solution_file,block_size, N)
    geh = metrics(A, b, x_hat)
    print flowerror(x_hat, x, flow)
    geh['flow_per_error'] = flowerror(x_hat, x, flow)
    return geh

def plot_GEH_vs_route_number_control():
    routes = [3, 10, 20, 30, 40, 50]
    outputfiles = ['%s/output_control%s.mat'% (BASE_DIR, i) for i in routes]
    matrixfiles = ['%s/experiment2_control_matrices_routes_%s.mat'% (BASE_DIR, i) for i in routes]
    GEH = [get_GEH_from(o, m) for o, m in zip(outputfiles, matrixfiles)]
    pyplot.plot(routes, GEH, '-o')
    pyplot.show()

def plot_GEH(routes, GEH, errorstat, xlabel, ylabel):
    GEH_mean = [g[errorstat] for g in GEH]
    pyplot.plot(routes, GEH_mean, '-o')
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.show()

def plot_GEH_vs_route_number_waypoints():
    routes = [3, 10, 20, 30, 40, 50]
    outputfiles = ['%s/output_waypoints%s.mat'% (BASE_DIR, i) for i in routes]
    matrixfiles = ['%s/experiment2_waypoints_matrices_routes_%s.mat'% (BASE_DIR, i) for i in routes]
    GEH = [get_GEH_from(o, m) for o, m in zip(outputfiles, matrixfiles)]
    plot_GEH(routes, GEH, 'mean_GEH', "Maximum Route Number", "Mean GEH")
    plot_GEH(routes, GEH, 'GEH_under_5', "Maximum Route Number", "Percent GEH under 5")
    plot_GEH(routes, GEH, 'GEH_under_0.5', "Maximum Route Number", "Percent GEH under .5")
    plot_GEH(routes, GEH, 'GEH_under_0.05', "Maximum Route Number", "Percent GEH under .05")
    plot_GEH(routes, GEH, 'flow_per_error', "Maximum Route Number", "Percent Flow Allocated Incorrectly")

if __name__== '__main__':
    plot_GEH_vs_route_number_waypoints()