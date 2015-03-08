__author__ = 'lei'

from matplotlib import pyplot
import numpy as np
import scipy.io as sio
import python.util as util
import python.config as config

# functions to load experiment files from disk
def problem_path(density, routes):
    return "{0}/{1}/experiment2_waypoints_matrices_routes_{2}.mat".format(config.EXPERIMENT_MATRICES_DIR, density, routes)

def solution_path(density, routes):
    return "{0}/{1}/output_waypoints{2}.mat".format(config.EXPERIMENT_MATRICES_DIR, density, routes)

def all_link_path(density, routes):
    return "{0}/AllLink/{1}/experiment2_all_link_matrices_routes_{2}.mat".format(config.EXPERIMENT_MATRICES_DIR, density, routes)

def all_link_matrices(path):
    matrices = sio.loadmat(path)
    A = matrices['A']
    b = matrices['b']
    return A, b

def modeled_flow(routes):
    directory = "{0}/modeled_flow.mat".format(config.EXPERIMENT_MATRICES_DIR)
    correction_factors = sio.loadmat(directory)
    total_flow = np.squeeze(correction_factors['total_flow'])
    captured_flow = np.squeeze(correction_factors['captured_flow'])[routes - 1]
    return total_flow, captured_flow

def read_problem_matrices(filepath):
    A, b, N, block_sizes, x_true, nz, flow, _= util.load_data(filepath, CP=True)
    return A, x_true, b, N, block_sizes, flow

def read_solution(filepath, block_sizes, N):
    matrices = sio.loadmat(filepath)
    x0 = np.array([util.block_e(block_sizes - 1, block_sizes)])
    x = matrices['x']
    fx = matrices['fx']
    x = x0.T + N*x.T
    return x, fx

def load_experiment(problemfile, solutionfile, total_flow, captured_flow):
    A, x_true, b, N, block_sizes, flow = read_problem_matrices(problemfile)
    matrices = sio.loadmat(problemfile)
    x_computed , fx = read_solution(solutionfile, block_sizes, N)

    return ExperimentResults(A, b, x_computed, x_true, matrices['U'], flow, N, total_flow, captured_flow)

def load_all_link(density, routes, experiment, load_high_route_b=True):
    path = all_link_path(density, routes)
    A, b = all_link_matrices(path)

    if load_high_route_b:
        path = all_link_path(density, 50)
        _, b = all_link_matrices(path)

    return ExperimentResults(A, b, experiment.x, experiment.x_true, experiment.U, experiment.f, experiment.N, experiment.total_flow, experiment.captured_flow)

class PlotTitles:
    degrees_of_freedom = 'Deg. of freedom from cell data'
    route_flow_error = 'Route flow error from cell data'
    corrected_route_flow_error = 'Model route flow error from cell data'
    geh = 'MATsim link flow error'

# ExperimentResults class to handle calculating numerical values for our problem
class ExperimentResults:
    def __init__(self, A, b, x, x_true, U, f, N, total_flow, captured_flow):
        self.A = A
        self.b = np.squeeze(b)
        self.x = np.squeeze(x)
        self.U = U
        self.f = np.squeeze(f)
        self.x_true = np.squeeze(x_true)
        self.N = N
        self.total_flow = total_flow
        self.captured_flow = captured_flow

    #flow error
    def flow_error(self):
        print np.sum(self.f * np.abs(self.x-self.x_true)) / np.sum(self.f * self.x_true)
        return np.sum(self.f * np.abs(self.x-self.x_true)) / np.sum(self.f * self.x_true)

    def model_corrected_flow_error(self):
        unmodeled_flow = self.total_flow - self.captured_flow
        flow_error = (self.flow_error()*self.total_flow + unmodeled_flow)/self.total_flow

        return flow_error

    #geh calculation
    @staticmethod
    def _GEH(b_estimate, b_true):
        diff = b_estimate - b_true
        plus = b_estimate + b_true

        GEH = np.sqrt(2 * diff**2 / (np.maximum(np.zeros(plus.shape),plus) + 1e-12))
        print GEH
        return GEH

    @staticmethod
    def _percent_under(n, xs):
        return sum(1.0 for x in xs if x < n)/len(xs)

    def geh(self, filter = lambda x,y: (x,y)):
        print self.A.shape, self.x.shape
        b_est = self.A.dot(self.x)
        b_est, b_true = filter(b_est, self.b)

        return ExperimentResults._GEH(np.array(b_est), np.array(b_true))

    # rank calculation
    # the approximate rank of the problem is the number of rows U and A have
    def nullity(self):
        print self.A.shape, self.U.shape
        return max(self.N.shape[1], 1)


def plot_format(plots, title, xlabel, ylabel):
    pyplot.title(title, fontweight='bold', fontsize=22)
    pyplot.legend(plots, ['{0} Routes'.format(i) for i in config.ROUTES], fontsize=18)
    pyplot.xlabel(xlabel,fontsize=22)
    pyplot.ylabel(ylabel,fontsize=22)
    pyplot.xticks(fontsize=18)
    pyplot.yticks(fontsize=18)
    pyplot.xlim([min(config.WAYPOINT_DENSITIES), max(config.WAYPOINT_DENSITIES)])

def geh_plot_format(plots, legend, title, xlabel, ylabel):
    pyplot.title(title, fontweight='bold', fontsize=22)
    pyplot.legend(plots, legend, fontsize=18, loc=4)
    pyplot.xlabel(xlabel,fontsize=22)
    pyplot.ylabel(ylabel,fontsize=22)
    pyplot.xticks(fontsize=18)
    pyplot.yticks(fontsize=18)

    pyplot.ylim([0,1.1])
    pyplot.xlim([0,60])

def slice_zero(xs, ys):
    index = xs.index(0)
    l = zip(*[(x, ys[i]) for i, x in enumerate(xs) if i != index])
    return ys[index], l[0], l[1]

def null_plot(experiment_results, plot):
    plots = []
    for i, r in enumerate(config.ROUTES):
        result = list([experiment_results[(d,r)].nullity() for d in config.WAYPOINT_DENSITIES])
        y, xs, ys = slice_zero(config.WAYPOINT_DENSITIES, result)
        p, = plot([0, max(config.WAYPOINT_DENSITIES)], [y,y], '--o' + config.COLORS[i], linewidth=2)
        #plots.append(p)
        p, = plot(xs, ys, '-o' + config.COLORS[i], linewidth=2)
        plots.append(p)
    plot_format(plots, PlotTitles.degrees_of_freedom, 'Cells', 'Degree of freedom')

def flow_error_plot(experiment_results, plot):
    plots = []
    for i, r in enumerate(config.ROUTES):
        result = list([experiment_results[(d,r)].flow_error() for d in config.WAYPOINT_DENSITIES])
        y, xs, ys = slice_zero(config.WAYPOINT_DENSITIES, result)
        p, = plot([0, max(config.WAYPOINT_DENSITIES)], [y,y], '--o' + config.COLORS[i], linewidth=2)
        #plots.append(p)
        p, = plot(xs, ys, '-o' + config.COLORS[i], linewidth=2)
        plots.append(p)
    plot_format(plots, PlotTitles.route_flow_error, 'Cells', 'Relative error')

def corrected_flow_error_plot(experiment_results, plot):
    plots = []
    for i, r in enumerate(config.ROUTES):
        result = list([experiment_results[(d,r)].model_corrected_flow_error() for d in config.WAYPOINT_DENSITIES])
        y, xs, ys = slice_zero(config.WAYPOINT_DENSITIES, result)
        p, = plot([0, max(config.WAYPOINT_DENSITIES)], [y,y], '--o' + config.COLORS[i], linewidth=2)
        #plots.append(p)
        p, = plot(xs, ys, '-o' + config.COLORS[i], linewidth=2)
        plots.append(p)
    plot_format(plots, PlotTitles.corrected_route_flow_error, 'Cells', 'Relative error')

def between(x, a, b):
    return a <= x < b

def select(l1, l2, predicate):
    return zip(*list((i1,i2) for i1, i2 in zip(l1,l2) if predicate(i1)))

def percentlessthan(n, xs):
    return sum(1.0 for x in xs if x < n)/len(xs)

def geh_plot(experiment_results, density, plot):
    #  We need to generate link flows for all the links, so we need to load new A and b matrices
    all_link = {}
    for r in config.ROUTES:
        all_link[(density, r)] = load_all_link(density, r, experiment_results[(density, r)])

    bins = [0, 700, 2700, float('inf')]
    plots = []
    colors = ['r', 'g', 'b']

    for i in range(len(bins) - 1):
        a = bins[i]
        b = bins[i+1]
        f = lambda l1, l2: select(l1, l2, lambda x: between(x, a, b))
        y = list(percentlessthan(5,all_link[(density, r)].geh(f)) for r in config.ROUTES)
        p, = pyplot.plot(config.ROUTES, y, '-o'+colors[i], linewidth=2)
        plots.append(p)

    all_link = {}
    for r in config.ROUTES:
        all_link[(density, r)] = load_all_link(density, r, experiment_results[(density, r)], False)

    for i in range(len(bins) - 1):
        a = bins[i]
        b = bins[i+1]
        f = lambda l1, l2: select(l1, l2, lambda x: between(x, a, b))
        y = list(percentlessthan(5,all_link[(density, r)].geh(f)) for r in config.ROUTES)
        p, = pyplot.plot(config.ROUTES, y, '--o'+colors[i], linewidth=2)
        #plots.append(p)

    plot([0,60],[.85,.85],'--k',linewidth=2)

    geh_plot_format(plots, ['<700vph', '700-2700vph', '>2700vph'], PlotTitles.geh, "Routes", "%(GEH < 5)")


def load_results(load_from_cache = True):
    #import os
    #import pickle
    experiment_results = {}

    # load all the experiments
    for density in config.WAYPOINT_DENSITIES:
        for route in config.ROUTES:
            print density, route
            problem = problem_path(density, route)
            solution = solution_path(density, route)
            total_flow, captured_flow = modeled_flow(route)
            experiment_results[(density, route)] = load_experiment(problem, solution, total_flow, captured_flow)
            experiment_results[(density, route)].flow_error()

    return experiment_results


def create_ISTTT_plots(plot):
    experiment_results = load_results()
    pyplot.figure()
    # Plot 1: The ranks of the matrices
    null_plot(experiment_results, pyplot.semilogy)
    pyplot.savefig('degrees_of_freedom.pdf')
    pyplot.savefig('degrees_of_freedom.png')
    #pyplot.show()
    pyplot.close()
    pyplot.figure()
    # Plot 2: Flow Error
    flow_error_plot(experiment_results, pyplot.semilogy)
    pyplot.savefig('flow_error.pdf')
    pyplot.savefig('flow_error.png')
    #pyplot.show()
    pyplot.close()
    pyplot.figure()
    # Plot 3: Corrected Flow Error
    corrected_flow_error_plot(experiment_results, pyplot.semilogy)
    pyplot.savefig('corrected_flow_error.pdf')
    pyplot.savefig('corrected_flow_error.png')
    #pyplot.show()
    pyplot.close()
    pyplot.figure()
    # Plot 4: GEH Plots
    geh_plot(experiment_results, 1000, pyplot.plot)
    pyplot.savefig('geh-1000.pdf')
    pyplot.savefig('geh-1000.png')
    #pyplot.show()
    pyplot.close()
    pyplot.figure()
    # Plot 4: GEH Plots
    geh_plot(experiment_results, 2000, pyplot.plot)
    pyplot.savefig('geh-2000.pdf')
    pyplot.savefig('geh-2000.png')
    #pyplot.show()
    pyplot.close()
    pyplot.figure()
    # Plot 4: GEH Plots
    geh_plot(experiment_results, 3000, pyplot.plot)
    pyplot.savefig('geh-3000.pdf')
    pyplot.savefig('geh-3000.png')
    #pyplot.show()
    pyplot.close()
if __name__ == "__main__":
    create_ISTTT_plots(pyplot.semilogy)
