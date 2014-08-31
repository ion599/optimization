__author__ = 'lei'
import config
import pickle
import scipy.io as sio
import numpy as np

from matplotlib import pyplot
def correct_for_unmodeled_flow(p, route_number):
    modeled_flow = sio.loadmat(config.PLOT_DIR+'/modeled_flow.mat')
    total_flow = modeled_flow['total_flow']
    captured_flow = modeled_flow['captured_flow']
    flow_in_routes = np.squeeze(captured_flow)[route_number - 1]
    unmodeled_flow = total_flow - flow_in_routes
    return ((p*flow_in_routes + unmodeled_flow)/total_flow).flat[0]

def convert_to_waypoint_vs_percent_error(stats, route_index):
    keys = sorted(stats.keys())
    values = [stats[d][route_index]['flow_per_error'] for d in keys]
    return keys, values

def plot_waypoint_density_vs_error(correction = lambda x, y: x):
    f = open(config.PLOT_DIR + '/stats.pkl')
    stats = pickle.load(f)
    colors = ['r', 'g', 'b','c','m','k']
    indexes = list(range(6))
    routes = [3,10,20,30,40,50]
    plots = []
    for c, i in zip(colors, indexes):
        x, y = convert_to_waypoint_vs_percent_error(stats, i)
        y = [correction(y_i, routes[i]) for y_i in y]
        p, = pyplot.loglog(x, y, '-o' + c)
        plots.append(p)
    pyplot.legend(plots, ['Routes-{0}'.format(i) for i in routes])
    pyplot.xlabel('Waypoint Density')
    pyplot.ylabel('Route Flow Percent Error')
    pyplot.ylim([0,5])
    pyplot.show()
def convert_to_route_vs_percent_error(stats, waypoint):
    routes = [3,10,20,30,40,50]
    return routes, [s['flow_per_error'] for s in stats[waypoint]]

def plot_routes_vs_error(plot, correction = lambda x, y: x):
    f = open(config.PLOT_DIR + '/stats.pkl')
    stats = pickle.load(f)
    colors = ['r', 'g', 'b','c','m','k','r','g','b']
    routes = [3,10,20,30,40,50]
    waypoints = [3800, 2850, 1900, 1425, 950, 713, 475, 238,0]
    plots = []
    for c, w in zip(colors, waypoints):
        x, y = convert_to_route_vs_percent_error(stats, w)
        y = [correction(y_i, routes[i]) for i,y_i in enumerate(y)]
        p, = plot(x, y, '-o' + c)
        plots.append(p)
    pyplot.legend(plots, ['Cell Density-{0}'.format(i) for i in waypoints])
    pyplot.xlabel('Routes')
    pyplot.ylabel('Route Flow Percent Error')
    pyplot.ylim([0,5])
    pyplot.show()
    pyplot.close('all')

plottypes = [pyplot.plot, pyplot.loglog, pyplot.semilogy]
for plot in plottypes:
    plot_routes_vs_error(plot,correct_for_unmodeled_flow)
    plot_routes_vs_error(plot)