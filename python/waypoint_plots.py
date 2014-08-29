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
    return keys, [stats[d][route_index]['flow_per_error'] for d in keys]

def plot_waypoint_density_vs_error(correction = lambda x, y: x):
    f = open(config.PLOT_DIR + '/stats.pkl')
    stats = pickle.load(f)
    colors = ['r', 'g', 'b','c','m','k']
    indexes = list(range(6))
    routes = [3,10,20,30,40,50]
    plots = []
    for c, i in zip(colors, indexes):
        x, y = convert_to_waypoint_vs_percent_error(stats, i)
        y = [correction(y_i, routes[i])*100.0 for y_i in y]
        p, = pyplot.plot(x, y, '-o' + c)
        plots.append(p)
    pyplot.legend(plots, ['Routes-{0}'.format(i) for i in routes])
    pyplot.xlabel('Waypoint Density')
    pyplot.ylabel('Route Flow Percent Error')
    pyplot.show()

plot_waypoint_density_vs_error(correct_for_unmodeled_flow)