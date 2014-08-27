__author__ = 'lei'
import config
import pickle
from matplotlib import pyplot

def convert_to_waypoint_vs_percent_error(stats, route_index):
    keys = sorted(stats.keys())
    return keys, [stats[d][route_index]['flow_per_error']*(100.0) for d in keys]

def plot_waypoint_density_vs_error():
    global f, stats, colors, indexes, plots, c, i, x, y, p
    f = open(config.PLOT_DIR + '/stats.pkl')
    stats = pickle.load(f)
    colors = ['r', 'g', 'b']
    indexes = [0, 2, 5]
    plots = []
    for c, i in zip(colors, indexes):
        x, y = convert_to_waypoint_vs_percent_error(stats, i)
        p, = pyplot.plot(x, y, '-o' + c)
        plots.append(p)
    pyplot.legend(plots, ('Routes-3', 'Routes-20', 'Routes-50'))
    pyplot.xlabel('Waypoint Density')
    pyplot.ylabel('Route Flow Percent Error')
    pyplot.show()

plot_waypoint_density_vs_error()