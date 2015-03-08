ACCEPTED_LOG_LEVELS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'WARN']
DATA_DIR = '/home/lei/traffic/datasets/Phi'
EXPERIMENT_DIR = '/home/lei/traffic/datasets/megacell-data'
#EXPERIMENT_DIR = '/home/lei/traffic/datasets/Phi/plots'
EXPERIMENT_MATRICES_DIR = EXPERIMENT_DIR + '/experiment_matrices/V3'
ALL_LINK_DIR = 'AllLinks'
ESTIMATION_INFO_DIR = 'estimation_info'
PLOT_DIR = DATA_DIR + '/plots'
test = False

WAYPOINT_DENSITIES = [8000,6000,4000,3500,3000,2500,2000,1500,1000,750,500,250,0]
#WAYPOINT_DENSITIES = [1000]
#WAYPOINT_DENSITIES = [3800,3325,2850,2375,1900,1425,950,713,475,238,0]

ROUTES = [50, 40, 30, 20, 10, 3]

if test:
    WAYPOINT_DENSITIES = [8000, 0]
    ROUTES = [50, 3]

COLORS = list(reversed(['m', 'c', 'b', 'k', 'g', 'r']))
#[3800,3325,2850,2375,1900,1425,950,713,475,238]
import os
# The directory must exist for other parts of this application to function properly
assert(os.path.isdir(DATA_DIR))
assert(os.path.isdir(EXPERIMENT_MATRICES_DIR))
