ACCEPTED_LOG_LEVELS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'WARN']
DATA_DIR = '/home/ubuntu/traffic/datasets/Phi'
EXPERIMENT_MATRICES_DIR = 'experiment_matrices/experiment_matrices'
ESTIMATION_INFO_DIR = 'estimation_info'

import os
# The directory must exist for other parts of this application to function properly
assert(os.path.isdir(DATA_DIR))
assert(os.path.isdir(DATA_DIR+'/'+EXPERIMENT_MATRICES_DIR))
