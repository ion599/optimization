% Sets up local test parameters
% Should be run from matlab/

DATA_DIR = sprintf('%s/data',pwd);
DATA_CV_DIR = sprintf('%s/CV',DATA_DIR);
if ~exist(DATA_DIR,'dir')
    mkdir(DATA_DIR);
end