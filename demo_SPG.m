%% Generate Some Synthetic Data
clear all

nInstances = 200;
nVars = 250;
sparsityFactor = .5;
flipFactor = .1;
X = [ones(nInstances,1) randn(nInstances,nVars-1)];
w = randn(nVars,1).*(rand(nVars,1) < sparsityFactor);
y = sign(X*w);
flipPos = rand(nInstances,1) < flipFactor;
y(flipPos) = -y(flipPos);
        
%% Set up optimization problem
w_init = zeros(nVars,1);

lambda = 1;
lambdaVect = lambda*[0;ones(nVars-1,1)];

funObj = @(w)LogisticLoss(w,X,y);

%% Set Optimization Options
gOptions.maxIter = 2000;
gOptions.verbose = 1; % Set to 0 to turn off output
options.corrections = 10; % Number of corrections to store for L-BFGS methods

%% Run Solvers

fprintf('Spectral Projected Gradient\n');
options = gOptions;
wSPG = L1General2_SPG(funObj,w_init,lambdaVect,options);