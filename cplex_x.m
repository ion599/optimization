function Result = cplex_x(varargin)
clc;

%% Load network dataset
load data/20140310T213327-cathywu-4.mat
A = p.Phi; b = p.f; n = p.n; U = p.L1; block_sizes = p.block_sizes;
noise = p.noise; epsilon = p.epsilon; lambda = p.lambda; w = p.w;
blocks = p.blocks;
x_init2 = rand(n,1);

%% Load small synthetic dataset
% load data/smaller_data.mat
% load data/stevesSmallData.mat

%% Set up problem
% n = size(A,2);
% 
% % Generate initial points
% fprintf('Generate initialization points\n\n')
% tic;
% [x_init,x_init2,x_init3,x_init4,~,~,~,~] = initXZ(n,N,x);
% fprintf('Time to generate initialization points %s sec\n',toc);
tic;

% Documentation: http://tomopt.com/docs/TOMLAB_CPLEX.pdf
Name  = 'QP for route split estimation';
F   = (A'*A);               % Matrix F in 1/2 * x' * F * x + c' * x
c   = -b'*A;                % Vector c in 1/2 * x' * F * x + c' * x
a   = U;                    % Constraint matrix
b_L = ones(size(a,1),1);    % Lower bounds on the linear constraints
b_U = b_L;                  % Upper bounds on the linear constraints
x_L = zeros(n,1);           % Lower bounds on the variables
x_U = inf * ones(n,1);      % Upper bounds on the variables
x_0 = x_init2;              % Starting point, routes by importance
fprintf('Time to set up problem %s\n',toc);
% assert(all(eigs(F) > 0), 'F not PSD');

%% Solver
% tic; load data/stevesSmallData-tomlab.mat; toc;

format compact
fprintf('=====================================================\n');
fprintf('Run very simple QP defined in the Tomlab Quick format\n');
fprintf('=====================================================\n');

Prob = qpAssign(F, c, a, b_L, b_U, x_L, x_U, x_0, 'QPRouteSplit');
% See Table 49 in TOMLAB_SOL.pdf for optPar parameters
Prob.optParam.MaxIter = 2;     % Setting maximum number of iterations
% Prob.optParam.xTol = 10;
% Prob.optParam.bTol = 10;
% Prob.optParam.wait = 0;
Prob.MIP.cpxControl.ITLIM = 2; % Setting maximum number of iterations
Prob.MIP.cpxControl.EPOPT = 10;% Setting optimality tolerance
Prob.SOL.PrintFile = 'QPRouteSplit-print.txt';
Prob.SOL.SummFile =  'QPRouteSplit-summary.txt';
Prob.PriLevOpt = 2;
% QP solver methods: http://www.cs.cornell.edu/w8/iisi/ilog/cplex101/
% usrcplex/solveQP9.html#637915
Prob.MIP.cpxControl.QPMETHOD = 4;

% Prob = ProbCheck(Prob,'cplex');
Result = tomRun('cplex',Prob);
% Result = cplexTL(Prob);
toc

PrintResult(Result);
fprintf('norm(Ax-b): %e\n', norm(A * Result.x_k - b))
fprintf('norm(Ux-1): %e\n', norm(U * Result.x_k - 1))

% Can't run simplex method (1) and can't figure out how to change max
% iterations

% Barrier QP (4) using init2, smaller_data.mat
% iter=9,   norm(Ax-b): 3.104257e-01, Time 390.36 sec (6.51 min)
% iter=5,   norm(Ax-b): 1.972266e-02, Time 9596.87 sec (159.95 min)