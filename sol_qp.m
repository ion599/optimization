function Result = sol_qp(varargin)
clc;

%% Load network dataset
% load data/20140310T213327-cathywu-4.mat
% A = p.Phi; b = p.f; n = p.n; U = p.L1; block_sizes = p.block_sizes;
% noise = p.noise; epsilon = p.epsilon; lambda = p.lambda; w = p.w;
% blocks = p.blocks;

%% Load small synthetic dataset
load('data/smaller_data.mat')
% load data/stevesSmallData.mat
n = size(A,2);

% Generate initial points
fprintf('Generate initialization points\n\n')
tic;
[x_init,x_init2,x_init3,x_init4,z_init,z_init2,z_init3,z_init4] = ...
    initXZ(n,N,x);
fprintf('Time to generate initialization points %s\n',toc);

tic;
% Documentation: http://tomopt.com/docs/TOMLAB_SNOPT.pdf
Name  = 'QP for route split estimation';
F   = A'*A;                 % Matrix F in 1/2 * x' * F * x + c' * x
c   = -b'*A;                % Vector c in 1/2 * x' * F * x + c' * x
a   = U;                    % Constraint matrix
b_L = ones(size(a,1),1);    % Lower bounds on the linear constraints
b_U = b_L;                  % Upper bounds on the linear constraints
x_L = zeros(n,1);           % Lower bounds on the variables
x_U = inf * ones(n,1);      % Upper bounds on the variables
x_0 = x_init2;              % Starting point, routes by importance
fprintf('Time to set up problem %s\n',toc);

%% Solver
% tic; load data/stevesSmallData-tomlab.mat; toc;

%   x_min and x_max only needed if doing plots
%   x_min = [-1 -1 ];      % Plot region lower bound parameters
%   x_max = [ 6  6 ];      % Plot region upper bound parameters

% Use the Tomlab Quick (TQ) format
%
% Call the Tomlab qpAssign routine that creates a structure with all
% problem information.

format compact
fprintf('=====================================================\n');
fprintf('Run very simple QP defined in the Tomlab Quick format\n');
fprintf('=====================================================\n');

Prob = qpAssign(F, c, a, b_L, b_U, x_L, x_U, x_0, 'QPRouteSplit');
% See Table 49 in TOMLAB_SOL.pdf for optPar parameters
% Prob.SOL.optPar(30)   = 200;     % Setting maximum number of iterations
Prob.SOL.optPar(5)    = 1;     % Setting print frequency
Prob.SOL.optPar(6)    = 1;     % Setting summary frequency
Prob.SOL.optPar(12)   = 10;    % Setting optimality tolerance
Prob.SOL.PrintFile = 'QPRouteSplit-print.txt';
Prob.SOL.SummFile =  'QPRouteSplit-summary.txt';
Prob.PriLevOpt = 2;

% Prob = ProbCheck(Prob,'qp-minos');
Result = tomRun('qp-minos',Prob);
% Result = qpSolve(Prob);       % Generic QP solver
toc

% When the Projected gradient gPr is very small, the minimum is found
% with good accuracy

PrintResult(Result);
fprintf('norm(Ax-b): %s\n', norm(A * Result.x_k - b))

% With random initial point
% iter=2,       L2 error: 5.996739e+05
% iter=20,      L2 error: 5.888152e+05
% iter=200,     L2 error: 6.035505e+05
% iter=25556,   L2 error: 1.545041e-06, Time: 6202.34 seconds (103.372 min)

% With heuristic initial point x_init2
% iter=1602,    L2 error: 1.730858e-06 , Time: 5915.45 seconds (98.591 min)
% iter=1602,    norm(Ax-b): 1.730858e-06,Time: 3050.89 seconds
