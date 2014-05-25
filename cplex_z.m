function Result = cplex_z(varargin)
clc;

%% Load network dataset
% load data/20140310T213327-cathywu-4.mat
% A = p.Phi; b = p.f; n = p.n; U = p.L1; block_sizes = p.block_sizes;
% noise = p.noise; epsilon = p.epsilon; lambda = p.lambda; w = p.w;
% blocks = p.blocks;

%% Load small synthetic dataset
load data/stevesSmallData.mat

%% Load large synthetic dataset
% load data/stevesData.mat
% x = x_true;
% b = bExact;

%% Set up problem
n = size(A,2);

% Generate initial points
fprintf('Generate initialization points\n\n')
tic;
[x_init,x_init2,x_init3,z_init,z_init2,z_init3] = initXZ(n,N,x);
fprintf('Time to generate initialization points %s sec\n',toc);

tic;
[x0,N2] = computeSparseParam(n,N);
fprintf('Time to compute sparse params %s sec\n',toc);

tic;
% Documentation: http://tomopt.com/docs/TOMLAB_CPLEX.pdf
Name  = 'QP for route split estimation';
temp = A*N2;
F   = (temp'*temp);             % Matrix F in 1/2 * x' * F * x + c' * x
c   = (A*x0-b)'*temp;           % Vector c in 1/2 * x' * F * x + c' * x
a   = N2;                       % Constraint matrix
b_L = -x0;                      % Lower bounds on the linear constraints
b_U = inf * ones(size(a,1),1);  % Upper bounds on the linear constraints
z_0 = z_init2;                  % Starting point
fprintf('Time to set up problem %s sec\n',toc);

%% Load large precomputed synthetic problem
% tic; load data/stevesSmallData-tomlab-z.mat; toc;

%% Solver

format compact
fprintf('=====================================================\n');
fprintf('Run very simple QP defined in the Tomlab Quick format\n');
fprintf('=====================================================\n');

Prob = qpAssign(F, c, a, b_L, b_U, [], [], z_0, 'QPRouteSplit');
% See Table 49 in TOMLAB_SOL.pdf for optPar parameters
% Prob.SOL.optPar(30)   = 200;     % Setting maximum number of iterations
Prob.SOL.optPar(5)    = 1;     % Setting print frequency
Prob.SOL.optPar(6)    = 1;     % Setting summary frequency
Prob.SOL.optPar(12)   = 1000;    % Setting optimality tolerance
Prob.SOL.optPar(48)   = 400;    % Setting superbasics limit 
% Superbasics limit: http://www.stanford.edu/group/SOL/software/snoptHelp/
% The_SPECS_file/Description_of_optional_parameters/
% snopt_Specs_Description_Superbasics_limit.htm
Prob.SOL.PrintFile = 'QPRouteSplit-print.txt';
Prob.SOL.SummFile =  'QPRouteSplit-summary.txt';
Prob.PriLevOpt = 2;
% QP solver methods: http://www.cs.cornell.edu/w8/iisi/ilog/cplex101/
% usrcplex/solveQP9.html#637915
Prob.MIP.cpxControl.QPMETHOD = 6;
toc;

%%
Prob = ProbCheck(Prob,'cplex');
Result = tomRun('cplex',Prob);
toc

% When the Projected gradient gPr is very small, the minimum is found
% with good accuracy

x_k = x0+N2*Result.x_k;
PrintResult(Result);
fprintf('norm(Ax-b): %s\n', norm(A * x_k - b))
fprintf('norm(Ux-1): %e\n', norm(U * x_k - 1))

% Barrier QP (4) using init2
% iter=8,   norm(Ax-b): 3.170721e-04, Time 6054.040000 sec (100.901 min)

% Sifting QP solver (5) using init2
% iter=8,   norm(Ax-b): 3.068989e-04, Time 5623.960000 sec


% Primal Simplex QP
% Error that Q is not PSD (weird)