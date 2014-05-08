function Results = sol_qp_z_loop(varargin)
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
[~,~,~,x_init4, ~,~,~,z_init4] = initXZ(n,N,x);
fprintf('Time to generate initialization points %s sec\n',toc);

tic;
[x0,N2] = computeSparseParam(n,N);
fprintf('Time to compute sparse params %s sec\n',toc);

%%
tic;
% Documentation: http://tomopt.com/docs/TOMLAB_SNOPT.pdf
Name  = 'QP for route split estimation';
F   = N2'*A'*A*N2;              % Matrix F in 1/2 * x' * F * x + c' * x
c   = (A*x0-b)'*A*N2;           % Vector c in 1/2 * x' * F * x + c' * x
a   = N2;                       % Constraint matrix
b_L = -x0;                      % Lower bounds on the linear constraints
b_U = inf * ones(size(a,1),1);  % Upper bounds on the linear constraints
z_0 = z_init4;                  % Starting point
x_0 = x_init4;
fprintf('Time to set up problem %s sec\n',toc);

%% Storing state
Results = struct;
Results.x = x_0;
Results.cost = norm(A * x_0 - b);
Results.Result = [];

%% Solve the first time

step = 10;

format compact
fprintf('=====================================================\n');
fprintf('Run very simple QP defined in the Tomlab Quick format\n');
fprintf('=====================================================\n');

Prob = qpAssign(F, c, a, b_L, b_U, [], [], z_0, 'QPRouteSplit');
% See Table 49 in TOMLAB_SOL.pdf for optPar parameters
Prob.SOL.optPar(30)   = step;     % Setting maximum number of iterations
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

Result = tomRun('qp-minos',Prob);
x_k = x0+N2*Result.x_k;
PrintResult(Result);
cost = norm(A * x_k - b);
fprintf('norm(Ax-b): %s\n', cost)

Results.cost = [Results.cost cost];
Results.x = [Results.x x_k];
Results.Result = [Results.Result Result];

%% Warm start

while Result.ExitFlag ~= 0
    Prob = WarmDefSOL('qp-minos', Prob, Result);
    Result = tomRun('qp-minos', Prob);

    x_k = x0+N2*Result.x_k;
    PrintResult(Result);
    cost = norm(A * x_k - b);
    fprintf('norm(Ax-b): %s\n', cost)
    
    Results.cost = [Results.cost cost];
    Results.x = [Results.x x_k];
    Results.Result = [Results.Result Result];
end

% Heuristic init point (by importance)
% iter=299,     norm(Ax-b): 4.389436e+02, Time 362.87 sec (6.0478 min)
% iter=567,     norm(Ax-b): 1.824042e+02, Time 669.89 sec (11.165 min)
% iter=800,     norm(Ax-b): 6.882624e+01, Time 928.07 sec
% iter=953,     norm(Ax-b): 3.510522e+01, Time 1196.22 sec (19.937 min)
% iter=1254,    norm(Ax-b): 2.727871e-08, Time 1622.74 sec (27.0457 min)
% iter=1254,    norm(Ax-b): 2.727871e-08, Time 1610.15 sec

% Heuristic init point (uniform)
% iter=800,     norm(Ax-b): 4.130360e+02, Time 930.65 sec
% iter=810,     norm(Ax-b): 4.116748e+02, Time +167.68 sec
% iter=820,     norm(Ax=b): 4.102483e+02, Time +21.51 sec