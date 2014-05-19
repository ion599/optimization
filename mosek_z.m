function res = mosek_z(varargin)
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
fprintf('Generate initialization points\n')
tic;
[x_init,x_init2,x_init3,z_init,z_init2,z_init3] = initXZ(n,N,x);
fprintf('Time to generate initialization points %s sec\n',toc);

tic;
[x0,N2] = computeSparseParam(n,N);
fprintf('Time to compute sparse params %s sec\n',toc);

tic;
% Documentation: http://tomopt.com/docs/TOMLAB_SNOPT.pdf
Name  = 'QP for route split estimation';
temp = A*N2;
q   = (temp'*temp);             % Matrix F in 1/2 * x' * F * x + c' * x
c   = (A*x0-b)'*temp;           % Vector c in 1/2 * x' * F * x + c' * x
a   = N2;                       % Constraint matrix
blc = -x0;                      % Lower bounds on the linear constraints
buc = inf * ones(size(a,1),1);  % Upper bounds on the linear constraints
z_0 = z_init2;                  % Starting point
fprintf('Time to set up problem %s sec\n',toc);

% Define the data.
% bas.skc      = repmat('LL', size(blc));
% bas.skx      = repmat('BS', size(z_0));
% bas.xc       = blc;
% bas.xx       = z_0;
% prob.sol.bas = bas;

% First the lower triangular part of q in the objective 
% is specified in a sparse format. The format is:
%
%   Q(prob.qosubi(t),prob.qosubj(t)) = prob.qoval(t), t=1,...,4

q_lower = tril(q);
[i,j,s] = find(q_lower);
prob.qosubi = i;
prob.qosubj = j;
prob.qoval  = s;

prob.c = c;
prob.a = a;
prob.blc  = blc;    % Lower bounds of constraints.
prob.buc  = buc;    % Upper bounds of constraints.
prob.blx  = [];     % Lower bounds of variables.
prob.bux = [];      % Upper bounds of variables. There are no bounds.

%% Load precomputed small synthetic dataset
% tic; load data/stevesSmallData-tomlab-mosek-z.mat; toc;
% tic; load data/stevesSmallData-tomlab-mosek-z-prob.mat; toc;

%% Solver

fprintf('=====================================================\n');
fprintf('Run very simple QP defined in MOSEK \n');
fprintf('=====================================================\n');

param.MSK_IPAR_OPTIMIZER = 'MSK_OPTIMIZER_FREE';
% param.MSK_IPAR_OPTIMIZER = 'MSK_OPTIMIZER_INTPNT';

% Optimize the problem.
[r,res] = mosekopt('minimize',prob, param);

% Display return code.
fprintf('Return code: %d\n',r);

% Display primal solution for the constraints.
res.sol.itr.xc'

% Display primal solution for the variables.
res.sol.itr.xx'

toc

% When the Projected gradient gPr is very small, the minimum is found
% with good accuracy

x_k = x0+N2*res.sol.itr.xx;
fprintf('norm(Ax-b): %s\n', norm(A * x_k - b))
