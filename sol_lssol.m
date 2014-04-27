function Result = sol_lssol(varargin)
clc;

%% Load network dataset
load data/20140310T213327-cathywu-4.mat
A = p.Phi; b = p.f; n = p.n; U = p.L1; block_sizes = p.block_sizes;
noise = p.noise; epsilon = p.epsilon; lambda = p.lambda; w = p.w;
blocks = p.blocks;

%% Load small synthetic dataset
% load data/stevesSmallData.mat
% n = size(A,2);
    
tic;
% Documentation: http://tomopt.com/docs/TOMLAB_SNOPT.pdf
Name  = 'QP for route split estimation';
C   = A;                    % Matrix F in 1/2 * x' * F * x + c' * x
d   = b;                % Vector c in 1/2 * x' * F * x + c' * x
a   = U;                    % Constraint matrix
b_L = ones(size(a,1),1);    % Lower bounds on the linear constraints
b_U = b_L;                  % Upper bounds on the linear constraints
x_L = zeros(n,1);           % Lower bounds on the variables
x_U = inf * ones(n,1);      % Upper bounds on the variables
x_0 = rand(n,1);            % Starting point
fprintf('Set up problem (%s)\n',toc);

%% Load direct inputs to TOMLAB
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

% Quickguide: http://tomopt.com/docs/quickguide/quickguide012.php
Prob = llsAssign(C, d, x_L, x_U, Name, x_0, ...
                            [], [], [], ...
                            a, b_L, b_U);
% See Table 49 in TOMLAB_SOL.pdf for optPar parameters
Prob.SOL.optPar(30)   = 2;     % Setting maximum number of iterations
Prob.SOL.optPar(5)    = 1;     % Setting print frequency
Prob.SOL.optPar(6)    = 1;     % Setting summary frequency
Prob.SOL.PrintFile = 'QPRouteSplit-print.txt';
Prob.SOL.SummFile =  'QPRouteSplit-summary.txt';
Prob.PriLevOpt = 2;

tic; Prob = ProbCheck(Prob,'lssol'); fprintf('Checked problem (%s)\n',toc);
Result = tomRun('lssol',Prob);
% Result = qpSolve(Prob);       % Generic QP solver
toc

% When the Projected gradient gPr is very small, the minimum is found
% with good accuracy

PrintResult(Result); 
fprintf('L2 error: %s\n', norm(A * Result.x_k - b))

% iter=2, L2 error: 5.996739e+05
