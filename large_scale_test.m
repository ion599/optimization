%% Generate Some Synthetic Data
clc; clear all

% Preprocessing U to Nf
% for i=1:102720 N(i)=sum(U(i,:)); end

load('data/stevesSmallData.mat')

% Dimensions of the problem
n = size(A,2);
m = size(A,1);

lenN = length(N);
assert(sum(N) == n) % Check that nullspace N accounts for number of routes
x_true = x;
z_true = x2z(x_true,N);

%% Generate x_init = rand

x_init = rand(n,1);
k=0;
for i=1:lenN
    x_init(k+1:k+N(i)) = x_init(k+1:k+N(i))/sum(x_init(k+1:k+N(i)));
    k = k+N(i);
end
z_init = x2z(x_init,N);

%% Generate x_init2,3 = routes by importance

fprintf('Generate initialization points\n\n')

x_init2 = zeros(n,1);
x_init3 = zeros(n,1);
k=0;
for i=1:lenN
    [~,id] = sort(x_true(k+1:k+N(i)));
    [~,id2] = sort(id);
    x_init2(k+1:k+N(i)) = id2/sum(id2);
    x_init3(k+1:k+N(i)) = 10.^(id2-1)/sum(10.^(id2-1));
    k = k+N(i);
end
z_init2 = x2z(x_init2,N);
z_init3 = x2z(x_init3,N);

%% Set up optimization problem
noise = 0; % if b=bExact

alpha = (100*(noise^2)*(noise>.1))*(1-x_init3);
b2 = b+normrnd(0,noise,m,1);
%funObj = @(z)objective(z,A,N,b2,zeros(n,1)); % no penalization (L2)
funObj = @(z)objective(z,A,N,b2,alpha);

%% Set Optimization Options
gOptions.maxIter = 1000;
gOptions.verbose = 1; % Set to 0 to turn off output
options.corrections = 10; % Number of corrections to store for L-BFGS methods
maxIter = 20;

%% Run Projected gradient

fprintf('\nProjected Gradient\n\n');
options = gOptions;
tic
x3 = z2x(SPG(funObj,z_init3,N,options),N);
timeSPG = toc;

%% Run l-BFGS

fprintf('\nl-BFGS\n\n');

tic
y3 = z2x(lbfgs2(funObj,z_init3,N,500,options),N);
timeLBFGS = toc;

%% Display performance

fprintf('\nProjected gradient without l2-regularization\n\n');

fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\nmax|x_init-x_true|: %.2f\n\n\n', ...
    norm(A*x3-b), norm(A*x_init3-b), max(abs(x3-x_true)), max(abs(x_true-x_init3)))

fprintf('\nLBFGS without l2-regularization\n\n');

fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\nmax|x_init-x_true|: %.2f\n\n\n', ...
    norm(A*y3-b), norm(A*x_init3-b), max(abs(y3-x_true)), max(abs(x_true-x_init3)))

%% Display results
%{
blocks = [];
for i=1:lenN
    blocks = [blocks; [i*ones(N(i),1) zeros(N(i),1)]];
    blocks(sum(N(1:i)),2) = 1;
end
results = [blocks(:,1) x x_true abs(x-x_true)];
fprintf('blocks x x_true |x-x_true|\n')
for i=1:n
    fprintf('%i      %.2f  %.2f  %.2f\n', results(i,:))
    if blocks(i,2)
        fprintf('\n')
    end
end
results = [blocks(:,1) x3 x_true abs(x3-x_true)];
fprintf('blocks x3 x_true |x3-x_true|\n')
for i=1:n
    fprintf('%i      %.2f  %.2f  %.2f\n', results(i,:))
    if blocks(i,2)
        fprintf('\n')
    end
end
%}