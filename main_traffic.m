%% Generate Some Synthetic Data
clc; clear all

%load('data/20140228T232249-cathywu-5.mat');
load('data/20140228T232250-cathywu-7.mat')
%load('data/20140228T232250-cathywu-8.mat');
%load('data/20140228T232251-cathywu-9.mat');

% Dimensions of the problem
n = size(p.Phi,2);
m = size(p.Phi,1);

N = p.block_sizes;
lenN = length(N);
assert(sum(N) == n) % Check that nullspace N accounts for number of routes

A = p.Phi;
x_true = p.real_a;
b = p.f;

z_true = x2z(x_true,N);

%% Set up optimization problem
alpha = 0; % no penalization (L2)
funObj = @(z)objective(z,A,N,b,alpha);

%% Compute three initial points
x_init = rand(n,1);
x_init2 = x_true+normrnd(0,.3,n,1);
x_init2(x_init2<0) = 0;
x_init3 = zeros(n,1);
k=0;
for i=1:lenN
    x_init(k+1:k+N(i)) = x_init(k+1:k+N(i))/sum(x_init(k+1:k+N(i)));
    if norm(x_init2(k+1:k+N(i))) == 0
        x_init2(k+1:k+N(i)) = ones(N(i),1);
    end
    x_init2(k+1:k+N(i)) = x_init2(k+1:k+N(i))/sum(x_init2(k+1:k+N(i)));
    [~,id] = sort(x_true(k+1:k+N(i)));
    [~,id2] = sort(id);
    x_init3(k+1:k+N(i)) = id2/sum(id2);
    k = k+N(i);
end
z_init = x2z(x_init,N);
z_init2 = x2z(x_init2,N);
z_init3 = x2z(x_init3,N);

%% Set Optimization Options
gOptions.maxIter = 100;
gOptions.verbose = 0; % Set to 0 to turn off output
options.corrections = 10; % Number of corrections to store for L-BFGS methods

%% Run Solver

fprintf('Spectral Projected Gradient\n\n');
options = gOptions;
x = z2x(SPG(funObj,z_init,N,options),N);
%x2 = z2x(SPG(funObj,z_init2,N,options),N);
x3 = z2x(SPG(funObj,z_init3,N,options),N);

%% Display performance

fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*x-b), norm(A*x_init-b), max(abs(x-x_true)))
%fprintf('norm(A*x2-b): %8.5e\nnorm(A*x_init2-b): %8.5e\nmax|x2-x_true|: %.2f\n\n\n', ...
%    norm(A*x2-b), norm(A*x_init2-b), max(abs(x2-x_true)))
fprintf('norm(A*x3-b): %8.5e\nnorm(A*x_init3-b): %8.5e\nmax|x3-x_true|: %.2f\n\n\n', ...
    norm(A*x3-b), norm(A*x_init3-b), max(abs(x3-x_true)))

%% Display results

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
