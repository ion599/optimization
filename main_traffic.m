%% Generate Some Synthetic Data
clc; clear all

load('model/data/params-traffic/20140228T232249-cathywu-6.mat');

% Dimensions of the problem
n = size(p.Phi,2);
m = size(p.Phi,1);

N = p.block_sizes;
lenN = length(N);
assert(sum(N) == n) % Check that nullspace N accounts for number of routes

A = p.Phi;
x_true = p.real_a;
b = p.f;

z_true = zeros(n-lenN,1);
ind = 1;
k = 0;
for i=1:lenN
    z_true(ind) = x_true(ind+k);
    ind = ind+1;
    for j=1:(N(i)-2)
        z_true(ind) = x_true(ind+k) + z_true(ind-1);
        ind = ind+1;
    end
    k = k+1;
end

%% Set up optimization problem
alpha = 0; % no penalization (L2)
funObj = @(z)objective(z,A,N,b,alpha);

%% Compute initial point
z_init = rand(n-lenN,1);
k=0;
for i=1:lenN
    z_init(k+1:k+N(i)-1) = PAValgo(z_init(k+1:k+N(i)-1),ones(N(i)-1,1),0,1);
    k = k+N(i)-1;
end

% Compute x from z projected onto the feasible set
x_init = zeros(n,1);
ind = 1;
k = 0;
for i=1:lenN
    x_init(ind) = z_init(ind-k);
    ind = ind+1;
    for j=2:(N(i)-1)
        x_init(ind) = z_init(ind-k)-z_init(ind-k-1);
        ind = ind+1;
    end
    x_init(ind) = -z_init(ind-k-1)+1;
    k = k+1;
    ind = ind+1;
end

%% Set Optimization Options
gOptions.maxIter = 100;
gOptions.verbose = 1; % Set to 0 to turn off output
options.corrections = 10; % Number of corrections to store for L-BFGS methods

%% Run Solver

fprintf('Spectral Projected Gradient\n');
options = gOptions;
z = SPG(funObj,z_init,N,options);

x = zeros(n,1);
ind = 1;
k = 0;
for i=1:lenN
    x(ind) = z(ind-k);
    ind = ind+1;
    for j=2:(N(i)-1)
        x(ind) = z(ind-k)-z(ind-k-1);
        ind = ind+1;
    end
    x(ind) = -z(ind-k-1)+1;
    k = k+1;
    ind = ind+1;
end
fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\n', norm(A*x-b), norm(A*x_init-b))
