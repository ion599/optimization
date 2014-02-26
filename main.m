%% Generate Some Synthetic Data
clc; clear all

% Dimensions of the problem
n = 100;
m = n/10;

%% Construct nullspace sparse matrix N
N = [];
% TODO more efficient random partitioning [2 to 5]
while sum(N) < n-10
    N = [N; 2+floor(4*rand)]; % Generate size of blocks randomly
end
N = [N; floor((n-sum(N))/2)];
N = [N; n-sum(N)];
%N = [floor(.1*n); floor(.2*n); floor(.1*n); floor(.3*n); floor(.1*n)];
%N = [N; n-sum(N)];
p = length(N);

assert(sum(N) == n) % Check that nullspace N accounts for number of routes

%% Generate synthetic data (A, x_true, b)
A = exprnd(100,m,n);
A(A<150) = 0;

% Generate x_true; for each OD pair, normalize each block of x
x_true = exprnd(1000,n,1);
k = 0; % block index into x_true
for i=1:p
    x_true(k+1:k+N(i)) = x_true(k+1:k+N(i))/sum(x_true(k+1:k+N(i)));
    k = k+N(i);
end
b = A*x_true;

z_true = zeros(n-p,1);
ind = 1;
k = 0;
for i=1:p
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
z_init = rand(n-p,1);
k=0;
for i=1:p
    z_init(k+1:k+N(i)-1) = PAValgo(z_init(k+1:k+N(i)-1),ones(N(i)-1,1),0,1);
    k = k+N(i)-1;
end
x_init = zeros(n,1);

% Compute x from z projected onto the feasible set
ind = 1;
k = 0;
for i=1:p
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
for i=1:p
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
