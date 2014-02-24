%% Generate Some Synthetic Data
clc; clear all

% dimensions of the problem
n = 100;
m = n/10;

% Generate synthetic data
A = exprnd(100,m,n);
A(A<150) = 0;
x_true = exprnd(1000,n,1);
N = [];
while sum(N) < n-10
    N = [N; 2+floor(4*rand)];
end
N = [N; floor((n-sum(N))/2)];
N = [N; n-sum(N)];
%N = [floor(.1*n); floor(.2*n); floor(.1*n); floor(.3*n); floor(.1*n)];
%N = [N; n-sum(N)];
p = length(N);
k = 0;

%% Set up optimization problem

z_init = zeros(n-p,1);
x_init = zeros(n,1);
for i=1:p
    x_true(k+1:k+N(i)) = x_true(k+1:k+N(i))/sum(x_true(k+1:k+N(i)));
    k = k+N(i);
end
b = A*x_true;
alpha = 0;
funObj = @(z)objective(z,A,N,b,alpha);

%% Compute initial point

z_init = rand(n-p,1);
z_init = PAValgo(z_init,ones(n-p,1),0,1);
x_init = zeros(n,1);

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
z = SPG(funObj,z_init,options);

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
