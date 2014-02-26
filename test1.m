%% Generate Some Synthetic Data
% Small test program: This runs like main, but initialized at x_true

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

for i=1:p
    x_true(k+1:k+N(i)) = x_true(k+1:k+N(i))/sum(x_true(k+1:k+N(i)));
    k = k+N(i);
end
b = A*x_true;
alpha = 0;

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

funObj = @(z)objective(z,A,N,b,alpha);

%% Set Optimization Options
gOptions.maxIter = 100;
gOptions.verbose = 1; % Set to 0 to turn off output
options.corrections = 10; % Number of corrections to store for L-BFGS methods

%% Run Solver

fprintf('Spectral Projected Gradient\n');
options = gOptions;
z = SPG(funObj,z_true,options);

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

