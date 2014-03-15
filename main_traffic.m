%% Generate Some Synthetic Data
clc; clear all

%load('data/20140228T232249-cathywu-5.mat');
%load('data/20140228T232250-cathywu-7.mat')
%load('data/20140228T232250-cathywu-8.mat');
%load('data/20140228T232251-cathywu-9.mat');
load('data/20140310T213327-cathywu-4.mat')

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

%% Generate x_init = rand

x_init = rand(n,1);
k=0;
for i=1:lenN
    x_init(k+1:k+N(i)) = x_init(k+1:k+N(i))/sum(x_init(k+1:k+N(i)));
    k = k+N(i);
end
z_init = x2z(x_init,N);

%{
%% Generate x_init2 = x_true + noise

sigma = 0.3; % noise added to x_true
x_init2 = x_true+normrnd(0,sigma,n,1);
x_init2(x_init2<0) = 0;
k = 0;
for i=1:lenN
    if norm(x_init2(k+1:k+N(i))) == 0
        x_init2(k+1:k+N(i)) = ones(N(i),1);
    end
    x_init2(k+1:k+N(i)) = x_init2(k+1:k+N(i))/sum(x_init2(k+1:k+N(i)));
    k = k+N(i);
end
z_init2 = x2z(x_init2,N);
%}
%% Generate x_init2,3 = routes by importance

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
noise = 0.3; %noise added to b

alpha = (100*(noise^2)*(noise>.1))*(1-x_init3);
b2 = b+normrnd(0,noise,m,1);
funObj = @(z)objective(z,A,N,b2,zeros(n,1)); % no penalization (L2)
funObj2 = @(z)objective(z,A,N,b2,alpha);

fprintf('Pre ADMM\n\n')
tau = 1.5;
[C,d] = preADMM(N,A,b2,zeros(n,1),tau);
[C2,d2] = preADMM(N,A,b2,alpha,tau);
eta = 0.999;

%% Set Optimization Options
gOptions.maxIter = 200;
gOptions.verbose = 0; % Set to 0 to turn off output
options.corrections = 10; % Number of corrections to store for L-BFGS methods
maxIter = 200;

%% Run Solver

fprintf('Spectral Projected Gradient\n\n');
options = gOptions;
tic
%x1 = z2x(SPG(funObj,z_init,N,options),N);
%x2 = z2x(SPG(funObj,z_init2,N,options),N);
%x3 = z2x(SPG(funObj,z_init3,N,options),N);

x12 = z2x(SPG(funObj2,z_init,N,options),N);
x22 = z2x(SPG(funObj2,z_init2,N,options),N);
x32 = z2x(SPG(funObj2,z_init3,N,options),N);
timeSPG = toc;

%% Run ADMM

fprintf('\nADMM\n\n');
%testADMM(N,A,b,alpha,1.5)
tic
%y1 = z2x(ADMM(funObj, z_init, C, d, N, tau, maxIter), N);
%y2 = z2x(ADMM(funObj, z_init2, C, d, N, tau, maxIter), N);
%y3 = z2x(ADMM(funObj, z_init3, C, d, N, tau, maxIter), N);

y12 = z2x(ADMM(funObj2, z_init, C2, d2, N, tau, maxIter), N);
y22 = z2x(ADMM(funObj2, z_init2, C2, d2, N, tau, maxIter), N);
y32 = z2x(ADMM(funObj2, z_init3, C2, d2, N, tau, maxIter), N);
timeADMM = toc;

%% Run FADMM

fprintf('\nFADMM\n\n');

tic
%z1 = z2x(FADMM(funObj, z_init, C, d, N, tau, maxIter), N);
%z2 = z2x(FADMM(funObj, z_init2, C, d, N, tau, maxIter), N);
%z3 = z2x(FADMM(funObj, z_init3, C, d, N, tau, maxIter), N);

z12 = z2x(FADMM(funObj2, z_init, C2, d2, N, tau, maxIter), N);
z22 = z2x(FADMM(funObj2, z_init2, C2, d2, N, tau, maxIter), N);
z32 = z2x(FADMM(funObj2, z_init3, C2, d2, N, tau, maxIter), N);
timeFADMM = toc;

%% Run FADMM with restart

fprintf('\nFADMM with restart\n\n');

tic
%a1 = z2x(FADMM(funObj, z_init, C, d, N, tau, maxIter), N);
%a2 = z2x(FADMM(funObj, z_init2, C, d, N, tau, maxIter), N);
%a3 = z2x(FADMM(funObj, z_init3, C, d, N, tau, maxIter), N);

a12 = z2x(FADMM2(funObj2, z_init, C2, d2, N, tau, eta, maxIter), N);
a22 = z2x(FADMM2(funObj2, z_init2, C2, d2, N, tau, eta, maxIter), N);
a32 = z2x(FADMM2(funObj2, z_init3, C2, d2, N, tau, eta, maxIter), N);
timeFADMM2 = toc;

%% Display performance SPG

fprintf('\nnoise=%.2f\n\n',noise)
%{
fprintf('\nSPG without l2-regularization with init 1,2,3\n\n')

fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*x1-b), norm(A*x_init-b), max(abs(x1-x_true)))
fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*x2-b), norm(A*x_init2-b), max(abs(x2-x_true)))
fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*x3-b), norm(A*x_init3-b), max(abs(x3-x_true)))
%}
fprintf('\nSPG with l2-regularization and init1,2,3\n\n')

fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*x12-b), norm(A*x_init-b), max(abs(x12-x_true)))
fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*x22-b), norm(A*x_init2-b), max(abs(x22-x_true)))
fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*x32-b), norm(A*x_init3-b), max(abs(x32-x_true)))

%% Display performance ADMM
%{
fprintf('\nADMM without l2-regularization with init 1,2,3\n\n');

fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*y1-b), norm(A*x_init-b), max(abs(y1-x_true)))
fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*y2-b), norm(A*x_init2-b), max(abs(y2-x_true)))
fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*y3-b), norm(A*x_init3-b), max(abs(y3-x_true)))
%}
fprintf('\nADMM with l2-regularization with init 1,2,3\n\n');

fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*y12-b), norm(A*x_init-b), max(abs(y12-x_true)))
fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*y22-b), norm(A*x_init2-b), max(abs(y22-x_true)))
fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*y32-b), norm(A*x_init3-b), max(abs(y32-x_true)))

%% Display performance FADMM
%{
fprintf('\nFADMM without l2-regularization with init 1,2,3\n\n');

fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*z1-b), norm(A*x_init-b), max(abs(z1-x_true)))
fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*z2-b), norm(A*x_init2-b), max(abs(z2-x_true)))
fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*z3-b), norm(A*x_init3-b), max(abs(z3-x_true)))
%}
fprintf('\nFADMM with l2-regularization with init 1,2,3\n\n');

fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*z12-b), norm(A*x_init-b), max(abs(z12-x_true)))
fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*z22-b), norm(A*x_init2-b), max(abs(z22-x_true)))
fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*z32-b), norm(A*x_init3-b), max(abs(z32-x_true)))

%% Display performance FADMM with restart

fprintf('\nFADMM with restart with l2-regularization with init 1,2,3\n\n');

fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*a12-b), norm(A*x_init-b), max(abs(a12-x_true)))
fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*a22-b), norm(A*x_init2-b), max(abs(a22-x_true)))
fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*a32-b), norm(A*x_init3-b), max(abs(a32-x_true)))

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