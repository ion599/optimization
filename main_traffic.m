%% Generate Some Synthetic Data
% clc; clear all

%load('data/20140228T232249-cathywu-5.mat');
% load('data/20140228T232250-cathywu-7.mat')
% load('data/20140228T232250-cathywu-8.mat');
% load('data/20140228T232251-cathywu-9.mat');
% load('data/20140310T213327-cathywu-4.mat');

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

%% Generate x_init3 = routes by importance

x_init3 = zeros(n,1);
k=0;
for i=1:lenN
    [~,id] = sort(x_true(k+1:k+N(i)));
    [~,id2] = sort(id);
    x_init3(k+1:k+N(i)) = id2/sum(id2);
    k = k+N(i);
end
z_init3 = x2z(x_init3,N);

%% Generate x_init4 = routes by 10 .^ importance

x_init4 = zeros(n,1);
k=0;
for i=1:lenN
    [~,id] = sort(x_true(k+1:k+N(i)));
    [~,id2] = sort(id);
    x_init4(k+1:k+N(i)) = 10.^id2/sum(10.^id2);
    k = k+N(i);
end
z_init4 = x2z(x_init4,N);

%% Set up optimization problem
noise = 0.0; %noise added to b

% alpha is for weighted L2 regularization
alpha = (100*(noise^2)*(noise>.1))*(1-x_init3);
b2 = b+normrnd(0,noise,m,1);
funObj = @(z)objective(z,A,N,b2,zeros(n,1)); % no penalization (L2)
funObj2 = @(z)objective(z,A,N,b2,alpha);

%% Set Optimization Options
gOptions.maxIter = 400;
gOptions.verbose = 0; % Set to 0 to turn off output
options.corrections = 10; % Number of corrections to store for L-BFGS methods

%% Run Solver

fprintf('Spectral Projected Gradient\n\n');
options = gOptions;
x = z2x(SPG(funObj,z_init,N,options),N); % random x_init, no L2
tic; x2 = z2x(SPG(funObj,z_init3,N,options),N); toc; % good x_init, no L2
x3 = z2x(SPG(funObj2,z_init3,N,options),N); % good x_init, L2
x4 = z2x(SPG(funObj,z_init4,N,options),N); % extreme x_init, no L2

%% Display performance

fprintf('noise %f\n\n', noise);
fprintf('random x_init, no L2\nnorm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\n\n\n', ...
    norm(A*x-b), norm(A*x_init-b), max(abs(x-x_true)))
fprintf('good x_init, no L2\nnorm(A*x2-b): %8.5e\nnorm(A*x_init3-b): %8.5e\nmax|x2-x_true|: %.2f\n\n\n', ...
    norm(A*x2-b), norm(A*x_init3-b), max(abs(x2-x_true)))
fprintf('good x_init, L2\nnorm(A*x3-b): %8.5e\nnorm(A*x_init3-b): %8.5e\nmax|x3-x_true|: %.2f\n\n\n', ...
    norm(A*x3-b), norm(A*x_init3-b), max(abs(x3-x_true)))
fprintf('extreme x_init, no L2\nnorm(A*x2-b): %8.5e\nnorm(A*x_init4-b): %8.5e\nmax|x4-x_true|: %.2f\n\n\n', ...
    norm(A*x4-b), norm(A*x_init4-b), max(abs(x4-x_true)))

%% Display results
%{
blocks = [];
for i=1:lenN
    blocks = [blocks; [i*ones(N(i),1) zeros(N(i),1)]];
    blocks(sum(N(1:i)),2) = 1;
end
results = [blocks(:,1) x_init x x_true abs(x-x_true)];
fprintf('blocks x_init x x_true |x-x_true|\n')
for i=1:n
    fprintf('%i      %.2f  %.2f  %.2f  %.2f\n', results(i,:))
    if blocks(i,2)
        fprintf('\n')
    end
end
results = [blocks(:,1) x_init3 x3 x_true abs(x3-x_true)];
fprintf('blocks x_init x3 x_true |x3-x_true|\n')
for i=1:n
    fprintf('%i      %.2f  %.2f  %.2f  %.2f\n', results(i,:))
    if blocks(i,2)
        fprintf('\n')
    end
end
%}
