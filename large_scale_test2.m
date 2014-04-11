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

%% Compute sparse matrices                                                        

N2 = sparse(n, n-lenN);
x0 = sparse(n,1);

ind = 1;
k = 0;
for i=1:lenN
    if N(i)>1
    N2(ind, ind-k) = 1;
    ind = ind+1;
    for j=2:(N(i)-1)
        N2(ind, ind-k-1) = -1;
        N2(ind, ind-k) = 1;
        ind = ind+1;
    end
    N2(ind, ind-k-1) = -1;
    end
    k = k+1;
    x0(ind) = 1;
    ind = ind+1;
end


%% Set up optimization problem
noise = 0.2; % if b=bExact

alpha = (100*(noise^2)*(noise>.1))*(1-x_init2);
b2 = b+normrnd(0,noise,m,1);
funObj = @(z)objectiveSparse(z,A,N,N2,x0,b2,zeros(n,1)); % no penalization (L2)
funObj2 = @(z)objectiveSparse(z,A,N,N2,x0,b2,alpha);

%% Set Optimization Options
gOptions.maxIter = 100;
gOptions.verbose = 1; % Set to 0 to turn off output
gOptions.suffDec = .3;
options.corrections = 10; % Number of corrections to store for L-BFGS methods
maxIter = 20;

%% Run Projected gradient

fprintf('\nProjected Gradient\n\n');
options = gOptions;
tic
[zSPG,histSPG] = SPG(funObj,z_init3,N,options);
xSPG = x0+N2*zSPG;
timeSPG = toc;

%% Run l-BFGS

fprintf('\nl-BFGS\n\n');

tic
[zLBFGS,histLBFGS] = lbfgs2(funObj,z_init3,N,500,options);
xLBFGS = x0+N2*zLBFGS;
timeLBFGS = toc;

%% Run noisy case

if noise>0.1
    % Run Projected gradient with reg.
    
    fprintf('\nProjected Gradient\n\n');
    options = gOptions;
    tic
    [zSPG2,histSPG2] = SPG(funObj2,z_init3,N,options);
    xSPG2 = x0+N2*zSPG2;
    timeSPG2 = toc;
    
    % Run l-BFGS with reg.
    
    fprintf('\nl-BFGS\n\n');
    
    tic
    [zLBFGS2,histLBFGS2] = lbfgs2(funObj2,z_init3,N,500,options);
    xLBFGS2 = x0+N2*zLBFGS2;
    timeLBFGS2 = toc;
end

%% Display performance

fprintf('\nProjected gradient without l2-regularization\n\n');

fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\nmax|x_init-x_true|: %.2f\n\n\n', ...
    norm(A*xSPG-b), norm(A*x_init3-b), max(abs(xSPG-x_true)), max(abs(x_true-x_init3)))

fprintf('\nLBFGS without l2-regularization\n\n');

fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\nmax|x_init-x_true|: %.2f\n\n\n', ...
    norm(A*xLBFGS-b), norm(A*x_init3-b), max(abs(xLBFGS-x_true)), max(abs(x_true-x_init3)))


fprintf('\nProjected gradient with l2-regularization\n\n');

fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\nmax|x_init-x_true|: %.2f\n\n\n', ...
    norm(A*xSPG2-b), norm(A*x_init3-b), max(abs(xSPG2-x_true)), max(abs(x_true-x_init3)))

fprintf('\nLBFGS with l2-regularization\n\n');

fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\nmax|x_init-x_true|: %.2f\n\n\n', ...
    norm(A*xLBFGS2-b), norm(A*x_init3-b), max(abs(xLBFGS2-x_true)), max(abs(x_true-x_init3)))
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

fLBFGS=zeros(size(histLBFGS,2),1);
fSPG=zeros(size(histSPG,2),1);
deltaLBFGS=zeros(size(histLBFGS,2),1);
deltaSPG=zeros(size(histSPG,2),1);
for i=1:length(fLBFGS)
    x = z2x(histLBFGS(:,i),N);
    fLBFGS(i) = norm(A*x-b);
    deltaLBFGS(i) = norm(x-x_true);
end
for i=1:length(fSPG)
    x = z2x(histSPG(:,i),N);
    fSPG(i) = norm(A*x-b);
    deltaSPG(i) = norm(x-x_true);
end

fLBFGS2 = zeros(size(histLBFGS2,2),1);
fSPG2 = zeros(size(histSPG2,2),1);
deltaLBFGS2 = zeros(size(histLBFGS2,2),1);
deltaSPG2 = zeros(size(histSPG2,2),1);
for i=1:length(fLBFGS2)
    x = z2x(histLBFGS2(:,i),N);
    fLBFGS2(i) = norm(A*x-b);
    deltaLBFGS2(i) = norm(x-x_true);
end
for i=1:length(fSPG2)
    x = z2x(histSPG2(:,i),N);
    fSPG2(i) = norm(A*x-b);
    deltaSPG2(i) = norm(x-x_true);
end

%% display results

figure;

plot(100*[1:9],fLBFGS)
title('Objective value vs. Iteration');
xlabel('Iterations');
ylabel('f value');
hold on
plot(100*[1:9],fSPG,'r')
hold on
plot(100*[1:9],fLBFGS2,'k')
hold on
plot(100*[1:9],fSPG2,'g')
legend('LBFGS','SPG','LBFGS reg','SPG reg')


figure;

plot(100*[1:9],deltaLBFGS)
title('|x-xtrue| vs. Iteration');
xlabel('Iterations');
ylabel('norm');
hold on
plot(100*[1:9],deltaSPG,'r')
hold on
plot(100*[1:9],deltaLBFGS2,'k')
hold on
plot(100*[1:9],deltaSPG2,'g')
legend('LBFGS','SPG','LBFGS reg','SPG reg')

%%

f = zeros(lenN,1);
k = 1;
r = 0;
for i=1:lenN
    f(i) = max(A(:,k));
    r = r + f(i)*max(abs(xLBFGS2(k:k+N(i)-1)-x_true(k:k+N(i)-1)));
    k = k+N(i);
end

r = r/sum(f);