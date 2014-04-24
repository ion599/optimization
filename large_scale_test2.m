%% Generate Some Synthetic Data
clc; clear all

%test = 'sparseObjX';
test = 'sparseObjZ';
%test = 'objZ';

% Preprocessing U to Nf
% for i=1:102720 N(i)=sum(U(i,:)); end

load('data/stevesSmallData.mat')
%load('data/stevesData.mat')

% Dimensions of the problem
n = size(A,2);
m = size(A,1);

lenN = length(N);
assert(sum(N) == n) % Check that nullspace N accounts for number of routes
x_true = x;
z_true = x2z(x_true,N);

%% Generate initial points
fprintf('Generate initialization points\n\n')
[x_init1,x_init2,x_init3,z_init1,z_init2,z_init3] = initXZ(n,N,x_true);

%% Compute sparse matrices

fprintf('Compute sparse x0 and sparse N')
[x0,N2] = computeSparseParam(n,N);

%% Set up optimization problem
noise = 0;

alpha = (100*(noise^2)*(noise>.1))*(1-x_init2);
b2 = b+normrnd(0,noise,m,1);
if strcmp(test,'sparseObjZ')
    funObj = @(z)objectiveSparse(z,A,N2,x0,b2,zeros(n,1)); % no penalization (L2)
    funObj2 = @(z)objectiveSparse(z,A,N2,x0,b2,alpha);
    funProj = @(z)zProject(z,N);
    init = z_init1;
elseif strcmp(test,'objZ')
    funObj = @(z)objective(z,A,N,b2,zeros(n,1)); % no penalization (L2)
    funObj2 = @(z)objective(z,A,N,b2,alpha);
    funProj = @(z)zProject(z,N);
    init = z_init1;
elseif strcmp(test,'sparseObjX')
    funObj = @(x)objectiveX(x,A,b2,zeros(n,1));
    funObj2 = @(x)objectiveX(x,A,b2,alpha);
    funProj = @(x)xProject(x,N);
    init = x_init1;
end

%% Set Optimization Options
gOptions.maxIter = 2000;
gOptions.verbose = 1; % Set to 0 to turn off output
gOptions.suffDec = .3;
gOptions.corrections = 500; % Number of corrections to store for L-BFGS methods

%% Run without regularization

options = gOptions;
%

fprintf('\nProjected Gradient\n\n');
tic
[zSPG,histSPG] = SPG(funObj,funProj,init,options);
timeSPG = toc;

fprintf('\Double Over-relaxation\n\n');
options = gOptions;
tic
[zDORE,histDORE] = DORE(funObj,funProj,A,b2,init,options);
timeDORE = toc;

fprintf('\nl-BFGS\n\n');

tic
[zLBFGS,histLBFGS] = lbfgs2(funObj,funProj,init,options);
timeLBFGS = toc;

if strcmp(test,'sparseObjZ') || strcmp(test,'objZ')
    xSPG = x0+N2*zSPG; xLBFGS = x0+N2*zLBFGS;
else
    xSPG = zSPG; xLBFGS = zLBFGS; xDORE = zDORE;
end
%
%% Run noisy case

if noise>0.1
    % Run Projected gradient with reg.
    %
    fprintf('\nProjected Gradient\n\n');
    options = gOptions;
    tic
    [zSPG2,histSPG2] = SPG(funObj2,funProj,init,options);
    timeSPG2 = toc;
    %
    % Run l-BFGS with reg.
    
    fprintf('\nl-BFGS\n\n');
    
    tic
    [zLBFGS2,histLBFGS2] = lbfgs2(funObj2,funProj,init,options);
    timeLBFGS2 = toc;
    
    if strcmp(test,'sparseObjZ') || strcmp(test,'objZ')
        xSPG2 = x0+N2*zSPG2; xLBFGS2 = x0+N2*zLBFGS2;
    else
        xSPG2 = zSPG2; xLBFGS2 = zLBFGS2;
    end
end

%% Display performance
%
fprintf('\nProjected gradient without l2-regularization\n\n');
fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\nmax|x_init-x_true|: %.2f\n\n\n', ...
    norm(s*A*xSPG/0.99-b), norm(s*A*x_init3/0.99-b), max(abs(xSPG-x_true)), max(abs(x_true-x_init3)))
fprintf(strcat('SPG (',sprintf('%.2f',timeSPG/60),' min)'));

fprintf('\Double over-relaxation without l2-regularization\n\n');
fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\nmax|x_init-x_true|: %.2f\n\n\n', ...
    norm(s*A*xDORE/0.99-b), norm(s*A*x_init3/0.99-b), max(abs(xDORE-x_true)), max(abs(x_true-x_init3)))
fprintf(strcat('DORE (',sprintf('%.2f',timeDORE/60),' min)'));

fprintf('\nLBFGS without l2-regularization\n\n');
fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\nmax|x_init-x_true|: %.2f\n\n\n', ...
    norm(s*A*xLBFGS/0.99-b), norm(s*A*x_init3/0.99-b), max(abs(xLBFGS-x_true)), max(abs(x_true-x_init3)))
fprintf(strcat('LBFGS (',sprintf('%.2f',timeLBFGS/60),' min)'));

if noise > 0.1
    
    fprintf('\nProjected gradient with l2-regularization\n\n');
    
    fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\nmax|x_init-x_true|: %.2f\n\n\n', ...
        norm(A*xSPG2-b), norm(A*x_init3-b), max(abs(xSPG2-x_true)), max(abs(x_true-x_init3)))
    
    fprintf('\nLBFGS with l2-regularization\n\n');
    
    fprintf('norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: %.2f\nmax|x_init-x_true|: %.2f\n\n\n', ...
        norm(A*xLBFGS2-b), norm(A*x_init3-b), max(abs(xLBFGS2-x_true)), max(abs(x_true-x_init3)))
    
end
%
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

f = zeros(lenN,1);
k = 1;
for i=1:lenN
    f(i) = max(A(:,k));
    k = k+N(i);
end

[fLBFGS,deltaLBFGS,delta2LBFGS] = computeHist(test,x0,N2,histLBFGS,x_true,N,f,A,b);
[fSPG,deltaSPG,delta2SPG] = computeHist(test,x0,N2,histSPG,x_true,N,f,A,b);
if noise > 0.1
    [fLBFGS2,deltaLBFGS2,delta2LBFGS2] = computeHist(test,x0,N2,histLBFGS2,x_true,N,f,A,b);
    [fSPG2,deltaSPG2,delta2SPG2] = computeHist(test,x0,N2,histSPG2,x_true,N,f,A,b);
end

%% display results

s1 = strcat('LBFGS (',sprintf('%.2f',timeLBFGS/60),' min)');
s2 = strcat('SPG (',sprintf('%.2f',timeSPG/60),' min)');
if noise>0.1
    s3 = strcat('LBFGS reg (',sprintf('%.2f',timeLBFGS2/60),' min)');
    s4 = strcat('SPG reg (',sprintf('%.2f',timeSPG2/60),' min)');
end
figure;

plot(100*[1:length(fLBFGS)],fLBFGS)
title('Objective value vs. Iteration');
xlabel('Iterations');
ylabel('f value');
hold on
plot(100*[1:length(fSPG)],fSPG,'r')
if noise > 0.1
    hold on
    plot(100*[1:length(fLBFGS2)],fLBFGS2,'k')
    hold on
    plot(100*[1:length(fSPG2)],fSPG2,'g')
    legend(s1,s2,s3,s4)
else
    legend(s1,s2)
end

figure;

plot(100*[1:length(fLBFGS)],deltaLBFGS)
title('|x-xtrue| vs. Iteration');
xlabel('Iterations');
ylabel('norm');
hold on
plot(100*[1:length(fSPG)],deltaSPG,'r')
if noise > 0.1
    hold on
    plot(100*[1:length(fLBFGS2)],deltaLBFGS2,'k')
    hold on
    plot(100*[1:length(fSPG2)],deltaSPG2,'g')
    legend(s1,s2,s3,s4)
else
    legend(s1,s2)
end

figure;

plot(100*[1:length(fLBFGS)],delta2LBFGS)
title('f*|x-xtrue| vs. Iteration');
xlabel('Iterations');
ylabel('norm');
hold on
plot(100*[1:length(fSPG)],delta2SPG,'r')
if noise > 0.1
    hold on
    plot(100*[1:length(fLBFGS2)],delta2LBFGS2,'k')
    hold on
    plot(100*[1:length(fSPG2)],delta2SPG2,'g')
    legend(s1,s2,s3,s4)
else
    legend(s1,s2)
end

%%

