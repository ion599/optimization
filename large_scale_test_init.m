%% Generate Some Synthetic Data
clc; clear all

%test = 'sparseObjX';
test = 'sparseObjZ';
%test = 'objZ';

noise = 0; % sets noise level

% Preprocessing U to Nf
% for i=1:102720 N(i)=sum(U(i,:)); end

% select data input
load('data/smaller_data.mat')
% load('data/stevesSmallData.mat')
% load('data/stevesData.mat')

%% Initialization
% Dimensions of the problem
n = size(A,2);
m = size(A,1);

lenN = length(N);
assert(sum(N) == n) % Check that nullspace N accounts for number of routes
x_true = x;
z_true = x2z(x_true,N);

%% Generate initial points
fprintf('Generate initialization points\n\n')
% 1: random
% 2: by importance (cheating-ish)
% 3: 10^importance (cheating-ish)
% 4: uniform
[x_init1,x_init2,x_init3,x_init4,z_init1,z_init2,z_init3,z_init4] = ...
    initXZ(n,N,x_true);

%% Compute sparse matrices
fprintf('Compute sparse x0 and sparse N\n')
[x0,N2] = computeSparseParam(n,N);

%% Set up optimization problem
alpha = (100*(noise^2)*(noise>.1))*(1-x_init2);
b2 = b+normrnd(0,noise,m,1);
if strcmp(test,'sparseObjZ')
    funObj = @(z)objectiveSparse(z,A,N2,x0,b2,zeros(n,1)); % no penalization (L2)
    funObj2 = @(z)objectiveSparse(z,A,N2,x0,b2,alpha);
    funProj = @(z)zProject(z,N);
elseif strcmp(test,'objZ')
    funObj = @(z)objective(z,A,N,b2,zeros(n,1)); % no penalization (L2)
    funObj2 = @(z)objective(z,A,N,b2,alpha);
    funProj = @(z)zProject(z,N);
elseif strcmp(test,'sparseObjX')
    funObj = @(x)objectiveX(x,A,b2,zeros(n,1));
    funObj2 = @(x)objectiveX(x,A,b2,alpha);
    funProj = @(x)xProject(x,N);
end

%% Set Optimization Options
gOptions.maxIter = 2000;
gOptions.verbose = 1; % Set to 0 to turn off output
gOptions.suffDec = .3;
gOptions.corrections = 500; % Number of corrections to store for L-BFGS methods

%% Run without regularization

options = gOptions;
%

fprintf('\nl-BFGS\n\n');
tic; t = cputime;
init = z_init1;
[zLBFGS1,histLBFGS1,timeLBFGS1] = lbfgs2(funObj,funProj,init,options);
timeLBFGS1 = toc; timeLBFGSCPU1 = cputime - t;

tic; t = cputime;
init = z_init2;
[zLBFGS2,histLBFGS2,timeLBFGS2] = lbfgs2(funObj,funProj,init,options);
timeLBFGS2 = toc; timeLBFGSCPU2 = cputime - t;

tic; t = cputime;
init = z_init3;
[zLBFGS3,histLBFGS3,timeLBFGS3] = lbfgs2(funObj,funProj,init,options);
timeLBFGS3 = toc; timeLBFGSCPU3 = cputime - t;

tic; t = cputime;
init = z_init4;
[zLBFGS4,histLBFGS4,timeLBFGS4] = lbfgs2(funObj,funProj,init,options);
timeLBFGS4 = toc; timeLBFGSCPU4 = cputime - t;

if strcmp(test,'sparseObjZ') || strcmp(test,'objZ')
    xLBFGS1 = x0+N2*zLBFGS1;
    xLBFGS2 = x0+N2*zLBFGS2;
    xLBFGS3 = x0+N2*zLBFGS3;
    xLBFGS4 = x0+N2*zLBFGS4;
else
    xLBFGS1 = zLBFGS1;
    xLBFGS2 = zLBFGS2;
    xLBFGS3 = zLBFGS3;
    xLBFGS4 = zLBFGS4;
end
%
%% Run with regularization (but only in noisy case)

if noise>0.1
    fprintf('\nl-BFGS\n\n');
    tic; t = cputime;
    init = z_init1;
    [zLBFGS1R,histLBFGS1R,timeLBFGS1R] = lbfgs2(funObj2,funProj,init,options);
    timeLBFGS1R = toc; timeLBFGSCPU1R = cputime - t;

    tic; t = cputime;
    init = z_init2;
    [zLBFGS2R,histLBFGS2R,timeLBFGS2R] = lbfgs2(funObj2,funProj,init,options);
    timeLBFGS2R = toc; timeLBFGSCPU2R = cputime - t;

    tic; t = cputime;
    init = z_init3;
    [zLBFGS3R,histLBFGS3R,timeLBFGS3R] = lbfgs2(funObj2,funProj,init,options);
    timeLBFGS3R = toc; timeLBFGSCPU3R = cputime - t;

    tic; t = cputime;
    init = z_init4;
    [zLBFGS4R,histLBFGS4R,timeLBFGS4R] = lbfgs2(funObj2,funProj,init,options);
    timeLBFGS4R = toc; timeLBFGSCPU4R = cputime - t;

    if strcmp(test,'sparseObjZ') || strcmp(test,'objZ')
        xLBFGS1R = x0+N2*zLBFGS1R;
        xLBFGS2R = x0+N2*zLBFGS2R;
        xLBFGS3R = x0+N2*zLBFGS3R;
        xLBFGS4R = x0+N2*zLBFGS4R;
    else
        xLBFGS1R = zLBFGS1R;
        xLBFGS2R = zLBFGS2R;
        xLBFGS3R = zLBFGS3R;
        xLBFGS4R = zLBFGS4R;
    end
end

%% Display performance
%
fprintf('\nProjected gradient without l2-regularization\n\n');

fprintf(['norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: ' ...
    '%.2f\nmax|x_init-x_true|: %.2f\n\n\n'], ...
    norm(A*xSPG-b), norm(A*x_init-b), max(abs(xSPG-x_true)), ...
    max(abs(x_true-x_init)))

fprintf('\nLBFGS without l2-regularization\n\n');

fprintf(['norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|: ' ...
    '%.2f\nmax|x_init-x_true|: %.2f\n\n\n'], ...
    norm(A*xLBFGS-b), norm(A*x_init-b), max(abs(xLBFGS-x_true)), ...
    max(abs(x_true-x_init)))

if noise > 0.1
    
    fprintf('\nProjected gradient with l2-regularization\n\n');
    
    fprintf(['norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|'...
        ': %.2f\nmax|x_init-x_true|: %.2f\n\n\n'], ...
        norm(A*xSPG2-b), norm(A*x_init-b), max(abs(xSPG2-x_true)), ...
        max(abs(x_true-x_init)))
    
    fprintf('\nLBFGS with l2-regularization\n\n');
    
    fprintf(['norm(A*x-b): %8.5e\nnorm(A*x_init-b): %8.5e\nmax|x-x_true|'...
        ': %.2f\nmax|x_init-x_true|: %.2f\n\n\n'], ...
        norm(A*xLBFGS2-b), norm(A*x_init-b), max(abs(xLBFGS2-x_true)), ...
        max(abs(x_true-x_init)))
    
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

[fLBFGS,deltaLBFGS,delta2LBFGS] = computeHist(test,x0,N2,histLBFGS,...
    x_true,N,f,A,b);
[fSPG,deltaSPG,delta2SPG] = computeHist(test,x0,N2,histSPG,x_true,N,f,A,b);
if noise > 0.1
    [fLBFGS2,deltaLBFGS2,delta2LBFGS2] = computeHist(test,x0,N2,...
        histLBFGS2,x_true,N,f,A,b);
    [fSPG2,deltaSPG2,delta2SPG2] = computeHist(test,x0,N2,histSPG2,...
        x_true,N,f,A,b);
end

%% display results

% s1 = strcat('LBFGS (',sprintf('%.2f',timeLBFGS/60),' min)');
% s2 = strcat('SPG (',sprintf('%.2f',timeSPG/60),' min)');
s1 = strcat('LBFGS (',sprintf('%.2f',timeLBFGSCPU/60),' min)');
s2 = strcat('SPG (',sprintf('%.2f',timeSPGCPU/60),' min)');
if noise>0.1
%     s3 = strcat('LBFGS reg (',sprintf('%.2f',timeLBFGS2/60),' min)');
%     s4 = strcat('SPG reg (',sprintf('%.2f',timeSPG2/60),' min)');
    s3 = strcat('LBFGS reg (',sprintf('%.2f',timeLBFGS2CPU/60),' min)');
    s4 = strcat('SPG reg (',sprintf('%.2f',timeSPG2CPU/60),' min)');
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

