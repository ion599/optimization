%% Generate Some Synthetic Data
clc; clear all
setup_params

% test = 'x';
test = 'z';
% test = 'dense-z';

noise = 0.02; % sets noise level

% select data input
% data_file = 'smaller_data';
data_file = 'stevesSmallData';
% data_file = 'stevesData';
data_fie = 'experiment1_matrices';

load(sprintf('%s/%s.mat', DATA_DIR, data_file))

%% Initialization
% Dimensions of the problem
n = size(A,2);
m = size(A,1);

% Preprocessing U to Nf
% for i=1:102720 N(i)=sum(U(i,:)); end

lenN = length(N);
assert(sum(N) == n) % Check that nullspace N accounts for number of routes
x_true = x;
z_true = x2z(x_true,N);

% Generate initial points
fprintf('Generate initialization points\n\n')
% 1: random
% 2: by importance (cheating-ish)
% 3: 10^importance (cheating-ish)
% 4: uniform
[x_init1,x_init2,x_init3,x_init4,z_init1,z_init2,z_init3,z_init4] = ...
    initXZ(n,N,x_true);

% Compute sparse matrices
fprintf('Compute sparse x0 and sparse N\n')
[x0,N2] = computeSparseParam(n,N);

%% Set up optimization problem
alpha = (100*(noise^2)*(noise>.1))*(1-x_init2);
b2 = b+normrnd(0,noise,m,1);
if strcmp(test,'z')
    funObj = @(z)objectiveSparse(z,A,N2,x0,b2,zeros(n,1)); % no penalization (L2)
    funObj2 = @(z)objectiveSparse(z,A,N2,x0,b2,alpha);
    funProj = @(z)zProject(z,N);
elseif strcmp(test,'dense-z')
    funObj = @(z)objective(z,A,N,b2,zeros(n,1)); % no penalization (L2)
    funObj2 = @(z)objective(z,A,N,b2,alpha);
    funProj = @(z)zProject(z,N);
elseif strcmp(test,'x')
    funObj = @(x)objectiveX(x,A,b2,zeros(n,1));
    funObj2 = @(x)objectiveX(x,A,b2,alpha);
    funProj = @(x)xProject(x,N);
end

%% Set Optimization Options
gOptions.maxIter = 550;
gOptions.verbose = 1; % Set to 0 to turn off output
gOptions.suffDec = .3;
gOptions.corrections = 500; % Number of corrections to store for L-BFGS methods

%% Run without regularization

options = gOptions;
%

fprintf('\nl-BFGS\n\n');

tic; t = cputime;
init = z_init1;
[zLBFGS1,histLBFGS1,timesLBFGS1] = lbfgs2(funObj,funProj,init,options);
timeLBFGS1 = toc; timeLBFGSCPU1 = cputime - t;

tic; t = cputime;
init = z_init2;
[zLBFGS2,histLBFGS2,timesLBFGS2] = lbfgs2(funObj,funProj,init,options);
timeLBFGS2 = toc; timeLBFGSCPU2 = cputime - t;

tic; t = cputime;
init = z_init3;
[zLBFGS3,histLBFGS3,timesLBFGS3] = lbfgs2(funObj,funProj,init,options);
timeLBFGS3 = toc; timeLBFGSCPU3 = cputime - t;

tic; t = cputime;
init = z_init4;
[zLBFGS4,histLBFGS4,timesLBFGS4] = lbfgs2(funObj,funProj,init,options);
timeLBFGS4 = toc; timeLBFGSCPU4 = cputime - t;

if strcmp(test,'z') || strcmp(test,'dense-z')
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
    [zLBFGS1R,histLBFGS1R,timesLBFGS1R] = lbfgs2(funObj2,funProj,init,options);
    timeLBFGS1R = toc; timeLBFGSCPU1R = cputime - t;

    tic; t = cputime;
    init = z_init2;
    [zLBFGS2R,histLBFGS2R,timesLBFGS2R] = lbfgs2(funObj2,funProj,init,options);
    timeLBFGS2R = toc; timeLBFGSCPU2R = cputime - t;

    tic; t = cputime;
    init = z_init3;
    [zLBFGS3R,histLBFGS3R,timesLBFGS3R] = lbfgs2(funObj2,funProj,init,options);
    timeLBFGS3R = toc; timeLBFGSCPU3R = cputime - t;

    tic; t = cputime;
    init = z_init4;
    [zLBFGS4R,histLBFGS4R,timesLBFGS4R] = lbfgs2(funObj2,funProj,init,options);
    timeLBFGS4R = toc; timeLBFGSCPU4R = cputime - t;

    if strcmp(test,'z') || strcmp(test,'dense-z')
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
% test,x0,N2,hist,timehist,initx,x_true,N,f,A,b

[fLBFGS1,deltaLBFGS1,delta2LBFGS1] = computeHist(test,x0,N2,histLBFGS1,...
    timesLBFGS1,x_init1,x_true,N,f,A,b);
[fLBFGS2,deltaLBFGS2,delta2LBFGS2] = computeHist(test,x0,N2,histLBFGS2,...
    timesLBFGS2,x_init2,x_true,N,f,A,b);
[fLBFGS3,deltaLBFGS3,delta2LBFGS3] = computeHist(test,x0,N2,histLBFGS3,...
    timesLBFGS3,x_init3,x_true,N,f,A,b);
[fLBFGS4,deltaLBFGS4,delta2LBFGS4] = computeHist(test,x0,N2,histLBFGS4,...
    timesLBFGS4,x_init4,x_true,N,f,A,b);
if noise > 0.1
    [fLBFGS1R,deltaLBFGS1R,delta2LBFGS1R] = computeHist(test,x0,N2,...
        histLBFGS1R,x_true,N,f,A,b);
    [fLBFGS2R,deltaLBFGS2R,delta2LBFGS2R] = computeHist(test,x0,N2,...
        histLBFGS2R,x_true,N,f,A,b);
    [fLBFGS3R,deltaLBFGS3R,delta2LBFGS3R] = computeHist(test,x0,N2,...
        histLBFGS3R,x_true,N,f,A,b);
    [fLBFGS4R,deltaLBFGS4R,delta2LBFGS4R] = computeHist(test,x0,N2,...
        histLBFGS4R,x_true,N,f,A,b);
end

%% display results

s1 = strcat('LBFGS rand (',sprintf('%.2f',timeLBFGSCPU1/60),' min)');
s2 = strcat('LBFGS popular (',sprintf('%.2f',timeLBFGSCPU2/60),' min)');
s3 = strcat('LBFGS 10^{pop} (',sprintf('%.2f',timeLBFGSCPU3/60),' min)');
s4 = strcat('LBFGS uniform (',sprintf('%.2f',timeLBFGSCPU4/60),' min)');
if noise>0.1
    s5 = strcat('LBFGS rand reg (',sprintf('%.2f',timeLBFGSCPU1R/60),' min)');
    s6 = strcat('LBFGS pop reg (',sprintf('%.2f',timeLBFGSCPU2R/60),' min)');
    s7 = strcat('LBFGS 10^{pop} reg (',sprintf('%.2f',timeLBFGSCPU3R/60),' min)');
    s8 = strcat('LBFGS uniform reg (',sprintf('%.2f',timeLBFGSCPU4R/60),' min)');
end


%% Iteration plots
figure;

plot(100*[1:length(fLBFGS1)],fLBFGS1,'b')
title('Objective value vs. Iteration');
xlabel('Iterations');
ylabel('f value');
hold on
plot(100*[1:length(fLBFGS2)],fLBFGS2,'r')
plot(100*[1:length(fLBFGS3)],fLBFGS3,'g')
plot(100*[1:length(fLBFGS4)],fLBFGS4,'k')
if noise > 0.1
    hold on
    plot(100*[1:length(fLBFGS1R)],fLBFGS1R,'b--')
    hold on
    plot(100*[1:length(fLBFGS2R)],fLBFGS2R,'r--')
    plot(100*[1:length(fLBFGS3R)],fLBFGS3R,'g--')
    plot(100*[1:length(fLBFGS4R)],fLBFGS4R,'k--')
    legend(s1,s2,s3,s4,s5,s6,s7,s8)
else
    legend(s1,s2,s3,s4)
end

figure;

plot(100*[1:length(fLBFGS1)],deltaLBFGS1,'b')
title('|x-xtrue| vs. Iteration');
xlabel('Iterations');
ylabel('norm');
hold on
plot(100*[1:length(fLBFGS2)],deltaLBFGS2,'r')
plot(100*[1:length(fLBFGS3)],deltaLBFGS3,'g')
plot(100*[1:length(fLBFGS4)],deltaLBFGS4,'k')
if noise > 0.1
    hold on
    plot(100*[1:length(fLBFGS1R)],deltaLBFGS1R,'b--')
    hold on
    plot(100*[1:length(fLBFGS2R)],deltaLBFGS2R,'r--')
    plot(100*[1:length(fLBFGS3R)],deltaLBFGS3R,'g--')
    plot(100*[1:length(fLBFGS4R)],deltaLBFGS4R,'k--')
    legend(s1,s2,s3,s4,s5,s6,s7,s8)
else
    legend(s1,s2,s3,s4)
end

figure;

plot(100*[1:length(fLBFGS1)],delta2LBFGS1,'b')
title('f*|x-xtrue| vs. Iteration');
xlabel('Iterations');
ylabel('norm');
hold on
plot(100*[1:length(fLBFGS2)],delta2LBFGS2,'r')
plot(100*[1:length(fLBFGS3)],delta2LBFGS3,'g')
plot(100*[1:length(fLBFGS4)],delta2LBFGS4,'k')
if noise > 0.1
    hold on
    plot(100*[1:length(fLBFGS1R)],delta2LBFGS1R,'b--')
    hold on
    plot(100*[1:length(fLBFGS2R)],delta2LBFGS2R,'r--')
    plot(100*[1:length(fLBFGS3R)],delta2LBFGS3R,'g--')
    plot(100*[1:length(fLBFGS4R)],delta2LBFGS4R,'k--')
    legend(s1,s2,s3,s4,s5,s6,s7,s8)
else
    legend(s1,s2,s3,s4)
end

%% Time plots

figure;

plot([0 cumsum(timesLBFGS1)/60],log(fLBFGS1),'b')
title('Objective value vs. time');
xlabel('Elapsed time (minutes)');
ylabel('log f value');
hold on
plot([0 cumsum(timesLBFGS2)/60],log(fLBFGS2),'r')
% plot([0 cumsum(timesLBFGS3)/60],fLBFGS3,'g')
plot([0 cumsum(timesLBFGS4)/60],log(fLBFGS4),'k')
if noise > 0.1
    plot([0 cumsum(timesLBFGS1R)/60],fLBFGS1R,'b--')
    hold on
    plot([0 cumsum(timesLBFGS2R)/60],fLBFGS2R,'r--')
%     plot([0 cumsum(timesLBFGS3R)/60],fLBFGS3R,'g--')
    plot([0 cumsum(timesLBFGS4R)/60],fLBFGS4R,'k--')
    legend(s1,s2,s3,s4,s5,s6,s7,s8)
else
    legend(s1,s2,s4)
end

figure;

plot([0 cumsum(timesLBFGS1)/60],deltaLBFGS1,'b')
title('|x-xtrue| vs. time');
xlabel('Elapsed time (minutes)');
ylabel('norm');
hold on
plot([0 cumsum(timesLBFGS2)/60],deltaLBFGS2,'r')
plot([0 cumsum(timesLBFGS3)/60],deltaLBFGS3,'g')
plot([0 cumsum(timesLBFGS4)/60],deltaLBFGS4,'k')
if noise > 0.1
    plot([0 cumsum(timesLBFGS1R)/60],deltaLBFGS1R,'b--')
    hold on
    plot([0 cumsum(timesLBFGS2R)/60],deltaLBFGS2R,'r--')
    plot([0 cumsum(timesLBFGS3R)/60],deltaLBFGS3R,'g--')
    plot([0 cumsum(timesLBFGS4R)/60],deltaLBFGS4R,'k--')
    legend(s1,s2,s3,s4,s5,s6,s7,s8)
else
    legend(s1,s2,s3,s4)
end

figure;

plot([0 cumsum(timesLBFGS1)/60],delta2LBFGS1,'b')
title('f*|x-xtrue| vs. time');
xlabel('Elapsed time (minutes)');
ylabel('norm');
hold on
plot([0 cumsum(timesLBFGS2)/60],delta2LBFGS2,'r')
plot([0 cumsum(timesLBFGS3)/60],delta2LBFGS3,'g')
plot([0 cumsum(timesLBFGS4)/60],delta2LBFGS4,'k')
if noise > 0.1
    plot([0 cumsum(timesLBFGS1R)/60],delta2LBFGS1R,'b--')
    hold on
    plot([0 cumsum(timesLBFGS2R)/60],delta2LBFGS2R,'r--')
    plot([0 cumsum(timesLBFGS3R)/60],delta2LBFGS3R,'g--')
    plot([0 cumsum(timesLBFGS4R)/60],delta2LBFGS4R,'k--')
    legend(s1,s2,s3,s4,s5,s6,s7,s8)
else
    legend(s1,s2,s3,s4)
end



