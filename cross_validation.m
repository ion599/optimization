%% Cross-validation script

%% Configuration
% test = 'x';
test = 'z';
% test = 'dense-z';

% method = 'BB';
method = 'LBFGS';

noise = 0.2; % sets noise level
k = 10; % k-fold cv

tag = sprintf('CV 10-fold %s-%s noise=%0.1f',method,test,noise);

% Preprocessing U to Nf
% for i=1:102720 N(i)=sum(U(i,:)); end

% select data input
load('data/smaller_data.mat')
% load('data/stevesSmallData.mat')
% load('data/stevesData.mat')

% hashmap to store errors
if exist('data/cv_results_smaller_data5.mat','file')
    load('data/cv_results_smaller_data5');
end
if ~exist('results','var')
    results = containers.Map();
    save('data/cv_results_smaller_data5','results');
end

%% Initialization
% Dimensions of the problem
n = size(A,2);
m = size(A,1);

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

% select initial point
z_init = z_init4;
x_init = x_init4;

% Compute sparse matrices
fprintf('Compute sparse x0 and sparse N\n')
[x0,N2] = computeSparseParam(n,N);

%% Perturb (+ noise)
alpha = (100*(noise^2)*(noise>.1))*(1-x_init);
bn = b+normrnd(0,noise,m,1);

%% Set up cross-validation
% Requires bioinformatics toolbox
% Indices = crossvalind('Kfold', n, k);
Indices = ones(m,1);
temp = randperm(m,m);
for i=1:k-1
    Indices(temp>i*m/k & temp<=(i+1)*m/k) = i+1;
end

%% Set Optimization Options
gOptions.maxIter = 850;
gOptions.verbose = 1; % Set to 0 to turn off output
gOptions.suffDec = .3;
gOptions.corrections = 500; % Number of corrections to store for L-BFGS methods

%% k-fold cross validation
record = struct
for q=1:k
    % Train
    b_train = bn(Indices~=q);
    A_train = A(Indices~=q,:);

    %% Set up optimization problem
    if strcmp(test,'z')
        funObj = @(z)objectiveSparse(z,A_train,N2,x0,b_train,zeros(n,1)); % no penalization (L2)
        funObj2 = @(z)objectiveSparse(z,A_train,N2,x0,b_train,alpha);
        funProj = @(z)zProject(z,N);
        init = z_init;
    elseif strcmp(test,'dense-z')
        funObj = @(z)objective(z,A_train,N,b_train,zeros(n,1)); % no penalization (L2)
        funObj2 = @(z)objective(z,A_train,N,b_train,alpha);
        funProj = @(z)zProject(z,N);
        init = z_init;
    elseif strcmp(test,'x')
        funObj = @(x)objectiveX(x,A_train,b_train,zeros(n,1));
        funObj2 = @(x)objectiveX(x,A_train,b_train,alpha);
        funProj = @(x)xProject(x,N);
        init = x_init;
    end
    
    % Select with or without regularization
    if noise>0.1
        fn = funObj2;
    else
        fn = funObj;
    end
    
    %% Optimize
    options = gOptions;
    tic; t = cputime;
    if strcmp(method,'BB')
        [z,hist,times] = SPG(fn,funProj,init,options);
    elseif strcmp(method,'LBFGS')
        [z,hist,times] = lbfgs2(fn,funProj,init,options);
    end
    time = toc; timeCPU = cputime - t;

    % Recover x from z
    if strcmp(test,'z') || strcmp(test,'dense-z')
        x_train = x0+N2*z;
        hist_train = repmat(x0,1,size(hist,2)) + N2*hist;
    else
        x_train = z;
        hist_train = hist;
    end
    
    %% Record
    record(q).timeCPU = timeCPU;
    record(q).times = times;
    record(q).hist = hist;
    record(q).z = z;
    record(q).x_train = x_train;
    record(q).hist_train = hist_train;
    
    % Progress
    delta_train = A_train * hist_train - repmat(b_train,1,size(hist,2));
    delta = A * hist_train - repmat(b,1,size(hist,2));
    record(q).error_hist_train = diag(sqrt(delta_train'*delta_train));
    record(q).error_hist = diag(sqrt(delta'*delta));

    % Test
    record(q).error_train = norm(A_train * x_train - b_train);
    record(q).error = norm(A * x_train - b);

    fprintf(['%d norm(A_train*x_train-b_train): %8.5e\n',...
        'norm(A*x_train-b): %8.5e\nnorm(A*x_init-b): %8.5e\n'], q, ...
        record(q).error_train, record(q).error, norm(A*x_init-b));
end

results(tag) = record;
save('data/cv_results_smaller_data5','results');

WarnWave = 0.1 * [sin(1:.6:400), sin(1:.7:400), sin(1:.4:400)];
Audio = audioplayer(repmat(WarnWave,1,10), 22050);
play(Audio);

% 2.7564  104.4283 | cv_10-fold_LBFGSz_noise0
% 2.8614  120.4514
% 2.9276  119.8896

% 13.3768  136.0122 | cv_10-fold_BBz_noise0
% 9.2688  136.6618 | cv_10-fold_LBFGSx_noise0
% 31.9126  124.5516 | cv_10-fold_BBx_noise0

% 4.1931  146.0264 | cv_10-fold_LBFGSz_noise02
% 13.4403  130.4493 | cv_10-fold_BBz_noise02
% 9.2850  116.4020 | cv_10-fold_LBFGSx_noise02
% 10.2494  120.4671
% 34.7648  141.9519 | cv_10-fold_BBx_noise02