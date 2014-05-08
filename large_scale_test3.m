%% Preprocess parameters

clc; clear all
load('data/smaller_data.mat')
n = size(A,2);
m = size(A,1);
x_true = x;
z_true = x2z(x_true,N);

fprintf('Generate initialization points\n\n')
[x_init1,x_init2,x_init3,z_init1,z_init2,z_init3] = initXZ(n,N,x_true);
fprintf('Compute sparse x0 and sparse N\n')
[x0,N2] = computeSparseParam(n,N);
initx = x_init1;
initz = z_init1;

%% Set up optimization problem

noise = 0.2;
alpha = (100*(noise^2)*(noise>.1))*(1-x_init2);
b2 = b+normrnd(0,noise,m,1);
funObjz = @(z)objectiveSparse(z,A,N2,x0,b2,alpha);
funObjx = @(x)objectiveX(x,A,b2,alpha);
funProjz = @(z)zProject(z,N);
funProjx = @(x)xProject(x,N);

%% Set Optimization Options
gOptions.maxIter = 1000;
gOptions.verbose = 1; % Set to 0 to turn off output
gOptions.suffDec = .3;
gOptions.corrections = 500; % Number of corrections to store for L-BFGS methods

%% Run algorithm

options = gOptions;

fprintf('\nProjected Gradient\n\n');
[x,histx,timehistx] = SPG(funObjx,funProjx,initx,options);
[z,histz,timehistz] = SPG(funObjz,funProjz,initz,options);

%%

lenN = length(N);
f = zeros(lenN,1);
k = 1;
for i=1:lenN
    f(i) = max(A(:,k));
    k = k+N(i);
end

[fx,deltax,delta2x] = computeHist('sparseObjX',x0,N2,histx,x_true,N,f,A,b);
[fz,deltaz,delta2z] = computeHist('sparseObjZ',x0,N2,histz,z_true,N,f,A,b);



