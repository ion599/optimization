%% Preprocess parameters

clc; clear all
% load('data/smaller_data.mat')
load data/stevesSmallData.mat
n = size(A,2);
m = size(A,1);
x_true = x;
z_true = x2z(x_true,N);

fprintf('Generate initialization points\n\n')
[x_init1,x_init2,x_init3,x_init4,z_init1,z_init2,z_init3,z_init4] = initXZ(n,N,x_true);
fprintf('Compute sparse x0 and sparse N\n')
[x0,N2] = computeSparseParam(n,N);
%initx = x_init4;
%initz = z_init4;
initx = x0;
initz = x2z(x0,N);

%% Set up optimization problem

noise = 0.2;
alpha = (100*(noise^2)*(noise>.1))*(1-x_init2);
b2 = b+normrnd(0,noise,m,1);
funObjz = @(z)objectiveSparse(z,A,N2,x0,b2,alpha);
funObjx = @(x)objectiveX(x,A,b2,alpha);
funProjz = @(z)zProject(z,N);
funProjx = @(x)xProject(x,N);

%% Set Optimization Options
gOptions.maxIter = 2000;
gOptions.verbose = 1; % Set to 0 to turn off output
gOptions.suffDec = .3;
gOptions.corrections = 500; % Number of corrections to store for L-BFGS methods

%% Run algorithm

options = gOptions;

%fprintf('\nProjected Gradient\n\n');
fprintf('\nProjected L-BFGS\n\n');

%[x,histx,timehistx] = SPG(funObjx,funProjx,initx,options);
%[z,histz,timehistz] = SPG(funObjz,funProjz,initz,options);
[x,histx,timehistx] = lbfgs2(funObjx,funProjx,initx,options);
[z,histz,timehistz] = lbfgs2(funObjz,funProjz,initz,options);

%% Compute results

lenN = length(N);
f = zeros(lenN,1);
k = 1;
for i=1:lenN
    f(i) = max(A(:,k));
    k = k+N(i);
end

[fx,deltax,delta2x,timex] = computeHist('sparseObjX',x0,N2,histx,timehistx,initx,x_true,N,f,A,b);
[fz,deltaz,delta2z,timez] = computeHist('sparseObjZ',x0,N2,histz,timehistz,initx,x_true,N,f,A,b);

%% Plot results

plot(timex,delta2x)
title('f*|x-xtrue| vs. time');
xlabel('cpu time');
ylabel('weighted norm');
hold on
plot(timez,delta2z,'r')
legend('without constraint elimination','with constraint elimination')

figure;

plot(timex,deltax)
title('|x-xtrue| vs. time');
xlabel('cpu time');
ylabel('norm');
hold on
plot(timez,deltaz,'r')
legend('without constraint elimination','with constraint elimination')

figure;

plot(timex,fx)
title('objective vs. time');
xlabel('cpu time');
ylabel('objective value');
hold on
plot(timez,fz,'r')
legend('without constraint elimination','with constraint elimination')

%% Plot results

figure;

plot(timexlbfgs,delta2xlbfgs,'Color','r','Marker','s','LineWidth',0.75)
title('f*|x-xtrue| vs. time');
xlabel('cpu time');
ylabel('weighted norm');
hold on
plot(timezlbfgs,delta2zlbfgs,'Color','k','Marker','^','LineWidth',0.75)
hold on
plot(timex(1:20),delta2x(1:20),'Marker','+','LineWidth',0.75)
hold on
plot(timez,delta2z,'Color','g','Marker','o','LineWidth',0.75)
legend('L-BFGS without constraint elimination',...
    'L-BFGS with constraint elimination',...
    'BB without constraint elimination',...
    'BB with constraint elimination')
%{
figure;

plot(timexlbfgs,deltaxlbfgs)
title('|x-xtrue| vs. time');
xlabel('cpu time');
ylabel('norm');
hold on
plot(timezlbfgs,deltazlbfgs,'r')
hold on
plot(timex,deltax,'k')
hold on
plot(timez,deltaz,'g')
legend('L-BFGS without constraint elimination',...
    'L-BFGS with constraint elimination',...
    'BB without constraint elimination',...
    'BB with constraint elimination')

figure;

plot(timexlbfgs,fxlbfgs)
title('objective vs. time');
xlabel('cpu time');
ylabel('objective value');
hold on
plot(timezlbfgs,fzlbfgs,'r')
hold on
plot(timex,fx,'k')
hold on
plot(timez,fz,'g')
legend('L-BFGS without constraint elimination',...
    'L-BFGS with constraint elimination',...
    'BB without constraint elimination',...
    'BB with constraint elimination')
%}
