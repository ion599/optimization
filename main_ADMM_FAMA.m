close all;
clear all;

%% download data
TestParameters;
load('data/20140228T232251-cathywu-9.mat');
%load('data/20140228T232249-cathywu-5.mat');

%% problem:
%  min |Ax-b|
%  s.t  Ux = 1 and x >= 0

% Dimensions of the problem
n = size(p.Phi,2);
m = size(p.Phi,1);

N = p.block_sizes;
lenN = length(N);
assert(sum(N) == n) % Check that nullspace N accounts for number of routes

A = p.Phi;
x_true = p.real_a;
b = p.f;

obj_true = (A*x_true-b)'*(A*x_true-b);
obj_true_re = (A*x_true-b)'*(A*x_true-b) + x_true'*x_true;

% generate random matrices
% m = 5;
% n = 20;
% A = rand(m,n);
% b = rand(m,1);
% 
% n_block = 5;
% N = n_block*ones(n/n_block,1);
% lenN = length(N);

U = [];
for i = 1:lenN
    U = blkdiag(U, ones(1,N(i)));
end
u = ones(lenN,1);

%% yalmip is used to check the accurancy of the results, 
% yal_x = sdpvar(n, 1);
% 
% yal_con = set(yal_x >= 0) + set(U*yal_x == u);
% yal_obj = (A*yal_x-b)'*(A*yal_x-b);
% 
% yal_sol = solvesdp(yal_con, yal_obj, sdpsettings('verbose', 0, 'solver', 'cplex'));
% yal_sol.x = double(yal_x);
% 
% if yal_sol.problem, error('YALMIP could not solve problem'); end
% 
% obj = double(yal_obj);

%% ADMM

x_ADMM = zeros(n,1);
z_ADMM = zeros(n,1);
lamb_ADMM = zeros(n,1);

num_step = 300;
tau = 1.5;

diff_ADMM = zeros(num_step,1);
obj_ADMM = zeros(num_step,1);

% pre-conpute helping matrix

H = (A'*A + tau/2*eye(n));
h = -2*A'*b;

KKT = [2*H U';
       U   zeros(size(U,1),size(U,1))];

inv_KKT = inv(KKT);

for i = 1:num_step
    
    % Step1:
    temp1 = [-(h + lamb_ADMM - tau*z_ADMM);u]; 
    temp2 = inv_KKT * temp1;
    x_ADMM = temp2(1:n);
    
    % Step 2:
    z_ADMM(:) = max(x_ADMM(:)+1/tau*lamb_ADMM,0);
    
    % Step 3:
    lamb_ADMM = lamb_ADMM + tau * (x_ADMM-z_ADMM);
       
    %diff_ADMM(i) = (yal_sol.x-x_ADMM)'*(yal_sol.x-x_ADMM);
    %obj_ADMM(i) = (A*x_ADMM-b)'*(A*x_ADMM-b) - obj;
    
    diff_ADMM(i) = (x_ADMM-x_true)'*(x_ADMM-x_true);    
    obj_ADMM(i) = (A*x_ADMM-b)'*(A*x_ADMM-b) - obj_true;
    
end


%% problem:
%  min |Ax-b| + |Cx|
%  s.t  Ux = 1 and x >= 0

C = eye(n);%diag(abs(rand(n,1)));

%% yalmip check
% yal_x_re = sdpvar(n, 1);
% 
% yal_con_re = set(yal_x_re >= 0) + set(U*yal_x_re == u);
% yal_obj_re = (A*yal_x_re-b)'*(A*yal_x_re-b) + ( C*yal_x_re)'*(C*yal_x_re);
% 
% yal_sol_re = solvesdp(yal_con_re, yal_obj_re, sdpsettings('verbose', 0, 'solver', 'cplex'));
% yal_sol_re.x = double(yal_x_re);
% 
% if yal_sol_re.problem, error('YALMIP could not solve problem'); end
% 
% obj_re = double(yal_obj_re);

%% FAMA

alpha = 1;
alphaPrev = 1;

x_FAMA = zeros(n,1);
z_FAMA = zeros(n,1);
lamb_FAMA = zeros(n,1);
lamb_FAMA_hat = zeros(n,1);
lamb_FAMA_Prev = zeros(n,1);

diff_FAMA = zeros(num_step,1);
obj_FAMA = zeros(num_step,1);

% pre-conpute helping matrix

H = (A'*A + C*C);
h = -2*A'*b;
tau_FAMA = min(eig(H));

KKT = [2*H U';
       U   zeros(size(U,1),size(U,1))];

inv_KKT = inv(KKT);

for i = 1:num_step
    
    % Step1:
    temp1 = [-(h + lamb_FAMA_hat);u]; 
    temp2 = inv_KKT * temp1;
    x_FAMA = temp2(1:n);
    
    % Step 2:
    z_FAMA(:) = max(x_FAMA(:)+1/tau_FAMA*lamb_FAMA_hat,0);
    
    % Step 3:
    lamb_FAMA_Prev = lamb_FAMA;
    lamb_FAMA = lamb_FAMA + tau_FAMA * (x_FAMA-z_FAMA);
    
    % Step 4: 
    alphaPrev = alpha;
    alpha = (1+sqrt(1+4*alpha^2))/2;
    lamb_FAMA_hat = lamb_FAMA + (alphaPrev-1)/alpha*(lamb_FAMA - lamb_FAMA_Prev);
     
    %diff_FAMA(i) = (yal_sol_re.x-x_FAMA)'*(yal_sol_re.x-x_FAMA);
    %obj_FAMA(i) = (A*x_FAMA-b)'*(A*x_FAMA-b) + (C*x_FAMA)'*(C*x_FAMA) - obj_re;
    
    diff_FAMA(i) = (x_FAMA-x_true)'*(x_FAMA-x_true);
    obj_FAMA(i) = (A*x_FAMA-b)'*(A*x_FAMA-b) + (C*x_FAMA)'*(C*x_FAMA) - obj_true_re;
end


%% plot
% figure(1);
% plot([1:num_step],diff_ADMM,'r');hold on;
% plot([1:num_step],diff_FAMA,'b');
% 
% figure(2);
% semilogy([1:num_step],diff_ADMM,'r');hold on;
% semilogy([1:num_step],diff_FAMA,'b');
% 
% figure(3);
% plot([1:num_step],obj_ADMM,'r');hold on;
% plot([1:num_step],obj_FAMA,'b');
% 
% figure(4);
% semilogy([1:num_step],obj_ADMM,'r');hold on;
% semilogy([1:num_step],obj_FAMA,'b');

figure(1);
semilogy([1:num_step],diff_ADMM,'r');hold on;
semilogy([1:num_step],diff_FAMA,'b');
title('difference |xEst - xTrue|')
legend('ADMM', 'FAMA')

figure(2);
semilogy([1:num_step],obj_ADMM,'r');hold on;
semilogy([1:num_step],obj_FAMA,'b');hold on;
title('objective')
legend('ADMM', 'FAMA')







