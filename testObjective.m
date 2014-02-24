clc; clear all

% dimensions of the problem
n = 10000;
m = n/10;
p = 6;

% Generate synthetic data
A = exprnd(100,m,n);
z = exprnd(10,(n-p),1);
b = exprnd(01,m,1);
N = [floor(.1*n); floor(.2*n); floor(.1*n); floor(.3*n); floor(.1*n)];
N = [N; n-sum(N)];
N2 = zeros(n, n-p);
x0 = zeros(n,1);

ind = 1;
k = 0;
for i=1:length(N)
    N2(ind, ind-k) = 1;
    ind = ind+1;
    for j=2:(N(i)-1)
        N2(ind, ind-k-1) = -1;
        N2(ind, ind-k) = 1;
        ind = ind +1;
    end
    N2(ind, ind-k-1) = -1;
    k = k+1;
    x0(ind) = 1;
    ind = ind+1;
end

tic
obj2 = A*(x0+N2*z)-b;
g2 = N2'*A'*obj2;
obj2 = .5 * (obj2'*obj2);
time1 = toc;

tic
[obj, g] = objective(z,A,N,b);
time2 = toc;

fprintf(['CPU time dense matrix computation: ' num2str(time1) ...
    '\nCPU time efficient matrix computation: ' num2str(time2) ...
    '\ndifference in objectives: ' num2str(norm(obj-obj2)) ...
    '\ndifference in gradients: ' num2str(norm(g - g2)) '\n'])
