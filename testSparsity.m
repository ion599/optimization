% Generate Some Synthetic Data
clc; clear all

% Preprocessing U to Nf
% for i=1:102720 N(i)=sum(U(i,:)); end

load('data/stevesData.mat')

% Dimensions of the problem
n = size(A,2);
m = size(A,1);

lenN = length(N);
assert(sum(N) == n) % Check that nullspace N accounts for number of routes
%x_true = x;
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

%% COnstruct objectives

funObj = @(z)objective(z,A,N,b,zeros(n,1)); % no penalization (L2)
funObj2 = @(z)objectiveSparse(z,A,N,N2,x0,b,zeros(n,1));

%%

tic
for i=1:100
test = funObj(z_init2);
end
toc

tic
for i=1:100
test = funObj2(z_init2);
end
toc