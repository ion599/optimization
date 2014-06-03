function [x1,x2,x3,x4,z1,z2,z3,z4] = initXZ(n,N,x_true)

lenN = length(N);

x1 = rand(n,1);
k=0;
for i=1:lenN
    x1(k+1:k+N(i)) = x1(k+1:k+N(i))/sum(x1(k+1:k+N(i)));
    k = k+N(i);
end

% Generate x_init2,3 = routes by importance
% 1: random
% 2: by importance (cheating-ish)
% 3: 10^importance (cheating-ish)
% 4: uniform

x2 = zeros(n,1);
x3 = zeros(n,1);
x4 = zeros(n,1);
k=0;
for i=1:lenN
    [~,id] = sort(x_true(k+1:k+N(i)));
    [~,id2] = sort(id);
    x2(k+1:k+N(i)) = id2/sum(id2);
    x3(k+1:k+N(i)) = 10.^(id2-1)/sum(10.^(id2-1));
    x4(k+1:k+N(i)) = 1/size(id,1);
    k = k+N(i);
end
z1 = x2z(x1,N);
z2 = x2z(x2,N);
z3 = x2z(x3,N);
z4 = x2z(x4,N);