function [x1,x2,x3,z1,z2,z3] = initXZ(n,N,x_true)

lenN = length(N);

x1 = rand(n,1);
k=0;
for i=1:lenN
    x1(k+1:k+N(i)) = x1(k+1:k+N(i))/sum(x1(k+1:k+N(i)));
    k = k+N(i);
end
z1 = x2z(x1,N);

% Generate x_init2,3 = routes by importance

x2 = zeros(n,1);
x3 = zeros(n,1);
k=0;
for i=1:lenN
    [~,id] = sort(x_true(k+1:k+N(i)));
    [~,id2] = sort(id);
    x2(k+1:k+N(i)) = id2/sum(id2);
    x3(k+1:k+N(i)) = 10.^(id2-1)/sum(10.^(id2-1));
    k = k+N(i);
end
z2 = x2z(x2,N);
z3 = x2z(x3,N);