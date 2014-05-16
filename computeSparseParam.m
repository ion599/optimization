function [x0,N2] = computeSparseParam(n,N)

lenN = length(N);

N2 = sparse(n, n-lenN);
x0 = sparse(n,1);
ind = 1;
k = 0;
for i=1:lenN
    ind = ind + N(i);
    x0(ind-1) = 1;
end