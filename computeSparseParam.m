function [x0,N2] = computeSparseParam(n,N)

lenN = length(N);

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