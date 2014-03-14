function testADMM(N,A,b)

% solve min||A(x0+Nz)-b||^2 s.t. x0+Nz>=0 using ADMM
[m,n] = size(A);
p = length(N);

% Preconpute matrices
AN = zeros(m,n-p);
Ax0 = zeros(m,1);
for l=1:m
    ind = 1;
    k = 0;
    for i=1:p
        for j=1:N(i)-1
            AN(l,ind) = A(l,ind+k)-A(l,ind+k+1);
            ind = ind+1;
        end
        Ax0(l) = Ax0(l)+A(l,ind+k);
        k = k+1;
    end
end

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
norm(AN-A*N2)
norm(Ax0-A*x0)
assert(norm(AN-A*N2)==0);
assert(norm(Ax0-A*x0)==0);