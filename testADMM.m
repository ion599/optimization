function testADMM(N,A,b,alpha,tau)

% check precomputations in implementation of ADMM
[m,n] = size(A);
p = length(N);

% Precompute matrices
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
C = AN'*AN;
d = -AN' * (Ax0-b);
ind = 1;
k = 0;
for i=1:p
    for j=1:N(i)-2
        C(ind,ind) = C(ind,ind) + alpha(ind+k)+alpha(ind+k+1)+tau;
        C(ind,ind+1) = C(ind,ind+1) -alpha(ind+k+1);
        C(ind+1,ind) = C(ind+1,ind) -alpha(ind+k+1);
        ind = ind+1;
    end
    C(ind,ind) = C(ind,ind) + alpha(ind+k)+alpha(ind+k+1)+tau;
    d(ind) = d(ind)+alpha(ind+k+1);
    ind = ind+1;
    k = k+1;
end
invC = inv(C);


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
N2'*diag(alpha)*N2
C2 = (A*N2)'*A*N2 + N2'*diag(alpha)*N2 + tau*eye(n-p);
invC2 = inv(C2);
d2 = -(A*N2)'*(A*x0-b)-N2'*diag(alpha)*x0;

norm(C-C2)
norm(d-d2)