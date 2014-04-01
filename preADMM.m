function [C,d] = preADMM(N,A,b,alpha,tau)

[m,n] = size(A);
p = length(N);

fprintf('Compute A X N ...');

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

fprintf('Compute AN^T X AN ...');

C = AN'*AN;

fprintf('Continue ...');

d = -AN' * (Ax0-b);
ind = 1;
k = 0;
for i=1:p
    if N(i) > 1
        for j=1:N(i)-2
            C(ind,ind) = C(ind,ind) + alpha(ind+k) + alpha(ind+k+1) + tau;
            C(ind,ind+1) = C(ind,ind+1) -alpha(ind+k+1);
            C(ind+1,ind) = C(ind+1,ind) -alpha(ind+k+1);
            ind = ind+1;
        end
        C(ind,ind) = C(ind,ind) + alpha(ind+k)+alpha(ind+k+1)+tau;
        d(ind) = d(ind)+alpha(ind+k+1);
        ind = ind+1;
    end
    k = k+1;
end
fprintf('Start inversion of matrix in preADMM ...');
C = inv(C);