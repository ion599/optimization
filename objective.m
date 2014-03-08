function [obj,g] = objective(z,A,N,b,alpha)

%{
Function that returns the the value of the objective and the gradient
N is given as the # of lines of each of its block
N = [n1; n2; ...; np]
%}

n = size(A,2);
p = length(N);
u = zeros(n,1);
g = zeros(n-p,1);

% Compute u=x0+Nz
ind = 1;
k = 0;
for i=1:p
    u(ind) = z(ind-k); % u = x_0 + Nz
    ind = ind+1;
    for j=2:(N(i)-1)
        u(ind) = z(ind-k)-z(ind-k-1);
        ind = ind+1;
    end
    u(ind) = -z(ind-k-1)+1;
    k = k+1;
    ind = ind+1;
end
obj = A*u-b;
temp = A' * obj + alpha.*u;
obj = .5 * ((obj'*obj) + alpha'*(u.*u));

ind = 1;
k = 0;
for i=1:p
    for j=1:N(i)-1
        g(ind) = temp(ind+k)-temp(ind+k+1);
        ind = ind+1;
    end
    k = k+1;
end