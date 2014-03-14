function z = ADMM(z_init,N,A,b,tau,maxIter)

optTol = 1e-5;

% solve min||A(x0+Nz)-b||^2 s.t. x0+Nz>=0 using ADMM
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
C = inv(AN'*AN + tau*eye(n-p));
d = -AN' * (Ax0-b);

% Main loop
lambda = zeros(n-p,1);
u = z_init;
for i = 1:maxIter
    % step 1
    z = C * (d - lambda + tau*u);
    % step 2
    u = project(z + lambda/tau,N);
    % step 3
    lambda = lambda + tau*(z-u);
    
    [~,g] = objective(z,A,N,b,zeros(n,1));
    if max(abs(g)) < optTol
        fprintf('First-order optimality below optTol, Iter=%i\n', i);
        break;
    end
end
z = project(z,N);

end

function w = project(w,N)

k = 0;
for i=1:length(N)
    w(k+1:k+N(i)-1) = PAValgo(w(k+1:k+N(i)-1),ones(N(i)-1,1),0,1);
    k = k+N(i)-1;
end

end