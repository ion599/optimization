function z = ADMM(funObj,z_init,C,d,N,tau,maxIter)

optTol = 1e-5;
progTol = 1e-9;

% solve min||A(x0+Nz)-b||^2 s.t. x0+Nz>=0 using ADMM

lambda = zeros(length(z_init),1);
u = z_init;
[f,~] = funObj(z_init);
for i = 1:maxIter
    % step 1
    z = C * (d - lambda + tau*u);
    % step 2
    u = project(z + lambda/tau,N);
    % step 3
    lambda = lambda + tau*(z-u);
    f_old = f;
    [f,g] = funObj(z);
    if max(abs(g)) < optTol
        fprintf('First order optimality below optTol, Iter=%i\n', i);
        break;
    end
    if abs(f-f_old) < progTol
        fprintf('Progress in parameters or objective below progTol, Iter=%i\n', i);
        break;
    end
    if i>=maxIter-1
        fprintf('Function evaluations reached maxIter\n');
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