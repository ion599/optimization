 function [x,f,g] = lbfgs(funObj,x0,m,maxIter,N)

optTol = 1e-5;
progTol = 1e-9;

stop = 0;
n = length(x0);
x = x0;
[f,g] = funObj(x);
ym = zeros(n,m);
sm = zeros(n,m);
rhom = zeros(1,m);
alpha = zeros(1,m);

[f,g_old] = funObj(x0);
x = x0 - g_old;
s = -g_old;
[f,g] = funObj(x);
y = g - g_old;
step = s'*y/(y'*y);

rho = (y'*s)^-1;
ym(:,1) = y;
sm(:,1) = s;
rhom(1) = rho;

for i = 1:(m-1)
    
    %fprintf('Iteration %i\n', i);
    
    s = -step*g;
    x = x + s;
    g_old = g;
    [f,g] = funObj(x);
    y = g - g_old;
    step = s'*y/(y'*y);
    
    rho = (y'*s)^-1;
    ym(:,i+1) = y;
    sm(:,i+1) = s;
    rhom(i+1) = rho;
    
    if max(abs(g)) < optTol
        fprintf('First-order optimality below optTol, Iter=%i\n', i+2);
        stop = 1;
        break;
    end
    
end

for i=1:(maxIter-m-1)
    
    if g'*g < optTol*(1+abs(f))
        if stop ==0
            fprintf('First order optimality below optTol, Iter=%i\n', i+m+1);
        end
        break;
    end
    
    q = g;
    
    for j=2:m
        alpha(m-j+1) = rhom(m-j+1)*sm(:,m-j+1)'*q;
        q = q - alpha(m-j+1)*ym(:,m-j+1);
    end
        
    gamma = (s'*y)/(y'*y);
    r = gamma*q;
    
    for j=1:(m-1)
        beta = rhom(j)*ym(:,j)'*r;
        r = r + sm(:,j)*(alpha(j)-beta);
    end
    
    d = -r;
    
    step = lineSearch(funObj,x,d);
    
    if max(abs(step*d)) < progTol
        fprintf('Step too small in line search, Iter=%i\n', i+m+1);
        break;
    end
    
    x_old = x;
    x = x + step*d;
    
    g_old = g;
    
    [f,g] = funObj(x);
    
    s = x - x_old;
    y = g - g_old;
    rho = (y'*s)^-1;
    ym(:,1) = [];
    sm(:,1) = [];
    rhom(1) = [];
    ym = [ym y];
    sm = [sm s];
    rhom = [rhom rho];
    
    if i==maxIter-m-1
        fprintf('Function evaluations reached maxIter\n');
        break;
    end
    
end

 end

%% Projection

function x = project(x,N)

k=0;
for i=1:length(N)
    x(k+1:k+N(i)-1) = PAValgo(x(k+1:k+N(i)-1),ones(N(i)-1,1),0,1);
    k = k+N(i)-1;
end

end
