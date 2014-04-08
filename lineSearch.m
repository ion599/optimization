function alpha = lineSearch(funObj,x,d)

progTol = 1e-9;
maxIter = 100;

c1 = 0.001;
c2 = 0.9;

alpha = 1;
mu = 0;
nu = inf;

[f0,g0] = funObj(x);
[f,g] = funObj(x+alpha*d);

for i=1:maxIter
    
    if f > f0 + alpha*c1*g0'*d
        nu = alpha;
    elseif g'*d < c2*g0'*d
        mu = alpha;
    else
        break;
    end
    if nu < inf
        alpha_old = alpha;
        alpha = (mu+nu)/2;
    else
        alpha_old = alpha;
        alpha = 2*alpha;
    end
    
    if alpha < progTol
        break;
    end
    if abs(alpha-alpha_old) < progTol
        break;
    end
    [f,g] = funObj(x+alpha*d);
end