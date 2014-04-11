function [w, hist] = lbfgs2(funObj,funProj,w,options)

hist = [];

%% Process Options
if nargin < 4
    options = [];
end

[verbose,optTol,progTol,maxIter,suffDec,memory,corrections] = ...
    myProcessOptions(options,'verbose',1,'optTol',1e-5,'progTol',1e-9,...
    'maxIter',1000,'suffDec',1e-4,'memory',10,'corrections',100);

if verbose
    fprintf('%6s %6s %12s %12s %12s %6s\n','Iter','fEvals','stepLen','fVal','optCond','nnz');
end

%% Evaluate Initial Point
n = length(w);
w = [w.*(w>0);-w.*(w<0)]; % size of w multiplied by 2
[f,g] = nonNegGrad(funObj,w,n);
funEvals = 1;

% Compute working set and check optimality
W = (w~=0) | (g < 0);
optCond = max(abs(g(W)));
if optCond < optTol
    if verbose
        fprintf('First-order optimality satisfied at initial point\n');
    end
    w = w(1:n)-w(n+1:end);
    return;
end

%% Initialize history sets

ym = zeros(n,corrections);
sm = zeros(n,corrections);
rhom = zeros(1,corrections);

%% Main loop of SPG
for i = 1:maxIter
    
    % Compute direction
    if i == 1
        d = -g;
        t = min(1,1/sum(abs(g)));
        old_fvals = repmat(-inf,[memory 1]);
        fr = f;
    else
        y = g-g_old;
        s = w-w_old;
        alpha = (y'*s)/(y'*y);
        if alpha <= 1e-10 || alpha > 1e10
            fprintf('BB update is having some trouble, implement fix!\n');
            pause;
        end
        
        if i <= corrections+1
            
            ym(:,i-1) = y(1:n);
            sm(:,i-1) = s(1:n)-s(n+1:end);
            rhom(i-1) = (ym(:,i-1)'*sm(:,i-1))^-1;
            d = -alpha*g;
            
        else
            
            ym(:,1) = [];
            sm(:,1) = [];
            rhom(1) = [];
            ym = [ym y(1:n)];
            sm = [sm s(1:n)-s(n+1:end)];
            rhom = [rhom (y(1:n)'*(s(1:n)-s(n+1:end)))^-1];
            d = lbfgs(y(1:n), s(1:n)-s(n+1:end), g(1:n), ym, sm, rhom, corrections);
            d = [d;-d];
            
        end
        
        t = 1;
        
        if i-1 <= memory
            old_fvals(i-1) = f;
        else
            old_fvals = [old_fvals(2:end);f];
        end
        fr = max(old_fvals);
    end
    f_old = f;
    g_old = g;
    w_old = w;
    
    % Compute directional derivative, check that we can make progress
    gtd = g'*d;
    if gtd > -progTol
        if verbose
            fprintf('Directional derivative below progTol\n');
        end
        break;
    end
    
    % Compute projected point
    w_new = project(funProj,w+t*d,n);
    [f_new,g_new] = nonNegGrad(funObj,w_new,n);
    funEvals = funEvals+1;
    
    % Line search along projection arc
    while f_new > fr + suffDec*g'*(w_new-w) || ~isLegal(f_new)
        t_old = t;
        
        % Backtracking
        if verbose
            fprintf('Backtracking...\n');
        end
        if ~isLegal(f_new)
            if verbose
                fprintf('Halving Step Size\n');
            end
            t = .5*t;
        else
            t = polyinterp([0 f gtd; t f_new g_new'*d]);
        end
        
        % Adjust if interpolated value near boundary
        if t < t_old*1e-3
            if verbose == 3
                fprintf('Interpolated value too small, Adjusting\n');
            end
            t = t_old*1e-3;
        elseif t > t_old*0.6
            if verbose == 3
                fprintf('Interpolated value too large, Adjusting\n');
            end
            t = t_old*0.6;
        end
        
        % Check whether step has become too small
        if max(abs(t*d)) < progTol
            if verbose
                fprintf('Step too small in line search\n');
            end
            t = 0;
            w_new = w;
            f_new = f;
            g_new = g;
            break;
        end
        
        % Compute projected point
        w_new = project(funProj,w+t*d,n);
        [f_new,g_new] = nonNegGrad(funObj,w_new,n);
        funEvals = funEvals+1;
    end
    
    % Take step
    w = w_new;
    f = f_new;
    g = g_new;
    
    % Compute new working set
    W = (w~=0) | (g < 0);
    
    % Output Log
    if verbose
        fprintf('%6d %6d %8.5e %8.5e %8.5e %6d\n',i,funEvals,t,f,max(abs(g(W))),nnz(w(1:n)-w(n+1:end)));
    end
    
    % Check Optimality
    optCond = max(abs(g(W)));
    if optCond < optTol
        if verbose
            fprintf('First-order optimality below optTol\n');
        end
        break;
    end
    
    % Check for lack of progress
    if max(abs(t*d)) < progTol || abs(f-f_old) < progTol
        if verbose
            fprintf('Progress in parameters or objective below progTol\n');
        end
        break;
    end
    
    % Check for iteration limit
    if funEvals >= maxIter
        if verbose
            fprintf('Function evaluations reached maxIter\n');
        end
        break;
    end
    
    if mod(i,100)==0
        hist = [hist, w(1:n)-w(n+1:end)];
    end
    
end

w = w(1:n)-w(n+1:end);

end

%% Non-negative variable gradient calculation
function [f,g] = nonNegGrad(funObj,w,n)
[f,g] = funObj(w(1:n)-w(n+1:end));
g = [g;-g];
end

function [w] = project(funProj,w,n)
w = funProj(w(1:n)-w(n+1:end));
w = [w.*(w>0);-w.*(w<0)];
end
