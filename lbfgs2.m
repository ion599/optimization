function [w] = lbfgs2(funObj,w,N,m,options)

%% Process Options
if nargin < 4
    options = [];
end

[verbose,optTol,progTol,maxIter,suffDec,memory] = ...
    myProcessOptions(options,'verbose',1,'optTol',1e-5,'progTol',1e-9,...
    'maxIter',1000,'suffDec',1e-4,'memory',10);

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

ym = zeros(n,m);
sm = zeros(n,m);
rhom = zeros(1,m);
alpha = zeros(1,m);
stop = 0;

%% Main loop of SPG
for i = 1:m+1
    
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
        ym(:,i-1) = y(1:n);
        sm(:,i-1) = s(1:n)-s(n+1:end);
        rhom(i-1) = (ym(:,i-1)'*sm(:,i-1))^-1;
        d = -alpha*g;
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
        stop = 1;
        break;
    end
    
    % Compute projected point
    w_new = project(w+t*d,n,N);
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
            stop = 1;
            break;
        end
        
        % Compute projected point
        w_new = project(w+t*d,n,N);
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
        stop = 1;
        break;
    end
    
    % Check for lack of progress
    if max(abs(t*d)) < progTol || abs(f-f_old) < progTol
        if verbose
            fprintf('Progress in parameters or objective below progTol\n');
        end
        stop = 1;
        break;
    end
    
    % Check for iteration limit
    if funEvals >= maxIter
        if verbose
            fprintf('Function evaluations reached maxIter\n');
        end
        stop = 1;
        break;
    end
end

% Main loop of l-BFGS

if stop == 0
    
    for i=m+2:maxIter
        
        % Compute direction
        y = g(1:n)-g_old(1:n);
        s = (w(1:n)-w(n+1:end))-(w_old(1:n)-w_old(n+1:end));
        q = g(1:n);
        
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
        d = [d;-d];
        t = 1;
        
        if i-1 <= memory
            old_fvals(i-1) = f;
        else
            old_fvals = [old_fvals(2:end);f];
        end
        fr = max(old_fvals);
        
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
        w_new = project(w+t*d,n,N);
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
            w_new = project(w+t*d,n,N);
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
        
    end
end

end

%% Non-negative variable gradient calculation
function [f,g,H] = nonNegGrad(funObj,w,n)

[f,g] = funObj(w(1:n)-w(n+1:end));

g = [g;-g];
end

function [w] = project(w,n,N)
w = w(1:n)-w(n+1:end);

k=0;
for i=1:length(N)
    w(k+1:k+N(i)-1) = PAValgo(w(k+1:k+N(i)-1),ones(N(i)-1,1),0,1);
    k = k+N(i)-1;
end

w = [w.*(w>0);-w.*(w<0)];
end
