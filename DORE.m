function [w, hist] = DORE(funObj,funProj,A,b,w,options)

hist = [];

%% Process Options
if nargin < 4
    options = [];
end

s = svds(A, 1);
A = 0.99*A / s;
b = 0.99*b / s;

[verbose,optTol,progTol,maxIter,suffDec,memory] = ...
    myProcessOptions(options,'verbose',1,'optTol',1e-9,'progTol',1e-9,...
    'maxIter',1000,'suffDec',1e-4,'memory',10);

if verbose
    fprintf('%4s    %8s    %8s    %8s\n','Iter','norm(f,2)','norm(g,2)', 'Step Len');
end

%% Evaluate Initial Point
n = length(w);
x = [w.*(w>0);-w.*(w<0)]; % size of w multiplied by 2

w_prev = w;
Aw = 0;
old_Aw = 0;
very_old_Aw = 0;

step_size = 0;

%% Main loop
for i = 1:maxIter
    very_old_Aw = old_Aw;
    old_Aw = Aw;
    [f, g] = funObj(w);
    if (i > 2 && dot(w-w_prev, w-w_prev)/n <= optTol)
        break;
    end
    Aw = A*w;
    err = b - Aw;
    w_new = w + A'*err;
    w_new = funProj(w_new);

    Aw = A*w_new;
    err = b - Aw;
    if i > 2
        dAw = Aw - old_Aw;
        dp = dot(dAw, dAw);
        if dp > 0
            a1 = dot(dAw, err)/dp;
            Aw_1 = (1 + a1)*Aw - a1*old_Aw;
            w_1 = w_new + a1*(w_new - w);
            err_1 = b - Aw_1;

            dAw = Aw_1 - very_old_Aw;
            dp = dot(dAw, dAw);
            if dp > 0
                a2 = dot(dAw, err_1)/dp;
                w_2 = w_1 + a2*(w_1 - w_prev);
                w_2 = funProj(w_2);
                Aw_2 = A*w_2;
                err_2 = b - Aw_2;

                if (dot(err_2, err_2) / dot(err, err)) < 1
                    w_select = w_2;
                    Aw = Aw_2;
                else
                    w_select = w_new;
                end
            else
                w_select = w_new;
            end
        else
            w_select = w_new;
        end
    else
        w_select = w_new;
    end
    
    fprintf('%4d %5d %5d %8d\n', i, norm(f,2), norm(g,2), norm(w_select - w_prev, 2));
    w_prev = w;
    w = w_select;
end
