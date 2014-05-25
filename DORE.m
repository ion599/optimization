function [w, hist, cv_error] = DORE(funObj,funProj,funApply_in,funApply_T_in,funCalcCVError,b,w,options)

hist = [];
cv_error = [funCalcCVError(w)];

%% Process Options
if nargin < 4
    options = [];
end

AT_A = @(x) funApply_T_in(funApply_in(x));

s = eigs(AT_A, size(w,1), 1)
s = sqrt(s / 0.99);

funApply = @(x) funApply_in(x)/s;
funApply_T = @(x) funApply_T_in(x)/s;

b = b/s;

[verbose,optTol,progTol,maxIter,suffDec,memory] = ...
    myProcessOptions(options,'verbose',1,'optTol',1e-9,'progTol',1e-9,...
    'maxIter',1000,'suffDec',1e-4,'memory',10);

if verbose
    fprintf('%4s    %8s    %8s     %8s O P\n','Iter','norm(f,2)','norm(g,2)', 'Step Len');
end

%% Evaluate Initial Point
n = length(w);
x = [w.*(w>0);-w.*(w<0)]; % size of w multiplied by 2

w_prev = w;
w_two_prev = w;
old_Aw = 0;
very_old_Aw = 0;

step_size = 0;
select_flag = 0;

Aw = funApply(w);
err = b - Aw;

%% Main loop
for i = 1:maxIter
    if (i > 2 && max(abs(w-w_prev)) <= progTol)
        break;
    end
    projection_needed = 0;
    w_two_prev = w_prev;
    w_prev = w;
    very_old_Aw = old_Aw;
    old_Aw = Aw;
    f = 0.5*(dot(err,err));
    g = funApply_T(err);
    w = w + g;
    w = funProj(w);

    Aw = funApply(w);
    err = b - Aw;
    if i > 2
        dAw = Aw - old_Aw;
        dp = dot(dAw, dAw);
        if dp > 0
            a1 = dot(dAw, err)/dp;
            Aw_1 = (1 + a1)*Aw - a1*old_Aw;
            w_1 = (1 + a1)*w - a1*w_prev;
            err_1 = b - Aw_1;

            dAw = Aw_1 - very_old_Aw;
            dp = dot(dAw, dAw);
            if dp > 0
                a2 = dot(dAw, err_1)/dp;
                w_2 = w_1 + a2*(w_1 - w_two_prev);
                w_2_new = funProj(w_2);
                if (w_2 == w_2_new)
                    Aw_2 = Aw_1 + a2*(Aw_1 - very_old_Aw);
                else
                    projection_needed = 1;
                    Aw_2 = funApply(w_2_new);
                end
                err_2 = b - Aw_2;

                if (dot(err_2, err_2) / dot(err, err)) < 1
                    select_flag = 1;
                    w = w_2_new;
                    err = err_2;
                    Aw = Aw_2;
                end
            end
        end
    end

    if mod(i,10)==0
        cv_error = [cv_error, funCalcCVError(w)];
    end
    
    if mod(i,10)==0
        hist = [hist, w];
    end

    fprintf('%4d %5d %5d %8d %d %d\n', i, (s*s)*norm(f,2), (s*s)*norm(g,2), norm(w - w_prev, 2), select_flag, projection_needed);
    select_flag = 0;
end
