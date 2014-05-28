function [obj,d,d2,time] = computeHist(test,x0,N2,hist,timehist,initx,x_true,N,f,A,b)

lenN = length(N);
lenHist = size(hist,2);
obj = zeros(lenHist+1,1);
d = zeros(lenHist+1,1);
d2 = zeros(lenHist+1,1);

obj(1) = norm(A*initx-b);
d(1) = norm(initx-x_true);
k = 1;
r = 0;
for j=1:lenN
    r = r + f(j)*max(abs(initx(k:k+N(j)-1)-x_true(k:k+N(j)-1)));
    k = k+N(j);
end
r = r/sum(f);
d2(1) = r;

for i=1:lenHist
    if strcmp(test,'z') || strcmp(test,'dense-z')
        x = x0+N2*hist(:,i);
    else
        x = hist(:,i);
    end
    obj(i+1) = norm(A*x-b);
    d(i+1) = norm(x-x_true);
    
    k = 1;
    r = 0;
    for j=1:lenN
        r = r + f(j)*max(abs(x(k:k+N(j)-1)-x_true(k:k+N(j)-1)));
        k = k+N(j);
    end
    r = r/sum(f);
    d2(i+1) = r;
end
time = [0; cumsum(timehist)'];
end
