function [obj,d,d2] = computeHist(test,x0,N2,hist,x_true,N,f,A,b)

lenN = length(N);
lenHist = size(hist,2);
obj = zeros(lenHist,1);
d = zeros(lenHist,1);
d2 = zeros(lenHist,1);
for i=1:lenHist
    if strcmp(test,'sparseObjZ') || strcmp(test,'objZ')
        x = x0+N2*hist(:,i);
    else
        x = hist(:,i);
    end
    obj(i) = norm(A*x-b);
    d(i) = norm(x-x_true);
    
    k = 1;
    r = 0;
    for j=1:lenN
        f(j) = max(A(:,k));
        r = r + f(j)*max(abs(x(k:k+N(j)-1)-x_true(k:k+N(j)-1)));
        k = k+N(j);
    end
    r = r/sum(f);
    d2(i) = r;
end